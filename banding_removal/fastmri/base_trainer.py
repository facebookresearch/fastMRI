import logging
import os
import sys
import pdb
import pickle
import math
import datetime
import random
import functools
from collections import OrderedDict, namedtuple
from timeit import default_timer as timer

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from fastmri.common.utils import CallbackDataset

from . import model
from . import optimizer
from .data.mri_data import SliceData

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.exp_dir = args.exp_dir
        self.exp_dir.mkdir(exist_ok=True, parents=True)

        self.presetup(args)
        self.initial_setup(args)
        self.transform_setup(args)
        self.data_setup(args)
        self.loader_setup(args)
        self.model_setup(args)
        self.parameter_groups_setup(args)
        self.optimizer_setup(args)
        self.loss_setup(args)
        self.runinfo_setup(args)

    def presetup(self, args):
        pass

    def transform_setup(self, args):
        pass

    def initial_setup(self, args):
        ############
        logging.info(f"run pid: {os.getpid()} parent: {os.getppid()}")
        logging.info("#########")
        logging.info(args.__dict__)
        logging.info(f"Rank: {args.rank} World_size: {args.world_size}, Run {args.run_name}")

        args.cuda = torch.cuda.is_available()
        logging.info(f"Pytorch version: {torch.__version__}")
        logging.info("Using CUDA: {} CUDA AVAIL: {} #DEVICES: {} VERSION: {}".format(
            args.cuda, torch.cuda.is_available(), torch.cuda.device_count(),
            torch.version.cuda))
        if not args.cuda:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
            cudnn.benchmark = True
            cudnn.enabled = True

        random.seed(args.seed) # The seed needs to be constant between processes.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    def data_setup(self, args):
        logging.info("Creating data objects")

        self.train_data = SliceData(
            root=self.args.data_path / f'{args.challenge}_train',
            transform=self.train_transform,
            args=self.args,
        )

        val_args = self.args
        val_args.max_kspace_width = None
        val_args.min_kspace_width = None
        val_args.max_kspace_height = None
        val_args.min_kspace_height = None
        val_args.start_slice = None
        val_args.end_slice = None
        val_args.acquisition_types = None
        val_args.acquisition_systems = None
        self.dev_data = SliceData(
            root=self.args.data_path / f'{args.challenge}_val',
            transform=self.dev_transform,
            args=val_args,
        )

        if self.args.resize_type == "none":
            # Only display the first size in the dataset.
            display_size, indices = list(self.dev_data.slice_indices_by_size.items())[0]
            self.display_data = CallbackDataset(
                    callback=functools.partial(data_for_index, self.dev_data, indices),
                    start=0,
                    end=len(indices),
                    increment=len(indices) // args.display_count)
        else:
            ndev = len(self.dev_data)
            indices = range(0, ndev)
            self.display_data = CallbackDataset(
                    callback=functools.partial(data_for_index, self.dev_data, indices),
                    start=0,
                    end=ndev,
                    increment=args.display_count)

    def loader_setup(self, args):
        logging.info("Creating samplers ...")
        train_sampler = RandomSampler(self.train_data)
        dev_sampler = RandomSampler(self.dev_data)


        logging.info("Creating data loaders ...")
        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            sampler=train_sampler,
        )
        self.dev_loader = DataLoader(
            dataset=self.dev_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            sampler=dev_sampler,
        )

        self.display_loader = DataLoader(
            dataset=self.display_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

        logging.debug("Determining batches ...")
        self.nbatches = len(self.train_loader)
        logging.info("Train Loader created, batches: {}".format(self.nbatches))

    def model_setup(self, args):
        if not args.gan:
            self.model = model.load(args.architecture, args)
            self.model.to(self.device)

    def parameter_groups_setup(self, args):
        self.parameter_groups = self.model.parameters()

    def optimizer_setup(self, args):
        self.optimizer = optimizer.load(self.parameter_groups, args)

    def loss_setup(self, args):
        pass

    def runinfo_setup(self, args):
        self.runinfo = {}
        self.runinfo["args"] = args
        self.runinfo["at_epoch"] = 0
        self.runinfo["seed"] = args.seed
        self.runinfo["best_dev_loss"] = 1e9
        self.runinfo["epoch"] = []
        self.runinfo["train_losses"] = []
        self.runinfo["train_fnames"] = []
        self.runinfo["dev_losses"] = []

    def serialize(self):
        return {
	    'runinfo': self.runinfo,
	    'epoch': self.runinfo["at_epoch"],
	    'args': self.args,
	    'model': self.model.state_dict(),
	    'optimizer': self.optimizer.state_dict(),
	    'best_dev_loss': self.runinfo["best_dev_loss"],
	    'exp_dir': self.exp_dir
	}

    ##################################################################################
    def train(self):
        beginning = timer()
        args = self.args
        for epoch in range(self.runinfo["at_epoch"], args.epochs):
            self.runinfo["at_epoch"] = epoch
            logging.info("Starting epoch {}".format(epoch))

            if self.args.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            seed = self.runinfo["seed"] + 1031*epoch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            self.start_of_epoch_hook(epoch)

            start = timer()
            self.run(epoch)
            sys.stdout.flush()
            end = timer()

            logging.info(f"TRAIN Epoch took: {datetime.timedelta(seconds=end-start)}")
            logging.info("")

            self.end_of_epoch_hook(epoch)

        end = timer()
        logging.info(f"Run took: {datetime.timedelta(seconds=end-beginning)}")
        logging.info("FINISHED")

        self.postrun()

    def backwards(self, loss):
        loss.backward()

    def start_of_epoch_hook(self, epoch):
        if self.args.eval_at_start:
            dev_loss = self.stats(epoch, self.dev_loader, "Dev")
            logging.info(f"EVAL Loss: {dev_loss}")

    def end_of_epoch_hook(self, epoch):
        self.end_of_epoch_eval_hook(epoch)

    def end_of_epoch_eval_hook(self, epoch):
        logging.info("Starting evaluation")
        start = timer()
        dev_loss = self.stats(epoch, self.dev_loader, "Dev")
        end = timer()
        logging.info(f"EVAL Loss: {dev_loss} time: {datetime.timedelta(seconds=end-start)}")

        if math.isnan(dev_loss) or math.isinf(dev_loss):
            logging.info("NaN or Inf detected, ending training")
            self.postrun()
            sys.exit(1)

        is_new_best = dev_loss < self.runinfo["best_dev_loss"]
        self.runinfo["best_dev_loss"] = min(self.runinfo["best_dev_loss"], dev_loss)

    def postrun(self):
        pass

    def preprocess_data_tensor(self, t):
        """ Override to cast """
        return t.to(self.device, non_blocking=True)

    def preprocess_data(self, tensors):
        """ Called on a batch returned by a dataloader, to do things like
            .cuda() and .half etc., Should be idempotent.
            Calls preprocess_data_tensor on tensors
        """
        if hasattr(tensors, '_fields'):
            # Skip if already processed to a named tuple
            return tensors
        elif isinstance(tensors, dict):
            ts = OrderedDict()
            for k,t in tensors.items():
                if isinstance(t, torch.Tensor):
                    ts[k] = self.preprocess_data_tensor(t)
                else:
                    ts[k] = t
            # Convert to a named tuple
            BatchTuple = namedtuple('Batch', ts.keys())
            return BatchTuple(**ts)
        elif isinstance(tensors, torch.Tensor):
            return self.preprocess_data_tensor(tensors)
        else:
            return tensors


# data_for_index must be a non-local function so it can be pickled correctly.
def data_for_index(dev_data, indices, x):
    return dev_data[indices[x]]
