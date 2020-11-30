"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import logging
import os
import sys
import pdb
import pickle
from pathlib import Path
from timeit import default_timer as timer
import time
import datetime

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data.volume_sampler import VolumeSampler

class DistributedMixin(object):

    def presetup(self, args):
        cwd = os.getcwd()
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])

        if world_size <= 8:
            tcp_address = "localhost"
        else:
            tcp_address_file = f"{cwd}/tcp_address.txt"
            print(f"({rank}) Waiting for {tcp_address_file} to exist")
            sys.stdout.flush()
            while not os.path.exists(tcp_address_file):
                time.sleep(30)
                print(f"({rank}) {datetime.datetime.now()} Not found yet")
                sys.stdout.flush()
            with open(tcp_address_file,'r') as f:
                tcp_address = f.read().rstrip()
        init_method = f"tcp://{tcp_address}:23457"

        if rank == 0:
            print(f"({rank}) Master address: {init_method}")

        print(f"({rank}) Waiting on init_process_group ...")
        sys.stderr.flush()

        dist.init_process_group(
            backend='nccl',
            init_method=init_method,
            world_size=world_size,
            rank=rank)
        print(f"({rank}) process group setup, syncing")
        sys.stdout.flush()
        dist.barrier()
        print(f"({rank}) Synced")
        sys.stdout.flush()

        args.rank = rank
        args.world_size = world_size
        #pdb.set_trace()
        #torch.cuda.set_device(args.rank)
        args.gpu = args.rank % 8 # gpus per machine
        sys.stdout.flush()
        torch.cuda.set_device(args.gpu)

        ## Only save to disk on the master process
        if rank != 0:
            args.save_info = False
            args.save_model = False

        self.progress_points = []

    def barrier(self):
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        dist.barrier(self.dist_group)
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()

    def model_setup(self, args):
        """ The model must be wrapped in a DistributedDataParallel instance
        """
        super().model_setup(args)

        logging.info("creating DPP instance")
        if args.apex_distributed:
            logging.info("Using Apex DPP")
        else:
            logging.info("Using Pytorch DPP")
        sys.stdout.flush()

        self.model = self.distribute_model_object(self.model)

        grank = torch.distributed.get_rank()
        gworld_size = torch.distributed.get_world_size()
        logging.info(f"g rank {grank}, world size: {gworld_size}")

        self.dist_group = torch.distributed.new_group(ranks=list(range(args.world_size))) 
        grank = torch.distributed.get_rank(self.dist_group)
        gworld_size = torch.distributed.get_world_size(self.dist_group)
        logging.debug(f"dist_group rank {grank}, world size: {gworld_size}")

    def distribute_model_object(self, mdl):
        args = self.args
        if args.apex_distributed:
            #TODO: try delay_allreduce=True
            from apex.parallel import DistributedDataParallel as ApexDDP
            mdl = ApexDDP(mdl, delay_allreduce=True)
        else:
            mdl = DDP(mdl, device_ids=[args.gpu], output_device=args.gpu)
        return mdl

    def loader_setup(self, args):
        """ A distributed sampler has to be used
        """
        train_sampler = DistributedSampler(self.train_data)

        self.train_loader = DataLoader(
            self.train_data, batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers, pin_memory=args.pin_memory, drop_last=True)

        logging.debug("Determining batches ...")
        self.nbatches = len(train_sampler)//args.batch_size
        logging.info("Distributed train Loader created, batches: {}".format(self.nbatches))

        self.dev_loader = DataLoader(
            self.dev_data, batch_size=args.batch_size,
            sampler=VolumeSampler(self.dev_data),
            num_workers=args.workers, pin_memory=args.pin_memory, drop_last=False)

        ### No need to use a distributed sampler for the display loader
        self.display_loader = DataLoader(
            dataset=self.display_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

    def start_of_batch_hook(self, progress, logging_epoch):
        super().start_of_batch_hook(progress, logging_epoch)
        # Tell monitoring process that we are still alive
        if logging_epoch:
            (self.exp_dir / "last_alive").touch()

    def start_of_test_batch_hook(self, progress, logging_epoch):
        """ We add syncronization here just to keep all the processes in lock-step.
            If you don't sync often enough NCCL can hit a timeout and essentially freeze
        """
        super().start_of_test_batch_hook(progress, logging_epoch)
        #logging.info(f"{self.args.rank} progress {progress} rounded: {round(progress % 1.0, 4)}")
        progress = progress % 1.0
        if round(progress, 4) == 0.0:
            # Points [0.1, 0.2, ..., 1.0]
            # these are poped off the list as the increments are reached.
            interval = 20
            self.progress_points = [(1.0/interval)*i for i in range(0, interval+1)] 

        self.sync(progress)
        
    def end_of_test_epoch_hook(self):
        super().end_of_test_epoch_hook()
        self.sync(progress=1.0)

    def sync(self, progress):
        while len(self.progress_points) > 0 and progress >= self.progress_points[0]:
            point = self.progress_points.pop(0)
            if self.args.use_barriers:
                logging.debug(f"({self.args.rank}) ({progress*100.0:2.2f}%) barrier syncing")
                self.barrier()
                logging.debug("synced")
                sys.stdout.flush()
                sys.stderr.flush()

    def start_of_epoch_hook(self, epoch):
        super().start_of_epoch_hook(epoch)
        if self.args.use_barriers:
            logging.debug("(Start of epoch) waiting at dist barrier for all tasks")
            self.barrier()
            logging.debug("(Start of epoch) barrier past")
            sys.stdout.flush()

    def end_of_epoch_hook(self, epoch):
        super().end_of_epoch_hook(epoch)
        if self.args.use_barriers:
            logging.debug("(End of epoch) waiting at dist barrier for all tasks")
            self.barrier()
            logging.debug("(End of epoch) barrier past")
            sys.stdout.flush()

    def save_model(self):
        """ Saving can take a while so we need barriers to keep the processes in lock-step"""
        if self.args.use_barriers:
            logging.debug("waiting at dist barrier for all tasks (pre-save)")
            self.barrier()
        super().save_model()
        if self.args.use_barriers:
            logging.debug("waiting at dist barrier for all tasks (post-save)")
            self.barrier()

    def stats(self, epoch, loader, setname):
        """ We use All-reduce to sync the losses across all the processes """
        losses = self.compute_stats(epoch, loader, setname)
        logging.debug(f'Epoch: {epoch}. process-local losses: {losses}')
        sys.stdout.flush()

        losses_tensor = torch.zeros(len(losses)).to(self.device)
        all_dataset_size = loader.sampler.total_size
        all_local_size = loader.sampler.num_samples

        for i, k in enumerate(sorted(losses.keys())):
            local_size = all_local_size
            if k in loader.sampler.system_acquisition_local_count:
                local_size = loader.sampler.system_acquisition_local_count[k]

            losses_tensor[i] = losses[k]*local_size

        dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
        logging.debug(f'({self.args.rank}) Loss all-reduce complete')
        sys.stdout.flush()

        for i, k in enumerate(sorted(losses.keys())):
            dataset_size = all_dataset_size
            if k in loader.sampler.system_acquisition_total_count:
                dataset_size = loader.sampler.system_acquisition_total_count[k]

            losses[k] = losses_tensor[i].item()/dataset_size # Average it

        self.test_loss_hook(losses)
        logging.info(f'Epoch: {epoch}. losses: {losses}')
        #print(f'Epoch: {epoch}. losses: {losses}')
        sys.stdout.flush()
        return losses["NMSE"]
