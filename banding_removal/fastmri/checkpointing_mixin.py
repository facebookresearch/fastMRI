"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import signal
import sys
from subprocess import call, Popen, PIPE
import torch
import logging
import pickle
import pdb
from pathlib import Path
import time

def remove_prefix_from_model(model_state_dict):
    prefix = next(iter(model_state_dict.keys())).split('.', 1)[0]
    if all(k.startswith(prefix) for k in model_state_dict.keys()):
        logging.info(f"Removing model prefix '{prefix}' from checkpoint")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
                name = k.split('.', 1)[1] #k[7:] # remove 'module.'
                new_state_dict[name] = v
        return new_state_dict
    else:
        return model_state_dict

class CheckpointingMixin(object):
    """
        Follows Slurm checkpointing advice in:
        https://fb.quip.com/Ov7NA1FD3mmx
    """

    def initial_setup(self, args):
        super().initial_setup(args)

        self.runinfo_path = self.exp_dir / "current_runinfo.pkl"
        self.model_path = self.exp_dir / 'current_model.mdl'

        if args.checkpoint_type == "resume":
            logging.info(f"RESUMING from checkpoint {args.checkpoint}")
            self.checkpoint = torch.load(args.checkpoint)
        if args.checkpoint_type == "restart":
            logging.info(f"RESTARTING from checkpoint {args.checkpoint}")
            self.checkpoint = torch.load(args.checkpoint)
        else:
            # Check for partial job as well
            if self.model_path.exists() and args.auto_requeue:
                logging.info("")
                logging.info(f"FOUND model file in working directory, resuming ...")
                logging.info("")
                self.checkpoint = torch.load(self.model_path)
                args.checkpoint_type = "resume"


        # Set up signal handler to detect Slurm signals
        if args.auto_requeue: #and 'SLURM_NODEID' in os.environ:
            signal.signal(signal.SIGUSR1, self.requeueHandler)
            signal.signal(signal.SIGTERM, self.termHandler)
            logging.info('Signal handler installed for automatic requeuing')
            self.MAIN_PID = os.getpid()
            self.HALT_filename = 'HALT'
            self.SIGNAL_RECEIVED = False
            '''Halt file is used as a sign of job completionself.
            Make sure no HALT file left from previous runs.
            '''
            if os.path.isfile(self.HALT_filename):
                os.remove(self.HALT_filename)

    def model_setup(self, args):
        super().model_setup(args)
        if args.checkpoint_type == "resume" or args.checkpoint_type == "restart":
            model_state_dict = self.checkpoint['model']
            #model_state_dict = remove_prefix_from_model(self.checkpoint['model'])
            # Should be compatible with models that do and do not use dataparallel
            try:
                self.model.load_state_dict(model_state_dict, strict=True)
            except:
                for _ in range(len(model_state_dict)):
                    k, v = model_state_dict.popitem(False)
                    idx = k.find('module.')
                    if idx >= 0:
                        newkey = k[:idx] + k[idx+len('module.'):]
                        model_state_dict[newkey] = v
                    else:
                        model_state_dict[k] = v
                self.model.load_state_dict(model_state_dict, strict=True)


    def optimizer_setup(self, args):
        super().optimizer_setup(args)
        if args.checkpoint_type == "resume":
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])

    def runinfo_setup(self, args):
        super().runinfo_setup(args)
        if args.checkpoint_type == "resume":
            self.runinfo = self.checkpoint["runinfo"]
            # at_epoch is last epoch completed (unless at beginning), so start next one.
            if len(self.runinfo["dev_losses"]) > 0:
                self.runinfo["at_epoch"] += 1

            del self.checkpoint # keep memory usage down
        else:
            # Save before first epoch so we can resume if preempted early
            self.save_info()
            self.save_model()

    def end_of_epoch_hook(self, epoch):
        super().end_of_epoch_hook(epoch)
        self.save_info()
        self.save_model()

    def save_info(self):
        if not self.args.save_info or self.args.rank != 0:
            return

        with open(self.runinfo_path, 'wb') as output:
            pickle.dump(self.runinfo, output)
        logging.info(f"Saved runinfo {self.runinfo_path.resolve()}")

    def save_model(self):
        if not self.args.save_model or self.args.rank != 0:
            return

        logging.debug("Saving model ...")
        # Avoid corruption if we crash during save by saving to a tmp and then moving
        tmp_model_path = self.model_path.with_suffix(".mdl.tmp")
        torch.save(self.serialize(), f = tmp_model_path)
        tmp_model_path.replace(self.model_path)
        logging.info(f"Saved model {self.model_path.resolve()}")

    def termHandler(self, signum, frame):
        """
         Slurm preemption sends a SIGTERM before the SIGUSR1, to give you a warning
         that the process is going to be preempted. This needs to be caught, otherwise
         the process will exit early.
        """
        print("SIGTERM caught and ignored", flush=True)

    def requeueHandler(self, signum, frame):
        """
            A USR1 signal is sent by slurm if the timelimit of the job is reached
            or if the job is about to be preempted
        """
        args = self.args
        print('Signal received', signum, time.time(), flush=True)
        self.SIGNAL_RECEIVED = True

        if os.path.isfile(self.HALT_filename):
            print('Job is done, exiting', flush=True)
            exit(0)

    def trigger_job_requeue(self):
        """ Submit a new job to resume from checkpoint.
            No need to checkpoint model or runinfo here since we don't
            currently support resuming from specific batches/iterations
            (only from epochs)
        """
        if self.args.rank == 0:
            ### This ensures that only the main processes (rank 0) requeues the job 
            print('Time is up, back to SLURM queue', flush=True)
            command = 'scontrol requeue ' + os.environ['SLURM_JOB_ID']
            print(command)
            if os.system(command):
                raise RuntimeError('requeue failed')
            print('Job successfully requeued', flush=True)
        else:
            print(f'Non-primary process {os.getpid()} waiting for requeue', flush=True)

        if self.args.is_distributed:
            self.barrier()
            logging.info(f"requeue synced")
        exit(0)

    def start_of_batch_hook(self, progress, logging_epoch):
        if self.args.auto_requeue and self.SIGNAL_RECEIVED:
            self.trigger_job_requeue()
        super().start_of_batch_hook(progress, logging_epoch)

    def start_of_test_batch_hook(self, progress, logging_epoch):
        if self.args.auto_requeue and self.SIGNAL_RECEIVED:
            self.trigger_job_requeue()
        super().start_of_test_batch_hook(progress, logging_epoch)
