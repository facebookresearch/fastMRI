"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

#from __future__ import print_function
import argparse
import pickle
import os
import pdb
import signal
import warnings
import time
from time import sleep

import sys
sys.path.append(sys.path[0] + "/..")
__package__ = "fastmri"

import multiprocessing
import subprocess
from .args import Args

def work(info):
    rank, ntasks, args = info
    # This must be imported here, in the child process, not globally.
    from .run import run as run_task
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(ntasks)
    run_task(args)

def run(args=None, ntasks=None):
    if args is None:
        args = Args().parse_args()
    if isinstance(args, dict):
        args = Args(**args).parse_args()

    # Some automatic ntask settings code
    if ntasks is None:
        try:
            devices = os.environ['CUDA_VISIBLE_DEVICES']
            ntasks = len(devices.split(','))
        except:
            try:
                ntasks = int(os.popen("nvidia-smi -L | wc -l").read())
            except:
                ntasks = 2

    args.is_distributed = True
    # Temp ignore for bug in pytorch dataloader, it leaks semaphores
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning,ignore::UserWarning'

    # Make this process the head of a process group.
    os.setpgrp()

    # Most important line in this file. CUDA fails horribly if we use the default fork
    multiprocessing.set_start_method('forkserver')

    processses = []
    for i in range(ntasks):
        p = multiprocessing.Process(target=work, args=[(i, ntasks, args)])
        p.start()

        if args.strace:
            # Addtional Monitoring process
            subprocess.Popen(["strace", "-tt" , 
                "-o", f"{args.exp_dir}/strace_{i}.log", 
                "-e", "trace=write", "-s256", 
                "-p", f"{p.pid}"])

        processses.append(p)

    def terminate(signum, frame):
        # Try terminate first
        print("terminating child processes")
        sys.stdout.flush()
        for i, p in enumerate(processses):
            if p.is_alive():
                p.terminate()

        # Wait a little
        for i in range(20):
            if any(p.is_alive() for p in processses):
                sleep(0.1)

        ## If they are still alive after that kill -9 them
        for i, p in enumerate(processses):
            if p.is_alive():
                print(f"Sending SIGKILL to process {i}")
                sys.stdout.flush()
                os.kill(p.pid, signal.SIGKILL)

        print("exiting")
        sys.stdout.flush()
        sys.exit(0)


    if args.auto_requeue:
        def forward_usr1_signal(signum, frame):
            print(f"Received USR1 signal in spawn_dist", flush=True)
            for i, p in enumerate(processses):
                if p.is_alive():
                    os.kill(p.pid, signal.SIGUSR1)
        
        def forward_term_signal(signum, frame):
            print(f"Received SIGTERM signal in spawn_dist", flush=True)
            for i, p in enumerate(processses):
                if p.is_alive():
                    os.kill(p.pid, signal.SIGTERM)

        # For requeing we need to ignore SIGTERM, and forward USR1
        signal.signal(signal.SIGUSR1, forward_usr1_signal)
        signal.signal(signal.SIGTERM, forward_term_signal)
        signal.signal(signal.SIGINT, terminate)

    else:
        signal.signal(signal.SIGINT, terminate)
        signal.signal(signal.SIGTERM, terminate)

    while True:
        sleep(0.5)
        if any(not p.is_alive() for p in processses):
            print("Detected an exited process, so exiting main")
            terminate(None, None)

    print("DONE")

if __name__ == "__main__":
    run()
