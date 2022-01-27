"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import importlib
import sys
import multiprocessing
import os
import logging

sys.path.append(sys.path[0] + "/..")
__package__ = "fastmri"

def run(args=None):
    from .args import Args

    try:
        if args is None:
            args = Args().parse_args()
        if isinstance(args, dict):
            args = Args(**args).parse_args()

        module_name, class_name = args.trainer_class.rsplit(".", 1)
        mdl = importlib.import_module(module_name, "fastmri")
        TrainerClass = getattr(mdl, class_name)

        trainer = TrainerClass(args)
        if args.eval:
            trainer.eval()
        else:
            trainer.train()

    except KeyboardInterrupt:
        pass # Hide traceback
    except Exception as e:
        logging.exception("Uncaught exception")
        raise


if __name__ == "__main__":
    run()
