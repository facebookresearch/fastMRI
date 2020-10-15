"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from argparse import ArgumentParser

import fastmri
import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.pl_modules import VarNetModule, configure_checkpoint


def cli_main(args):
    """Main training routine."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    pl.seed_everything(args.seed)
    model = VarNetModule(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING OR TEST
    # ------------------------
    if args.mode == "train":
        trainer.fit(model)
    elif args.mode == "test":
        assert args.resume_from_checkpoint is not None
        outputs = trainer.test(model)
        fastmri.save_reconstructions(outputs, args.default_root_dir / "reconstructions")
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    path_config = pathlib.Path.cwd() / ".." / ".." / "fastmri_dirs.yaml"
    knee_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    parent_parser = ArgumentParser(add_help=False)

    parser = VarNetModule.add_model_specific_args(parent_parser)
    parser = pl.Trainer.add_argparse_args(parser)

    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    checkpoint_callback, resume_from_checkpoint = configure_checkpoint(default_root_dir)

    # module config
    config = dict(
        num_cascades=8,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        mask_type="equispaced",
        center_fractions=[0.08],
        accelerations=[4],
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=knee_path,
        challenge="multicoil",
        test_split="test",
        batch_size=batch_size,
    )
    parser.set_defaults(**config)

    # trainer config
    parser.set_defaults(
        default_root_dir=default_root_dir,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=num_gpus,
        replace_sampler_ddp=False,
        distributed_backend=backend,
        seed=42,
        deterministic=True,
    )

    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
