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
from fastmri.pl_modules import FastMriDataModule, UnetModule, configure_checkpoint
from fastmri.data.transforms import UnetDataTransform
from fastmri.data.subsample import create_mask_for_mask_type


def cli_main(args):
    # ------------
    # data
    # ------------
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform = UnetDataTransform(args.challenge, mask_func=mask)
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator == "ddp"),
    )

    # ------------
    # model
    # ------------
    pl.seed_everything(args.seed)
    model = UnetModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, data_module)
    elif args.mode == "test":
        assert args.resume_from_checkpoint is not None
        outputs = trainer.test(model, data_module)
        fastmri.save_reconstructions(outputs, args.default_root_dir / "reconstructions")
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    parser = ArgumentParser(add_help=False)

    parser = UnetModule.add_model_specific_args(parser)
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    num_gpus = 2
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # data config
    parser.set_defaults(
        data_path=None,
        challenge="singlecoil",
        test_split="test",
        sample_rate=1.0,
        batch_size=batch_size,
        num_workers=4,
        distributed_sampler=(backend == "ddp"),
    )

    # module config
    parser.set_defaults(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
    )

    # trainer config
    parser.set_defaults(
        gpus=num_gpus,
        replace_sampler_ddp=False,
        accelerator=backend,
        seed=42,
        deterministic=True,
    )

    # client arguments
    parser.add_argument(
        "--path_config", default="../../fastmri_dirs.yaml", type=pathlib.Path
    )
    parser.add_argument("--log_path", default=None, type=pathlib.Path)
    parser.add_argument("--mode", default="train", type=str)

    # data transform params
    parser.add_argument("--sample_rate", default=1.0, type=float)
    parser.add_argument(
        "--mask_type", choices=("random", "equispaced"), default="random", type=str
    )
    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int)

    args = parser.parse_args()

    if args.path_config is not None:
        if args.data_path is None:
            args.data_path = fetch_dir("knee_path", args.path_config)
        if args.log_path is None:
            args.default_root_dir = (
                fetch_dir("log_path", args.path_config) / "unet" / "unet_demo"
            )
        else:
            args.default_root_dir = args.log_path

    if args.default_root_dir is None:
        args.default_root_dir = pathlib.Path.cwd()

    args.checkpoint_callback, resume_from_checkpoint = configure_checkpoint(
        args.default_root_dir
    )
    if args.resume_from_checkpoint is None:
        args.resume_from_checkpoint = resume_from_checkpoint

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
