"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import fastmri
import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform(mask_func=mask)
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
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    if args.mode == "test":
        model.load_from_checkpoint(args.resume_from_checkpoint)
        args.resume_from_checkpoint = None

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
        outputs = trainer.test(model, data_module)
        fastmri.save_reconstructions(outputs, args.default_root_dir / "reconstructions")
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"
    checkpoint_path = default_root_dir / "checkpoints"

    # set default checkpoint if one exists in our default directory
    resume_from_checkpoint = None
    if checkpoint_path.exists():
        ckpt_list = sorted(checkpoint_path.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            resume_from_checkpoint = str(ckpt_list[-1])

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,
        mask_type="equispaced",
        challenge="multicoil",
        batch_size=batch_size,
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=8,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,
        replace_sampler_ddp=False,
        accelerator=backend,
        seed=42,
        deterministic=True,
        default_root_dir=default_root_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    args = parser.parse_args()

    # configure checkpointing
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.default_root_dir / "checkpoints",
        save_top_k=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
