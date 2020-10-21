"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import pytest
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform, VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule, VarNetModule
from pytorch_lightning import Trainer


def build_unet_args(tmp_path):
    knee_path = fetch_dir("knee_path")
    logdir = tmp_path / "unet_test_dir"

    parser = ArgumentParser()

    num_gpus = 1
    backend = "dp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
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
    parser.set_defaults(data_path=knee_path, batch_size=batch_size)

    # module config
    parser = UnetModule.add_model_specific_args(parser)
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
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        accelerator=backend,
    )

    parser.add_argument("--mode", default="train", type=str)

    args = parser.parse_args([])

    return args


def build_varnet_args(tmp_path):
    knee_path = fetch_dir("knee_path")
    logdir = tmp_path / "varnet_test_dir"

    parser = ArgumentParser()

    backend = "dp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

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
        data_path=knee_path,
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
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        accelerator=backend,
    )

    parser.add_argument("--mode", default="train", type=str)

    args = parser.parse_args([])

    return args


@pytest.mark.parametrize("backend", [(None)])
def test_unet_trainer(backend, skip_module_test, tmp_path):
    if skip_module_test:
        pytest.skip("config set to skip")

    args = build_unet_args(tmp_path)
    args.fast_dev_run = True
    args.backend = backend

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

    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, data_module)


@pytest.mark.parametrize("backend", [(None)])
def test_varnet_trainer(backend, skip_module_test, tmp_path):
    if skip_module_test:
        pytest.skip("config set to skip")

    args = build_varnet_args(tmp_path)
    args.fast_dev_run = True
    args.backend = backend

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
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, data_module)
