import pathlib
from argparse import ArgumentParser

import pytest
from pytorch_lightning import Trainer

from experimental.unet.unet_module import UnetModule
from experimental.varnet.varnet_module import VarNetModule
from fastmri.data.mri_data import fetch_dir


def build_unet_args():
    knee_path = fetch_dir("knee_path")
    logdir = fetch_dir("log_path") / "test_dir"

    parent_parser = ArgumentParser(add_help=False)

    parser = UnetModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 1
    backend = "dp"
    batch_size = 1 if backend == "ddp" else num_gpus

    config = dict(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        mask_type="random",
        center_fractions=[0.08],
        accelerations=[4],
        resolution=384,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=knee_path,
        challenge="singlecoil",
        exp_dir=logdir,
        exp_name="unet_demo",
        test_split="test",
        batch_size=batch_size,
    )
    parser.set_defaults(**config)

    parser.set_defaults(
        gpus=num_gpus,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
    )

    parser.add_argument("--mode", default="train", type=str)

    args = parser.parse_args([])

    return args


def build_varnet_args():
    knee_path = fetch_dir("knee_path")
    logdir = fetch_dir("log_path") / "test_dir"

    parent_parser = ArgumentParser(add_help=False)

    parser = VarNetModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    backend = "dp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    config = dict(
        num_cascades=8,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        mask_type="equispaced",
        center_fractions=[0.08],
        accelerations=[4],
        resolution=384,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=knee_path,
        challenge="multicoil",
        exp_dir=logdir,
        exp_name="varnet_demo",
        test_split="test",
        batch_size=batch_size,
    )
    parser.set_defaults(**config)

    parser.set_defaults(
        gpus=num_gpus,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
    )

    parser.add_argument("--mode", default="train", type=str)

    args = parser.parse_args([])

    return args


@pytest.mark.parametrize("backend", [(None)])
def test_unet_trainer(backend, skip_module_test):
    if skip_module_test:
        pytest.skip("config set to skip")

    args = build_unet_args()
    args.fast_dev_run = True
    args.backend = backend

    model = UnetModule(**vars(args))
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model)


@pytest.mark.parametrize("backend", [(None)])
def test_varnet_trainer(backend, skip_module_test):
    if skip_module_test:
        pytest.skip("config set to skip")

    args = build_varnet_args()
    args.fast_dev_run = True
    args.backend = backend

    model = VarNetModule(**vars(args))
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model)
