"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import pytest
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform, VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule, VarNetModule
from pytorch_lightning import Trainer


def build_unet_args(data_path, logdir, backend):
    parser = ArgumentParser()

    num_gpus = 0
    batch_size = 1

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
    parser.set_defaults(data_path=data_path, batch_size=batch_size)

    # module config
    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,
        out_chans=1,
        chans=8,
        num_pool_layers=2,
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


def build_varnet_args(data_path, logdir, backend):
    parser = ArgumentParser()

    num_gpus = 0
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
        data_path=data_path,
        mask_type="equispaced",
        challenge="multicoil",
        batch_size=batch_size,
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=4,
        pools=2,
        chans=8,
        sens_pools=2,
        sens_chans=4,
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


@pytest.mark.parametrize("backend", [None])
def test_unet_trainer(fastmri_mock_dataset, backend, tmp_path, monkeypatch):
    knee_path, _, metadata = fastmri_mock_dataset

    def retrieve_metadata_mock(a, fname):
        return metadata[str(fname)]

    monkeypatch.setattr(SliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    params = build_unet_args(knee_path, tmp_path, backend)
    params.fast_dev_run = True
    params.backend = backend

    mask = create_mask_for_mask_type(
        params.mask_type, params.center_fractions, params.accelerations
    )
    train_transform = UnetDataTransform(
        params.challenge, mask_func=mask, use_seed=False
    )
    val_transform = UnetDataTransform(params.challenge, mask_func=mask)
    test_transform = UnetDataTransform(params.challenge)
    data_module = FastMriDataModule(
        data_path=params.data_path,
        challenge=params.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=params.test_split,
        sample_rate=params.sample_rate,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        distributed_sampler=(params.accelerator == "ddp"),
        use_dataset_cache_file=False,
    )

    model = UnetModule(
        in_chans=params.in_chans,
        out_chans=params.out_chans,
        chans=params.chans,
        num_pool_layers=params.num_pool_layers,
        drop_prob=params.drop_prob,
        lr=params.lr,
        lr_step_size=params.lr_step_size,
        lr_gamma=params.lr_gamma,
        weight_decay=params.weight_decay,
    )

    trainer = Trainer.from_argparse_args(params)

    trainer.fit(model, data_module)


@pytest.mark.parametrize("backend", [None])
def test_varnet_trainer(fastmri_mock_dataset, backend, tmp_path, monkeypatch):
    knee_path, _, metadata = fastmri_mock_dataset

    def retrieve_metadata_mock(a, fname):
        return metadata[str(fname)]

    monkeypatch.setattr(SliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    params = build_varnet_args(knee_path, tmp_path, backend)
    params.fast_dev_run = True
    params.backend = backend

    mask = create_mask_for_mask_type(
        params.mask_type, params.center_fractions, params.accelerations
    )
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()
    data_module = FastMriDataModule(
        data_path=params.data_path,
        challenge=params.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=params.test_split,
        sample_rate=params.sample_rate,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        distributed_sampler=(params.accelerator == "ddp"),
        use_dataset_cache_file=False,
    )
    model = VarNetModule(
        num_cascades=params.num_cascades,
        pools=params.pools,
        chans=params.chans,
        sens_pools=params.sens_pools,
        sens_chans=params.sens_chans,
        lr=params.lr,
        lr_step_size=params.lr_step_size,
        lr_gamma=params.lr_gamma,
        weight_decay=params.weight_decay,
    )
    trainer = Trainer.from_argparse_args(params)

    trainer.fit(model, data_module)
