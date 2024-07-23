"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from feature_varnet import (
    AttentionFeatureVarNet_n_sh_w,
    E2EVarNet,
    FeatureVarNet_n_sh_w,
    FeatureVarNet_sh_w,
    FIVarNet,
    IFVarNet,
)
from pytorch_lightning.loggers import TensorBoardLogger

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules.data_module import FastMriDataModule

from .feature_varnet_module import FIVarNetModule

torch.set_float32_matmul_precision("high")


def check_gpu_availability():
    command = "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l"
    output = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    return int(output)


def reload_state_dict(
    module: FIVarNetModule, fname: Path, module_name: str = "fi_varnet."
):
    print(f"loading model from {fname}")
    lm = len(module_name)
    state_dict = torch.load(fname, map_location=torch.device("cpu"))["state_dict"]
    state_dict = {k[lm:]: v for k, v in state_dict.items() if k[:lm] == module_name}
    module.fi_varnet.load_state_dict(state_dict)
    return module


def fetch_model(args, acceleration):
    if args.varnet_type == "fi_varnet":
        print(f"BUILDING FI VARNET, chans={args.chans}")
        return FIVarNet(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            acceleration=acceleration,
        )
    if args.varnet_type == "if_varnet":
        print(f"BUILDING IF VARNET, chans={args.chans}")
        return IFVarNet(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            acceleration=acceleration,
        )
    elif args.varnet_type == "attention_feature_varnet_sh_w":
        print(
            f"BUILDING ATTENTION FEATURE VARNET WITH WEIGHT SHARING, chans={args.chans}"
        )
        return AttentionFeatureVarNet_n_sh_w(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            acceleration=acceleration,
        )
    elif args.varnet_type == "feature_varnet_n_sh_w":
        print(f"BUILDING FEATURE VARNET WITHOUT WEIGHT SHARING, chans={args.chans}")
        return FeatureVarNet_n_sh_w(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
        )
    elif args.varnet_type == "feature_varnet_sh_w":
        print(f"BUILDING FEATURE VARNET WITH WEIGHT SHARING, chans={args.chans}")
        return FeatureVarNet_sh_w(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
        )
    elif args.varnet_type == "e2e_varnet":
        print(f"BUILDING E2E VARNET, chans={args.chans}")
        return E2EVarNet(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
        )
    else:
        raise ValueError("Unrecognized varnet_type")


def cli_main(args):
    pl.seed_everything(args.seed)

    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)

    if args.mode == "test_val":
        args.mode = "test"
        test_transform = VarNetDataTransform(mask_func=mask)
    else:
        test_transform = VarNetDataTransform()

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=True,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    acceleration_mean = int(round(sum(args.accelerations) / len(args.accelerations)))
    print(acceleration_mean)
    pl_module = FIVarNetModule(
        fi_varnet=fetch_model(args, acceleration_mean),
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        ramp_steps=args.ramp_steps,
        cosine_decay_start=args.cosine_decay_start,
    )

    if args.resume_from_checkpoint is not None:
        pl_module = reload_state_dict(pl_module, args.resume_from_checkpoint)
    trainer = pl.Trainer.from_argparse_args(args)
    if args.mode == "train":
        trainer.fit(pl_module, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(pl_module, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args(
    model_name: Optional[str] = "VarNet DDP x4", cluster_launch: bool = True
):
    parser = ArgumentParser()
    path_config = pathlib.Path("./fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = check_gpu_availability() if backend == "ddp" else 1
    batch_size = 1
    data_path = fetch_dir("data_path", path_config)
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test", "test_val"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced", "equispaced_fraction"),
        default="equispaced_fraction",
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
    parser.add_argument(
        "--varnet_type",
        choices=(
            "fi_varnet",
            "if_varnet",
            "feature_varnet_sh_w",
            "feature_varnet_n_sh_w",
            "attention_feature_varnet_sh_w",
            "e2e_varnet",
        ),
        default="fi_varnet",
        type=str,
        help="Type of VarNet to use",
    )

    parser = FastMriDataModule.add_data_specific_args(parser)

    args, _ = parser.parse_known_args()
    if args.mode == "test" or args.mode == "test_val":
        num_gpus = 1
    if args.varnet_type == "e2e_varnet":
        default_root_dir = fetch_dir("log_path", path_config) / "e2e_varnet"
    if args.varnet_type == "fi_varnet":
        default_root_dir = fetch_dir("log_path", path_config) / "fi_varnet"
    if args.varnet_type == "if_varnet":
        default_root_dir = fetch_dir("log_path", path_config) / "if_varnet"
    elif args.varnet_type == "feature_varnet_sh_w":
        default_root_dir = fetch_dir("log_path", path_config) / "feature_varnet_sh_w"
    elif args.varnet_type == "feature_varnet_n_sh_w":
        default_root_dir = fetch_dir("log_path", path_config) / "feature_varnet_n_sh_w"
    elif args.varnet_type == "attention_feature_varnet_sh_w":
        default_root_dir = (
            fetch_dir("log_path", path_config) / "attention_feature_varnet_sh_w"
        )

    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="equispaced_fraction",  # knee uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    parser = FIVarNetModule.add_model_specific_args(parser)

    parser.set_defaults(
        num_cascades=12,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=32,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.0003,  # Adam learning rate
        ramp_steps=7500,
        cosine_decay_start=150000,  # 150000,
        weight_decay=0.0,  # weight regularization strength
    )
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        devices=num_gpus,  # number of gpus to use
        replace_sampler_ddp=True,  # this is necessary for volume dispatch during val
        accelerator="gpu",  # what distributed version to use
        strategy="ddp_find_unused_parameters_false",  # what distributed version to use
        seed=42,  # random seed
        # deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_steps=210000,  # 210000,  # number of steps for 50 knee epochs
        detect_anomaly=False,
        gradient_clip_val=1.0,
    )
    args = parser.parse_args()
    print(f"MODEL NAME: {model_name}")
    args.logger = TensorBoardLogger(
        save_dir=args.default_root_dir, version=f"{model_name}"
    )
    checkpoint_dir = args.default_root_dir / "checkpoints" / f"{model_name}"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(),
    ]
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
    return args


def run_cli():
    args = build_args(cluster_launch=True)
    cli_main(args)


if __name__ == "__main__":
    run_cli()
