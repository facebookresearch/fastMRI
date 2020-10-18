"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
from argparse import ArgumentParser

import fastmri
import pytorch_lightning as pl
import torch


class FastMriDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        challenge,
        train_transform,
        val_transform,
        test_transform,
        test_split="test",
        sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        distributed_sampler=False,
    ):
        """
        Args:
            data_path (pathlib.Path): Path to root data directory. For example,
                if knee/path is the root directory with subdirectories
                multicoil_train and multicoil_val, you would input knee/path
                for data_path.
            challenge (str): Name of challenge from ('multicoil',
                'singlecoil').
            test_split (str): Name of test split from ("test", "challenge").
            sample_rate (float, default=1.0): Fraction of models from the
                dataset to use.
            batch_size (int, default=1): Batch size.
            num_workers (int, default=4): Number of workers for PyTorch
                dataloader.
        """
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.test_split = test_split
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(self, data_transform, data_partition, sample_rate=None):
        if data_partition == "train":
            is_train = True
            sample_rate = sample_rate or self.sample_rate
        else:
            is_train = False
            sample_rate = 1.0

        dataset = fastmri.data.SliceDataset(
            root=self.data_path / f"{self.challenge}_{data_partition}",
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=self.challenge,
        )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=sampler,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform, data_partition=self.test_split, sample_rate=1.0,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=pathlib.Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--test_split",
            choices=("test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=1.0,
            type=float,
            help="Fraction of data set to use (train split only)",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=float,
            help="Number of workers to use in data loader",
        )

        return parser
