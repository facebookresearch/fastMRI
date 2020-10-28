"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import fastmri
import pytorch_lightning as pl
import torch
from fastmri.data import CombinedSliceDataset, SliceDataset


class FastMriDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp this module will create data loaders after ddp
    processes have launched. This correctly donfigures distributed sampling
    ranks.
    """

    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_test_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: float = 1.0,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_test_val: Whether to combine test and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            sample_rate: Fraction of of the training data split to use. Can be
                set to less than 1.0 for rapid prototyping.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_test_val = combine_test_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = sample_rate or self.sample_rate
        else:
            is_train = False
            sample_rate = 1.0

        dataset: Union[SliceDataset, CombinedSliceDataset]
        if data_partition == "train" and self.combine_test_val:
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates = [sample_rate, sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{self.challenge}_{data_partition}"

            dataset = SliceDataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
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
        return self._create_data_loader(
            self.val_transform, data_partition="val", sample_rate=1.0
        )

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform,
            data_partition=self.test_split,
            sample_rate=1.0,
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
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
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
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
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
