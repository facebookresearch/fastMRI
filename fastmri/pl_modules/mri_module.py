"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from fastmri import evaluate
from fastmri.data import SliceDataset
from fastmri.data.volume_sampler import VolumeSampler
from torch.utils.data import DataLoader, DistributedSampler


class DistributedMetricSum(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch):
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - fastMRI data loaders
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - train_data_transform, val_data_transform, test_data_transform:
            Create and return data transformer objects for each data split
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(
        self,
        data_path,
        challenge,
        test_split="test",
        sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        num_log_images=16,
        **kwargs,
    ):
        """
        Args:
            data_path (pathlib.Path): Path to root data directory. For example, if
                knee/path is the root directory with subdirectories
                multicoil_train and multicoil_val, you would input knee/path for
                data_path.
            challenge (str): Name of challenge from ('multicoil', 'singlecoil').
            test_split (str): Name of test split from ("test", "challenge").
            sample_rate (float, default=1.0): Fraction of models from the
                dataset to use.
            batch_size (int, default=1): Batch size.
            num_workers (int, default=4): Number of workers for PyTorch dataloader.
        """
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.test_split = test_split
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def _create_data_loader(self, data_transform, data_partition, sample_rate=None):
        sample_rate = sample_rate or self.sample_rate
        dataset = SliceDataset(
            root=self.data_path / f"{self.challenge}_{data_partition}",
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=self.challenge,
        )

        is_train = data_partition == "train"

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.use_ddp:
            if is_train:
                sampler = DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=sampler,
        )

        return dataloader

    def train_data_transform(self):
        raise NotImplementedError

    def train_dataloader(self):
        return self._create_data_loader(
            self.train_data_transform(), data_partition="train"
        )

    def val_data_transform(self):
        raise NotImplementedError

    def val_dataloader(self):
        return self._create_data_loader(self.val_data_transform(), data_partition="val")

    def test_data_transform(self):
        raise NotImplementedError

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_data_transform(), data_partition=self.test_split, sample_rate=1.0,
        )

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step"
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.logger.experiment.add_image(f"{key}/target", target)
                self.logger.experiment.add_image(f"{key}/reconstruction", output)
                self.logger.experiment.add_image(f"{key}/error", error)

        # compute evaluation metrics
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            nmse_vals[fname][slice_num] = torch.tensor(evaluate.nmse(target, output))
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target, output, maxval=maxval)
            )
            psnr_vals[fname][slice_num] = torch.tensor(evaluate.psnr(target, output))

        return {
            "val_loss": val_logs["val_loss"],
            "nmse_vals": nmse_vals,
            "ssim_vals": ssim_vals,
            "psnr_vals": psnr_vals,
        }

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["nmse_vals"].keys():
                nmse_vals[k].update(val_log["nmse_vals"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["psnr_vals"].keys():
                psnr_vals[k].update(val_log["psnr_vals"][k])

        # check to make sure we have all files in all metrics
        assert nmse_vals.keys() == ssim_vals.keys() == psnr_vals.keys()

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in nmse_vals.keys():
            local_examples = local_examples + 1
            metrics["nmse"] = metrics["nmse"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["psnr"] = metrics["psnr"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("val_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(list)

        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname].append((slice_num, log["output"][i]))

        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # data arguments
        parser.add_argument(
            "--data_path", default=pathlib.Path("datasets/"), type=pathlib.Path
        )
        parser.add_argument(
            "--challenge",
            choices=["singlecoil", "multicoil"],
            default="singlecoil",
            type=str,
        )
        parser.add_argument(
            "--sample_rate", default=1.0, type=float,
        )
        parser.add_argument(
            "--batch_size", default=1, type=int,
        )
        parser.add_argument(
            "--num_workers", default=4, type=float,
        )
        parser.add_argument(
            "--seed", default=42, type=int,
        )

        # logging params
        parser.add_argument(
            "--test_split", default="test", type=str,
        )
        parser.add_argument("--num_log_images", default=16, type=int)

        return parser
