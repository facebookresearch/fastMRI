"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, DistributedSampler

import fastmri
from fastmri.data import SliceDataset
from fastmri.data.volume_sampler import VolumeSampler
from fastmri import evaluate
from fastmri.evaluate import DistributedMetricAverage


class BaseModule(pl.LightningModule):
    """Abstract super class for deep larning reconstruction models.
    
    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - fastMRI data loaders
        - Evaluating reconstructions
        - Visualization
        - Saving test reconstructions

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

    Args:
        data_path (pathlib.Path): Path to root data directory. For example, if
            knee/path is the root directory with subdirectories
            multicoil_train and multicoil_val, you would input knee/path for
            data_path.
        challenge (str): Name of challenge from ('multicoil', 'singlecoil').
        exp_dir (pathlib.Path): Top directory for where you want to store log
            files.
        exp_name (str): Name of this experiment - this will store logs in
            exp_dir / {exp_name}.
        sample_rate (float, default=1.0): Sampling rate for this experiment.
        batch_size (int, default=1): Batch size.
        num_workers (int, default=4): Number of workers for PyTorch dataloader.
        use_ddp (boolean, default=False): Set this to true if you use a 'ddp'
            backend for the PyTorch Lightning trainer - this will make
            aggregation for ssim and other metrics perform as expected. 
    """

    def __init__(
        self,
        data_path,
        challenge,
        exp_dir,
        exp_name,
        sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        use_ddp=False,
    ):
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.sample_rate = sample_rate
        self.use_ddp = use_ddp

        self.distr_metric_funs = {
            "nmse": DistributedMetricAverage(),
            "ssim": DistributedMetricAverage(),
            "psnr": DistributedMetricAverage(),
        }

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
            drop_last=is_train,
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
            self.test_data_transform(), data_partition="test", sample_rate=1.0,
        )

    def _evaluate(self, val_logs):
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)

        for log in val_logs:
            losses.append(log["val_loss"].cpu().numpy())
            for i, (fname, slice_ind) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname].append((slice_ind, log["output"][i]))
                targets[fname].append((slice_ind, log["target"][i]))

        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics["nmse"].append(evaluate.nmse(target, output))
            metrics["ssim"].append(evaluate.ssim(target, output))
            metrics["psnr"].append(evaluate.psnr(target, output))

        metrics = {
            metric: self.distr_metric_funs[metric](np.mean(values))
            for metric, values in metrics.items()
        }

        return dict(log=metrics, **metrics)

    def _visualize(self, val_logs):
        def _normalize(image):
            image = image[np.newaxis]
            image = image - image.min()
            return image / image.max()

        def _save_image(image, tag):
            grid = torchvision.utils.make_grid(torch.Tensor(image), nrow=4, pad_value=1)
            self.logger.experiment.add_image(tag, grid)

        # only process first size to simplify visualization.
        visualize_size = val_logs[0]["output"].shape
        val_logs = [x for x in val_logs if x["output"].shape == visualize_size]
        num_logs = len(val_logs)
        num_viz_images = 16
        step = (num_logs + num_viz_images - 1) // num_viz_images
        outputs, targets = [], []

        for i in range(0, num_logs, step):
            outputs.append(_normalize(val_logs[i]["output"][0]))
            targets.append(_normalize(val_logs[i]["target"][0]))

        outputs = np.stack(outputs)
        targets = np.stack(targets)
        _save_image(targets, "Target")
        _save_image(outputs, "Reconstruction")
        _save_image(np.abs(targets - outputs), "Error")

    def validation_epoch_end(self, val_logs):
        self._visualize(val_logs)
        return self._evaluate(val_logs)

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(list)

        for log in test_logs:
            for i, (fname, slice) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname].append((slice, log["output"][i]))

        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

        fastmri.save_reconstructions(
            outputs, self.exp_dir / self.exp_name / "reconstructions"
        )

        return dict()
