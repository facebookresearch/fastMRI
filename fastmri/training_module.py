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
from common.utils import save_reconstructions
from data.mri_data import SliceData
from data.volume_sampler import VolumeSampler
from fastmri import evaluate
from .evaluate import DistributedMetric


class TrainingModule(pl.LightningModule):
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
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        val_dataloader = self.val_dataloader()

        nmse_metrics = dict()
        ssim_metrics = dict()
        psnr_metrics = dict()
        slice_counts = dict()
        for (fname, slice_ind, _, _) in val_dataloader.dataset.examples:
            if nmse_metrics.get(fname) is None:
                nmse_metrics[fname] = DistributedMetric()
                ssim_metrics[fname] = DistributedMetric()
                psnr_metrics[fname] = DistributedMetric()

                slice_counts[fname] = slice_ind + 1
            elif slice_ind + 1 > slice_counts[fname]:
                slice_counts[fname] = slice_ind + 1

        self.nmse_metrics = nmse_metrics
        self.ssim_metrics = ssim_metrics
        self.psnr_metrics = psnr_metrics
        self.slice_counts = slice_counts

    def _create_data_loader(self, data_transform, data_partition, sample_rate=None):
        sample_rate = sample_rate or self.hparams.sample_rate
        dataset = SliceData(
            root=self.hparams.data_path / f"{self.hparams.challenge}_{data_partition}",
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=self.hparams.challenge,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            drop_last=(data_partition == "train"),
        )

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

        metrics = {metric: np.mean(values) for metric, values in metrics.items()}

        return dict(log=metrics, **metrics)

    def _visualize(self, val_logs):
        def _normalize(image):
            image = image[np.newaxis]
            image -= image.min()
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
        save_reconstructions(
            outputs, self.hparams.exp_dir / self.hparams.exp / "reconstructions"
        )
        return dict()
