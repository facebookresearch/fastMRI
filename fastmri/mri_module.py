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
import torchvision
from pytorch_lightning import _logger as log
from torch.utils.data import DataLoader, DistributedSampler

import fastmri
from fastmri import evaluate
from fastmri.data import SliceDataset
from fastmri.data.volume_sampler import VolumeSampler
from fastmri.evaluate import DistributedMetricSum


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.
    
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
        **kwargs,
    ):
        """
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
            sample_rate (float, default=1.0): Fraction of models from the
                dataset to use.
            batch_size (int, default=1): Batch size.
            num_workers (int, default=4): Number of workers for PyTorch dataloader.
            use_ddp (boolean, default=False): Set this to true if you use a 'ddp'
                backend for the PyTorch Lightning trainer - this will make
                aggregation for ssim and other metrics perform as expected. 
        """
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_ddp = use_ddp

        self.distr_metric_funs = {
            "tot_num_examples": DistributedMetricSum("tot_num_examples"),
            "val_loss": DistributedMetricSum("val_loss"),
            "nmse": DistributedMetricSum("nmse"),
            "ssim": DistributedMetricSum("ssim"),
            "psnr": DistributedMetricSum("psnr"),
        }

        self.othermetric = DistributedMetricSum("bleh")

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

    def _visualize(self, val_outputs, val_targets):
        def _normalize(image):
            image = image[np.newaxis]
            image = image - image.min()
            return image / image.max()

        def _save_image(image, tag):
            grid = torchvision.utils.make_grid(torch.Tensor(image), nrow=4, pad_value=1)
            self.logger.experiment.add_image(tag, grid)

        # only process first size to simplify visualization.
        visualize_size = val_outputs[0].shape
        val_outputs = [x for x in val_outputs if x.shape == visualize_size]
        val_targets = [x for x in val_targets if x.shape == visualize_size]

        num_logs = len(val_outputs)
        assert num_logs == len(val_targets)

        num_viz_images = 16
        step = (num_logs + num_viz_images - 1) // num_viz_images
        outputs, targets = [], []

        for i in range(0, num_logs, step):
            outputs.append(_normalize(val_outputs[i]))
            targets.append(_normalize(val_targets[i]))

        outputs = np.stack(outputs)
        targets = np.stack(targets)
        _save_image(targets, "Target")
        _save_image(outputs, "Reconstruction")
        _save_image(np.abs(targets - outputs), "Error")

    def validation_epoch_end(self, val_logs):
        # run the visualizations
        assert val_logs[0]["output"].ndim == 3
        self._visualize(
            val_outputs=np.concatenate([x["output"].cpu().numpy() for x in val_logs]),
            val_targets=np.concatenate([x["target"].cpu().numpy() for x in val_logs]),
        )

        # aggregate losses
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        base_tensor = val_logs[0]["output"][0]

        for val_log in val_logs:
            losses.append(val_log["val_loss"].cpu().numpy())
            for i, (fname, slice_ind) in enumerate(
                zip(val_log["fname"], val_log["slice"])
            ):
                outputs[int(fname)].append(
                    (int(slice_ind), val_log["output"][i].cpu().numpy())
                )
                targets[int(fname)].append(
                    (int(slice_ind), val_log["target"][i].cpu().numpy())
                )

        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        bleh = 0
        for fname in outputs:
            output = np.concatenate([out for _, out in sorted(outputs[fname])])
            target = np.concatenate([tgt for _, tgt in sorted(targets[fname])])
            metrics["nmse"].append(evaluate.nmse(target, output))
            metrics["ssim"].append(evaluate.ssim(target, output))
            metrics["psnr"].append(evaluate.psnr(target, output))
            bleh = bleh + self.othermetric(
                torch.tensor(evaluate.nmse(target, output)).to(base_tensor)
            )
            print(f"bleh: {bleh}")

        # handle aggregation for distributed case with pytorch_lightning metrics
        # aggregation has the mean-of-means problem, which
        num_examples = torch.tensor(len(outputs)).to(base_tensor)
        tot_examples = self.distr_metric_funs["tot_num_examples"](num_examples)
        weight = num_examples.to(tot_examples) / tot_examples

        metrics = {
            metric: self.distr_metric_funs[metric](
                (np.mean(values) * weight).to(base_tensor)
            )
            .cpu()
            .numpy()
            for metric, values in metrics.items()
        }

        return dict(log=metrics, **metrics)

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
        parser.add_argument("--use_ddp", default=False, type=bool)

        # logging params
        parser.add_argument(
            "--exp_dir", default=pathlib.Path("logs/"), type=pathlib.Path
        )
        parser.add_argument(
            "--exp_name", default="my_experiment", type=str,
        )

        return parser
