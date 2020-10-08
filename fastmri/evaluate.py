"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from pytorch_lightning.metrics.metric import NumpyMetric, TensorMetric
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.distributed import ReduceOp

from fastmri.data import transforms


class MSE(NumpyMetric):
    """Calculates MSE and aggregates by summing across distr processes."""

    def __init__(self, name="MSE", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, gt, pred):
        return mse(gt, pred)


class NMSE(NumpyMetric):
    """Calculates NMSE and aggregates by summing across distr processes."""

    def __init__(self, name="NMSE", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, gt, pred):
        return nmse(gt, pred)


class PSNR(NumpyMetric):
    """Calculates PSNR and aggregates by summing across distr processes."""

    def __init__(self, name="PSNR", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, gt, pred):
        return psnr(gt, pred)


class SSIM(NumpyMetric):
    """Calculates SSIM and aggregates by summing across distr processes."""

    def __init__(self, name="SSIM", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, gt, pred, maxval=None):
        return ssim(gt, pred, maxval=maxval)


class DistributedMetricSum(TensorMetric):
    """Used for summing parameters across distr processes."""

    def __init__(self, name="DistributedMetricSum", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, x):
        return x.clone()


def mse(gt, pred):
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]

    return ssim


METRIC_FUNCS = dict(MSE=mse, NMSE=nmse, PSNR=psnr, SSIM=ssim,)


class Metrics(object):
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(
            args.predictions_path / tgt_file.name, "r"
        ) as recons:
            if args.acquisition and args.acquisition != target.attrs["acquisition"]:
                continue

            if args.acceleration and target.attrs["acceleration"] != args.acceleration:
                continue

            target = target[recons_key][()]
            recons = recons["reconstruction"][()]
            target = transforms.center_crop(
                target, (target.shape[-1], target.shape[-1])
            )
            recons = transforms.center_crop(
                recons, (target.shape[-1], target.shape[-1])
            )
            metrics.push(target, recons)

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        required=True,
        help="Which challenge",
    )
    parser.add_argument("--acceleration", type=int, default=None)
    parser.add_argument(
        "--acquisition",
        choices=[
            "CORPD_FBK",
            "CORPDFS_FBK",
            "AXT1",
            "AXT1PRE",
            "AXT1POST",
            "AXT2",
            "AXFLAIR",
        ],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    args = parser.parse_args()

    recons_key = (
        "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"
    )
    metrics = evaluate(args, recons_key)
    print(metrics)
