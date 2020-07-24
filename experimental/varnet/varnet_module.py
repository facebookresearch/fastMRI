"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

import fastmri
from fastmri import MriModule
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import VarNet


class VarNetModule(MriModule):
    """
    Unet training module.
    """

    def __init__(
        self,
        num_cascades=12,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        mask_type="equispaced",
        center_fractions=[0.08],
        accelerations=[4],
        resolution=384,
        lr=0.0003,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades (int, default=12): Number of cascades (i.e., layers)
                for variational network.
            sens_chans (int, default=8): Number of channels for sensitivity map
                U-Net.
            sens_pools (int, default=8): Number of downsampling and upsampling
                layers for sensitivity map U-Net.
            chans (int, default=18): Number of channels for cascade U-Net.
            pools (int, default=4): Number of downsampling and upsampling
                layers for cascade U-Net.
            mask_type (str, default="equispaced"): Type of mask from ("random",
                "equispaced").
            center_fractions (list, default=[0.08]): Fraction of all samples to
                take from center (i.e., list of floats).
            accelerations (list, default=[4]): List of accelerations to apply
                (i.e., list of ints).
            resolution (int, default=384): Reconstruction resolution.
            lr (float, default=0.0003): Learning rate.
            lr_step_size (int, default=40): Learning rate step size.
            lr_gamma (float, default=0.1): Learning rate gamma decay.
            weight_decay (float, default=0): Parameter for penalizing weights
                norm.
        """
        super().__init__(**kwargs)

        if self.batch_size != 1:
            raise NotImplementedError(
                f"Only batch_size=1 allowed for {self.__class__.__name__}"
            )

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.mask_type = mask_type
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.resolution = resolution
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask):
        return self.varnet(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, _, _, max_value = batch

        output = self(masked_kspace, mask)

        target, output = T.center_crop_to_smallest(target, output)
        loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)

        return {"loss": loss, "log": {"train_loss": loss.item()}}

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, slice_num, max_value = batch

        output = self.forward(masked_kspace, mask)
        target, output = T.center_crop_to_smallest(target, output)

        fnumber = torch.zeros(len(fname)).to(output)
        for i, fn in enumerate(fname):
            fnumber[i] = int(fn.split("file")[1].split(".h5")[0])

        return {
            "fname": fnumber,
            "slice": slice_num,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, _, fname, slice_num, _ = batch

        output = self(masked_kspace, mask)

        _, _, w = output.shape

        crop_size = min(w, self.resolution)
        output = T.center_crop(output, (crop_size, crop_size))

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def train_data_transform(self):
        mask = create_mask_for_mask_type(
            self.mask_type, self.center_fractions, self.accelerations
        )

        return DataTransform(self.resolution, mask, use_seed=False)

    def val_data_transform(self):
        mask = create_mask_for_mask_type(
            self.mask_type, self.center_fractions, self.accelerations
        )

        return DataTransform(self.resolution, mask)

    def test_data_transform(self):
        mask = create_mask_for_mask_type(
            self.mask_type, self.center_fractions, self.accelerations
        )

        return DataTransform(self.resolution, mask)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument("--num_cascades", default=12, type=int)
        parser.add_argument("--pools", default=4, type=int)
        parser.add_argument("--chans", default=18, type=int)
        parser.add_argument("--sens_pools", default=4, type=int)
        parser.add_argument("--sens_chans", default=8, type=float)

        # data params
        parser.add_argument(
            "--mask_type", choices=["random", "equispaced"], default="random", type=str
        )
        parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)
        parser.add_argument("--accelerations", nargs="+", default=[4], type=int)
        parser.add_argument("--resolution", default=384, type=int)

        # training params (opt)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--lr_step_size", default=40, type=int)
        parser.add_argument("--lr_gamma", default=0.1, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)

        return parser


class DataTransform(object):
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, resolution, mask_func=None, use_seed=True):
        """
        Args:
            resolution (int): Resolution of the image.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
        """
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
            mask (numpy.array): Mask from the test dataset.
            target (numpy.array): Target image.
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch
                    Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        if target is not None:
            target = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        if self.mask_func:
            masked_kspace, mask = T.apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask[:, :, :acq_start] = 0
            mask[:, :, acq_end:] = 0

        return masked_kspace, mask.byte(), target, fname, slice_num, max_value
