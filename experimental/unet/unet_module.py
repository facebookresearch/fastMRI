"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import hashlib
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

import fastmri
from fastmri import MriModule
from fastmri.data import transforms
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet


class UnetModule(MriModule):
    """
    Unet training module.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        mask_type="random",
        center_fractions=[0.08],
        accelerations=[4],
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net
                model.
            chans (int): Number of output channels of the first convolution
                layer.
            num_pool_layers (int): Number of down-sampling and up-sampling
                layers.
            drop_prob (float): Dropout probability.
            mask_type (str): Type of mask from ("random", "equispaced").
            center_fractions (list): Fraction of all samples to take from
                center (i.e., list of floats).
            accelerations (list): List of accelerations to apply (i.e., list
                of ints).
            lr (float): Learning rate.
            lr_step_size (int): Learning rate step size.
            lr_gamma (float): Learning rate gamma decay.
            weight_decay (float): Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.mask_type = mask_type
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        image, target, _, _, _, _ = batch
        output = self(image)
        loss = F.l1_loss(output, target)
        logs = {"loss": loss.detach()}

        return dict(loss=loss, log=logs)

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num = batch
        output = self(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        # hash strings to int so pytorch can concat them
        fnumber = torch.zeros(len(fname), dtype=torch.long, device=output.device)
        for i, fn in enumerate(fname):
            fnumber[i] = (
                int(hashlib.sha256(fn.encode("utf-8")).hexdigest(), 16) % 10 ** 12
            )

        return {
            "fname": fnumber,
            "slice": slice_num,
            "output": output * std + mean,
            "target": target * std + mean,
            "val_loss": F.l1_loss(output, target),
        }

    def test_step(self, batch, batch_idx):
        image, _, mean, std, fname, slice_num = batch
        output = self.forward(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def train_data_transform(self):
        mask = create_mask_for_mask_type(
            self.mask_type, self.center_fractions, self.accelerations,
        )

        return DataTransform(self.challenge, mask, use_seed=False)

    def val_data_transform(self):
        mask = create_mask_for_mask_type(
            self.mask_type, self.center_fractions, self.accelerations,
        )
        return DataTransform(self.challenge, mask)

    def test_data_transform(self):
        return DataTransform(self.challenge)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument("--in_chans", default=1, type=int)
        parser.add_argument("--out_chans", default=1, type=int)
        parser.add_argument("--chans", default=1, type=int)
        parser.add_argument("--num_pool_layers", default=4, type=int)
        parser.add_argument("--drop_prob", default=0.0, type=float)

        # data params
        parser.add_argument(
            "--mask_type", choices=["random", "equispaced"], default="random", type=str
        )
        parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)
        parser.add_argument("--accelerations", nargs="+", default=[4], type=int)

        # training params (opt)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--lr_step_size", default=40, type=int)
        parser.add_argument("--lr_gamma", default=0.1, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)

        return parser


class DataTransform(object):
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting
                the dataset.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.which_challenge = which_challenge
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
                fname (str): File name.
                slice_num (int): Serial number of the slice.
        """
        kspace = transforms.to_tensor(kspace)

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = transforms.complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size)
            target = transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, mean, std, fname, slice_num
