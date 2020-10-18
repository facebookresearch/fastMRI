"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms
from fastmri.models import VarNet

from .mri_module import MriModule


class VarNetModule(MriModule):
    """
    VarNet training module.
    """

    def __init__(
        self,
        num_cascades=12,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
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
            lr (float, default=0.0003): Learning rate.
            lr_step_size (int, default=40): Learning rate step size.
            lr_gamma (float, default=0.1): Learning rate gamma decay.
            weight_decay (float, default=0): Parameter for penalizing weights
                norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
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
        masked_kspace, mask, target, _, _, max_value, _ = batch

        output = self(masked_kspace, mask)

        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)

        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, slice_num, max_value, _ = batch

        output = self.forward(masked_kspace, mask)
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, _, fname, slice_num, _, crop_size = batch
        crop_size = crop_size[0]  # always have a batch size of 1 for varnet

        output = self(masked_kspace, mask)

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

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

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades", default=12, type=int, help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sensitivity map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sensitivity map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
