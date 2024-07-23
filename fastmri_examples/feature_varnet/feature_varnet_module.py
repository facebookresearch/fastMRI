"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from argparse import ArgumentParser

import torch
from feature_varnet import FIVarNet

from fastmri.data.transforms import center_crop, center_crop_to_smallest
from fastmri.losses import SSIMLoss
from fastmri.pl_modules.mri_module import MriModule

torch.set_float32_matmul_precision("high")


class FIVarNetModule(MriModule):
    def __init__(
        self,
        fi_varnet: FIVarNet,
        lr: float = 0.0003,
        weight_decay: float = 0.0,
        max_steps: int = 65450,
        ramp_steps: int = 2618,
        cosine_decay_start: int = 32725,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = lr
        self.max_steps = max_steps
        self.ramp_steps = ramp_steps
        self.cosine_decay_start = cosine_decay_start
        self.weight_decay = weight_decay
        self.fi_varnet = fi_varnet
        self.loss = SSIMLoss()

    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.fi_varnet(masked_kspace, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        output = self.fi_varnet(
            batch.masked_kspace,
            batch.mask,
            batch.num_low_frequencies,
            crop_size=batch.crop_size,
        )
        target, output = center_crop_to_smallest(batch.target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1).float(), data_range=batch.max_value
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        for name, param in self.fi_varnet.named_parameters():
            if param.grad is not None:
                self.log(f"grads/{name}", torch.norm(param.grad))

    def validation_step(self, batch, batch_idx):
        output = self.fi_varnet(
            batch.masked_kspace,
            batch.mask,
            batch.num_low_frequencies,
            crop_size=batch.crop_size,
        )
        target, output = center_crop_to_smallest(batch.target, output)
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1),
                target.unsqueeze(1).float(),
                data_range=batch.max_value,
            ),
        }

    def test_step(self, batch, batch_idx):
        output = self.fi_varnet(
            batch.masked_kspace,
            batch.mask,
            batch.num_low_frequencies,
            crop_size=batch.crop_size,
        )
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size
        output = center_crop(output, crop_size)
        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        cosine_steps = self.max_steps - self.cosine_decay_start

        def step_fn(step):
            if step < self.cosine_decay_start:
                return min(step / self.ramp_steps, 1.0)
            else:
                angle = (step - self.cosine_decay_start) / cosine_steps * math.pi / 2
                return max(math.cos(angle), 1e-8)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, step_fn),
            "interval": "step",
        }
        return [optimizer], [lr_scheduler_config]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNetFiLM blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNetFiLM blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNetFiLM",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNetFiLM",
        )
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--ramp_steps",
            default=2618,
            type=int,
            help="Number of steps for ramping learning rate",
        )
        parser.add_argument(
            "--cosine_decay_start",
            default=32725,
            type=int,
            help="Step at which to start cosine lr decay",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        return parser
