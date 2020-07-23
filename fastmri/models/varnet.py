"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms as T

from .unet import Unet


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(self, chans, num_pools, in_chans=2, out_chans=2, drop_prob=0):
        """
        Args:
            chans (int): Number of output channels of the first convolution
                layer.
            num_pools (int): Number of down-sampling and up-sampling layers.
            in_chans (int, default=2): Number of channels in the input to the
                U-Net model.
            out_chans (int, default=2): Number of channels in the output to the
                U-Net model.
            drop_prob (float, default=0): Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

    def norm(self, x):
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)

        mean = (
            x.mean(dim=2)
            .view(b, 2, 1, 1, 1)
            .expand(b, 2, c // 2, 1, 1)
            .contiguous()
            .view(b, c, 1, 1)
        )
        std = (
            x.std(dim=2)
            .view(b, 2, 1, 1, 1)
            .expand(b, 2, c // 2, 1, 1)
            .contiguous()
            .view(b, c, 1, 1)
        )

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def pad(self, x):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x, h_pad, w_pad, h_mult, w_mult):
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x):
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(self, chans, num_pools, in_chans=2, out_chans=2, drop_prob=0):
        """
        Args:
            chans (int): Number of output channels of the first convolution
                layer.
            num_pools (int): Number of down-sampling and up-sampling layers.
            in_chans (int, default=2): Number of channels in the input to the
                U-Net model.
            out_chans (int, default=2): Number of channels in the output to the
                U-Net model.
            drop_prob (float, default=0): Dropout probability.
        """
        super().__init__()

        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x):
        b, c, *other = x.shape
        return x.contiguous().view(b * c, 1, *other), b

    def batch_chans_to_chan_dim(self, x, batch_size):
        bc, _, *other = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, *other)

    def divide_root_sum_of_squares(self, x):
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace, mask):
        def get_low_frequency_lines(mask):
            l = r = mask.shape[-2] // 2
            while mask[..., r, :]:
                r += 1

            while mask[..., l, :]:
                l -= 1

            return l + 1, r

        l, r = get_low_frequency_lines(mask)
        num_low_freqs = r - l
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        x = T.mask_center(masked_kspace, pad, pad + num_low_freqs)
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)

        return x


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(self, num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4):
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
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace, mask):
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): Module for "regularization" component of
                variational network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def forward(self, current_kspace, ref_kspace, mask, sens_maps):
        def sens_expand(x):
            return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

        def sens_reduce(x):
            x = fastmri.ifft2c(x)
            return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
                dim=1, keepdim=True
            )

        def soft_dc(x):
            return torch.where(mask, x - ref_kspace, self.zero) * self.dc_weight

        return (
            current_kspace
            - soft_dc(current_kspace)
            - sens_expand(self.model(sens_reduce(current_kspace)))
        )
