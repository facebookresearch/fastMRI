"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F
from torch import nn
from .data import transforms


class SSIM(nn.Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]

        C1 = (self.k1 * data_range)**2
        C2 = (self.k2 * data_range)**2

        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2)
        D = B1 * B2
        S = (A1 * A2) / D
        return S.mean()


class SSIMLossMixin(object):
    def loss_setup(self, args):
        self.ssim = SSIM().to(self.device)
        self.ssim_l1_coefficient = args.ssim_l1_coefficient

        super().loss_setup(args)

    def training_loss(self, batch):
        output, target = self.predict(batch)
        max_value = batch['attrs_dict']['max'].float().to(self.device)
        output, target = transforms.center_crop_to_smallest(output, target)
        l1_loss = F.l1_loss(output, target)
        output_ = self.unnorm(output, batch)
        target_ = self.unnorm(target, batch)
        ssim_loss = 1 - self.ssim(output_, target_, data_range=max_value)
        loss = ssim_loss.add(self.ssim_l1_coefficient, l1_loss)
        loss_dict = {
            'train_loss': loss,
            'ssim_loss': ssim_loss,
            'l1_loss': l1_loss,
        }
        return loss_dict, output, target
