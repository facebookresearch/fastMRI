"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import pdb
from fastmri.model.classifiers.resnet_r1 import ResnetBlock

kernel_size = 5

class WideDiscriminator(nn.Module):
    """
        Known to work well as a GAN discriminator
        
    """
    def __init__(self, num_classes=1, args=None):
        super().__init__()
        nf = self.nf = 128

        # Submodules
        nlayers = 1
        self.nf0 = nf * 2**nlayers

        blocks = [
            ResnetBlock(nf, nf),
            ResnetBlock(nf, nf),
            ResnetBlock(nf, nf),
        ]

        for i in range(nlayers):
            nf0 = nf * 2**i
            nf1 = nf * 2**(i+1)
            blocks += [
                nn.MaxPool2d(4, stride=4, padding=0),
                ResnetBlock(nf0, nf1),
                ResnetBlock(nf1, nf1),
                ResnetBlock(nf1, nf1),
            ]

        # Initial up-channeling conv
        self.conv_img = nn.Conv2d(3, 1*nf, kernel_size=kernel_size, padding=kernel_size//2)

        self.resnet = nn.Sequential(*blocks)

        # Final stage is standard avg-pool followed by linear
        self.pool_max = nn.MaxPool2d(4, stride=4, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.nf0, num_classes)
        self.norm = nn.InstanceNorm2d(3, affine=False, eps=0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        out = x
        
        out = self.norm(out)
        out = self.conv_img(out)
        out = self.resnet(out)
        out = self.pool_max(out)
        out = self.pool(out)
        out = out.view(batch_size, self.nf0)
        out = self.fc(actvn(out))
        
        return out

def actvn(x):
    return F.relu(x)
