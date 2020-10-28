"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pytest
import torch
from fastmri.data import transforms
from fastmri.data.subsample import RandomMaskFunc
from fastmri.models import Unet, VarNet

from .conftest import create_input


@pytest.mark.parametrize(
    "shape, out_chans, chans",
    [
        ([1, 1, 32, 16], 5, 1),
        ([5, 1, 15, 12], 10, 32),
        ([3, 2, 13, 18], 1, 16),
        ([1, 2, 17, 19], 3, 8),
    ],
)
def test_unet(shape, out_chans, chans):
    x = create_input(shape)

    num_chans = x.shape[1]

    unet = Unet(in_chans=num_chans, out_chans=out_chans, chans=chans, num_pool_layers=2)

    y = unet(x)

    assert y.shape[1] == out_chans


@pytest.mark.parametrize(
    "shape, out_chans, chans, center_fractions, accelerations",
    [
        ([1, 3, 32, 16, 2], 2, 1, [0.08], [4]),
        ([5, 5, 15, 12, 2], 2, 32, [0.04], [8]),
        ([3, 8, 13, 18, 2], 2, 16, [0.08], [4]),
        ([1, 2, 17, 19, 2], 2, 8, [0.08], [4]),
    ],
)
def test_varnet(shape, out_chans, chans, center_fractions, accelerations):
    mask_func = RandomMaskFunc(center_fractions, accelerations)
    x = create_input(shape)
    output, mask = transforms.apply_mask(x, mask_func, seed=123)

    varnet = VarNet(num_cascades=2, sens_chans=4, sens_pools=2, chans=4, pools=2)

    y = varnet(output, mask.byte())

    assert y.shape[1:] == x.shape[2:4]


def test_unet_scripting():
    model = Unet(
        in_chans=1,
        out_chans=1,
        chans=8,
        num_pool_layers=2,
        drop_prob=0.0,
    )
    scr = torch.jit.script(model)
    assert scr is not None


def test_varnet_scripting():
    model = VarNet(
        num_cascades=4,
        pools=2,
        chans=8,
        sens_pools=2,
        sens_chans=4,
    )
    scr = torch.jit.script(model)
    assert scr is not None
