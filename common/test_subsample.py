"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pytest
import torch

from common.subsample import RandomMaskFunc


@pytest.mark.parametrize("center_fracs, accelerations, batch_size, dim", [
    ([0.2], [4], 4, 320),
    ([0.2, 0.4], [4, 8], 2, 368),
])
def test_random_mask_reuse(center_fracs, accelerations, batch_size, dim):
    mask_func = RandomMaskFunc(center_fracs, accelerations)
    shape = (batch_size, dim, dim, 2)
    mask1 = mask_func(shape, seed=123)
    mask2 = mask_func(shape, seed=123)
    mask3 = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize("center_fracs, accelerations, batch_size, dim", [
    ([0.2], [4], 4, 320),
    ([0.2, 0.4], [4, 8], 2, 368),
])
def test_random_mask_low_freqs(center_fracs, accelerations, batch_size, dim):
    mask_func = RandomMaskFunc(center_fracs, accelerations)
    shape = (batch_size, dim, dim, 2)
    mask = mask_func(shape, seed=123)
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = dim
    assert list(mask.shape) == mask_shape

    num_low_freqs_matched = False
    for center_frac in center_fracs:
        num_low_freqs = int(round(dim * center_frac))
        pad = (dim - num_low_freqs + 1) // 2
        if np.all(mask[pad:pad + num_low_freqs].numpy() == 1):
            num_low_freqs_matched = True
    assert num_low_freqs_matched
