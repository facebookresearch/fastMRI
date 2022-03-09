"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import fastmri
import numpy as np
import pytest
import torch
from fastmri.data import transforms

from .conftest import create_input


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3],
        [4, 6],
        [10, 8, 4],
    ],
)
def test_fft2(shape):
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fastmri.fft2c(x).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = transforms.tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3],
        [4, 6],
        [10, 8, 4],
    ],
)
def test_ifft2(shape):
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fastmri.ifft2c(x).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = transforms.tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3],
        [4, 6],
        [10, 8, 4],
    ],
)
def test_complex_abs(shape):
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fastmri.complex_abs(x).numpy()
    input_numpy = transforms.tensor_to_complex_np(x)
    out_numpy = np.abs(input_numpy)

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape, dim",
    [
        [[3, 3], 0],
        [[4, 6], 1],
        [[10, 8, 4], 2],
    ],
)
def test_root_sum_of_squares(shape, dim):
    x = create_input(shape)
    out_torch = fastmri.rss(x, dim).numpy()
    out_numpy = np.sqrt(np.sum(x.numpy() ** 2, dim))

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shift, dim",
    [
        (0, 0),
        (1, 0),
        (-1, 0),
        (100, 0),
        ([1, 2], [1, 2]),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [5, 6, 2],
        [3, 4, 5],
    ],
)
def test_roll(shift, dim, shape):
    x = np.arange(np.product(shape)).reshape(shape)
    if isinstance(shift, int) and isinstance(dim, int):
        torch_shift = [shift]
        torch_dim = [dim]
    else:
        torch_shift = shift
        torch_dim = dim
    out_torch = fastmri.roll(torch.from_numpy(x), torch_shift, torch_dim).numpy()
    out_numpy = np.roll(x, shift, dim)

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [5, 3],
        [2, 4, 6],
    ],
)
def test_fftshift(shape):
    x = np.arange(np.product(shape)).reshape(shape)
    out_torch = fastmri.fftshift(torch.from_numpy(x)).numpy()
    out_numpy = np.fft.fftshift(x)

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [5, 3],
        [2, 4, 5],
        [2, 7, 5],
    ],
)
def test_ifftshift(shape):
    x = np.arange(np.product(shape)).reshape(shape)
    out_torch = fastmri.ifftshift(torch.from_numpy(x)).numpy()
    out_numpy = np.fft.ifftshift(x)

    assert np.allclose(out_torch, out_numpy)
