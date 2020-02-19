"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
sys.path.append(sys.path[0] + "/../../..")
import numpy as np
import pytest
import torch

from fastmri.common import utils
from fastmri.common.subsample import RandomMask
from fastmri.data import transforms


def create_input(shape):
    input = np.arange(np.product(shape)).reshape(shape)
    input = torch.from_numpy(input).float()
    return input


@pytest.mark.parametrize('shape, num_low_frequencies, accelerations', [
    ([4, 32, 32, 2], [round(0.08 * 368)], [4]),
    ([2, 64, 64, 2], [round(0.04 * 368), round(0.08 * 368)], [8, 4]),
])
def test_apply_mask(shape, num_low_frequencies, accelerations):
    mask_func = RandomMask(num_low_frequencies, accelerations)
    expected_mask, _ = mask_func(shape, seed=123)
    input = create_input(shape)
    output, mask, _ = transforms.apply_mask(input, mask_func, seed=123)
    assert output.shape == input.shape
    assert mask.shape == expected_mask.shape
    assert np.all(expected_mask.numpy() == mask.numpy())
    assert np.all((output * mask).numpy() == output.numpy())


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_fft2(shape):
    shape = shape + [2]
    input = create_input(shape)
    out_torch = transforms.fft2(input).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = utils.tensor_to_complex_np(input)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_ifft2(shape):
    shape = shape + [2]
    input = create_input(shape)
    out_torch = transforms.ifft2(input).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = utils.tensor_to_complex_np(input)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_complex_abs(shape):
    shape = shape + [2]
    input = create_input(shape)
    out_torch = transforms.complex_abs(input).numpy()
    input_numpy = utils.tensor_to_complex_np(input)
    out_numpy = np.abs(input_numpy)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape, dim', [
    [[3, 3], 0],
    [[4, 6], 1],
    [[10, 8, 4], 2],
])
def test_root_sum_of_squares(shape, dim):
    input = create_input(shape)
    out_torch = transforms.root_sum_of_squares(input, dim).numpy()
    out_numpy = np.sqrt(np.sum(input.numpy() ** 2, dim))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape, target_shape', [
    [[10, 10], [4, 4]],
    [[4, 6], [2, 4]],
    [[8, 4], [4, 4]],
])
def test_center_crop(shape, target_shape):
    input = create_input(shape)
    out_torch = transforms.center_crop(input, target_shape).numpy()
    assert list(out_torch.shape) == target_shape

@pytest.mark.parametrize('x_shape, y_shape, target_shape', [
    [[320, 320], [380, 380], [320, 320]],
    [[320, 320], [220, 220], [220, 220]],
])
def test_center_crop_to_smallest(x_shape, y_shape, target_shape):
    input_x = create_input(x_shape)
    input_y = create_input(y_shape)
    x_out, y_out = transforms.center_crop_to_smallest(input_x, input_y)
    assert x_out.shape == y_out.shape
    assert list(x_out.shape) == target_shape

@pytest.mark.parametrize('data_shape, shape, target_shape', [
    [[384, 384], [320, 320], [320, 320]],
    [[220, 220], [320, 320], [320, 320]],
    [[384, 220], [320, 320], [320, 320]],
])
def test_center_crop_or_pad(data_shape, shape, target_shape):
    input = create_input(data_shape)
    result = transforms.center_crop_or_pad(input, shape)
    assert list(result.shape) == target_shape

@pytest.mark.parametrize('shape, target_shape', [
    [[10, 10], [4, 4]],
    [[4, 6], [2, 4]],
    [[8, 4], [4, 4]],
])
def test_complex_center_crop(shape, target_shape):
    shape = shape + [2]
    input = create_input(shape)
    out_torch = transforms.complex_center_crop(input, target_shape).numpy()
    assert list(out_torch.shape) == target_shape + [2, ]

@pytest.mark.parametrize('shape, mean, stddev', [
    [[10, 10], 0, 1],
    [[4, 6], 4, 10],
    [[8, 4], 2, 3],
])
def test_normalize(shape, mean, stddev):
    input = create_input(shape)
    output = transforms.normalize(input, mean, stddev).numpy()
    assert np.isclose(output.mean(), (input.numpy().mean() - mean) / stddev)
    assert np.isclose(output.std(), input.numpy().std() / stddev)


@pytest.mark.parametrize('shape', [
    [10, 10],
    [20, 40, 30],
])
def test_normalize_instance(shape):
    input = create_input(shape)
    output, mean, stddev = transforms.normalize_instance(input)
    output = output.numpy()
    assert np.isclose(input.numpy().mean(), mean, rtol=1e-2)
    assert np.isclose(input.numpy().std(), stddev, rtol=1e-2)
    assert np.isclose(output.mean(), 0, rtol=1e-2, atol=1e-3)
    assert np.isclose(output.std(), 1, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('shift, dim', [
    (0, 0),
    (1, 0),
    (-1, 0),
    (100, 0),
    ((1, 2), (1, 2)),
])
@pytest.mark.parametrize('shape', [
    [5, 6, 2],
    [3, 4, 5],
])
def test_roll(shift, dim, shape):
    input = np.arange(np.product(shape)).reshape(shape)
    out_torch = transforms.roll(torch.from_numpy(input), shift, dim).numpy()
    out_numpy = np.roll(input, shift, dim)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [5, 3],
    [2, 4, 6],
])
def test_fftshift(shape):
    input = np.arange(np.product(shape)).reshape(shape)
    out_torch = transforms.fftshift(torch.from_numpy(input)).numpy()
    out_numpy = np.fft.fftshift(input)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [5, 3],
    [2, 4, 5],
    [2, 7, 5],
])
def test_ifftshift(shape):
    input = np.arange(np.product(shape)).reshape(shape)
    out_torch = transforms.ifftshift(torch.from_numpy(input)).numpy()
    out_numpy = np.fft.ifftshift(input)
    assert np.allclose(out_torch, out_numpy)
