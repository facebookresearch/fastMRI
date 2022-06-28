"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pytest

from fastmri.data import transforms
from fastmri.data.subsample import MaskFunc, RandomMaskFunc, create_mask_for_mask_type

from .conftest import create_input


@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [([4, 150, 75, 2], [0.08], [4]), ([2, 120, 60, 2], [0.04, 0.08], [8, 4])],
)
def test_apply_mask(shape, center_fractions, accelerations):
    state = np.random.get_state()

    mask_func = RandomMaskFunc(center_fractions, accelerations)
    expected_mask, expected_num_low_frequencies = mask_func(shape, seed=123)
    assert expected_num_low_frequencies in [
        round(cf * shape[-2]) for cf in center_fractions
    ]
    x = create_input(shape)
    output, mask, num_low_frequencies = transforms.apply_mask(x, mask_func, seed=123)

    assert (state[1] == np.random.get_state()[1]).all()
    assert output.shape == x.shape
    assert mask.shape == expected_mask.shape
    assert np.all(expected_mask.numpy() == mask.numpy())
    assert np.all(np.where(mask.numpy() == 0, 0, output.numpy()) == output.numpy())
    assert num_low_frequencies == expected_num_low_frequencies


@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [([150, 75, 2], [0.08], [4]), ([120, 60, 2], [0.04, 0.08], [8, 4])],
)
def test_apply_mask_with_padding(shape, center_fractions, accelerations):
    state = np.random.get_state()

    padding = [3, shape[-2] - 3]
    mask_func = RandomMaskFunc(center_fractions, accelerations)
    expected_mask, expected_num_low_frequencies = mask_func(shape, seed=123)
    assert expected_num_low_frequencies in [
        round(cf * shape[-2]) for cf in center_fractions
    ]
    x = create_input(shape)
    output, mask, num_low_frequencies = transforms.apply_mask(
        x,
        mask_func,
        seed=123,
        padding=padding,
    )

    assert (state[1] == np.random.get_state()[1]).all()
    assert output.shape == x.shape
    assert mask.shape == expected_mask.shape
    assert np.unique(mask.numpy()).tolist() == [0.0, 1.0]
    assert np.all(
        expected_mask[..., padding[0] : padding[1], :].numpy()
        == mask[..., padding[0] : padding[1], :].numpy()
    )
    assert np.all(np.where(mask.numpy() == 0, 0, output.numpy()) == output.numpy())
    assert num_low_frequencies == expected_num_low_frequencies


@pytest.mark.parametrize(
    "mask_type",
    ["random", "equispaced", "equispaced_fraction", "magic", "magic_fraction"],
)
def test_mask_types(mask_type):
    shape_list = ((4, 32, 32, 2), (2, 64, 32, 2), (1, 33, 24, 2))
    center_fraction_list = ([0.08], [0.04], [0.04, 0.08])
    acceleration_list = ([4], [8], [4, 8])
    state = np.random.get_state()

    for shape in shape_list:
        for center_fractions, accelerations in zip(
            center_fraction_list, acceleration_list
        ):
            mask_func = create_mask_for_mask_type(
                mask_type, center_fractions, accelerations
            )
            expected_mask, expected_num_low_frequencies = mask_func(shape, seed=123)
            x = create_input(shape)
            output, mask, num_low_frequencies = transforms.apply_mask(
                x, mask_func, seed=123
            )

            assert (state[1] == np.random.get_state()[1]).all()
            assert output.shape == x.shape
            assert mask.shape == expected_mask.shape
            assert np.all(expected_mask.numpy() == mask.numpy())
            assert np.all(
                np.where(mask.numpy() == 0, 0, output.numpy()) == output.numpy()
            )
            assert num_low_frequencies == expected_num_low_frequencies


@pytest.mark.parametrize(
    (
        "allow_any_combination, center_fractions, accelerations, seed, "
        "choose_acceleration_output"
    ),
    [
        (True, [0.04, 0.08], [8, 4], 2, (0.04, 4)),
        (False, [0.04, 0.08], [8, 4], 2, (0.04, 8)),
        (True, [0.04, 0.08], [16, 8, 4], 8, (0.08, 16)),
    ],
)
def test_fraction_accel_sync(
    allow_any_combination,
    center_fractions,
    accelerations,
    seed,
    choose_acceleration_output,
):
    """For this test we have manually checked choose_acceleration_output."""
    f = MaskFunc(
        center_fractions=center_fractions,
        accelerations=accelerations,
        allow_any_combination=allow_any_combination,
        seed=seed,
    )
    if not allow_any_combination:
        for _ in range(50):
            center_fraction, acceleration = f.choose_acceleration()
            assert center_fractions.index(center_fraction) == accelerations.index(
                acceleration
            )
    else:
        assert f.choose_acceleration() == choose_acceleration_output


@pytest.mark.parametrize(
    "shape, target_shape", [[[10, 10], [4, 4]], [[4, 6], [2, 4]], [[8, 4], [4, 4]]]
)
def test_center_crop(shape, target_shape):
    x = create_input(shape)
    out_torch = transforms.center_crop(x, target_shape).numpy()

    assert list(out_torch.shape) == target_shape


@pytest.mark.parametrize(
    "shape, target_shape", [[[10, 10], [4, 4]], [[4, 6], [2, 4]], [[8, 4], [4, 4]]]
)
def test_complex_center_crop(shape, target_shape):
    shape = shape + [2]
    x = create_input(shape)
    out_torch = transforms.complex_center_crop(x, target_shape).numpy()

    assert list(out_torch.shape) == target_shape + [
        2,
    ]


@pytest.mark.parametrize(
    "shape, mean, stddev", [[[10, 10], 0, 1], [[4, 6], 4, 10], [[8, 4], 2, 3]]
)
def test_normalize(shape, mean, stddev):
    x = create_input(shape)
    output = transforms.normalize(x, mean, stddev).numpy()

    assert np.isclose(output.mean(), (x.numpy().mean() - mean) / stddev)
    assert np.isclose(output.std(), x.numpy().std() / stddev)


@pytest.mark.parametrize("shape", [[10, 10], [20, 40, 30]])
def test_normalize_instance(shape):
    x = create_input(shape)
    output, mean, stddev = transforms.normalize_instance(x)
    output = output.numpy()

    assert np.isclose(x.numpy().mean(), mean, rtol=1e-2)
    assert np.isclose(x.numpy().std(), stddev, rtol=1e-2)
    assert np.isclose(output.mean(), 0, rtol=1e-2, atol=1e-3)
    assert np.isclose(output.std(), 1, rtol=1e-2, atol=1e-3)
