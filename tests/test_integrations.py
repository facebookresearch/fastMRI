"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pytest
from fastmri.data.mri_data import CombinedSliceDataset, SliceDataset, fetch_dir


def test_knee_dataset_lengths(knee_split_lens, skip_integration_tests):
    if skip_integration_tests:
        pytest.skip("config set to skip")

    knee_path = fetch_dir("knee_path")

    for split, data_len in knee_split_lens.items():
        challenge = "multicoil" if "multicoil" in split else "singlecoil"
        dataset = SliceDataset(knee_path / split, transform=None, challenge=challenge)

        assert len(dataset) == data_len


def test_brain_dataset_lengths(brain_split_lens, skip_integration_tests):
    if skip_integration_tests:
        pytest.skip("config set to skip")

    brain_path = fetch_dir("brain_path")

    for split, data_len in brain_split_lens.items():
        dataset = SliceDataset(
            brain_path / split, transform=None, challenge="multicoil"
        )

        assert len(dataset) == data_len


def test_combined_dataset_lengths(
    knee_split_lens, brain_split_lens, skip_integration_tests
):
    if skip_integration_tests:
        pytest.skip("config set to skip")

    knee_path = fetch_dir("knee_path")
    brain_path = fetch_dir("brain_path")

    for knee_split, knee_data_len in knee_split_lens.items():
        for brain_split, brain_data_len in brain_split_lens.items():
            dataset = CombinedSliceDataset(
                [knee_path / knee_split, brain_path / brain_split],
                transforms=[None, None],
                challenges=["multicoil", "multicoil"],
            )

            assert len(dataset) == knee_data_len + brain_data_len
