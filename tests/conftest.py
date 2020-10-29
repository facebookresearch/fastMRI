"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pytest
import torch
from .create_temp_data import create_temp_data

# these are really slow - skip by default
SKIP_INTEGRATIONS = True


def create_input(shape):
    x = np.arange(np.product(shape)).reshape(shape)
    x = torch.from_numpy(x).float()

    return x


@pytest.fixture(scope="session")
def fastmri_mock_dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("fastmri_data")

    return create_temp_data(path)


@pytest.fixture
def skip_integration_tests():
    return SKIP_INTEGRATIONS


@pytest.fixture
def knee_split_lens():
    split_lens = {
        "multicoil_train": 34742,
        "multicoil_val": 7135,
        "multicoil_test": 4092,
        "singlecoil_train": 34742,
        "singlecoil_val": 7135,
        "singlecoil_test": 3903,
    }

    return split_lens


@pytest.fixture
def brain_split_lens():
    split_lens = {
        "multicoil_train": 70748,
        "multicoil_val": 21842,
        "multicoil_test": 8852,
    }

    return split_lens
