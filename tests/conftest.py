"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib

import numpy as np
import pytest
import torch
import yaml

# these are really slow - skip by default
skip_module_test_flag = True
skip_data_test_flag = True


def create_input(shape):
    x = np.arange(np.product(shape)).reshape(shape)
    x = torch.from_numpy(x).float()

    return x


@pytest.fixture
def skip_module_test():
    return skip_module_test_flag


@pytest.fixture
def skip_data_test():
    return skip_data_test_flag


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
