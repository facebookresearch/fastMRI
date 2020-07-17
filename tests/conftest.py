import pathlib

import pytest
import yaml

# knee data parameters
@pytest.fixture
def knee_path():
    with open("tests/fastmri_dirs.yaml", "r") as f:
        data_dir = yaml.safe_load(f)["knee_path"]

    if data_dir is not None:
        data_dir = pathlib.Path(data_dir)

    return data_dir


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


# brain data parameters
@pytest.fixture
def brain_path():
    with open("tests/fastmri_dirs.yaml", "r") as f:
        data_dir = yaml.safe_load(f)["brain_path"]

    if data_dir is not None:
        data_dir = pathlib.Path(data_dir)

    return data_dir


@pytest.fixture
def brain_split_lens():
    split_lens = {
        "multicoil_train": 92590,
        "multicoil_val": 21842,
        "multicoil_test": 8852,
    }

    return split_lens
