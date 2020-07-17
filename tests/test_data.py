import pytest

from fastmri.data.mri_data import SliceDataset


def test_knee_dataset_lengths(knee_path, knee_split_lens):
    if knee_path is None:
        pytest.skip("knee_path not set in conftest.py")

    for split, data_len in knee_split_lens.items():
        challenge = "multicoil" if "multicoil" in split else "singlecoil"
        dataset = SliceDataset(knee_path / split, transform=None, challenge=challenge)

        assert len(dataset) == data_len


def test_brain_dataset_lengths(brain_path, brain_split_lens):
    if brain_path is None:
        pytest.skip("brain_path not set in conftest.py")

    for split, data_len in brain_split_lens.items():
        dataset = SliceDataset(
            brain_path / split, transform=None, challenge="multicoil"
        )

        assert len(dataset) == data_len
