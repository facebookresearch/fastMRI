import pytest

from fastmri.data.mri_data import SliceDataset, CombinedSliceDataset


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


def test_combined_dataset_lengths(
    knee_path, knee_split_lens, brain_path, brain_split_lens
):
    if knee_path is None or brain_path is None:
        pytest.skip("knee_path or brain_path not set in conftest.py")

    for knee_split, knee_data_len in knee_split_lens.items():
        for brain_split, brain_data_len in brain_split_lens.items():
            dataset = CombinedSliceDataset(
                [knee_path / knee_split, brain_path / brain_split],
                transforms=[None, None],
                challenges=["multicoil", "multicoil"],
            )

            assert len(dataset) == knee_data_len + brain_data_len
