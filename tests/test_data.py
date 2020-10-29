"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from fastmri.data.mri_data import SliceDataset, CombinedSliceDataset


def test_slice_datasets(fastmri_mock_dataset, monkeypatch):
    knee_path, brain_path, metadata = fastmri_mock_dataset

    def retrieve_metadata_mock(a, fname):
        return metadata[str(fname)]

    monkeypatch.setattr(SliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for challenge in ("multicoil", "singlecoil"):
        for split in ("train", "val", "test", "challenge"):
            dataset = SliceDataset(
                knee_path / f"{challenge}_{split}", transform=None, challenge=challenge
            )

            assert len(dataset) > 0
            assert dataset[0] is not None
            assert dataset[-1] is not None

    for challenge in ("multicoil",):
        for split in ("train", "val", "test", "challenge"):
            dataset = SliceDataset(
                brain_path / f"{challenge}_{split}", transform=None, challenge=challenge
            )

            assert len(dataset) > 0
            assert dataset[0] is not None
            assert dataset[-1] is not None


def test_combined_slice_dataset(fastmri_mock_dataset, monkeypatch):
    knee_path, brain_path, metadata = fastmri_mock_dataset

    def retrieve_metadata_mock(a, fname):
        return metadata[str(fname)]

    monkeypatch.setattr(SliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    roots = [knee_path / "multicoil_train", knee_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = SliceDataset(
        root=roots[0], challenge=challenges[0], transform=transforms[0]
    )
    dataset2 = SliceDataset(
        root=roots[1], challenge=challenges[1], transform=transforms[1]
    )
    comb_dataset = CombinedSliceDataset(
        roots=roots, challenges=challenges, transforms=transforms
    )

    assert len(comb_dataset) == len(dataset1) + len(dataset2)
    assert comb_dataset[0] is not None
    assert comb_dataset[-1] is not None

    roots = [brain_path / "multicoil_train", brain_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = SliceDataset(
        root=roots[0], challenge=challenges[0], transform=transforms[0]
    )
    dataset2 = SliceDataset(
        root=roots[1], challenge=challenges[1], transform=transforms[1]
    )
    comb_dataset = CombinedSliceDataset(
        roots=roots, challenges=challenges, transforms=transforms
    )

    assert len(comb_dataset) == len(dataset1) + len(dataset2)
    assert comb_dataset[0] is not None
    assert comb_dataset[-1] is not None
