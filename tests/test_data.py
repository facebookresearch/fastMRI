"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree. 
"""

from fastmri.data.mri_data import (
    SliceDataset,
    CombinedSliceDataset,
    AnnotatedSliceDataset,
)


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


def test_annotated_slice_dataset(
    fastmri_mock_dataset, fastmri_mock_annotation, monkeypatch
):
    knee_path, brain_path, metadata = fastmri_mock_dataset
    annotation_knee_csv, annotation_brain_csv = fastmri_mock_annotation

    def download_csv_mock(a,version,subsplit,path):
        if subsplit == "knee":
            return annotation_knee_csv
        else:
            return annotation_brain_csv

    def retrieve_metadata_mock(a, fname):
        return metadata[str(fname)]

    monkeypatch.setattr(AnnotatedSliceDataset, "download_csv", download_csv_mock)

    monkeypatch.setattr(SliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for challenge in ("multicoil", "singlecoil"):
        for split in ("train", "val", "test", "challenge"):
            for multiple_annotation_policy in ("first", "random", "all"):
                dataset = AnnotatedSliceDataset(
                    knee_path / f"{challenge}_{split}",
                    challenge=challenge,
                    subsplit="knee",
                    multiple_annotation_policy=multiple_annotation_policy,
                )

                assert len(dataset) > 0
                assert dataset[0] is not None
                assert dataset[-1] is not None
                assert dataset[0][3]["annotation"] is not None
                assert dataset[-1][3]["annotation"] is not None

    for challenge in ("multicoil",):
        for split in ("train", "val", "test", "challenge"):
            for multiple_annotation_policy in ("first", "random", "all"):
                dataset = AnnotatedSliceDataset(
                    brain_path / f"{challenge}_{split}",
                    challenge=challenge,
                    subsplit="brain",
                    multiple_annotation_policy=multiple_annotation_policy,
                )

                assert len(dataset) > 0
                assert dataset[0] is not None
                assert dataset[-1] is not None
                assert dataset[0][3]["annotation"] is not None
                assert dataset[-1][3]["annotation"] is not None
