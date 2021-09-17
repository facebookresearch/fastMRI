"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import h5py
import numpy as np


def create_temp_data(path):
    rg = np.random.default_rng(seed=1234)
    max_num_slices = 15
    max_num_coils = 15
    data_splits = {
        "knee_data": [
            "multicoil_train",
            "multicoil_val",
            "multicoil_test",
            "multicoil_challenge",
            "singlecoil_train",
            "singlecoil_val",
            "singlecoil_test",
            "singlecoil_challenge",
        ],
        "brain_data": [
            "multicoil_train",
            "multicoil_val",
            "multicoil_test",
            "multicoil_challenge",
        ],
    }

    enc_sizes = {
        "train": [(1, 128, 64), (1, 128, 49), (1, 150, 67)],
        "val": [(1, 128, 64), (1, 170, 57)],
        "test": [(1, 128, 64), (1, 96, 96)],
        "challenge": [(1, 128, 64), (1, 96, 48)],
    }
    recon_sizes = {
        "train": [(1, 64, 64), (1, 49, 49), (1, 67, 67)],
        "val": [(1, 64, 64), (1, 57, 47)],
        "test": [(1, 64, 64), (1, 96, 96)],
        "challenge": [(1, 64, 64), (1, 48, 48)],
    }

    metadata = {}
    for dataset in data_splits:
        for split in data_splits[dataset]:
            fcount = 0
            (path / dataset / split).mkdir(parents=True)
            encs = enc_sizes[split.split("_")[-1]]
            recs = recon_sizes[split.split("_")[-1]]
            for i in range(len(encs)):
                fname = path / dataset / split / f"file{fcount}.h5"
                num_slices = rg.integers(2, max_num_slices)
                if "multicoil" in split:
                    num_coils = rg.integers(2, max_num_coils)
                    enc_size = (num_slices, num_coils, encs[i][-2], encs[i][-1])
                    recon_size = (num_slices, recs[i][-2], recs[i][-1])
                else:
                    enc_size = (num_slices, encs[i][-2], encs[i][-1])
                    recon_size = (num_slices, recs[i][-2], recs[i][-1])

                data = rg.normal(size=enc_size) + 1j * rg.normal(size=enc_size)

                if split.split("_")[-1] in ("train", "val"):
                    recon = np.absolute(rg.normal(size=recon_size)).astype(
                        np.dtype("<f4")
                    )
                else:
                    mask = rg.integers(0, 2, size=recon_size[-1]).astype(bool)

                with h5py.File(fname, "w") as hf:
                    hf.create_dataset("kspace", data=data.astype(np.complex64))
                    if split.split("_")[-1] in ("train", "val"):
                        hf.attrs["max"] = recon.max()
                        if "singlecoil" in split:
                            hf.create_dataset("reconstruction_esc", data=recon)
                        else:
                            hf.create_dataset("reconstruction_rss", data=recon)
                    else:
                        hf.create_dataset("mask", data=mask)

                enc_size = encs[i]

                enc_limits_center = enc_size[1] // 2 + 1
                enc_limits_max = enc_size[1] - 2

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max

                metadata[str(fname)] = (
                    {
                        "padding_left": padding_left,
                        "padding_right": padding_right,
                        "encoding_size": enc_size,
                        "recon_size": recon_size,
                    },
                    num_slices,
                )

                fcount += 1

    return path / "knee_data", path / "brain_data", metadata
