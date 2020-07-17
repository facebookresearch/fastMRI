"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import h5py
import numpy as np
from torch.utils.data import Dataset


def fetch_data_dir(split, knee_path=None, brain_path=None):
    """Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        split (str): A string specifying the split, one of ("multicoil_train",
            "multicoil_val", "multicoil_test", "singlecoil_train",
            "singlecoil_val", "singlecoil_test").
        knee_path (pathlib.Path): The path to the fastmri knee data.
        brain_path (pathlib.Path): The path to the fastmri brain data.

    Returns:
        pathlib.Path: The path to the specified dataset.
    """
    data_dirs = dict()

    if knee_path is not None:
        knee_dirs = dict(
            multicoil_knee_train=knee_path / "multicoil_train",
            multicoil_knee_val=knee_path / "multicoil_val",
            multicoil_knee_test=knee_path / "multicoil_test",
            singlecoil_knee_train=knee_path / "singlecoil_train",
            singlecoil_knee_val=knee_path / "singlecoil_val",
            singlecoil_knee_test=knee_path / "singlecoil_test",
        )
        data_dirs.update(knee_dirs)

    if brain_path is not None:
        brain_dirs = dict(
            multicoil_brain_train=brain_path / "multicoil_train",
            multicoil_brain_val=brain_path / "multicoil_val",
            multicoil_brain_test=brain_path / "multicoil_test",
        )
        data_dirs.update(brain_dirs)

    data_dir = data_dirs[split]

    return data_dir


class SliceDataset(Dataset):
    """A PyTorch Dataset that provides access to MR image slices.

    Args:
        root (pathlib.Path): Path to the dataset.
        transform (callable): A callable object that pre-processes the raw data
            into appropriate form. The transform function should take 'kspace',
            'target', 'attributes', 'filename', and 'slice' as inputs. 'target'
            may be null for test data.
        challenge (str): "singlecoil" or "multicoil" depending on which
            challenge to use.
        sample_rate (float, optional): A float between 0 and 1. This controls
            what fraction of the volumes should be loaded.
    """

    def __init__(self, root, transform, challenge, sample_rate=1):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        for fname in sorted(files):
            data = h5py.File(fname, "r")

            # Compute the size of zero padding in k-space
            # We really should have stored this as an attribute in the hdf5 file
            try:
                import ismrmrd

                hdr = ismrmrd.xsd.CreateFromDocument(data["ismrmrd_header"][()])
                enc = hdr.encoding[0]
                enc_size = (
                    enc.encodedSpace.matrixSize.x,
                    enc.encodedSpace.matrixSize.y,
                    enc.encodedSpace.matrixSize.z,
                )
                enc_limits_center = enc.encodingLimits.kspace_encoding_step_1.center
                enc_limits_max = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
            except Exception as e:
                padding_left = None
                padding_right = None
                raise e

            kspace = data["kspace"]
            num_slices = kspace.shape[0]
            self.examples += [
                (fname, slice, padding_left, padding_right)
                for slice in range(num_slices)
            ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, padding_left, padding_right = self.examples[i]

        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice]
            mask = np.asarray(data["mask"]) if "mask" in data else None
            target = data[self.recons_key][slice] if self.recons_key in data else None
            attrs = dict(data.attrs)
            attrs["padding_left"] = padding_left
            attrs["padding_right"] = padding_right

        return self.transform(kspace, mask, target, attrs, fname.name, slice)
