"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import pickle
import random

import h5py
import ismrmrd
import numpy as np
import yaml
from torch.utils.data import Dataset


def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path, 
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/path/to/knee",
            brain_path="/path/to/brain",
            log_path="/path/to/log",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


class CombinedSliceDataset(Dataset):
    """
    A container for combining slice datasets.

    Args:
        roots (list of pathlib.Path): Paths to the datasets.
        transforms (list of callable): A callable object that pre-processes the
            raw data into appropriate form. The transform function should take
            'kspace', 'target', 'attributes', 'filename', and 'slice' as
            inputs. 'target' may be null for test data.
        challenges (list of str): "singlecoil" or "multicoil" depending on which
            challenge to use.
        sample_rates (list of float, optional): A float between 0 and 1. This
            controls what fraction of the volumes should be loaded.
        num_cols (tuple(int), optional): if provided, only slices with the desired
            number of columns will be considered.
    """

    def __init__(self, roots, transforms, challenges, sample_rates=None, num_cols=None):
        assert len(roots) == len(transforms) == len(challenges)
        if sample_rates is not None:
            assert len(sample_rates) == len(roots)
        else:
            sample_rates = [1] * len(roots)

        self.datasets = list()
        for i in range(len(roots)):
            self.datasets.append(
                SliceDataset(
                    roots[i],
                    transforms[i],
                    challenges[i],
                    sample_rates[i],
                    num_cols=num_cols,
                )
            )

    def __len__(self):
        length = 0
        for dataset in self.datasets:
            length = length + len(dataset)

        return length

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)


class SliceDataset(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.

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
        dataset_cache_file (pathlib.Path). A file in which to cache dataset
            information for faster load times. Default: dataset_cache.pkl.
        num_cols (tuple(int), optional): if provided, only slices with the desired
            number of columns will be considered.
    """

    def __init__(
        self,
        root,
        transform,
        challenge,
        sample_rate=1,
        dataset_cache_file=pathlib.Path("dataset_cache.pkl"),
        num_cols=None,
    ):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.dataset_cache_file = dataset_cache_file

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        if self.dataset_cache_file.exists():
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        if dataset_cache.get(root) is None:
            files = list(pathlib.Path(root).iterdir())
            for fname in sorted(files):
                with h5py.File(fname, "r") as hf:
                    hdr = ismrmrd.xsd.CreateFromDocument(hf["ismrmrd_header"][()])
                    enc = hdr.encoding[0]

                    enc_size = (
                        enc.encodedSpace.matrixSize.x,
                        enc.encodedSpace.matrixSize.y,
                        enc.encodedSpace.matrixSize.z,
                    )
                    recon_size = (
                        enc.reconSpace.matrixSize.x,
                        enc.reconSpace.matrixSize.y,
                        enc.reconSpace.matrixSize.z,
                    )

                    enc_limits_center = enc.encodingLimits.kspace_encoding_step_1.center
                    enc_limits_max = (
                        enc.encodingLimits.kspace_encoding_step_1.maximum + 1
                    )
                    padding_left = enc_size[1] // 2 - enc_limits_center
                    padding_right = padding_left + enc_limits_max

                    num_slices = hf["kspace"].shape[0]

                metadata = {
                    "padding_left": padding_left,
                    "padding_right": padding_right,
                    "encoding_size": enc_size,
                    "recon_size": recon_size,
                }

                self.examples += [
                    (fname, slice_ind, metadata) for slice_ind in range(num_slices)
                ]

            dataset_cache[root] = self.examples
            logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
            with open(self.dataset_cache_file, "wb") as f:
                pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]

        if num_cols:
            self.examples = [
                ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols
            ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice, metadata = self.examples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        return self.transform(kspace, mask, target, attrs, fname.name, dataslice)
