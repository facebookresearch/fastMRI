"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import os
import re
import random
import numpy as np
import pdb
import logging
from collections import defaultdict

import h5py
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, args, fnames=None):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        challenge = args.challenge
        min_width = args.min_kspace_width
        max_width = args.max_kspace_width
        min_height = args.min_kspace_height
        max_height = args.max_kspace_height
        sample_rate = args.sample_rate
        acquisitions = args.acquisition_types
        systems = args.acquisition_systems
        min_target_width = args.min_target_width
        min_target_height = args.min_target_height
        max_target_width = args.max_target_width
        max_target_height = args.max_target_height
        min_num_coils = args.min_num_coils
        max_num_coils = args.max_num_coils
        start_slice = args.start_slice
        end_slice = args.end_slice
        only_square_targets = args.only_square_targets
        magnet = args.magnet
        filter_acceleration = args.filter_acceleration

        # picks `self.before_slices` and `self.after_slices` slices
        self.before_slices = args.before_slices
        self.after_slices = args.after_slices

        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(
                'challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        files = list(pathlib.Path(root).iterdir())
        self.slice_indices_by_size = defaultdict(list)
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        self.system_acquisitions = set()
        self.examples = []
        for fname in sorted(files):
            basename = os.path.basename(fname)

            if fnames is not None and basename not in fnames:
                continue

            # These are malformed brain images that should be removed from the
            # brain dataset. All of them are in the train set except the last.
            fnames_filter = ['file_brain_AXT2_200_2000446.h5',
                             'file_brain_AXT2_201_2010556.h5',
                             'file_brain_AXT2_208_2080135.h5',
                             'file_brain_AXT2_207_2070275.h5',
                             'file_brain_AXT2_208_2080163.h5',
                             'file_brain_AXT2_207_2070549.h5',
                             'file_brain_AXT2_207_2070254.h5',
                             'file_brain_AXT2_202_2020292.h5',
                             ]
            if fnames_filter is not None and basename in fnames_filter:
                continue

            data = h5py.File(fname, 'r')

            system_model = get_system_from_volume(data)
            if systems is not None and system_model not in systems:
                continue

            acquisition = 'AXT1' if data.attrs['acquisition'] == 'AXT1PRE' else data.attrs['acquisition']
            if acquisitions is not None and acquisition not in acquisitions:
                continue

            fs = float(re.search(
                '<systemFieldStrength_T>([^<]+)', data['ismrmrd_header'][()].decode()).group(1))
            if magnet is not None and abs(fs - magnet) > 1:
                continue

            if filter_acceleration is not None:
                if 'mask' in data:
                    inds = np.nonzero(data['mask'][()])[0]
                    if inds[1] - inds[0] == filter_acceleration:
                        continue

            min_num_coils_filter = min_num_coils is not None and data[
                'kspace'].shape[1] < min_num_coils
            max_num_coils_filter = max_num_coils is not None and data[
                'kspace'].shape[1] > max_num_coils
            if min_num_coils_filter or max_num_coils_filter:
                continue

            min_width_filter = min_width is not None and data['kspace'].shape[-1] < min_width
            max_width_filter = max_width is not None and data['kspace'].shape[-1] > max_width
            if min_width_filter or max_width_filter:
                continue

            min_height_filter = min_height is not None and data['kspace'].shape[-2] < min_height
            max_height_filter = max_height is not None and data['kspace'].shape[-2] > max_height
            if min_height_filter or max_height_filter:
                continue

            min_target_width_filter = (min_target_width is not None
                                       and data[self.recons_key].shape[-1] < min_target_width)
            min_target_height_filter = (min_target_height is not None and
                                        data[self.recons_key].shape[-2] < min_target_height)
            if min_target_width_filter or min_target_height_filter:
                continue

            max_target_width_filter = (max_target_width is not None
                                       and data[self.recons_key].shape[-1] > max_target_width)
            max_target_height_filter = (max_target_height is not None and
                                        data[self.recons_key].shape[-2] > min_target_height)
            if max_target_width_filter or max_target_height_filter:
                continue

            only_square_targets_filter = data[self.recons_key].shape[-1] != data[self.recons_key].shape[-2]
            if only_square_targets and only_square_targets_filter:
                continue

            # Compute the size of zero padding in k-space
            # We really should have stored this as an attribute in the hdf5 file
            try:
                import ismrmrd
                hdr = ismrmrd.xsd.CreateFromDocument(
                    data['ismrmrd_header'][()])
                enc = hdr.encoding[0]
                enc_size = (enc.encodedSpace.matrixSize.x,
                            enc.encodedSpace.matrixSize.y,
                            enc.encodedSpace.matrixSize.z)
                enc_limits_center = enc.encodingLimits.kspace_encoding_step_1.center
                enc_limits_max = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
            except:
                padding_left = 0
                padding_right = 0

            kspace = data['kspace']

            if end_slice is not None:
                end_slice = min(end_slice, kspace.shape[0])
                if end_slice < 0:
                    end_slice = kspace.shape[0] - end_slice
            else:
                end_slice = kspace.shape[0]

            if start_slice is None:
                start_slice = 0

            num_slices = end_slice - start_slice

            lower = self.before_slices + start_slice
            upper = end_slice - self.after_slices
            self.examples += [(fname, slice, padding_left, padding_right, num_slices, acquisition, system_model)
                              for slice in range(lower, upper)]

            slice_size = (kspace.shape[-2], kspace.shape[-1])
            slice_indices = range(len(self.examples) -
                                  num_slices, len(self.examples))
            self.slice_indices_by_size[slice_size].extend(slice_indices)

            self.system_acquisitions.add(system_model + '_' + acquisition)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, real_slice, padding_left, padding_right, _, _, _ = self.examples[i]
        slices = [real_slice] + list(range(real_slice - self.before_slices, real_slice)) \
            + list(range(real_slice + 1, real_slice + self.after_slices + 1))

        with h5py.File(fname, 'r') as data:
            if len(slices) > 1:
                kspace = np.stack([data['kspace'][slice] for slice in slices])
                target = np.stack([data[self.recons_key][slice]
                                   for slice in slices])
            else:
                kspace = data['kspace'][real_slice]
                target = data[self.recons_key][real_slice] if self.recons_key in data else None

            attrs = dict(data.attrs)
            attrs['system'] = get_system_from_volume(data)
            attrs['padding_left'] = padding_left
            attrs['padding_right'] = padding_right
            attrs['acquisition'] = 'AXT1' if attrs['acquisition'] == 'AXT1PRE' else attrs['acquisition']

            attrs['mask_offset'] = 0
            if 'mask' in data:
                ipat2_mask = data['mask'][()]
                inds = np.nonzero(ipat2_mask)[0]
                attrs['mask_offset'] = inds[0]

            return self.transform(kspace, target, attrs, fname.name, slices)


def get_system_from_volume(data):
    ismrmrd_header = data['ismrmrd_header'][()].decode('UTF-8')
    root = ET.fromstring(ismrmrd_header)
    system = root.findall(
        "./{0}acquisitionSystemInformation/{0}systemModel".format("{http://www.ismrm.org/ISMRMRD}"))[0]
    return system.text
