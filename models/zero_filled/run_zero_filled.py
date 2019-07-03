"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib

import h5py

from common.args import Args
from common.utils import save_reconstructions
from data import transforms


def save_zero_filled(data_dir, out_dir, which_challenge, resolution):
    reconstructions = {}

    for file in data_dir.iterdir():
        with h5py.File(file) as hf:
            masked_kspace = transforms.to_tensor(hf['kspace'][()])
            # Inverse Fourier Transform to get zero filled solution
            image = transforms.ifft2(masked_kspace)
            # Crop input image
            image = transforms.complex_center_crop(image, (resolution, resolution))
            # Absolute value
            image = transforms.complex_abs(image)
            # Apply Root-Sum-of-Squares if multicoil data
            if which_challenge == 'multicoil':
                image = transforms.root_sum_of_squares(image, dim=1)

            reconstructions[file.name] = image
    save_reconstructions(reconstructions, out_dir)


def create_arg_parser():
    parser = Args()
    parser.add_argument('--out-path', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    save_zero_filled(args.data_path, args.out_path, args.challenge, args.resolution)
