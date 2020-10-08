"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import sys
from argparse import ArgumentParser

import h5py
import ismrmrd

sys.path.append("../../")  # noqa: E402

import fastmri
from fastmri.data import transforms


def save_zero_filled(data_dir, out_dir, which_challenge):
    reconstructions = {}

    for f in data_dir.iterdir():
        with h5py.File(f, "r") as hf:
            enc = ismrmrd.xsd.CreateFromDocument(hf["ismrmrd_header"][()]).encoding[0]
            masked_kspace = transforms.to_tensor(hf["kspace"][()])

            # extract target image width, height from ismrmrd header
            crop_size = (enc.reconSpace.matrixSize.x, enc.reconSpace.matrixSize.y)

            # inverse Fourier Transform to get zero filled solution
            image = fastmri.ifft2c(masked_kspace)

            # check for FLAIR 203
            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            # crop input image
            image = transforms.complex_center_crop(image, crop_size)

            # absolute value
            image = fastmri.complex_abs(image)

            # apply Root-Sum-of-Squares if multicoil data
            if which_challenge == "multicoil":
                image = fastmri.rss(image, dim=1)

            reconstructions[f.name] = image

    fastmri.save_reconstructions(reconstructions, out_dir)


def create_arg_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_path", type=pathlib.Path, required=True, help="Path to the data",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        required=True,
        help="Path to save the reconstructions to",
    )
    parser.add_argument(
        "--challenge", type=str, required=True, help="Which challenge",
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    save_zero_filled(args.data_path, args.out_path, args.challenge)
