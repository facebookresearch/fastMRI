"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import multiprocessing
import pathlib
import time
from argparse import ArgumentParser
from collections import defaultdict

import bart
import fastmri
import numpy as np
import torch
import yaml
from fastmri import tensor_to_complex_np
from fastmri.data import SliceDataset
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type


class DataTransform(object):
    """
    Data Transformer that masks input k-space.
    """

    def __init__(self, split, reg_wt=None, mask_func=None, use_seed=True):
        if split in ("train", "val"):
            self.retrieve_acc = False
            self.mask_func = mask_func
        else:
            self.retrieve_acc = True
            self.mask_func = None

        self.reg_wt = reg_wt
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Data Transformer that simply returns the input masked k-space data and
        relevant attributes needed for running MRI reconstruction algorithms
        implemented in BART.

        Args:
            masked_kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
            target (numpy.array, optional): Target image.
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            tuple: tuple containing:
                masked_kspace (torch.Tensor): Sub-sampled k-space with the same
                    shape as kspace.
                reg_wt (float): Regularization parameter.
                fname (str): File name containing the current data item.
                slice_num (int): The index of the current slice in the volume.
                crop_size (tuple): Size of the image to crop to given ISMRMRD
                    header.
                num_low_freqs (int): Number of low-resolution lines acquired.
        """
        kspace = T.to_tensor(kspace)

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        if self.retrieve_acc:
            num_low_freqs = attrs["num_low_frequency"]
        else:
            num_low_freqs = None

        if self.retrieve_acc and self.reg_wt is None:
            acquisition = attrs["acquisition"]
            acceleration = attrs["acceleration"]

            with open("cs_config.yaml", "r") as f:
                param_dict = yaml.safe_load(f)

            if acquisition not in param_dict[args.challenge]:
                raise ValueError(f"Invalid acquisition protocol: {acquisition}")
            if acceleration not in (4, 8):
                raise ValueError(f"Invalid acceleration factor: {acceleration}")

            reg_wt = param_dict[args.challenge][acquisition][acceleration]
        else:
            reg_wt = self.reg_wt

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        return (masked_kspace, reg_wt, fname, slice_num, crop_size, num_low_freqs)


def cs_total_variation(args, kspace, reg_wt, crop_size, num_low_freqs):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization
    based reconstruction algorithm using the BART toolkit.

    Args:
        args (argparse.Namespace): Arguments including ESPIRiT parameters.
        reg_wt (float): Regularization parameter.
        crop_size (tuple): Size to crop final image to.

    Returns:
        np.array: Reconstructed image.
    """
    if args.challenge == "singlecoil":
        kspace = kspace.unsqueeze(0)

    kspace = kspace.permute(1, 2, 0, 3).unsqueeze(0)
    kspace = tensor_to_complex_np(kspace)

    # estimate sensitivity maps
    if num_low_freqs is None:
        sens_maps = bart.bart(1, "ecalib -d0 -m1", kspace)
    else:
        sens_maps = bart.bart(1, f"ecalib -d0 -m1 -r {num_low_freqs}", kspace)

    # use Total Variation Minimization to reconstruct the image
    pred = bart.bart(
        1, f"pics -d0 -S -R T:7:0:{reg_wt} -i {args.num_iters}", kspace, sens_maps
    )
    pred = torch.from_numpy(np.abs(pred[0]))

    # check for FLAIR 203
    if pred.shape[1] < crop_size[1]:
        crop_size = (pred.shape[1], pred.shape[1])

    return T.center_crop(pred, crop_size)


def save_outputs(outputs, output_path):
    """Saves reconstruction outputs to output_path."""
    reconstructions = defaultdict(list)

    for fname, slice_num, pred in outputs:
        reconstructions[fname].append((slice_num, pred))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    fastmri.save_reconstructions(reconstructions, output_path)


def run_model(idx):
    """
    Run BART on idx index from dataset.

    Args:
        idx (int): The index of the dataset.

    Returns:
        tuple: tuple with
            fname: Filename
            slice_num: Slice number.
            prediction: Reconstructed image.
    """
    masked_kspace, reg_wt, fname, slice_num, crop_size, num_low_freqs = dataset[idx]

    prediction = cs_total_variation(
        args, masked_kspace, reg_wt, crop_size, num_low_freqs
    )

    return fname, slice_num, prediction


def run_bart(args):
    """Run the BART reconstruction on the given data set."""
    if args.num_procs == 0:
        start_time = time.perf_counter()
        outputs = []
        for i in range(len(dataset)):
            outputs.append(run_model(i))
        time_taken = time.perf_counter() - start_time
    else:
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_model, range(len(dataset)))
            time_taken = time.perf_counter() - start_time

    logging.info(f"Run Time = {time_taken:} s")
    save_outputs(outputs, args.output_path)


def create_arg_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--output_path",
        type=pathlib.Path,
        required=True,
        help="Path to save the reconstructions to",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        required=True,
        help="Which challenge",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Percent of data to run",
    )
    parser.add_argument(
        "--mask_type", choices=["random", "equispaced"], default="random", type=str
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "challenge"],
        default="val",
        type=str,
    )
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int)
    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)

    # bart args
    parser.add_argument(
        "--num_iters",
        type=int,
        default=200,
        help="Number of iterations to run the reconstruction algorithm",
    )
    parser.add_argument(
        "--reg_wt", type=float, default=0.01, help="Regularization weight parameter"
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=4,
        help="Number of processes. Set to 0 to disable multiprocessing.",
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()

    if args.split in ("train", "val"):
        mask = create_mask_for_mask_type(
            args.mask_type,
            args.center_fractions,
            args.accelerations,
        )
    else:
        mask = None
        args.reg_wt = None

    # need this global for multiprocessing
    dataset = SliceDataset(
        root=args.data_path / f"{args.challenge}_{args.split}",
        transform=DataTransform(split=args.split, mask_func=mask, reg_wt=args.reg_wt),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
    )

    run_bart(args)
