"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import multiprocessing
import pathlib
import random
import time
from collections import defaultdict

import numpy as np
import torch

import bart
from common import utils
from common.args import Args
from common.subsample import MaskFunc
from common.utils import tensor_to_complex_np
from data import transforms
from data.mri_data import SliceData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer that masks input k-space.
    """

    def __init__(self, mask_func):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        self.mask_func = mask_func

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array, optional): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                masked_kspace (torch.Tensor): Sub-sampled k-space with the same shape as kspace.
                fname (str): File name containing the current data item
                slice (int): The index of the current slice in the volume
        """
        kspace = transforms.to_tensor(kspace)
        seed = tuple(map(ord, fname))
        # Apply mask to raw k-space
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        return masked_kspace, fname, slice


def create_data_loader(args):
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    data = SliceData(
        root=args.data_path / f'{args.challenge}_val',
        transform=DataTransform(dev_mask),
        challenge=args.challenge,
        sample_rate=args.sample_rate
    )
    return data


def cs_total_variation(args, kspace):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization based
    reconstruction algorithm using the BART toolkit.
    """

    if args.challenge == 'singlecoil':
        kspace = kspace.unsqueeze(0)
    kspace = kspace.permute(1, 2, 0, 3).unsqueeze(0)
    kspace = tensor_to_complex_np(kspace)

    # Estimate sensitivity maps
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', kspace)

    # Use Total Variation Minimization to reconstruct the image
    pred = bart.bart(
        1, f'pics -d0 -S -R T:7:0:{args.reg_wt} -i {args.num_iters}', kspace, sens_maps
    )
    pred = torch.from_numpy(np.abs(pred[0]))

    # Crop the predicted image to the correct size
    return transforms.center_crop(pred, (args.resolution, args.resolution))


def run_model(i):
    masked_kspace, fname, slice = data[i]
    prediction = cs_total_variation(args, masked_kspace)
    return fname, slice, prediction


def main():
    if args.num_procs == 0:
        start_time = time.perf_counter()
        outputs = []
        for i in range(len(data)):
            outputs.append(run_model(i))
        time_taken = time.perf_counter() - start_time
    else:
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_model, range(len(data)))
            time_taken = time.perf_counter() - start_time
    logging.info(f'Run Time = {time_taken:}s')
    save_outputs(outputs, args.output_path)


def save_outputs(outputs, output_path):
    reconstructions = defaultdict(list)
    for fname, slice, pred in outputs:
        reconstructions[fname].append((slice, pred))
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    utils.save_reconstructions(reconstructions, output_path)


if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--output-path', type=pathlib.Path, default=None,
                        help='Path to save the reconstructions to')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--reg-wt', type=float, default=0.01,
                        help='Regularization weight parameter')
    parser.add_argument('--num-procs', type=int, default=20,
                        help='Number of processes. Set to 0 to disable multiprocessing.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = create_data_loader(args)
    main()
