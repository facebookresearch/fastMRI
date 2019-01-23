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

import numpy as np
import torch

import bart
from common.args import Args
from common.utils import tensor_to_complex_np
from data import transforms
from data.mri_data import SliceData
from models.cs.run_bart_val import save_outputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default parameters
REG_PARAM = {
    'singlecoil': {
        'CORPD_FBK': {
            4: 0.01,
            8: 0.1
        },
        'CORPDFS_FBK': {
            4: 0.01,
            8: 0.01
        },
    },

    'multicoil': {
        'CORPD_FBK': {
            4: 0.01,
            8: 0.01
        },
        'CORPDFS_FBK': {
            4: 0.001,
            8: 0.01
        },
    },
}


def data_transform(masked_kspace, target, attrs, fname, slice):
    """
    Data Transformer that simply returns the input masked k-space data and relevant attributes
    needed for running MRI reconstruction algorithms implemented in BART.

    Args:
        masked_kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for
            multi-coil data or (rows, cols, 2) for single coil data.
        target (numpy.array, optional): Target image.
        attrs (dict): Acquisition related information stored in the HDF5 object.
        fname (str): File name
        slice (int): Serial number of the slice.
    Returns:
        (tuple): tuple containing:
            masked_kspace (torch.Tensor): Sub-sampled k-space with the same shape as kspace.
            acquisition (str): CORPD_FBK or CORPDFS_FBK denoting the type of MR acquisition
            acceleration (int): The rate of acceleration
            num_low_freqs (int): Number of low frequency columns selected
            fname (str): File name containing the current data item
            slice (int): The index of the current slice in the volume
    """
    masked_kspace = transforms.to_tensor(masked_kspace)
    acquisition = attrs['acquisition']
    acceleration = attrs['acceleration']
    num_low_freqs = attrs['num_low_frequency']
    return masked_kspace, acquisition, acceleration, num_low_freqs, fname, slice


def create_data_loader(args):
    data = SliceData(
        root=args.data_path / f'{args.challenge}_test',
        transform=data_transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate
    )
    return data


def cs_total_variation(args, kspace, acquisition, acceleration, num_low_freqs):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization based
    reconstruction algorithm using the BART toolkit.
    """

    if acquisition not in {'CORPD_FBK', 'CORPDFS_FBK'}:
        raise ValueError(f'Invalid acquisition protocol: {acquisition}')
    if acceleration not in {4, 8}:
        raise ValueError(f'Invalid acceleration factor: {acceleration}')

    if args.challenge == 'singlecoil':
        kspace = kspace.unsqueeze(0)
    kspace = kspace.permute(1, 2, 0, 3).unsqueeze(0)
    kspace = tensor_to_complex_np(kspace)

    # Estimate sensitivity maps
    sens_maps = bart.bart(1, f'ecalib -d0 -m1 -r {num_low_freqs}', kspace)

    # Use Total Variation Minimization to reconstruct the image
    reg_wt = REG_PARAM[args.challenge][acquisition][acceleration]
    pred = bart.bart(1, f'pics -d0 -S -R T:7:0:{reg_wt} -i {args.num_iters}', kspace, sens_maps)
    pred = torch.from_numpy(np.abs(pred[0]))

    # Crop the predicted image to the correct size
    return transforms.center_crop(pred, (args.resolution, args.resolution))


def run_model(i):
    masked_kspace, acquisition, acceleration, num_low_freqs, fname, slice = data[i]
    prediction = cs_total_variation(args, masked_kspace, acquisition, acceleration, num_low_freqs)
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


if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--output-path', type=pathlib.Path, default=None,
                        help='Path to save the reconstructions to')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--num-procs', type=int, default=20,
                        help='Number of processes. Set to 0 to disable multiprocessing.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = create_data_loader(args)
    main()
