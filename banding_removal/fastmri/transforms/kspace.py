import h5py
import torch
from collections import OrderedDict
from fastmri.data import transforms

import numpy as np
import random
import pdb

def est_sens_maps(kspace, start, end, apodize_hori=0.07):
    num_coils, height, width = kspace.shape
    mask = np.zeros(width, dtype=kspace.dtype)
    mask[start:end] = 1
    kspace = np.where(mask, kspace, 0)
    mask = np.exp(-(np.linspace(-1, 1, width) / apodize_hori) ** 2, dtype=kspace.dtype)
    kspace = kspace * mask
    sens_maps = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm='ortho'))
    sens_maps /= np.sqrt(np.sum(np.abs(sens_maps) ** 2, axis=0, keepdims=True))
    return sens_maps


class KSpaceDataTransform(object):
    def __init__(self, args, mask_func, partition, use_seed=True):
        self.args = args
        self.mask_func = mask_func
        self.partition = partition
        self.use_seed = use_seed

    def __call__(self, target_ksp, target_im, attrs, fname, slice):
        kspace_np = target_ksp
        target_im = transforms.to_tensor(target_im)
        target_ksp = transforms.to_tensor(target_ksp)

        if self.args.coil_compress_coils:
            target_ksp = transforms.coil_compress(target_ksp, self.args.coil_compress_coils)

        if self.args.calculate_offsets_directly:
            krow = kspace_np.sum(axis=(0,1)) # flatten to a single row
            width = len(krow)
            offset = (krow != 0).argmax()
            acq_start = offset
            acq_end = width - (krow[::-1] != 0).argmax() #exclusive
        else:
            offset = None # Mask will pick randomly
            if self.partition == 'val' and 'mask_offset' in attrs:
                offset = attrs['mask_offset']

            acq_start = attrs['padding_left']
            acq_end = attrs['padding_right']

        #pdb.set_trace()

        seed = None if not self.use_seed else tuple(map(ord, fname))
        input_ksp, mask, num_lf = transforms.apply_mask(
            target_ksp, self.mask_func, 
            seed, offset,
            (acq_start, acq_end))

        #pdb.set_trace()

        sens_map = torch.Tensor(0)
        if self.args.compute_sensitivities:
            start_of_center_mask = (kspace_np.shape[-1] - num_lf + 1) // 2
            end_of_center_mask = start_of_center_mask + num_lf
            sens_map = est_sens_maps(kspace_np, start_of_center_mask, end_of_center_mask)
            sens_map = transforms.to_tensor(sens_map)

        if self.args.grappa_input:
            with h5py.File(self.args.grappa_input_path / self.partition / fname, 'r') as hf:
                kernel = transforms.to_tensor(hf['kernel'][slice])
                input_ksp = transforms.apply_grappa(input_ksp, kernel, target_ksp, mask)

        grappa_kernel = torch.Tensor(0)
        if self.args.grappa_path is not None:
            with h5py.File(self.args.grappa_path / self.partition / fname, 'r') as hf:
                grappa_kernel = transforms.to_tensor(hf['kernel'][slice])

        if self.args.grappa_target:
            with h5py.File(self.args.grappa_target_path / self.partition / fname, 'r') as hf:
                kernel = transforms.to_tensor(hf['kernel'][slice])
                target_ksp = transforms.apply_grappa(target_ksp.clone(), kernel, target_ksp, mask, sample_accel=2)
                target_im = transforms.root_sum_of_squares(transforms.complex_abs(transforms.ifft2(target_ksp)))

        input_im = transforms.ifft2(input_ksp)
        if not self.args.scale_inputs:
            scale = torch.Tensor([1.])
        else:
            abs_input = transforms.complex_abs(input_im)
            if self.args.scale_type == 'max':
                scale = torch.max(abs_input)
            else:
                scale = torch.mean(abs_input)

            input_ksp /= scale
            target_ksp /= scale
            target_im /= scale

        scale = scale.view([1, 1, 1])
        attrs_dict = dict(**attrs)

        return OrderedDict(
            input = input_ksp,
            target = target_ksp,
            target_im = target_im,
            mask = mask,
            grappa_kernel = grappa_kernel,
            scale = scale,
            attrs_dict = attrs_dict,
            fname = fname,
            slice = slice,
            num_lf = num_lf,
            sens_map = sens_map,
        )
