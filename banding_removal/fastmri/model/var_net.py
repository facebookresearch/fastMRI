"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import re

from torch import nn
import math
import numpy as np
import torch
import torch.nn.functional as F

from fastmri.data import transforms as T
from fastmri.model.public_unet import UnetModel, UnetModel2
from torch.utils.checkpoint import checkpoint

def get_num_slices():
    return 1

def merge_multi_slice(data, cat_dim=-1):
    if args.before_slices == 0 and args.after_slices == 0:
        return data[:, :, 0]

    merged_data = data[:, :, 0]
    for i in range(0, args.before_slices):
        merged_data = torch.cat(
            (data[:, :, i + 1], merged_data), dim=cat_dim)

    for i in range(0, args.after_slices):
        merged_data = torch.cat((
            merged_data, data[:, :, args.before_slices + i + 1]), dim=cat_dim)

    return merged_data

def unmerge_multi_slice(data, cat_dim=-1):
    if args.before_slices == 0 and args.after_slices == 0:
        return data

    num_slices = 1 + args.before_slices + args.after_slices
    b, c, one, h, w, two = data.shape
    slice_width = w // num_slices

    middle_slice_start = (args.before_slices * slice_width)
    middle_slice_end = (args.before_slices * slice_width) + slice_width
    slices = [data[..., middle_slice_start:middle_slice_end, :]]
    for i in range(0, args.before_slices):
        slice_start = (i) * slice_width
        slice_end = (i + 1) * slice_width
        slices.append(data[:, :, :, :, slice_start:slice_end])

    for i in range(args.before_slices+1, args.after_slices + args.before_slices + 1):
        slice_start = (i) * slice_width
        slice_end = (i + 1) * slice_width
        slices.append(data[:, :, :, :, slice_start:slice_end])

    return torch.stack(slices, dim=cat_dim).squeeze(3)

class Abs(nn.Module):
    def forward(self, x):
        return T.complex_abs(x)

def complex_mul(x, y):
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)

class MaskCenter(nn.Module):
    def forward(self, x, input):
        s = x.size(2)
        mask = torch.zeros_like(x)
        for j in range(s):
            lf = input['num_lf'][j]
            mask[:, :, j, ...] = T.mask_center(x[:, :, j, ...], lf)
        return mask

class RSS(nn.Module):
    def forward(self, x):
        return T.root_sum_of_squares_complex(x, dim=1)

class dRSS(nn.Module):
    def forward(self, x):
        return x / T.root_sum_of_squares_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

class Fm2Batch(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        b, c, *other = x.shape
        x = x.contiguous().view(b * c, 1, *other)
        x = merge_multi_slice(x, cat_dim=-2).unsqueeze(1).contiguous()
        x = self.model(x)
        x = unmerge_multi_slice(x, 2).contiguous()
        bc, one, *other = x.shape
        c = bc // b
        x = x.view(b, c, *other)
        return x

class Complex2Fm(nn.Module):
    def forward(self, x):
        b, c, s, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 5, 1, 2, 3, 4).contiguous().view(b, 2 * c * s, h, w)

class Fm2Complex(nn.Module):
    def forward(self, x):
        s = get_num_slices()
        b, cs2, h, w = x.shape
        assert cs2 % (2*s) == 0
        c = cs2 // (2*s)
        return x.view(b, 2, c, s, h, w).permute(0, 2, 3, 4, 5, 1)

class Polar(nn.Module):
    def forward(self, x):
        r = complex_abs2(x).sqrt()
        phi = torch.atan2(x[..., 1], x[..., 0])
        return torch.stack((r, phi), dim=-1)

class Cartesian(nn.Module):
    def forward(self, x):
        r, phi = x[..., 0], x[..., 1]
        return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)

class FT(nn.Module):
    def forward(self, x):
        return T.fft2(x)

class IFT(nn.Module):
    def forward(self, x):
        return T.ifft2(x)

class Push(nn.Module):
    pass

class Pop(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method

class Norm(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return norm(x, self.model, args.norm_type, args.norm_mean, args.norm_std)

def norm(x, model, norm_type, norm_mean, norm_std):
    b, c, h, w = x.shape
    mean, std = 0, 1
    if norm_type == 'layer':
        x = x.contiguous().view(b, c * h * w)
        if norm_mean:
            mean = x.mean(dim=1).view(b, 1, 1, 1)
        if norm_std:
            std = x.std(dim=1).view(b, 1, 1, 1)
    elif norm_type == 'instance':
        x = x.contiguous().view(b, c, h * w)
        if norm_mean:
            mean = x.mean(dim=2).view(b, c, 1, 1)
        if norm_std:
            std = x.std(dim=2).view(b, c, 1, 1)
    elif norm_type == 'group':
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        if norm_mean:
            mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        if norm_std:
            std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
    x = x.view(b, c, h, w)
    x = (x - mean) / std
    x = model(x)
    x = x * std + mean
    return x

class DC(nn.Module):
    def forward(self, x, input):
        return dc(x, input['mask'], input['kspace'])

def dc(x, mask, kspace):
    return torch.where(mask, kspace, x)

class SensExpand(nn.Module):
    def forward(self, x, input):
        return sens_expand(x, input['sens_maps'])

def sens_expand(x, sens_maps):
    return complex_mul(x, sens_maps)

class SensReduce(nn.Module):
    def forward(self, x, input):
        return sens_reduce(x, input['sens_maps'])

def sens_reduce(x, sens_maps):
    return complex_mul(x, T.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

class GRAPPA(nn.Module):
    def __init__(self, acceleration):
        super().__init__()
        self.acceleration = acceleration

    def forward(self, x, input):
        return T.apply_grappa(x, input[f'grappa_{self.acceleration}'], input['kspace'], input['mask'].float(), sample_accel=self.acceleration)

class SequentialPlus(nn.Sequential):
    def forward(self, input):
        stack = []
        x = input['kspace'].clone() if isinstance(input, dict) else input

        nmodules = len(self._modules.values())

        for module_idx, module in enumerate(self._modules.values()):
            last_module_flag = module_idx == nmodules - 1

            if isinstance(module, Push):
                stack.append(x)
            elif isinstance(module, Pop):
                if module.method == 'concat':
                    x = torch.cat((x, stack.pop()), 1)
                elif module.method == 'add':
                    x = x + stack.pop()
                else:
                    assert False
            else:
                if isinstance(module, (DC, SensExpand, SensReduce, SoftDC, GRAPPA, MaskCenter)):
                    x = module(x, input)
                else:
                    if args.gradient_checkpointing and not last_module_flag:
                        x.requires_grad_()
                        x = checkpoint(module, x)
                    else:
                        x = module(x)
        return x

class Pad(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return pad16(x, self.model)

def pad16(x, func):
    def floor_ceil(n):
        return math.floor(n), math.ceil(n)
    b, c, h, w = x.shape
    w_mult = ((w - 1) | 15) + 1
    h_mult = ((h - 1) | 15) + 1
    w_pad = floor_ceil((w_mult - w) / 2)
    h_pad = floor_ceil((h_mult - h) / 2)
    x = F.pad(x, w_pad + h_pad)
    x = func(x)
    x = x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]
    return x

def unet_grappa(in_chans, out_chans, chans):
    return UnetModel(
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        variant=None,
        kernel_size = args.kernel_size,
        padding = args.kernel_size // 2,
        dilation = args.dilation,
        groups = args.groups,
    )

def create_model(chans, in_chans, out_chans):
    if args.var_net_model == 'unet':
        return unet_grappa(in_chans, out_chans, chans)
    else:
        raise ValueError(f'Model type: {type}')

def unet(f, in_chans, out_chans=None):
    if out_chans is None:
        out_chans = in_chans

    return Pad(Norm(SequentialPlus(create_model(f, in_chans, out_chans))))

class CombineSlices(nn.Module):
    def __init__(self, slice_dim=2):
        super().__init__()
        self.slice_dim = slice_dim

    def forward(self, x):
        return torch.index_select(x, dim=self.slice_dim, index=torch.tensor(0, device=x.device))

def coord(module, out_chans):
    return nn.Sequential(Complex2Fm(), module, Fm2Complex())

def cunet(f, fm_in, out_chans=None):
    return coord(unet(f, fm_in, out_chans), out_chans)

def polar(module):
    return nn.Sequential(Polar(), Complex2Fm(), module, Fm2Complex(), Cartesian())

class SoftDC(nn.Module):
    def __init__(self, net, space='k-space', mode='parallel'):
        super().__init__()
        assert space in {'img-space', 'k-space'}
        assert mode in {'parallel', 'sequential'}
        self.net = net
        self.space = space
        self.mode = mode
        self.lambda_ = nn.Parameter(torch.ones(1))
        self.register_buffer('zero', torch.zeros(1, 1, 1, 1, 1))

    def soft_dc(self, x, input):
        if self.space == 'img-space':
            x = T.fft2(sens_expand(x, input['sens_maps']))
        x = torch.where(input['mask'], x - input['kspace'], self.zero)
        if self.space == 'img-space':
            x = sens_reduce(T.ifft2(x), input['sens_maps'])
        return self.lambda_ * x

    def net_forward(self, x, input):
        if self.space == 'k-space':
            x = sens_reduce(T.ifft2(x), input['sens_maps'])

        x = merge_multi_slice(x, cat_dim=-2).unsqueeze(1).contiguous()
        x = self.net(x)
        x = unmerge_multi_slice(x, 2).contiguous()

        if self.space == 'k-space':
            x = T.fft2(sens_expand(x, input['sens_maps']))
        return x

    def forward(self, x, input):
        if self.mode == 'parallel':
            return x - self.soft_dc(x, input) - self.net_forward(x, input)
        elif self.mode == 'sequential':
            x = self.net_forward(x, input)
            return x - self.soft_dc(x, input)

def parse_model(s):
    if s is None:
        return None
    s = re.sub('(\d+)\[(.*?)\]', lambda m: int(m.group(1)) * m.group(2), s)
    return eval(f'SequentialPlus({s})')

class SensModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_sens = parse_model(args.sens_method_str)
        self.model = parse_model(args.method_str)

    def forward(self, input):
        if self.model_sens is not None:
            input['sens_maps'] = self.model_sens(input)
        return self.model(input)

def var_net(args_local):
    global args
    args = args_local
    return SensModel()

def var_net_explicit_sens(args_local):
    global args
    args = args_local
    return parse_model(args.method_str)

