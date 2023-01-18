"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import collections
import json
import math
import os
import pathlib
import pickle
import random
import re
import signal
import sys
import time

import h5py
import ismrmrd
import numpy as np
import skimage
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torchvision import utils

from fastmri.data import transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--desc")
parser.add_argument("--start-epoch", type=int, default=0)
parser.add_argument("--resume", type=pathlib.Path)
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--dicom", action="store_true")
parser.add_argument("--dicom-noise", type=eval, default=[0.015])
parser.add_argument("--magnet", type=float)
parser.add_argument(
    "--data-dir", type=pathlib.Path, default="/datasets01_101/fastMRI/041219/pat2/"
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("-l", "--learning-rate", type=float, default=0.0003)
parser.add_argument("-m", "--method", default="12[SoftDC(cunet(18,2)),]IFT(),RSS()")
parser.add_argument(
    "-s", "--method-sens", default="MaskCenter(),IFT(),Fm2Batch(cunet(8,2)),dRSS()"
)
parser.add_argument(
    "--norm-type", choices=("layer", "instance", "group"), default="group"
)
parser.add_argument("--norm-mean", type=int, default=1)
parser.add_argument("--norm-std", type=int, default=1)
parser.add_argument(
    "--acq",
    choices=(
        "CORPD",
        "CORPDFS",
        "AXT2FS",
        "SAGPD",
        "SAGT2FS",
        "CORPD_FBK",
        "CORPDFS_FBK",
    ),
)
parser.add_argument(
    "--loss", choices=("L1Loss", "NMSELoss", "SSIMLoss"), default="SSIMLoss"
)
parser.add_argument("-w", "--tr-max-width", type=int, default=740)
parser.add_argument("--tr-subset", type=float, default=1)
parser.add_argument("--va-subset", type=float, default=1)
parser.add_argument("--tr-all-data", action="store_true")
parser.add_argument("--save-all", action="store_true")
parser.add_argument("--tr-accel", type=eval, default="range(2,11)")
parser.add_argument("--tr-lf", type=eval, default="range(8,27)")
parser.add_argument("--va-accel", type=int, default=4)
parser.add_argument("--va-lf", type=int, default=16)
parser.add_argument(
    "--experiment-dir",
    type=pathlib.Path,
    default="tmp/" + re.sub("[^-.a-zA-Z0-9_]+", "_", "_".join(sys.argv)),
)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--kernel-size", type=int, default=3)


class Abs(nn.Module):
    def forward(self, x):
        return complex_abs2(x).sqrt()


def complex_abs2(x):
    assert x.shape[-1] == 2
    return x[..., 0] ** 2 + x[..., 1] ** 2


def complex_conj(x):
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_mul(x, y):
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)


class MaskCenter(nn.Module):
    def forward(self, x, input):
        return mask_center(x, input["num_lf"].item())


def mask_center(x, num_lf):
    b, c, h, w, two = x.shape
    assert b == 1
    pad = (w - num_lf + 1) // 2
    y = torch.zeros_like(x)
    y[:, :, :, pad : pad + num_lf] = x[:, :, :, pad : pad + num_lf]
    return y


class RSS(nn.Module):
    def forward(self, x):
        return rss(x)


class dRSS(nn.Module):
    def forward(self, x):
        return x / rss(x).unsqueeze(-1)


def rss(x):
    b, c, h, w, two = x.shape
    assert two == 2
    return complex_abs2(x).sum(dim=1, keepdim=True).sqrt()


class Fm2Batch(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        b, c, *other = x.shape
        x = x.contiguous().view(b * c, 1, *other)
        x = self.model(x)
        x = x.view(b, c, *other)
        return x


class Complex2Fm(nn.Module):
    def forward(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)


class Fm2Complex(nn.Module):
    def forward(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        return x.view(b, 2, c2 // 2, h, w).permute(0, 2, 3, 4, 1)


class Polar(nn.Module):
    def forward(self, x):
        r = complex_abs2(x).sqrt()
        phi = torch.atan2(x[..., 1], x[..., 0])
        return torch.stack((r, phi), dim=-1)


class Cartesian(nn.Module):
    def forward(self, x):
        r, phi = x[..., 0], x[..., 1]
        return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)


def crop_to(x, y):
    assert x.shape[:2] == (1, 1)
    assert y.shape[:2] == (1, 1)
    assert x.dim() == 4
    assert y.dim() == 4
    d0 = (x.shape[2] - y.shape[2]) // 2
    d1 = (x.shape[3] - y.shape[3]) // 2
    return x[..., d0 : d0 + y.shape[2], d1 : d1 + y.shape[3]]


class FT(nn.Module):
    def forward(self, x):
        return ft(x)


class IFT(nn.Module):
    def forward(self, x):
        return ift(x)


def roll(x, shift, dim):
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ft(x):
    assert x.shape[-1] == 2
    x = fftshift(x, dim=(-3, -2))
    x = torch.fft(x, 2, normalized=True)
    x = fftshift(x, dim=(-3, -2))
    return x


def ift(x):
    assert x.shape[-1] == 2
    x = fftshift(x, dim=(-3, -2))
    x = torch.ifft(x, 2, normalized=True)
    x = fftshift(x, dim=(-3, -2))
    return x


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
    if norm_type == "layer":
        x = x.contiguous().view(b, c * h * w)
        if norm_mean:
            mean = x.mean(dim=1).view(b, 1, 1, 1)
        if norm_std:
            std = x.std(dim=1).view(b, 1, 1, 1)
    elif norm_type == "instance":
        x = x.contiguous().view(b, c, h * w)
        if norm_mean:
            mean = x.mean(dim=2).view(b, c, 1, 1)
        if norm_std:
            std = x.std(dim=2).view(b, c, 1, 1)
    elif norm_type == "group":
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        if norm_mean:
            mean = (
                x.mean(dim=2)
                .view(b, 2, 1, 1, 1)
                .expand(b, 2, c // 2, 1, 1)
                .contiguous()
                .view(b, c, 1, 1)
            )
        if norm_std:
            std = (
                x.std(dim=2)
                .view(b, 2, 1, 1, 1)
                .expand(b, 2, c // 2, 1, 1)
                .contiguous()
                .view(b, c, 1, 1)
            )
    x = x.view(b, c, h, w)
    x = (x - mean) / std
    x = model(x)
    x = x * std + mean
    return x


class DC(nn.Module):
    def forward(self, x, input):
        return dc(x, input["mask"], input["kspace"])


def dc(x, mask, kspace):
    return torch.where(mask, kspace, x)


class SensExpand(nn.Module):
    def forward(self, x, input):
        return sens_expand(x, input["sens_maps"])


def sens_expand(x, sens_maps):
    return complex_mul(x, sens_maps)


class SensReduce(nn.Module):
    def forward(self, x, input):
        return sens_reduce(x, input["sens_maps"])


def sens_reduce(x, sens_maps):
    return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)


class GRAPPA(nn.Module):
    def __init__(self, acceleration):
        super().__init__()
        self.acceleration = acceleration

    def forward(self, x, input):
        return T.apply_grappa(
            x,
            input[f"grappa_{self.acceleration}"],
            input["kspace"],
            input["mask"].float(),
            sample_accel=self.acceleration,
        )


class SequentialPlus(nn.Sequential):
    def forward(self, input):
        stack = []
        x = input["kspace"].clone() if isinstance(input, dict) else input
        for module in self._modules.values():
            if isinstance(module, Push):
                stack.append(x)
            elif isinstance(module, Pop):
                if module.method == "concat":
                    x = torch.cat((x, stack.pop()), 1)
                elif module.method == "add":
                    x = x + stack.pop()
                else:
                    assert False
            elif isinstance(
                module, (DC, SensExpand, SensReduce, SoftDC, GRAPPA, MaskCenter)
            ):
                x = module(x, input)
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
    x = x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]
    return x


def unet(f, fm_in, fm_out=None):
    if fm_out is None:
        fm_out = fm_in

    def conv(in_channels, out_channels, transpose=False):
        if transpose:
            yield nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False)
        else:
            yield nn.Conv2d(
                in_channels,
                out_channels,
                args.kernel_size,
                1,
                args.kernel_size // 2,
                bias=False,
            )
        yield nn.InstanceNorm2d(out_channels)
        yield nn.LeakyReLU(0.2, True)

    return Pad(
        Norm(
            SequentialPlus(
                *conv(fm_in, 1 * f),
                *conv(1 * f, 1 * f),
                Push(),
                nn.AvgPool2d(2, 2),
                *conv(1 * f, 2 * f),
                *conv(2 * f, 2 * f),
                Push(),
                nn.AvgPool2d(2, 2),
                *conv(2 * f, 4 * f),
                *conv(4 * f, 4 * f),
                Push(),
                nn.AvgPool2d(2, 2),
                *conv(4 * f, 8 * f),
                *conv(8 * f, 8 * f),
                Push(),
                nn.AvgPool2d(2, 2),
                *conv(8 * f, 16 * f),
                *conv(16 * f, 16 * f),
                *conv(16 * f, 8 * f, transpose=True),
                Pop("concat"),
                *conv(16 * f, 8 * f),
                *conv(8 * f, 8 * f),
                *conv(8 * f, 4 * f, transpose=True),
                Pop("concat"),
                *conv(8 * f, 4 * f),
                *conv(4 * f, 4 * f),
                *conv(4 * f, 2 * f, transpose=True),
                Pop("concat"),
                *conv(4 * f, 2 * f),
                *conv(2 * f, 2 * f),
                *conv(2 * f, 1 * f, transpose=True),
                Pop("concat"),
                *conv(2 * f, 1 * f),
                *conv(1 * f, 1 * f),
                nn.Conv2d(1 * f, fm_out, 1),
            )
        )
    )


def coord(module):
    return nn.Sequential(Complex2Fm(), module, Fm2Complex())


def cunet(f, fm_in):
    return coord(unet(f, fm_in))


def polar(module):
    return nn.Sequential(Polar(), Complex2Fm(), module, Fm2Complex(), Cartesian())


class SoftDC(nn.Module):
    def __init__(self, net, space="k-space", mode="parallel"):
        super().__init__()
        assert space in {"img-space", "k-space"}
        assert mode in {"parallel", "sequential"}
        self.net = net
        self.space = space
        self.mode = mode
        self.lambda_ = nn.Parameter(torch.ones(1))
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def soft_dc(self, x, input):
        if self.space == "img-space":
            x = ft(sens_expand(x, input["sens_maps"]))
        x = torch.where(input["mask"], x - input["kspace"], self.zero)
        if self.space == "img-space":
            x = sens_reduce(ift(x), input["sens_maps"])
        return self.lambda_ * x

    def net_forward(self, x, input):
        if self.space == "k-space":
            x = sens_reduce(ift(x), input["sens_maps"])
        x = self.net(x)
        if self.space == "k-space":
            x = ft(sens_expand(x, input["sens_maps"]))
        return x

    def forward(self, x, input):
        if self.mode == "parallel":
            return x - self.soft_dc(x, input) - self.net_forward(x, input)
        elif self.mode == "sequential":
            x = self.net_forward(x, input)
            return x - self.soft_dc(x, input)


def parse_model(s):
    if s is None:
        return None
    s = re.sub("(\d+)\[(.*?)\]", lambda m: int(m.group(1)) * m.group(2), s)  # noqa
    return eval(f"SequentialPlus({s})")


class SensModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_sens = parse_model(args.method_sens)
        self.model = parse_model(args.method)

    def forward(self, input):
        if self.model_sens is not None:
            input["sens_maps"] = self.model_sens(input)
        return self.model(input)


def input_cuda(input):
    input["kspace"] = input["kspace"].cuda(args.gpu, non_blocking=True)
    input["mask"] = input["mask"].cuda(args.gpu, non_blocking=True)
    input["max"] = input["max"].float().cuda(args.gpu, non_blocking=True)
    input["norm"] = input["norm"].float().cuda(args.gpu, non_blocking=True)
    if "grappa_2" in input:
        input["grappa_2"] = input["grappa_2"].float().cuda(args.gpu, non_blocking=True)
        input["grappa_4"] = input["grappa_4"].float().cuda(args.gpu, non_blocking=True)


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        win_size = 7
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, norm, max):
        # scale to prevent underflow in the derivateive of D
        q = 1e5
        X, Y, max = X * q, Y * q, max * q

        data_range = max[:, None, None, None]
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return -S.mean()


class NMSELoss(nn.Module):
    def forward(self, X, Y, norm, max):
        se = torch.sum((X - Y) ** 2, dim=(1, 2, 3))
        return torch.sum(se / norm**2)


class L1Loss(nn.Module):
    def forward(self, X, Y, norm, max):
        return F.l1_loss(X, Y)


class Slice(torch.utils.data.Dataset):
    def __init__(self, subdirs, mode, fnames=None, subset=1, max_width=None):
        self.args = args
        self.mode = mode
        self.examples = []

        subdirs = list(map(pathlib.Path, subdirs))
        if fnames:
            assert len(subdirs) == 1
            fnames = [subdirs[0] / fname for fname in fnames]
        else:
            fnames = []
            for subdir in subdirs:
                for fname in sorted((args.data_dir / subdir).iterdir()):
                    with h5py.File(fname, "r") as f:
                        if max_width and f["kspace"].shape[3] > max_width:
                            continue
                        if args.acq and f.attrs["acquisition"] != args.acq:
                            continue
                        fs = float(
                            re.search(
                                "<systemFieldStrength_T>([^<]+)",
                                f["ismrmrd_header"][()].decode(),
                            ).group(1)
                        )
                        if args.magnet is not None and abs(fs - args.magnet) > 1:
                            continue
                        inds = np.nonzero(f["mask"][()])[0]
                        if inds[1] - inds[0] == 1:
                            continue
                    fnames.append(subdir / fname.name)
            random.Random(42).shuffle(fnames)
            fnames = fnames[: max(1, int(len(fnames) * subset))]
        for fname in fnames:
            with h5py.File(args.data_dir / fname, "r") as f:
                # get padding info
                hdr = ismrmrd.xsd.CreateFromDocument(f["ismrmrd_header"][()])
                enc = hdr.encoding[0]
                enc_size = (
                    enc.encodedSpace.matrixSize.x,
                    enc.encodedSpace.matrixSize.y,
                    enc.encodedSpace.matrixSize.z,
                )
                enc_limits_center = enc.encodingLimits.kspace_encoding_step_1.center
                enc_limits_max = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
                acq_start = enc_size[1] // 2 - enc_limits_center
                acq_end = acq_start + enc_limits_max
                self.examples.extend(
                    (fname, i, acq_start, acq_end) for i in range(f["kspace"].shape[0])
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, acq_start, acq_end = self.examples[i]
        if (
            hasattr(self.args, "data_dir_cache_complete")
            and self.args.data_dir_cache_complete.is_file()
        ):
            dir = self.args.data_dir_cache
        else:
            dir = self.args.data_dir
        with h5py.File(dir / fname, "r") as f:
            kspace = f["kspace"][slice]
            ipat2_mask = f["mask"][()]
            target = f["reconstruction_rss"][slice]
            max = f.attrs["max"]
            norm = f.attrs["norm"]

        # mask
        num_cols = kspace.shape[2]
        if self.mode == "train":
            acceleration = random.choice(self.args.tr_accel)
            num_lf = random.choice(self.args.tr_lf)
            offset = random.randrange(acceleration)
        elif self.mode == "val":
            acceleration = self.args.va_accel
            num_lf = self.args.va_lf
            inds = np.nonzero(ipat2_mask)[0]
            offset = inds[0]
        mask = np.zeros(num_cols, dtype=np.uint8)
        mask[offset::acceleration] = 1
        pad = (num_cols - num_lf + 1) // 2
        mask[pad : pad + num_lf] = 1
        mask[:acq_start] = mask[acq_end:] = 0

        kspace_masked = np.where(mask, kspace, 0)
        input = dict(
            kspace=torch.from_numpy(
                np.stack((kspace_masked.real, kspace_masked.imag), axis=-1)
            ),
            mask=torch.from_numpy(mask[None, None, :, None]),
            max=max,
            norm=norm,
            num_lf=num_lf,
            fname=fname.name,
            slice=slice,
        )
        if "GRAPPA" in self.args.method:
            grappa_root = pathlib.Path(
                "/datasets01/fastMRI/041219/pat2_grappa_kernels/"
            )
            input["grappa_2"] = torch.from_numpy(
                h5py.File(grappa_root / "grappa_2x" / self.mode / fname.name, "r")[
                    "kernel"
                ][slice]
            )
            input["grappa_4"] = torch.from_numpy(
                h5py.File(grappa_root / "grappa_4x" / self.mode / fname.name, "r")[
                    "kernel"
                ][slice]
            )
        return input, torch.from_numpy(target[None])


def main_worker(gpu, args_local):
    global args
    args = args_local
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    model = model_1gpu = SensModel().cuda(args.gpu)
    if args.num_gpus > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:23456",
            world_size=args.num_gpus,
            rank=gpu,
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model_1gpu, device_ids=[args.gpu]
        )
    loss_func = eval(args.loss)().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    if args.gpu == 0:
        stdout = open(args.experiment_dir / "stdout", "a", buffering=1)
        print(" ".join(sys.argv))
        stdout.write(" ".join(sys.argv) + "\n")
    if args.resume:
        if args.gpu == 0:
            print("Resuming from checkpoint")
        checkpoint = torch.load(args.resume, map_location="cpu")
        args.start_epoch = checkpoint["epoch"]
        model_1gpu.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        del checkpoint
    torch.backends.cudnn.benchmark = True

    if args.dicom:
        assert args.num_gpus == 1
        dicom(model_1gpu)
        return

    if args.tr_all_data:
        tr_subdirs = ["multicoil_train", "multicoil_val"]
        va_subdirs = []
    else:
        tr_subdirs = ["multicoil_train"]
        va_subdirs = ["multicoil_val"]
    tr_dataset = Slice(
        tr_subdirs, "train", subset=args.tr_subset, max_width=args.tr_max_width
    )
    va_dataset = Slice(va_subdirs, "val", subset=args.va_subset)
    # pickle.dump((tr_dataset, va_dataset), open('db/dataset.pkl', 'wb'))
    # tr_dataset, va_dataset = pickle.load(open('db/dataset.pkl', 'rb')); print('loading dataset from pickle')
    if args.gpu == 0:
        print(f"len(tr_dataset)={len(tr_dataset)}; len(va_dataset)={len(va_dataset)}")
    disp_dataset = torch.utils.data.Subset(
        va_dataset, np.random.RandomState(42).permutation(len(va_dataset))[:28]
    )
    kwargs = dict(batch_size=1, num_workers=args.num_workers, pin_memory=True)
    tr_sampler = (
        torch.utils.data.distributed.DistributedSampler(tr_dataset)
        if args.num_gpus > 1
        else None
    )
    tr_loader = torch.utils.data.DataLoader(
        tr_dataset, sampler=tr_sampler, shuffle=tr_sampler is None, **kwargs
    )
    va_loader = torch.utils.data.DataLoader(va_dataset, **kwargs)
    disp_loader = torch.utils.data.DataLoader(disp_dataset, **kwargs)

    if args.visualize:
        assert args.num_gpus == 1
        visualize(disp_loader, model_1gpu)
        return

    if args.evaluate:
        assert args.num_gpus == 1
        loss_va, time_va, nmse, psnr, ssim, metadata = val(
            va_loader, model_1gpu, loss_func
        )
        d = collections.defaultdict(list)
        for m, v in zip(metadata, ssim):
            d[m["acquisition"]].append(v)
        for k, v in sorted(d.items()):
            print(f"{np.mean(v)} * {len(v)} {k}")
        print(np.mean(ssim))
        return

    if args.num_gpus == 8 and "SLURM_JOB_ID" in os.environ:
        args.data_dir_cache = (
            pathlib.Path("/scratch/slurm_tmpdir/") / os.environ["SLURM_JOB_ID"]
        )
        args.data_dir_cache_complete = args.data_dir_cache / "TRANSFER_COMPLETE"
        if args.gpu == 0:
            os.system(
                f"cp -r {args.data_dir}/* {args.data_dir_cache} && touch {args.data_dir_cache_complete} &"
            )
            print(f"transfer dataset to {args.data_dir_cache}")

    for epoch in range(args.start_epoch, args.epochs):
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)
        loss_tr, time_tr = train(tr_loader, model, loss_func, optimizer, epoch)
        if args.gpu == 0:
            if va_loader:
                loss_va, time_va, nmse, psnr, ssim, metadata = val(
                    va_loader, model_1gpu, loss_func
                )
                visualize(disp_loader, model_1gpu)
            else:
                loss_va, time_va, nmse, psnr, ssim, metadata = (
                    [0],
                    0,
                    [0],
                    [0],
                    [0],
                    None,
                )
            state = dict(
                epoch=epoch + 1,
                model=model_1gpu.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.experiment_dir / "model.pth")
            if args.save_all:
                torch.save(state, args.experiment_dir / f"model_{epoch:04d}.pth")
            stdout.write(
                json.dumps(
                    dict(
                        epoch=epoch,
                        loss_tr=np.mean(loss_tr),
                        loss_va=np.mean(loss_va),
                        nmse=np.mean(nmse),
                        psnr=np.mean(psnr),
                        ssim=np.mean(ssim),
                        time_tr=time_tr,
                        time_va=time_va,
                    )
                )
                + "\n"
            )


def train(tr_loader, model, loss_func, optimizer, epoch):
    model.train()
    time_tr = time_step = time.perf_counter()
    loss_tr = []
    for i, (input, target) in enumerate(tr_loader):
        input_cuda(input)
        target = target.cuda(args.gpu, non_blocking=True)
        output = model(input)
        output = crop_to(output, target)
        loss = loss_func(output, target, input["norm"], input["max"])
        loss_tr.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.gpu == 0 and sys.stdout.isatty():
            time_step = time.perf_counter() - time_step
            print(
                f" {epoch} {100 * i / len(tr_loader):.1f}% {np.mean(loss_tr):.7e} {time_step:.4f}s ",
                end="\r",
            )
            time_step = time.perf_counter()
    time_tr = time.perf_counter() - time_tr
    return loss_tr, time_tr


def val(va_loader, model, loss_func):
    model.eval()
    time_va = time.perf_counter()
    loss_va = []
    preds = collections.defaultdict(list)
    with torch.no_grad():
        for i, (input, target) in enumerate(va_loader):
            input_cuda(input)
            target = target.cuda(args.gpu, non_blocking=True)
            output = model(input)
            output = crop_to(output, target)
            loss = loss_func(output, target, input["norm"], input["max"])
            loss_va.append(loss.item())
            output = output.cpu().numpy()
            for j in range(output.shape[0]):
                preds[input["fname"][j]].append(output[j])

    nmse, psnr, ssim, metadata = [], [], [], []
    for fname in preds.keys():
        with h5py.File(args.data_dir / "multicoil_val" / fname, "r") as f:
            gt = f["reconstruction_rss"][()]
            acquisition = f.attrs["acquisition"]
            system = re.search(
                "<systemModel>(.*)</systemModel>", f["ismrmrd_header"][()].decode()
            ).group(1)
        pred = preds[fname] = np.concatenate(preds[fname])
        nmse.append(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)
        psnr.append(skimage.measure.compare_psnr(gt, pred, data_range=gt.max()))
        ssim.append(
            skimage.measure.compare_ssim(
                gt.transpose(1, 2, 0),
                pred.transpose(1, 2, 0),
                multichannel=True,
                data_range=gt.max(),
            )
        )
        metadata.append(dict(system=system, acquisition=acquisition, fname=fname))
    time_va = time.perf_counter() - time_va

    pickle.dump(
        (loss_va, time_va, nmse, psnr, ssim, preds, metadata),
        open(args.experiment_dir / "val.pkl", "wb"),
    )
    return loss_va, time_va, nmse, psnr, ssim, metadata


def visualize(disp_loader, model):
    model.eval()
    with torch.no_grad():
        outputs, targets = [], []
        for i, (input, target) in enumerate(disp_loader):
            input_cuda(input)
            output = model(input)
            output = crop_to(output, target)
            kwargs = dict(size=(320, 320), mode="bilinear", align_corners=False)
            target = F.interpolate(target, **kwargs)
            output = F.interpolate(output, **kwargs)
            target = target.cpu().numpy()
            output = output.cpu().numpy()
            min = target.min(axis=(2, 3), keepdims=True)
            max = target.max(axis=(2, 3), keepdims=True)
            target = ((target - min) / (max - min) * 1.4).clip(0, 1)
            output = ((output - min) / (max - min) * 1.4).clip(0, 1)
            target = np.ascontiguousarray(target[:, :, ::-1])
            output = np.ascontiguousarray(output[:, :, ::-1])
            targets.append(target)
            outputs.append(output)
        utils.save_image(
            torch.from_numpy(np.concatenate(targets)),
            args.experiment_dir / "img_gt.png",
            nrow=7,
            pad_value=1,
        )
        utils.save_image(
            torch.from_numpy(np.concatenate(outputs)),
            args.experiment_dir / "img_rec.png",
            nrow=7,
            pad_value=1,
        )


def dicom(model, which_dataset="multicoil_test"):
    import pydicom

    def write_dicom(
        x,
        accession_number,
        acquisition,
        series_description,
        series_number,
        system_model,
        field_strength,
        min,
        max,
    ):
        print(accession_number, acquisition, series_description, series_number)
        assert x.ndim == 3
        assert isinstance(accession_number, int)
        assert isinstance(series_number, int)

        def uid(seed=None):
            prng = np.random if seed is None else np.random.RandomState(seed=seed)
            return "1.2.840.113654.2.70.1." + "".join(
                str(prng.randint(0, 10)) for _ in range(37)
            )

        x = (x - min) / (max - min)
        ds = pydicom.dcmread("/checkpoint/jzb/FO-1011186284685752066.dcm")
        ds.Rows = x.shape[1]
        ds.Columns = x.shape[2]
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.WindowCenter = 1024
        ds.WindowWidth = 2048
        ds.SeriesDescription = series_description
        ds.SeriesNumber = series_number
        ds.ManufacturerModelName = system_model
        ds.MagneticFieldStrength = field_strength
        ds.PatientName = str(accession_number)
        ds.PatientID = str(accession_number)
        ds.AccessionNumber = str(accession_number)
        ds.StudyInstanceUID = uid(accession_number)
        ds.SeriesInstanceUID = uid()
        for i in range(x.shape[0]):
            ds.InstanceNumber = i + 1
            ds.SOPInstanceUID = uid()
            if acquisition in {"corpd", "corpdfs", "axt2fs"}:
                x[i] = np.flipud(np.fliplr(x[i]))
            elif acquisition in {"sagpd", "sagt2fs"}:
                x[i] = np.rot90(np.flipud(x[i]))
            else:
                assert False
            ds.PixelData = (x[i] * 2048).clip(0, 2**16 - 1).astype(np.uint16)
            ds.save_as(
                str(
                    args.experiment_dir
                    / f"{accession_number}_{series_number:04d}_{i:04d}.dcm"
                )
            )

    def add_noise_adaptive_v2(volume, noise_level=0.0):
        if noise_level == 0:
            return volume
        from scipy.ndimage.filters import median_filter

        result = []
        for i in range(volume.shape[0]):
            slice = volume[i]
            max = slice.max()
            slice = slice / max
            noise = np.random.normal(
                0, noise_level * np.sqrt(median_filter(slice, 11, mode="constant"))
            )
            result.append((slice + noise) * max)
        return np.stack(result)

    model.eval()
    slurm_ntasks = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    slurm_procid = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    accession_numbers = sorted(
        {
            fname.name.split("_")[3]
            for fname in sorted((args.data_dir / which_dataset).iterdir())
        }
    )
    accession_numbers = accession_numbers[
        slurm_procid : len(accession_numbers) : slurm_ntasks
    ]
    for accession_number in accession_numbers:
        series_number = 1
        for fname in sorted(
            (args.data_dir / which_dataset).glob(
                f"knee_pat2_test_{accession_number}*.h5"
            )
        ):
            fname = fname.name
            with h5py.File(args.data_dir / which_dataset / fname, "r") as f:
                gt = f["reconstruction_rss"][()]
                acquisition = f.attrs["acquisition"]
                assert acquisition.endswith("_FBP")
                acquisition = acquisition[:-4].lower().replace("_", "")
                hdr = f["ismrmrd_header"][()].decode()
                field_strength = float(
                    re.search("<systemFieldStrength_T>([^<]+)", hdr).group(1)
                )
                system_model = re.search("<systemModel>(.*?)</systemModel>", hdr).group(
                    1
                )
                if abs(field_strength - args.magnet) > 1:
                    continue
            min = 0
            max = dict(
                corpd=0.0003,
                sagpd=0.0003,
                sagt2fs=0.0001,
                axt2fs=0.0001,
                corpdfs=0.0001,
            )[acquisition]
            accession_number = int(fname.split("_")[3])
            write_dicom(
                gt,
                accession_number,
                acquisition,
                "ground_truth",
                series_number,
                system_model,
                field_strength,
                min,
                max,
            )
            series_number += 1

            di_dataset = Slice([which_dataset], "val", fnames=[fname])
            di_loader = torch.utils.data.DataLoader(
                di_dataset, batch_size=1, num_workers=1, pin_memory=True
            )
            pred = []
            with torch.no_grad():
                for i, (input, target) in enumerate(di_loader):
                    input_cuda(input)
                    output = model(input)
                    output = crop_to(output, target)
                    output = output.cpu().numpy()
                    assert output.shape[:2] == (1, 1)
                    pred.append(output[0, 0])
            pred = np.stack(pred)

            for noise_lvl in args.dicom_noise:
                write_dicom(
                    add_noise_adaptive_v2(pred, noise_lvl),
                    accession_number,
                    acquisition,
                    f"n{noise_lvl}",
                    series_number,
                    system_model,
                    field_strength,
                    min,
                    max,
                )
                series_number += 1


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    args.num_gpus = torch.cuda.device_count()

    signal.signal(signal.SIGUSR1, handle_sigusr1)
    signal.signal(signal.SIGTERM, handle_sigterm)

    os.system(f"mkdir -p {args.experiment_dir}")
    if (args.experiment_dir / "model.pth").is_file():
        args.resume = args.experiment_dir / "model.pth"
    if args.num_gpus == 1:
        main_worker(0, args)
    else:
        torch.multiprocessing.spawn(main_worker, nprocs=args.num_gpus, args=(args,))
