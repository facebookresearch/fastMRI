"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import pathlib
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F

from common.args import Args
from common.subsample import create_mask_for_mask_type
from data import transforms as T
from models.mri_model import MRIModel
from models.unet.unet_model import UnetModel


class DataTransform:
    """
    Data Transformer for training Var Net models.
    """

    def __init__(self, resolution, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                masked_kspace (torch.Tensor): Masked k-space
                mask (torch.Tensor): Mask
                target (torch.Tensor): Target image converted to a torch Tensor.
                fname (str): File name
                slice (int): Serial number of the slice.
                max_value (numpy.array): Maximum value in the image volume
        """
        if target is not None:
            target = T.to_tensor(target)
            max_value = attrs['max']
        else:
            target = torch.tensor(0)
            max_value = 0.0
        kspace = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs['padding_left']
        acq_end = attrs['padding_right']
        if self.mask_func:
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed, (acq_start, acq_end))
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask[:,:,:acq_start] = 0
            mask[:,:,acq_end:] = 0
        return masked_kspace, mask.byte(), target, fname, slice, max_value


class SSIM(nn.Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2)
        D = B1 * B2
        S = (A1 * A2) / D
        return 1 - S.mean()


class NormUnet(nn.Module):
    def __init__(self, chans, num_pools):
        super().__init__()
        self.unet = UnetModel(
                in_chans=2,
                out_chans=2,
                chans=chans,
                num_pool_layers=num_pools,
                drop_prob=0
            )

    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

    def norm(self, x):
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def pad(self, x):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        b, c, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x, h_pad, w_pad, h_mult, w_mult):
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, x):
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)
        return x


class VarNetBlock(nn.Module):
    def __init__(self, model):
        super(VarNetBlock, self).__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.register_buffer('zero', torch.zeros(1, 1, 1, 1, 1))

    def forward(self, current_kspace, ref_kspace, mask, sens_maps):
        def sens_expand(x):
            return T.fft2(T.complex_mul(x, sens_maps))

        def sens_reduce(x):
            x = T.ifft2(x)
            return T.complex_mul(x, T.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

        def soft_dc(x):
            return torch.where(mask, x - ref_kspace, self.zero) * self.dc_weight

        return current_kspace - \
                soft_dc(current_kspace) - \
                sens_expand(self.model(sens_reduce(current_kspace)))


class SensitivityModel(nn.Module):
    def __init__(self, chans, num_pools):
        super().__init__()
        self.norm_unet = NormUnet(chans, num_pools)

    def chans_to_batch_dim(self, x):
        b, c, *other = x.shape
        return x.contiguous().view(b * c, 1, *other), b

    def batch_chans_to_chan_dim(self, x, batch_size):
        bc, one, *other = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, *other)

    def divide_root_sum_of_squares(self, x):
        return x / T.root_sum_of_squares_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace, mask):
        def get_low_frequency_lines(mask):
            l = r = mask.shape[-2] // 2
            while mask[..., r, :]:
                r += 1

            while mask[..., l, :]:
                l -= 1

            return l + 1, r

        l, r = get_low_frequency_lines(mask)
        num_low_freqs = r - l
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        x = T.mask_center(masked_kspace, pad, pad + num_low_freqs)
        x = T.ifft2(x)
        x, b = self.chans_to_batch_dim(x)
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        return x


class VariationalNetworkModel(MRIModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.sens_net = SensitivityModel(hparams.sens_chans, hparams.sens_pools)
        self.cascades = nn.ModuleList([
            VarNetBlock(NormUnet(hparams.chans, hparams.pools))
            for _ in range(hparams.num_cascades)
        ])
        self.ssim_loss = SSIM()

    def forward(self, masked_kspace, mask):
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        return T.root_sum_of_squares(T.complex_abs(T.ifft2(kspace_pred)), dim=1)

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, _, max_value = batch
        output = self.forward(masked_kspace, mask)
        target, output = T.center_crop_to_smallest(target, output)
        return {'loss': self.ssim_loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)}

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, slice, max_value = batch
        output = self.forward(masked_kspace, mask)
        target, output = T.center_crop_to_smallest(target, output)
        return {
            'fname': fname,
            'slice': slice,
            'output': output.cpu().numpy(),
            'target': target.cpu().numpy(),
            'val_loss': self.ssim_loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value),
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, _, fname, slice, _ = batch
        output = self.forward(masked_kspace, mask)
        output = T.center_crop(output,(self.hparams.resolution,self.hparams.resolution))
        return {
            'fname': fname,
            'slice': slice,
            'output': output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.lr_step_size, self.hparams.lr_gamma)
        return [optim], [scheduler]

    def train_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, mask, use_seed=False)

    def val_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, mask)

    def test_data_transform(self):
        return DataTransform(self.hparams.resolution)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--num-cascades', type=int, default=12, help='Number of U-Net channels')
        parser.add_argument('--pools', type=int, default=4, help='Number of U-Net pooling layers')
        parser.add_argument('--chans', type=int, default=18, help='Number of U-Net channels')
        parser.add_argument('--sens-pools', type=int, default=4, help='Number of U-Net pooling layers')
        parser.add_argument('--sens-chans', type=int, default=8, help='Number of U-Net channels')

        parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
        parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
        parser.add_argument('--lr-step-size', type=int, default=40,
                            help='Period of learning rate decay')
        parser.add_argument('--lr-gamma', type=float, default=0.1,
                            help='Multiplicative factor of learning rate decay')
        parser.add_argument('--weight-decay', type=float, default=0.,
                            help='Strength of weight decay regularization')
        return parser


def create_trainer(args):
    return Trainer(
        default_save_path=args.exp_dir,
        checkpoint_callback=True,
        max_epochs=args.num_epochs,
        gpus=args.gpus,
        num_nodes=args.nodes,
        weights_summary=None,
        distributed_backend='ddp',
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        early_stop_callback=False,
        num_sanity_val_steps=0,
    )

def run(args):
    cudnn.benchmark = True
    cudnn.enabled = True
    if args.mode == 'train':
        trainer = create_trainer(args)
        model = VariationalNetworkModel(args)
        trainer.fit(model)
    else:  # args.mode == 'test' or args.mode == 'challenge'
        assert args.checkpoint is not None
        model = VariationalNetworkModel.load_from_checkpoint(str(args.checkpoint))
        model.hparams.sample_rate = 1.
        trainer = create_trainer(args)
        model.hparams = args
        trainer.test(model)


def main(args=None):
    parser = Args()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--exp-dir', type=pathlib.Path, default='experiments',
                        help='Path where model and results should be saved')
    parser.add_argument('--exp', type=str, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        help='Path to pre-trained model. Use with --mode test')
    parser = VariationalNetworkModel.add_model_specific_args(parser)
    if args is not None:
        parser.set_defaults(**args)

    args, _ = parser.parse_known_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run(args)

if __name__ == '__main__':
    main()
