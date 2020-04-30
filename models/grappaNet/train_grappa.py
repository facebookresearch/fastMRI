"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import RMSprop
from torch.optim import Adam

from common.args import Args
from common.evaluate import ssim
from common.subsample import create_mask_for_mask_type
from data import transforms
from models.mri_model import MRIModel
from models.grappaNet.grappa_unet_model import UnetModel
import torch.nn as nn
from RAKI.RAKI_trainer import RAKI_trainer
import pytorch_ssim

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        kspace = transforms.to_tensor(kspace)
        ref_ksp = kspace.clone()
        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask, acceleration = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace, mask = kspace, None
        

        
        # Crop 320x320 region. No normalization used.
        # NOTE: self.resolution is 320 by default. 
        image = transforms.ifft2(masked_kspace)
        image_ref = transforms.ifft2(ref_ksp)
        
        smallest_width = min(self.resolution, image.shape[-2])
        smallest_height = min(self.resolution, image.shape[-3])
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])
        crop_size = (smallest_height, smallest_width)
        image = transforms.complex_center_crop(image, crop_size)
        image_ref = transforms.complex_center_crop(image_ref, crop_size)

        # Fix mask size
        h_from = (mask.shape[-2] - crop_size[0]) // 2
        h_to = h_from + crop_size[0]
        mask = mask[..., h_from:h_to, :]

        # Crop target
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size)

        else:
            target = torch.Tensor([0])

        masked_kspace = transforms.fft2(image)
        ref_ksp = transforms.fft2(image_ref)

        return masked_kspace, target, ref_ksp, mask, acceleration, fname, slice

class GrappaModel(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.grappa_kernel = nn.Parameter(torch.ones(kernel_size), requires_grad=True)
        
    def loss(self, input_ksp, ref_ksp, mask):
        pred = transforms.apply_grappa(input_ksp=input_ksp, kernel=self.grappa_kernel, ref_ksp=ref_ksp, mask=mask.float())
        return F.mse_loss(pred, ref_ksp)

    def get_grappa_kernel(self):
        return self.grappa_kernel



class UnetMRIModel(MRIModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.use_grappa = hparams.use_grappa
        print(hparams)
        print(self.use_grappa)
        if self.use_grappa:
            self.unet_kspace_f1 = UnetModel(
                in_chans=30,
                out_chans=30,
                chans=hparams.num_chans,
                num_pool_layers=hparams.num_pools,
                drop_prob=hparams.drop_prob
            )
            self.unet_image_f1 = UnetModel(
                in_chans=30,
                out_chans=30,
                chans=hparams.num_chans,
                num_pool_layers=hparams.num_pools,
                drop_prob=hparams.drop_prob
            )
        self.unet_kspace_f2 = UnetModel(
            in_chans=30,
            out_chans=30,
            chans=hparams.num_chans,
            num_pool_layers=hparams.num_pools,
            drop_prob=hparams.drop_prob
        )
        self.unet_image_f2 = UnetModel(
            in_chans=30,
            out_chans=30,
            chans=hparams.num_chans,
            num_pool_layers=hparams.num_pools,
            drop_prob=hparams.drop_prob
        )

    def forward(self, input_ksp, ref_ksp, mask, acceleration):
        input_ksp = input_ksp.squeeze(1)
        unet_size = [input_ksp.size(0), input_ksp.size(1) * input_ksp.size(-1), input_ksp.size(2), input_ksp.size(3)]
        if self.use_grappa:
            second_block_input = self.train_grappa_kernel(input_ksp, ref_ksp, mask, unet_size)
        else:
            second_block_input = []
            for input_ksp_i, ref_ksp_i, acceleration_i, mask_i in zip(input_ksp, ref_ksp, acceleration, mask):
                second_block_input.append(RAKI_trainer(acceleration_i).train(input_ksp_i.unsqueeze(0), ref_ksp_i.unsqueeze(0), mask_i.unsqueeze(0)))
            second_block_input = torch.cat(second_block_input)
                
        # Second blue block CNN
        unet_kspace = self.unet_kspace_f2(second_block_input.view(unet_size)) # Dim mismatch?
        unet_kspace = unet_kspace.view(input_ksp.size())
        unet_image_space = transforms.ifft2(transforms.kspace_dc(unet_kspace, ref_ksp, mask))
        unet_image_space = self.unet_image_f2(unet_image_space.view(unet_size))
        unet_image_space = unet_image_space.view(input_ksp.size())
        unet_kspace = transforms.kspace_dc(transforms.fft2(unet_image_space), ref_ksp, mask)

        # IFT + RSS

        image = transforms.ifft2(unet_kspace)
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares 
        image = transforms.root_sum_of_squares(image, 1)

        return image

    def train_grappa_kernel(self, input_ksp, ref_ksp, mask, unet_size):
        # First blue block CNN

        unet_kspace = self.unet_kspace_f1(input_ksp.view(unet_size))
        unet_kspace = unet_kspace.view(input_ksp.size())
        unet_image_space = transforms.ifft2(transforms.kspace_dc(unet_kspace, ref_ksp, mask))
        unet_image_space = self.unet_image_f1(unet_image_space.view(unet_size))
        unet_image_space = unet_image_space.view(input_ksp.size())
        unet_kspace = transforms.kspace_dc(transforms.fft2(unet_image_space), ref_ksp, mask)

        # Sample the unet in 2x. To apply GRAPPA in the 2x unet space. R'-fold.
        mask_fun_2x = create_mask_for_mask_type(self.hparams.mask_type, 0.16, 2)
        unet_kspace, _, _ = transforms.apply_mask(unet_kspace, mask_fun_2x)

        # input is already masked, need to do least squares between input and input['kspace'] grappa is 5x4 kernel.
        # scipy.optimize.minimize use this and flatten input kernel grappa for f callable. Need to find mingrappa
        batch, coils, height, width, cmplx = input_ksp.size()

        kernel_size = (batch, coils*cmplx, coils*cmplx, 5, 5)
        grappa_model = GrappaModel(kernel_size).to(unet_kspace.device)
        optimizer = Adam(grappa_model.parameters())
        
        # Optimization over a set for Grappa estimation
        second_block_input = []
        for unet_ksp_i, input_ksp_i, ref_ksp_i, mask_i in zip(unet_ksp, input_ksp, ref_ksp, mask):
            with torch.enable_grad():
                grappa_model.train()
                for epoch in range(200):
                    loss = grappa_model.loss(input_kspace_i, ref_ksp_i, mask_i)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
            min_grappa = grappa_model.get_grappa_kernel()
            second_block_input.append(transforms.apply_grappa(input_ksp=unet_ksp_i, kernel=min_grappa, ref_ksp=ref_ksp_i, mask=mask_i.float()))

        second_block_input = torch.cat(second_block_input)

        

        # Use min grappa for kernel
        return second_block_input

    def training_step(self, batch, batch_idx):
        input, target, ref_ksp, mask, acceleration, _, _ = batch
        print(input.size())
        # The output is normalized during the forward pass
        output = self.forward(input, ref_ksp, mask, acceleration)
        # Loss as stated in the paper! J(x) = - SSIM(x, \hat{x}) + \lambda*||x - \hat{x}|| -> \lamda = 0.001
        #loss = 0.001*F.l1_loss(output, target) - ssim(output.detach().cpu().numpy(), target.cpu().numpy())
        loss = .001*F.l1_loss(output, target) - pytorch_ssim.ssim(target.unsqueeze(1), output.unsqueeze(1))
        logs = {'loss': loss.item()}
        return dict(loss=loss, log=logs)

    def validation_step(self, batch, batch_idx):
        input, target, ref_ksp, mask, acceleration, fname, slice = batch
        print(input.size())
        output = self.forward(input, ref_ksp, mask, acceleration)
        return {
            'fname': fname,
            'slice': slice,
            'output': output.detach().cpu().numpy(),
            'target': target.cpu().numpy(),
            'val_loss': .001*F.l1_loss(output, target) - pytorch_ssim.ssim(target.unsqueeze(1), output.unsqueeze(1)),
            #'val_loss': 0.001*F.l1_loss(output, target) - ssim(output.cpu().numpy(), target.cpu().numpy()),
        }

    def test_step(self, batch, batch_idx):
        input, _, ref_ksp, mask, acceleration, fname, slice = batch
        output = self.forward(input, ref_ksp, mask, acceleration)
        return {
            'fname': fname,
            'slice': slice,
            'output': output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.lr_step_size, self.hparams.lr_gamma)
        return [optim], [scheduler]

    def train_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, self.hparams.challenge, mask, use_seed=False)

    def val_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, self.hparams.challenge, mask)

    def test_data_transform(self):
        return DataTransform(self.hparams.resolution, self.hparams.challenge)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--num-pools', type=int, default=2, help='Number of U-Net pooling layers')
        parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
        parser.add_argument('--num-chans', type=int, default=4, help='Number of U-Net channels')
        parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--lr-step-size', type=int, default=40,
                            help='Period of learning rate decay')
        parser.add_argument('--lr-gamma', type=float, default=0.1,
                            help='Multiplicative factor of learning rate decay')
        parser.add_argument('--weight-decay', type=float, default=0.,
                            help='Strength of weight decay regularization')
        return parser


def create_trainer(args, logger):
    return Trainer(
        logger=logger,
        default_save_path=args.exp_dir,
        checkpoint_callback=True,
        max_nb_epochs=args.num_epochs,
        gpus=args.gpus,
        distributed_backend='ddp',
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        early_stop_callback=False
    )


def main(args):
    if args.mode == 'train':
        load_version = 0 if args.resume else None
        logger = TestTubeLogger(save_dir=args.exp_dir, name=args.exp, version=load_version)
        trainer = create_trainer(args, logger)
        model = UnetMRIModel(args)
        trainer.fit(model)
    else:  # args.mode == 'test'
        assert args.checkpoint is not None
        model = UnetMRIModel.load_from_checkpoint(str(args.checkpoint))
        model.hparams.sample_rate = 1.
        trainer = create_trainer(args, logger=False)
        trainer.test(model)



if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--exp-dir', type=pathlib.Path, default='experiments',
                        help='Path where model and results should be saved')
    parser.add_argument('--exp', type=str, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        help='Path to pre-trained model. Use with --mode test')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. ')
    parser.add_argument('--use-grappa', type=int, help='Decide method to approximate kspace. Set True for Grappa and False for RAKI')
    parser = UnetMRIModel.add_model_specific_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
