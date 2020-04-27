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
from torch.nn import functional as F
from torch.optim import RMSprop

from scipy.optimize import minimize
from common.args import Args
from common.evaluate import ssim
from common.subsample import create_mask_for_mask_type
from data import transforms
from models.mri_model import MRIModel
from models.grappaNet.grappa_unet_model import UnetModel


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
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace, mask = kspace, None
        

        
        # Crop 320x320 region. No normalization used.
        # NOTE: self.resolution is 320 by default. 
        image = transforms.ifft2(masked_kspace)
        smallest_width = min(self.resolution, image.shape[-2])
        smallest_height = min(self.resolution, image.shape[-3])
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])
        crop_size = (smallest_height, smallest_width)
        image = transforms.complex_center_crop(image, crop_size)

        # Crop target
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size)

        else:
            target = torch.Tensor([0])

        masked_kspace = transforms.fft2(image)

        return masked_kspace, target, ref_ksp, mask, fname, slice


class UnetMRIModel(MRIModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.unet_kspace_f1 = UnetModel(
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
        self.unet_image_f1 = UnetModel(
            in_chans=15,
            out_chans=15,
            chans=hparams.num_chans,
            num_pool_layers=hparams.num_pools,
            drop_prob=hparams.drop_prob
        )
        self.unet_image_f2 = UnetModel(
            in_chans=15,
            out_chans=15,
            chans=hparams.num_chans,
            num_pool_layers=hparams.num_pools,
            drop_prob=hparams.drop_prob
        )

    
    def residuals(self, g, input, ref_ksp, mask):
        # Reshape G to its original shape.
        g = g.reshape(5, 4) 
        # Apply grappa and calculate norm
        kspace_grappa = transforms.apply_grappa(input_ksp=input, kernel=g, ref_ksp=ref_ksp, mask=mask.float())
        return np.linalg.norm(kspace_grappa - input['kspace'])**2


    def forward(self, input, ref_ksp, mask):
        

        # First blue block CNN
        input = input.squeeze(1)
        unet_kspace = self.unet_kspace_f1(input.view(input.size(0), input.size(1)*input.size(-1), input.size(2), input.size(3)))
        unet_kspace = unet_kspace.view(input.size())
        unet_image_space = transforms.ifft2(transforms.kspace_dc(unet_kspace, ref_ksp, mask))
        unet_image_space = self.unet_image_f1(unet_image_space)
        unet_kspace = transforms.kspace_dc(transforms.fft2(unet_image_space), ref_ksp, mask)


        # input is already masked, need to do least squares between input and input['kspace'] grappa is 5x4 kernel.
        # scipy.optimize.minimize use this and flatten input kernel grappa for f callable. Need to find mingrappa
        grappa = np.random.rand(20)
        res = minimize(residuals, grappa, args=(input), tol=1e-6)
        min_grappa = res.x
        # Use min grappa for kernel 
        kspace_grappa = transforms.apply_grappa(input_ksp=unet_kspace, kernel=min_grappa, ref_ksp=ref_ksp, mask=mask.float())
        
        # Send blue block CNN
        unet_kspace = self.unet_kspace_f2(kspace_grappa.view(input.size(0), input.size(1)*input.size(-1), input.size(2), input.size(3)))
        unet_kspace = unet_kspace.view(input.size())
        unet_image_space = transforms.ifft2(transforms.kspace_dc(unet_kspace, ref_ksp, mask))
        unet_image_space = self.unet_image_f2(unet_image_space)
        unet_kspace = transforms.kspace_dc(transforms.fft2(unet_image_space), ref_ksp, mask)

        # IFT + RSS

        image = transforms.ifft2(unet_kspace)

        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares 
        image = transforms.root_sum_of_squares(image)

        return image

    def training_step(self, batch, batch_idx):
        input, target, ref_ksp, mask, _, _ = batch

        # The output is normalized during the forward pass
        output = self.forward(input, ref_ksp, mask)

        # Loss as stated in the paper! J(x) = - SSIM(x, \hat{x}) + \lambda*||x - \hat{x}|| -> \lamda = 0.001
        loss = 0.001*F.l1_loss(output, target) - ssim(output, target)
        logs = {'loss': loss.item()}
        return dict(loss=loss, log=logs)

    def validation_step(self, batch, batch_idx):
        input, target, ref_ksp, mask, fname, slice = batch
        output = self.forward(input, ref_ksp, mask)
        return {
            'fname': fname,
            'slice': slice,
            'output': output.numpy(),
            'target': target.cpu().numpy(),
            'val_loss': 0.001*F.l1_loss(output, target) - ssim(output, target),
        }

    def test_step(self, batch, batch_idx):
        input, _, ref_ksp, mask, fname, slice = batch
        output = self.forward(input, ref_ksp, mask)
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
        parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
        parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
        parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
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
    parser = UnetMRIModel.add_model_specific_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
