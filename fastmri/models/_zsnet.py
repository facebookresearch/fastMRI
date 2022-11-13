"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms
from fastmri.models.varnet import SensitivityModel

from ._zsnet_unet import ZSNetUnet


def _create_zero_tensor(tensor_type: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        (1, 1, 1, 1, 1), dtype=tensor_type.dtype, device=tensor_type.device
    )


class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = ZSNetUnet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        return x.permute(0, 4, 1, 2, 3).reshape(b, two * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape

        x = x.contiguous().view(b, c, c // c * h * w)
        mean = (
            x.mean(dim=2)
            .view(b, c, 1, 1, 1)
            .expand(b, c, c // c, 1, 1)
            .contiguous()
            .view(b, c, 1, 1)
        )
        std = (
            x.std(dim=2)
            .view(b, c, 1, 1, 1)
            .expand(b, c, c // c, 1, 1)
            .contiguous()
            .view(b, c, 1, 1)
        )

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        if not mean.shape[1] == x.shape[1]:
            mean = mean[:, :2]
        if not std.shape[1] == x.shape[1]:
            std = std[:, :2]
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949

        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, pad_sizes = self.pad(x)
        x, mean, std = self.norm(x)

        x = self.unet(x.contiguous())

        # get shapes back and unnormalize
        x = self.unnorm(x, mean, std)
        x = self.unpad(x, *pad_sizes)
        x = self.chan_complex_to_last_dim(x)

        return x


class ZSNetSensitivityModel(SensitivityModel):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        super().__init__(chans, num_pools, in_chans, out_chans, drop_prob, mask_center)
        # overwrite unet
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )


class ZSNet(nn.Module):
    def __init__(
        self,
        image_crop_size: int,
        num_concat_cascades: int = 18,
        sens_chans: int = 10,
        sens_pools: int = 5,
        chans: int = 20,
        pools: int = 5,
        mask_center: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        self.sens_net = ZSNetSensitivityModel(
            sens_chans, sens_pools, mask_center=mask_center
        )
        self.init_layer = ZSNetBlock(NormUnet(chans, pools), crop_size=image_crop_size)
        self.cascades = nn.ModuleList()
        for i in range(1, num_concat_cascades):
            chan_mult = 2
            self.cascades.append(
                ZSNetConcatBlock(
                    NormUnet(chans=chans, num_pools=pools, in_chans=2 * chan_mult),
                    crop_size=image_crop_size,
                )
            )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
        only_center: bool = False,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred, image = self.init_layer(
            masked_kspace.clone(), masked_kspace, mask, sens_maps
        )
        previous_images = [image]
        for cascade in self.cascades:
            kspace_pred, image = cascade(
                kspace_pred,
                masked_kspace,
                mask,
                sens_maps,
                previous_images,
            )

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)


class ZSNetBaseBlock(nn.Module):
    def __init__(self, model: nn.Module, crop_size: int):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.crop_size = crop_size
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def image_crop(self, image: torch.Tensor) -> torch.Tensor:
        input_shape = image.shape
        crop_size = (min(self.crop_size, input_shape[-3]), input_shape[-2])

        return transforms.complex_center_crop(image, crop_size)

    def image_uncrop(
        self, image: torch.Tensor, original_image: torch.Tensor
    ) -> torch.Tensor:
        """Insert values back into original image."""
        in_shape = original_image.shape
        crop_height = image.shape[-3]
        in_height = in_shape[-3]
        pad_height = (in_height - crop_height) // 2
        if (in_height - crop_height) % 2 != 0:
            pad_height_top = pad_height + 1
        else:
            pad_height_top = pad_height

        original_image[..., pad_height_top:-pad_height, :, :] = image[...]  # type: ignore

        return original_image

    def apply_model_with_crop(self, image: torch.Tensor) -> torch.Tensor:
        if self.crop_size is not None:
            image = self.image_uncrop(
                self.model(self.image_crop(image)), image[..., :2].clone()
            )
        else:
            image = self.model(image)

        return image


class ZSNetBlock(ZSNetBaseBlock):
    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = _create_zero_tensor(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        image = self.sens_reduce(current_kspace, sens_maps)
        model_term = self.sens_expand(self.apply_model_with_crop(image), sens_maps)

        return current_kspace - soft_dc - model_term, image


class ZSNetConcatBlock(ZSNetBaseBlock):
    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        previous_images: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = _create_zero_tensor(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        image = self.sens_reduce(current_kspace, sens_maps)

        model_term = self.sens_expand(
            self.apply_model_with_crop(torch.cat([image] + previous_images, dim=-1)),
            sens_maps,
        )
        return current_kspace - soft_dc - model_term, image
