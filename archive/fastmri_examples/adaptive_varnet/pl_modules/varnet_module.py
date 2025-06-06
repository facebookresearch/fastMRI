"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import fastmri
from fastmri import evaluate
from fastmri.data import transforms
from fastmri.data.transforms import VarNetSample
from fastmri.models.adaptive_varnet import AdaptiveSensitivityModel, AdaptiveVarNetBlock
from fastmri.models.varnet import NormUnet
from fastmri.pl_modules.mri_module import MriModule

from .metrics import DistributedArraySum, DistributedMetricSum


class VarNet(nn.Module):
    """
    A full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        num_sense_lines: Optional[int] = None,
        hard_dc: bool = False,
        dc_mode: str = "simul",
        sparse_dc_gradients: bool = True,
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
            num_sense_lines: Number of low-frequency lines to use for
                sensitivity map computation, must be even or `None`. Default
                `None` will automatically compute the number from masks.
                Default behaviour may cause some slices to use more
                low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
            hard_dc: Whether to do hard DC layers instead of soft (learned).
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows. This should change
                nothing for the non-active VarNet.
        """
        super().__init__()

        self.num_sense_lines = num_sense_lines
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode

        self.sparse_dc_gradients = sparse_dc_gradients

        self.sens_net = AdaptiveSensitivityModel(
            sens_chans, sens_pools, num_sense_lines=num_sense_lines
        )
        self.cascades = nn.ModuleList(
            [
                AdaptiveVarNetBlock(
                    NormUnet(chans, pools),
                    hard_dc=hard_dc,
                    dc_mode=dc_mode,
                    sparse_dc_gradients=sparse_dc_gradients,
                )
                for _ in range(num_cascades)
            ]
        )

    def forward(
        self, kspace: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor
    ):

        extra_outputs = defaultdict(list)

        sens_maps = self.sens_net(masked_kspace, mask)
        extra_outputs["sense"].append(sens_maps.detach().cpu())

        kspace_pred = masked_kspace.clone()

        extra_outputs["masks"].append(mask.detach().cpu())
        # Store current reconstruction
        current_recon = fastmri.complex_abs(
            self.sens_reduce(masked_kspace, sens_maps)
        ).squeeze(1)
        extra_outputs["recons"].append(current_recon.detach().cpu())

        for cascade in self.cascades:
            kspace_pred = cascade(
                kspace_pred, masked_kspace, mask, sens_maps, kspace=kspace
            )

            # Store current reconstruction
            current_recon = fastmri.complex_abs(
                self.sens_reduce(masked_kspace, sens_maps)
            ).squeeze(1)

            extra_outputs["recons"].append(current_recon.detach().cpu())

        # Could presumably do complex_abs(complex_rss()) instead and get same result?
        output = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        return output, extra_outputs

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )


class VarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        num_sense_lines: int = None,
        hard_dc: bool = False,
        dc_mode: str = "simul",
        sparse_dc_gradients: bool = True,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
            hard_dc: Whether to do hard DC layers instead of soft (learned).
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by using torch.where()
                with the mask: this essentially removes gradients for the policy on unsampled rows. This should
                change nothing for the non-active VarNet.
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.num_sense_lines = num_sense_lines
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode
        self.sparse_dc_gradients = sparse_dc_gradients

        # logging functions
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        self.ValMargDist = DistributedArraySum()
        self.ValCondEnt = DistributedMetricSum()

        self.TrainNMSE = DistributedMetricSum()
        self.TrainSSIM = DistributedMetricSum()
        self.TrainPSNR = DistributedMetricSum()
        self.TrainLoss = DistributedMetricSum()
        self.TrainTotExamples = DistributedMetricSum()
        self.TrainTotSliceExamples = DistributedMetricSum()
        self.TrainMargDist = DistributedArraySum()
        self.TrainCondEnt = DistributedMetricSum()

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
            num_sense_lines=self.num_sense_lines,
            hard_dc=self.hard_dc,
            dc_mode=self.dc_mode,
            sparse_dc_gradients=self.sparse_dc_gradients,
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, kspace, masked_kspace, mask):
        return self.varnet(kspace, masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        output, extra_outputs = self(batch.kspace, batch.masked_kspace, batch.mask)

        target, output = transforms.center_crop_to_smallest(batch.target, output)
        # NOTE: Using max value here...
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("train_loss", loss)

        # Return same stuff as on validation step here
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "loss": loss,
            "extra_outputs": extra_outputs,
        }

    def training_step_end(self, train_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "loss",
            "extra_outputs",
        ):
            if k not in train_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by training_step."
                )
        if train_logs["output"].ndim == 2:
            train_logs["output"] = train_logs["output"].unsqueeze(0)
        elif train_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from training_step.")
        if train_logs["target"].ndim == 2:
            train_logs["target"] = train_logs["target"].unsqueeze(0)
        elif train_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from training_step.")

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(train_logs["fname"]):
            slice_num = int(train_logs["slice_num"][i].cpu())
            maxval = train_logs["max_value"][i].cpu().numpy()
            output = train_logs["output"][i].detach().cpu().numpy()
            target = train_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "loss": train_logs["loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "max_vals": max_vals,
        }

    def validation_step(self, batch, batch_idx):
        batch: VarNetSample
        output, extra_outputs = self.forward(
            batch.kspace, batch.masked_kspace, batch.mask
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            ),
            "extra_outputs": extra_outputs,
        }

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "max_vals": max_vals,
        }

    def training_epoch_end(self, train_logs):
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for train_log in train_logs:
            losses.append(train_log["loss"].data.view(-1))

            for k in train_log["mse_vals"].keys():
                mse_vals[k].update(train_log["mse_vals"][k])
            for k in train_log["target_norms"].keys():
                target_norms[k].update(train_log["target_norms"][k])
            for k in train_log["ssim_vals"].keys():
                ssim_vals[k].update(train_log["ssim_vals"][k])
            for k in train_log["max_vals"]:
                max_vals[k] = train_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.TrainNMSE(metrics["nmse"])
        metrics["ssim"] = self.TrainSSIM(metrics["ssim"])
        metrics["psnr"] = self.TrainPSNR(metrics["psnr"])
        tot_examples = self.TrainTotExamples(torch.tensor(local_examples))
        train_loss = self.TrainLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TrainTotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("training_loss", train_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"train_metrics/{metric}", value / tot_examples)

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

    def test_step(self, batch, batch_idx):
        kspace, masked_kspace, mask, _, fname, slice_num, _, crop_size = batch
        crop_size = crop_size[0]  # always have a batch size of 1 for varnet

        output, extra_outputs = self(kspace, masked_kspace, mask)

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        # This needs to be a class attribute for storing of gradients workaround
        self.optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, self.lr_step_size, self.lr_gamma
        )

        return [self.optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
