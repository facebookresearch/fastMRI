"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import pathlib
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from pl_modules import AdaptiveVarNetModule, VarNetModule
from pytorch_lightning.callbacks import Callback
from subsample import create_mask_for_mask_type

from fastmri.data.mri_data import fetch_dir
from fastmri.data.transforms import MiniCoilTransform
from fastmri.pl_modules import FastMriDataModule


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None
        else 0
    )


def count_untrainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if model is not None
        else 0
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2none(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return v


def int2none(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return int(v)


def float2none(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return float(v)


def make_wandb_run_name(args):
    name = ""

    # Create base name
    if args.learn_acquisition:
        if args.loupe_mask:
            name += "loupe-"
        else:
            name += "act-"
    else:
        name += "Nact-"
    assert len(args.accelerations) == 1
    name += str(args.accelerations[0])
    name += "-cas"
    name += str(args.num_cascades)
    name += "-"

    if args.learn_acquisition and not args.loupe_mask:
        name += "p"
        name += str(args.cascades_per_policy)
        name += "-"

    if args.chans != 18:
        name += "ch"
        name += str(args.chans)
        name += "-"

    if args.num_compressed_coils == 1:
        name += "singlecoil-"

    if args.sparse_dc_gradients:
        name += "dcsparse-"
    else:
        name += "dcmultip-"

    if args.learn_acquisition:
        if args.use_softplus:
            name += f"softplus{args.slope}b-"
        else:
            name += f"sigmoid{args.slope}s-"
        if args.straight_through_slope != 10:
            name += f"stslope{args.straight_through_slope}-"
        if args.hard_dc:
            if not args.dc_mode == "first":
                name += f"hdc{args.dc_mode}-"
        else:
            name += "sdc-"

        if args.st_clamp:
            name += "stclamp-"

        if not args.loupe_mask:  # Policy runs
            if args.policy_num_fc_layers != 3 and args.policy_fc_size != 256:
                name += f"{args.policy_num_fc_layers}fc{args.policy_fc_size}-"
            elif args.policy_num_fc_layers != 3:
                name += f"{args.policy_num_fc_layers}fc-"
            elif args.policy_fc_size != 256:
                name += f"fc{args.policy_fc_size}-"

            if args.policy_drop_prob != 0.0:
                name += f"drop{args.policy_drop_prob}-"

            if args.policy_activation != "leakyrelu":
                name += "elu-"
        else:  # LOUPE runs
            pass
    else:  # Non-active runs
        if args.mask_type != "adaptive_equispaced_fraction":
            name += f"{args.mask_type}-"

    name += "seed"
    name += str(args.seed)

    if args.lr != 0.001:
        name += "-lr{}".format(args.lr)

    if args.sample_rate is None and args.volume_sample_rate is not None:
        if args.volume_sample_rate != 1.0:
            name += "-"
            name += "vsr"
            name += str(args.volume_sample_rate)
    elif args.sample_rate is not None and args.volume_sample_rate is None:
        if args.sample_rate != 1.0:
            name += "-"
            name += "sr"
            name += str(args.sample_rate)

    return name


class WandbLoggerCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.resume_from_checkpoint:
            # Get wandb id from file in checkpoint dir
            # resume_from_checkpoint = default_root_dir / checkpoints / model.ckpt
            # wandb_id is stored in default_root_dir / wandb_id.txt
            with open(
                pathlib.Path(args.resume_from_checkpoint).parent.parent
                / "wandb_id.txt",
                "r",
            ) as f:
                id = f.read()
            with open(
                pathlib.Path(args.resume_from_checkpoint).parent.parent
                / "wandb_dir.txt",
                "r",
            ) as f:
                dir = pathlib.Path(f.read())
        else:
            id = wandb.util.generate_id()

            base_dir = pathlib.Path.cwd() / "wandb"
            now = datetime.now()
            if args.learn_acquisition:
                if args.loupe_mask:
                    algo = "loupe"
                else:
                    algo = "adaptive"
            else:
                algo = "non_adaptive"
            dir = base_dir / now.strftime("%Y_%m_%d") / algo
            dir.mkdir(parents=True, exist_ok=True)

        wandb.init(
            entity=self.args.wandb_entity,
            project=self.args.project,
            config=self.args,
            resume="allow",
            id=id,
            dir=dir,
        )

        if not wandb.run.resumed:
            # Extract run index from wandb name
            wandb_index = wandb.run.name.split("-")[-1]
            # Overwrite wandb run name
            wandb_name = make_wandb_run_name(args)
            wandb.run.name = wandb_name + "-" + wandb_index

            # Save wandb info
            with open(pathlib.Path(args.default_root_dir) / wandb.run.name, "w") as f:
                f.write(wandb.run.id)
            with open(pathlib.Path(args.default_root_dir) / wandb.run.id, "w") as f:
                f.write(wandb.run.name)
            with open(pathlib.Path(args.default_root_dir) / "wandb_id.txt", "w") as f:
                f.write(wandb.run.id)
            with open(pathlib.Path(args.default_root_dir) / "wandb_dir.txt", "w") as f:
                f.write(str(dir))

    def on_pretrain_routine_start(self, trainer, pl_module):
        train_loader_len = len(trainer.datamodule.train_dataloader())
        val_loader_len = len(trainer.datamodule.val_dataloader())
        test_loader_len = len(trainer.datamodule.test_dataloader())

        print(f"Train loader batches: {train_loader_len}")
        print(f"Val loader batches: {val_loader_len}")
        print(f"Test loader batches: {test_loader_len}")

        wandb.log(
            {
                "epoch": -1,
                "train_batches": train_loader_len,
                "val_batches": val_loader_len,
                "test_batches": test_loader_len,
            }
        )

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        epoch = trainer.current_epoch

        tot_ex = pl_module.TrainTotExamples.compute().item()
        tot_slice_ex = pl_module.TrainTotSliceExamples.compute().item()
        ssim = pl_module.TrainSSIM.compute().item() / tot_ex
        psnr = pl_module.TrainPSNR.compute().item() / tot_ex
        nmse = pl_module.TrainNMSE.compute().item() / tot_ex
        train_loss = pl_module.TrainLoss.compute().item() / tot_slice_ex
        wandb_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ssim": ssim,
            "train_psnr": psnr,
            "train_nmse": nmse,
            "train_tot_ex": tot_ex,
            "train_tot_slice_ex": tot_slice_ex,
        }

        wandb.log(wandb_dict)

        # For some reason tot_ex is not the correct number to divide by, due to some
        #  kind of weird issue with how we count it. Fortunately, we know the sum of
        #  val_marg_dist should be 1, so we can compute the correct normalisation
        #  number from that constraint. Probably this means that we're overcounting
        #  some examples relative to others for the entropy calculations?
        # NOTE: This is not really the correct distribution, since the policy is a
        #  bunch of independent Bernoullis (+ rejection sampling), not a policy over
        #  a single acquisition.
        # NOTE: These are not the entropies reported in the paper.
        train_marg_dist = pl_module.TrainMargDist.compute()
        norm_ex = train_marg_dist.sum()
        train_marg_dist = train_marg_dist / norm_ex
        if train_marg_dist.shape != torch.Size([1]):  # Check that we didn't skip
            W = len(train_marg_dist)
            plt.imshow(
                train_marg_dist.expand(W, W).cpu().numpy(),
                cmap="gist_gray",
            )
            plt.colorbar()
            wandb.log({"train_marg_dist": plt, "epoch": epoch})
            plt.close()
            train_marg_ent = torch.sum(
                -1 * train_marg_dist * torch.log(train_marg_dist + 1e-8)
            )
            train_cond_ent = pl_module.TrainCondEnt.compute() / norm_ex
            train_mut_inf = train_marg_ent - train_cond_ent
            wandb.log(
                {
                    "epoch": epoch,
                    "train_marg_ent": train_marg_ent,
                    "train_cond_ent": train_cond_ent,
                    "train_mut_inf": train_mut_inf,
                }
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        # See MriModule.validation_epoch_end()
        tot_ex = pl_module.TotExamples.compute().item()
        tot_slice_ex = pl_module.TotSliceExamples.compute().item()
        ssim = pl_module.SSIM.compute().item() / tot_ex
        psnr = pl_module.PSNR.compute().item() / tot_ex
        nmse = pl_module.NMSE.compute().item() / tot_ex
        val_loss = pl_module.ValLoss.compute().item() / tot_slice_ex
        wandb_dict = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_ssim": ssim,
            "val_psnr": psnr,
            "val_nmse": nmse,
            "val_tot_ex": tot_ex,
            "val_tot_slice_ex": tot_slice_ex,
        }

        wandb.log(wandb_dict)

        # For some reason tot_ex is not the correct number to divide by, due to some
        #  kind of weird issue with how we count it. Fortunately, we know the sum of
        #  val_marg_dist should be 1, so we can compute the correct normalisation
        #  number from that constraint. Probably this means that we're overcounting
        #  some examples relative to others for the entropy calculations?
        # NOTE: This is not really the correct distribution, since the policy is a
        #  bunch of independent Bernoullis (+ rejection sampling), not a policy over
        #  a single acquisition.
        # NOTE: These are not the entropies reported in the paper.
        val_marg_dist = pl_module.ValMargDist.compute()
        norm_ex = val_marg_dist.sum().item()
        val_marg_dist = val_marg_dist / norm_ex
        if val_marg_dist.shape != torch.Size([1]):  # Check that we didn't skip
            W = len(val_marg_dist)
            plt.imshow(
                val_marg_dist.expand(W, W).cpu().numpy(),
                cmap="gist_gray",
            )
            plt.colorbar()
            wandb.log({"val_marg_dist": plt, "epoch": epoch})
            plt.close()
            val_marg_ent = torch.sum(
                -1 * val_marg_dist * torch.log(val_marg_dist + 1e-8)
            )
            val_cond_ent = pl_module.ValCondEnt.compute() / norm_ex
            val_mut_inf = val_marg_ent - val_cond_ent
            wandb.log(
                {
                    "epoch": epoch,
                    "val_marg_ent": val_marg_ent,
                    "val_cond_ent": val_cond_ent,
                    "val_mut_inf": val_mut_inf,
                }
            )


def cli_main(args):
    if args.num_sense_lines is not None:
        assert (
            args.num_sense_lines % 2 == 0
        ), "`num_sense_lines` must be even, not {}".format(args.num_sense_lines)
        assert (
            len(args.accelerations) == 1 and len(args.center_fractions) == 1
        ), "Cannot use multiple accelerations when `num_sense_lines` is set."

    if args.seed is not None:
        pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type,
        args.center_fractions,
        args.accelerations,
        args.skip_low_freqs,
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = MiniCoilTransform(
        mask_func=mask,
        use_seed=False,  # Set this to True to get deterministic results for Equispaced and Random.
        num_compressed_coils=args.num_compressed_coils,
        crop_size=args.crop_size,
    )
    val_transform = MiniCoilTransform(
        mask_func=mask,
        num_compressed_coils=args.num_compressed_coils,
        crop_size=args.crop_size,
    )
    if args.test_split in ("test", "challenge"):
        mask = None
    test_transform = MiniCoilTransform(
        mask_func=mask,
        num_compressed_coils=args.num_compressed_coils,
        crop_size=args.crop_size,
    )

    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    if args.learn_acquisition:
        model = AdaptiveVarNetModule(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            budget=args.budget,
            cascades_per_policy=args.cascades_per_policy,
            loupe_mask=args.loupe_mask,
            use_softplus=args.use_softplus,
            crop_size=args.crop_size,
            num_actions=args.crop_size[1],
            num_sense_lines=args.num_sense_lines,
            hard_dc=args.hard_dc,
            dc_mode=args.dc_mode,
            slope=args.slope,
            sparse_dc_gradients=args.sparse_dc_gradients,
            straight_through_slope=args.straight_through_slope,
            st_clamp=args.st_clamp,
            policy_fc_size=args.policy_fc_size,
            policy_drop_prob=args.policy_drop_prob,
            policy_num_fc_layers=args.policy_num_fc_layers,
            policy_activation=args.policy_activation,
        )
    else:
        model = VarNetModule(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            num_sense_lines=args.num_sense_lines,
            hard_dc=args.hard_dc,
            dc_mode=args.dc_mode,
            sparse_dc_gradients=args.sparse_dc_gradients,
        )

    # ------------
    # trainer
    # ------------

    if args.wandb:
        trainer = pl.Trainer.from_argparse_args(
            args, num_sanity_val_steps=0, callbacks=[WandbLoggerCallback(args)]
        )
    else:
        trainer = pl.Trainer.from_argparse_args(args, num_sanity_val_steps=0)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")

    # Finish up wandb groups
    if args.wandb and args.accelerator == "DDP":
        wandb.finish()


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = (
        fetch_dir("log_path", path_config) / "varnet" / "vscode_default_dir"
    )

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int2none,
        help="Random seed to use. `None` for no seed.",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "adaptive_equispaced_fraction"),
        default="adaptive_equispaced_fraction",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb",
        default=False,
        type=str2bool,
        help="Whether to use wandb logging.",
    )
    parser.add_argument(
        "--project",
        default="varmri",
        type=str,
        help="Project name for wandb logging.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        type=str2none,
        help="wandb entity to use.",
    )

    parser.add_argument(
        "--num_compressed_coils",
        default=4,
        type=int2none,
        help="How many coils to use in coil compression..",
    )
    parser.add_argument(
        "--crop_size",
        default=(128, 128),
        nargs="+",
        type=int2none,
        help="Crop size of images for MiniCoilTransform.",
    )

    # Active acquisition arguments
    parser.add_argument(
        "--learn_acquisition",
        default=False,
        type=str2bool,
        help="Whether to do mask design (e.g. LOUPE, Policy) or not.",
    )
    parser.add_argument(
        "--budget",
        default=1,
        type=int,
        help="Number of acquisitions to do when doing active acquisition.",
    )
    parser.add_argument(
        "--cascades_per_policy",
        default=1,
        type=int,
        help=(
            "How many cascades to do per policy. `num_cascades` must be "
            "a multiple of this + 1 when learning a Policy model. Else "
            "this argument is ignored."
        ),
    )

    # LOUPE arguments
    parser.add_argument(
        "--loupe_mask",
        default=False,
        type=str2bool,
        help="Whether to use LOUPE mask, for non-adaptive acquisition.",
    )
    parser.add_argument(
        "--use_softplus",
        default=False,
        type=str2bool,
        help="Whether to use softplus or sigmoid in LOUPE and Policy.",
    )

    # Mask arguments
    parser.add_argument(
        "--skip_low_freqs",
        default=True,
        type=str2bool,
        help="Whether skip low-frequency lines when computing equispaced mask.",
    )
    parser.add_argument(
        "--num_sense_lines",
        default=None,
        type=int2none,
        help=(
            "Number of low-frequency lines to use for computation of sensitivity maps."
            "Default `None` will compute it automatically from masks. Must be even."
        ),
    )

    # DC arguments
    parser.add_argument(
        "--hard_dc",
        default=True,
        type=str2bool,
        help="Whether to do hard DC layers instead of learned soft DC.",
    )
    parser.add_argument(
        "--dc_mode",
        default="first",
        type=str,
        help=(
            "Whether to do DC before ('first'), after ('last') or simultaneously "
            "('simul') with Refinement step. Default 'first'."
        ),
    )

    # Gradient arguments
    parser.add_argument(
        "--slope",
        default=10,
        type=float,
        help="Slope to use for LOUPE and Policy sigmoid, or beta to use in softplus.",
    )
    parser.add_argument(
        "--sparse_dc_gradients",
        default=True,
        type=str2bool,
        help=(
            "Whether to sparsify the gradients in DC by using torch.where() "
            "with the mask: this essentially removes gradients for the policy "
            "on unsampled rows."
        ),
    )

    parser.add_argument(
        "--straight_through_slope",
        default=10,
        type=float,
        help="Slope to use in Straight Through estimator.",
    )

    parser.add_argument(
        "--st_clamp",
        default=False,
        type=str2bool,
        help="Whether to clamp gradients between -1 and 1 in straight through estimator.",
    )

    parser.add_argument(
        "--policy_fc_size",
        default=256,
        type=int,
        help="Size of intermediate Policy fc-layers.",
    )
    parser.add_argument(
        "--policy_drop_prob",
        default=0.0,
        type=float,
        help="Dropout probability of Policy convolutional layers.",
    )
    parser.add_argument(
        "--policy_num_fc_layers",
        default=3,
        type=int,
        help="Number of Policy fc-layers.",
    )
    parser.add_argument(
        "--policy_activation",
        default="leakyrelu",
        choices=["leakyrelu", "elu"],
        help="Activation function to use in between Policy fc-layers.",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="adaptive_equispaced_fraction",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=1,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
        num_workers=20,
    )

    # module config
    # NOTE: Should technically also add defaults here for ActiveVarNetModule
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=5,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.001,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=1,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=None,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # Save arguments
    args_dir = pathlib.Path(args.default_root_dir)
    if not args_dir.exists():
        args_dir.mkdir(parents=True)

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_top_k=True,
        verbose=True,
        monitor="validation_loss",
        mode="min",
        prefix="",
    )

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    args_dict = {key: str(value) for key, value in args.__dict__.items()}
    with open(args_dir / "args_dict.json", "w") as f:
        json.dump(args_dict, f)

    return args


def run_cli():
    args = build_args()

    # Prevent Lightning pre-emption
    for fname in pathlib.Path(args.default_root_dir).iterdir():
        if fname.name[: len("hpc_ckpt")] == "hpc_ckpt" and fname.suffix == ".ckpt":
            fname.unlink()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
