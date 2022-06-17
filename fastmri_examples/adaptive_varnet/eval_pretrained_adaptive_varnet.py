import pathlib
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pl_modules import AdaptiveVarNetModule, VarNetModule
from subsample import create_mask_for_mask_type

from fastmri import evaluate
from fastmri.data.mri_data import fetch_dir
from fastmri.data.transforms import MiniCoilTransform
from fastmri.pl_modules import FastMriDataModule


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def entropy(prob_mask: torch.Tensor):
    ent = -(prob_mask * prob_mask.log() + (1 - prob_mask) * (1 - prob_mask).log())
    ent[prob_mask == 0] = 0
    ent[prob_mask == 1] = 0
    return ent


def load_model(
    module_class: pl.LightningModule,
    fname: pathlib.Path,
):
    print(f"loading model from {fname}")
    checkpoint = torch.load(fname, map_location=torch.device("cpu"))

    # Initialise model with stored params
    module = module_class(**checkpoint["hyper_parameters"])

    # Load stored weights: this will error if the keys don't match the model weights, which will happen
    #  when we are loading a VarNet instead of an AdaptiveVarNet or vice-versa.
    module.load_state_dict(checkpoint["state_dict"])

    return module


def cli_main(args):
    pl.seed_everything(0)

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

    # Assumes coil compression with 4 coils
    val_transform = MiniCoilTransform(
        mask_func=mask,
        num_compressed_coils=4,
        crop_size=args.crop_size,
    )

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=val_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------
    # model
    # ------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    non_adaptive = False
    try:
        # Try to load as AdaptiveVarNetModule, if this fails, then the model is probably a VarNetModule instead
        print("Trying to load as AdaptiveVarNetModule...")
        model = load_model(AdaptiveVarNetModule, args.load_checkpoint)

        print("... Success!")
    except RuntimeError:
        # If this still fails, then probably the state dict
        print(
            "Loading as AdaptiveVarNetModule failed, trying to load as VarNetModule..."
        )
        model = load_model(VarNetModule, args.load_checkpoint)
        non_adaptive = True
        print("... Success!")

    model.to(device)

    data_loader = (
        data_module.val_dataloader()
        if args.data_mode == "val"
        else data_module.train_dataloader()
    )

    # --------------------------------------------------------------------------------
    # We loop over the whole dataset and store information per-volume[
    # The metrics will be computed in a different loop
    # --------------------------------------------------------------------------------
    vol_info = {}
    seen_slices = set()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == args.num_batches:
                break

            output, extra_outputs = model(
                batch.kspace.to(device),
                batch.masked_kspace.to(device),
                batch.mask.to(device),
            )

            prob_masks_list = extra_outputs["prob_masks"]
            if not non_adaptive:
                assert (
                    len(prob_masks_list) == 1
                ), "Found more than one prob mask... Multiple policies in this checkpoint?"
                batch_prob_masks = (
                    prob_masks_list[0].squeeze().detach().cpu()
                )  # b x 1 x 1 x 128 x 1 --> b x 128
                assert np.isclose(
                    batch_prob_masks[0].sum(), model.budget
                ), f"Sum of a prob mask should match budget {model.budget}!"

            for i, f in enumerate(batch.fname):
                if f not in vol_info:
                    vol_info[f] = []
                prob_mask = None if non_adaptive else batch_prob_masks[i]
                slice_id = (f, batch.slice_num[i])
                assert slice_id not in seen_slices
                seen_slices.add(slice_id)
                vol_info[f].append(
                    (
                        output[i].cpu(),
                        batch.masked_kspace[i].cpu(),
                        batch.slice_num[i],
                        batch.target[i].cpu(),
                        batch.max_value[i],
                        prob_mask,
                    )
                )

        # --------------------------------------------------------------------------------
        # Now we compute metrics per volume
        # --------------------------------------------------------------------------------
        all_prob_masks = []
        all_ssims, all_psnrs, all_nmses = [], [], []
        for vol_name, vol_data in vol_info.items():
            # slice_data is (output, masked_kspace, slice_num, target, max_value, prob_mask)
            output = torch.stack([slice_data[0] for slice_data in vol_data]).numpy()
            target = torch.stack([slice_data[3] for slice_data in vol_data]).numpy()

            if not non_adaptive:
                all_prob_masks.append(
                    torch.stack([slice_data[-1] for slice_data in vol_data])
                )

            # ----- Metrics calculation -----
            if args.vol_based:
                # Note that SSIMLoss computes average SSIM over the entire batch
                ssim = evaluate.ssim(target, output)
                psnr = evaluate.psnr(target, output)
                nmse = evaluate.nmse(target, output)
                all_ssims.append(ssim)
                all_psnrs.append(psnr)
                all_nmses.append(nmse)
            else:
                for gt, rec in zip(target, output):
                    gt = gt[np.newaxis, :]
                    rec = rec[np.newaxis, :]
                    ssim = evaluate.ssim(gt, rec)
                    psnr = evaluate.psnr(gt, rec)
                    nmse = evaluate.nmse(gt, rec)
                    all_ssims.append(ssim)
                    all_psnrs.append(psnr)
                    all_nmses.append(nmse)

        # --------------------------------------------------------------------------------
        # Aggregate everything
        # --------------------------------------------------------------------------------
        ssim_array = np.concatenate(np.array(all_ssims)[:, None], axis=0)
        psnr_array = np.concatenate(np.array(all_psnrs)[:, None], axis=0)
        nmse_array = np.concatenate(np.array(all_nmses)[:, None], axis=0)

        return_dict = {
            "ssim": ssim_array.mean().item(),
            "psnr": psnr_array.mean().item(),
            "nmse": nmse_array.mean().item(),
        }

        if all_prob_masks:
            # Each row sums to model.budget.
            prob_mask_tensor = torch.cat(all_prob_masks, dim=0).double()

            print(
                f"Computed {prob_mask_tensor.shape[0]} masks of size {prob_mask_tensor.shape[1]}"
            )

            marg_prob = prob_mask_tensor.mean(dim=0, keepdim=True)
            marg_entropy = entropy(marg_prob).sum(dim=1)
            avg_cond_entropy = entropy(prob_mask_tensor).sum(dim=1).mean()
            mut_inf = marg_entropy - avg_cond_entropy

            return_dict.update(
                {
                    "cond_ent_ind": avg_cond_entropy.item(),
                    "marg_ent_ind": marg_entropy.item(),
                    "mi_ind": mut_inf.item(),
                }
            )
        print(return_dict)


def build_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--load_checkpoint",
        type=pathlib.Path,
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help=(
            "Number of center lines to use in mask. 0.08 for acceleration 4, 0.04 for acceleration 8 models.",
        ),
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help=(
            "Acceleration rates to use. This and `center_fractions` matter mostly for evaluating the equispaced "
            "models (the other models only care that the Auto-Calibration Region is fully sampled). Regardless, good "
            "practice is to set these parameters to the values specified in the `center_fractions` help, for the "
            "corresponding acceleration.",
        ),
    )
    parser.add_argument(
        "--crop_size",
        default=(128, 128),
        type=int,
        nargs="+",
        help="Crop size used by checkpoint.",
    )
    parser.add_argument(
        "--num_batches",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--data_mode",
        default="val",
        type=str,
        choices=["train", "val"],
    )
    parser.add_argument(
        "--skip_low_freqs",
        default=True,
        type=str2bool,
        help="Whether skip low-frequency lines when computing equispaced mask.",
    )
    parser.add_argument(
        "--vol_based",
        default=True,
        type=str2bool,
        help="Whether to do volume-based evaluation (otherwise slice-based).",
    )

    parser = AdaptiveVarNetModule.add_model_specific_args(parser)

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="adaptive_equispaced_fraction",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=64,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
        num_workers=20,
    )

    args = parser.parse_args()
    assert (
        len(args.crop_size) == 2
    ), f"Crop size must be of length 2, not {len(args.crop_size)}."

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
