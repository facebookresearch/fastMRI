import pathlib
import sys
from argparse import ArgumentParser

from pytoqrch_lightning import Trainer

sys.path.append("../../")  # noqa: E402

from fastmri.training_modules import UnetModule


def main(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = UnetModule(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    knee_path = pathlib.Path("./datasets/knee_data")
    parent_parser = ArgumentParser(add_help=False)

    parser = UnetModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    backend = "ddp"
    config = dict(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        mask_type="random",
        center_fractions=[0.08],
        accelerations=[4],
        resolution=384,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=knee_path,
        challenge="singlecoil",
        exp_dir=pathlib.Path("logs/unet"),
        exp_name="unet_demo",
        use_ddp=(backend == "ddp"),
    )
    parser.set_defaults(**config)

    parser.set_defaults(gpus=2, backend=backend)

    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
