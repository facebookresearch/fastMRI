import pathlib
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer

sys.path.append("../../")  # noqa: E402

from unet_module import UnetModule


def main(args):
    """Main training routine."""
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
    # this assumes data is stored in a folder parallel to the code repository
    knee_path = pathlib.Path.cwd() / "../../../datasets/knee_data"
    logdir = pathlib.Path.cwd() / "../../../logs/unet/unet_demo"
    parent_parser = ArgumentParser(add_help=False)

    parser = UnetModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 2
    backend = "dp"
    batch_size = 1 if backend == "dp" else num_gpus

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
        exp_dir=logdir,
        exp_name="unet_demo",
        batch_size=batch_size,
    )
    parser.set_defaults(**config)

    parser.set_defaults(
        gpus=num_gpus,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
    )

    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
