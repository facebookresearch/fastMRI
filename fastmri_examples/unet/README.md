# U-Net Model for MRI Reconstruction

This directory contains a reference U-Net implementation for MRI reconstruction
in PyTorch.

The files `unet_knee_sc_leaderboard_20201111.py`,
`unet_knee_mc_leaderboard_20201111.py`, and
`unet_brain_leaderboard_20201111.py` contain code and hyperparameters that were
used to train the "fastMRI Repo U-Net" model on the
[fastMRI leaderboards](https://fastmri.org/leaderboards/).
`train_unet_demo.py` contains a basic demo designed for lower memory
consumption.

To start training the model, run:

```bash
python train_unet_demo.py
```

You can also pass options at the command line:

```bash
python train_unet_demo.py --challenge CHALLENGE --data_path DATA --mask_type MASK_TYPE
```

where `CHALLENGE` is either `singlecoil` or `multicoil` and `MASK_TYPE` is
either `random` (for knee) or `equispaced` (for brain). Training logs and
checkpoints are saved in the current working directory by default.

To run the model on test data:

```bash
python models/unet/train_unet.py --mode test --test_split TESTSPLIT --challenge CHALLENGE --data-path DATA --resume_from_checkpoint MODEL
```

where `MODEL` is the path to the model checkpoint.`TESTSPLIT` should specify
the test split you want to run on - either `test` or `challenge`.

The outputs will be saved to `reconstructions` directory which can be uploaded
for submission.

## Pretrained Models

We have pretrained models uploaded on AWS. For an example on how to download
and use them, please see the `run_pretrained_unet_inference.py` script. The
script is a stripped-down version of model creation, `state_dict` downloading
and loading, and model inference. To run the script, type

```bash
python run_pretrained_unet_inference.py --data-path DATA_PATH --output-path OUTPUT_PATH --challenge CHALLENGE
```

where in this case CHALLENGE is `unet_knee_sc` for the single-coil knee U-Net,
`unet_knee_mc` for the multi-coil knee U-Net, or `unet_brain_mc` for the multi-
coil brain U-Net. The script will download the model and run on your GPU.

## Implementation Notes

The leaderboard model was trained where the `train` split included both the
`train` and `val` splits from the public data.
