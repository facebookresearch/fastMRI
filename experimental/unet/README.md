# U-Net Model for MRI Reconstruction

This directory contains a reference U-Net implementation for MRI reconstruction in PyTorch.

The file `unet_brain_leaderboard_submission_2020-08-18.py` contains code and hyperparameters that were used to train the "fastMRI Repo U-Net" model on the [fastMRI Multi-Coil Brain Leaderboard](https://fastmri.org/leaderboards/). `train_unet_demo.py` contains a basic demo designed for lower memory consumption.

To start training the model, run:

```bash
python train_unet_demo.py
```

You can also pass options at the command line:

```bash
python train_unet_demo.py --challenge CHALLENGE --data_path DATA --mask_type MASK_TYPE
```

where `CHALLENGE` is either `singlecoil` or `multicoil`. And `MASK_TYPE` is either `random` (for knee) or `equispaced` (for brain). Training logs and checkpoints are saved in `experiments/unet` directory.

To run the model on test data:

```bash
python models/unet/train_unet.py --mode test --test_split TESTSPLIT --challenge CHALLENGE --data-path DATA --resume_from_checkpoint MODEL
```

where `MODEL` is the path to the model checkpoint from `{logdir}/lightning_logs/version_0/checkpoints/`. Subsequent model runs will increment the version count. `TESTSPLIT` should specify the test split you want to run on - either `test` or `challenge`.

The outputs will be saved to `reconstructions` directory which can be uploaded for submission.

## Implementation Notes

The leaderboard model was trained where the `train` split included both the `train` and `val` splits from the public data.
