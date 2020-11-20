# End-to-End Variational Networks for Accelerated MRI Reconstruction Model

This directory contains a PyTorch implementation and code for running
pretrained models for reproducing the paper:

[End-to-End Variational Networks for Accelerated MRI Reconstruction ({A. Sriram*, J. Zbontar*} et al., 2020)][e2evarnet]

The following files

- `varnet_knee_leaderboard_20201111.py`
- `varnet_brain_leaderboard_20201111.py`

contain code and hyperparameters that were used to train the "fastMRI Repo
End-to-End VarNet" model on the [fastMRI leaderboards][leadlink].
`train_varnet_demo.py` contains a basic demo designed for lower memory
consumption.

To start training demo the model, run:

```bash
python train_varnet_demo.py
```

If you run with no arguments, the script will create a `fastmri_dirs.yaml` file
in the root directory that you can use to store paths for your system. You can
also pass options at the command line:

```bash
python train_varnet_demo.py \
    --challenge CHALLENGE \
    --data_path DATA \
    --mask_type MASK_TYPE
```

where `CHALLENGE` is either `singlecoil` or `multicoil` and `MASK_TYPE` is
either `random` (for knee) or `equispaced` (for brain). Training logs and
checkpoints are saved in the current working directory by default.

To run the model on test data:

```bash
python train_varnet_demo.py \
    --mode test \
    --test_split TESTSPLIT \
    --challenge CHALLENGE \
    --data_path DATA \
    --resume_from_checkpoint MODEL
```

where `MODEL` is the path to the model checkpoint. `TESTSPLIT` should specify
the test split you want to run on - either `test` or `challenge`.

The outputs will be saved to `reconstructions` directory which can be uploaded
for submission.

## Pretrained Models

We have pretrained models uploaded on AWS. For an example on how to download
and use them, please see the `run_pretrained_varnet_inference.py` script. The
script is a stripped-down version of model creation, `state_dict` downloading
and loading, and model inference. To run the script, type

```bash
python run_pretrained_varnet_inference.py \
    --data_path DATA \
    --output_path OUTPUTS \
    --challenge CHALLENGE
```

where in this case `CHALLENGE` is `varnet_knee_mc` for the multi-coil knee
VarNet or `varnet_brain_mc` for the multi-coil brain VarNet. The script will
download the model and run on your GPU.

## Citing

If you use this this code in your research, please cite the corresponding
paper:

```BibTeX
@inproceedings{sriram2020endtoend,
    title={End-to-End Variational Networks for Accelerated MRI Reconstruction},
    author={Anuroop Sriram and Jure Zbontar and Tullie Murrell and Aaron Defazio and C. Lawrence Zitnick and Nafissa Yakubova and Florian Knoll and Patricia Johnson},
    year={2020},
    eprint={2004.06688},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## Implementation Notes

The leaderboard model was trained where the `train` split included both the
`train` and `val` splits from the public data.

There are a few differences between this implementation and the
[paper][e2evarnet].

- The paper model used a fixed number of center lines, whereas this model uses
the `center_fractions` variable that might change depending on image size.
- The paper model was trained separately on 4x and 8x, whereas this model
trains on both of them together.

These differences have been left partly for backwards compatibility and partly
due to the number of areas in the code base that would have go be tweaked and
tested to get them working.

[leadlink]: https://fastmri.org/leaderboards/
[e2evarnet]: https://doi.org/10.1007/978-3-030-59713-9_7
