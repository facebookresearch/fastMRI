# Compressed Sensing with Total Variation Minimization

This directory contains code for compressed sensing baselines. The baselines
are based on the following paper:

[ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA (M. Uecker et al., 2013)](https://doi.org/10.1002/mrm.24751)

which was used as a baseline model in

[fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)

The implementation uses the BART toolkit. To install BART, please follow the
[installation instructions][bartlink].

Once BART is installed, set the `TOOLBOX_PATH` environment variable to point to the location where the repo was cloned and `PYTHONPATH` to the python wrapper for BART:

```bash
export TOOLBOX_PATH=/path/to/bart
export PYTHONPATH=${TOOLBOX_PATH}/python:${PYTHONPATH}
```

where `/path/to/bart` is the path to the cloned BART repository, not your OS installed BART program.

To run the reconstruction algorithm on the validation data, run:

```bash
python run_bart.py \
    --challenge CHALLENGE \
    --data_path DATA \
    --output_path reconstructions_val \
    --reg_wt 0.01 \
    --mask_type MASK_TYPE \
    --split val
```

where `CHALLENGE` is either `singlecoil` or `multicoil`. And `MASK_TYPE` is
either `random` (for knee) or `equispaced` (for brain). The outputs are saved
in a directory called `reconstructions_val`. To evaluate the results, run:

```bash
python fastmri/evaluate.py \
    --target-path TARGET_DATA \
    --predictions-path reconstructions_val \
    --challenge CHALLENGE
```

To apply the reconstruction algorithm to the test data, run:

```bash
python run_bart.py \
    --challenge CHALLENGE \
    --data_path DATA \
    --output_path reconstructions_test \
    --split test
```

The outputs will be saved to `reconstructions_test` directory which can be
uploaded for submission.

Note: for the 2020 Brain Challenge we have opted to not include compressed
sensing as a FAIR/NYU baseline for the leaderboard. The 2020 Challenge uses
equispaced masks, which are not supported by compressed sensing theory.

[bartlink]: https://mrirecon.github.io/bart/
