# Compressed Sensing with Total Variation Minimization

This directory contains code to apply the ESPIRiT algorithm for coil sensitivity estimation and Total Variation minimization based reconstruction using the BART toolkit.

To install BART, please follow the [installation instructions](https://mrirecon.github.io/bart/).

Once BART is installed, set the `TOOLBOX_PATH` environment variable and point `PYTHONPATH` to the python wrapper for BART:

```bash
export TOOLBOX_PATH=/path/to/bart
export PYTHONPATH=${TOOLBOX_PATH}/python:${PYTHONPATH}
```

To run the reconstruction algorithm on the validation data, run:

```bash
python models/cs/run_bart.py --challenge CHALLENGE --data_path DATA --output_path reconstructions_val --reg_wt 0.01 --mask_type MASK_TYPE --split val
```

where `CHALLENGE` is either `singlecoil` or `multicoil`. And `MASK_TYPE` is either `random` (for knee) or `equispaced` (for brain). The outputs are saved in a directory called `reconstructions_val`. To evaluate the results, run:

```bash
python common/evaluate.py --target-path TARGET_DATA --predictions-path reconstructions_val --challenge CHALLENGE
```

To apply the reconstruction algorithm to the test data, run:

```bash
python models/cs/run_bart.py --challenge CHALLENGE --data_path DATA --output_path reconstructions_test --split test
```

The outputs will be saved to `reconstructions_test` directory which can be uploaded for submission.

Note: for the 2020 Brain Challenge we have opted to not include compressed sensing as a FAIR/NYU baseline for the leaderboard. The 2020 Challenge uses equispaced masks, which are not supported by compressed sensing theory.
