## Compressed Sensing with Total Variation Minimization

This directory contains code to apply the ESPIRiT algorithm for coil sensitivity 
estimation and Total Variation minimization based reconstruction using the 
BART toolkit.

To install BART, please follow the installation instructions at 
https://mrirecon.github.io/bart/.

Once BART is installed, set the `TOOLBOX_PATH` environment variable and point
`PYTHONPATH` to the python wrapper for BART:

```bash
export TOOLBOX_PATH=/path/to/bart
export PYTHONPATH=${TOOLBOX_PATH}/python:${PYTHONPATH}
```

To run the reconstruction algorithm on the validation data, run:
```bash
python models/cs/run_bart_val.py --challenge CHALLENGE --data-path DATA --output-path reconstructions_val --reg-wt 0.01
```
where `CHALLENGE` is either `singlecoil` or `multicoil`. The outputs are saved in a directory called 
`reconstructions_val`. To evaluate the results, run:
```bash
python common/evaluate.py --target-path TARGET_DATA --predictions-path reconstructions_val --challenge CHALLENGE
```

To apply the reconstruction algorithm to the test data, run:
```bash
python models/cs/run_bart_test.py --challenge CHALLENGE --data-path DATA --output-path reconstructions_test
``` 
The outputs will be saved to `reconstructions_test` directory which can be uploaded for submission.

