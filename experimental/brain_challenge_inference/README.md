# Brain Challenge Inference Scripts

This directory contains example scripts for running models on the 2020 fastMRI Challenge
Data split. The scripts are designed to be minimal - all that is needed is a model
checkpoint, a path to the data directory, and a path where you want to save your
reconstructions. In order to submit after saving, you'll want to compress your images
to a `.zip` or a `.tar.gz` file and put a link to the file in the submission form at
[fastmri.org](fastmri.org).

## Note on the brain challenge transfer track

The "Transfer" track contains data from two major MRI vendors: Philips and GE. This
contrasts with the main fastMRI data set, which was collected on Siemens scanners. When
submitting to the Transfer track you are required to only train your model weights with
provided data - i.e., if you have Philips or GE data outside of the fastMRI set, you may
not use it for the training phase (Note: we do not prohibit participants from using
their own data for validation.)

One important aspect is that the GE data is provided without frequency oversampling.
This may affect some models (such as those in this repository) that were trained with
frequency-oversampled inputs into CNNs. Extra work must be done to get these models to
successfully process the Transfer data. In `run_brain_challenge_transfer_inference.py`,
we simulate the oversampling with image-domain zero-padding prior to model input. We
believe there are better ways to augment the Transfer data with regards to frequency
oversampling. Our only goal with providing this code is to illustrate the issue to
challenge participants. We leave decisions on best ways to handle this to the
participants, as they are best-informed on the properties of their models.

## Examples

To run inference on the 4x/8x data split, run the following command:

```bash
python run_brain_challenge_inference.py \
    --checkpoint CHECKPOINT \
    --data-path DATA_PATH \
    --output-path OUTPUT_PATH
```

where `CHECKPOINT` is your model `CHECKPOINT`, `DATA_PATH` is a path to the
`brain_multicoil_challenge` split of the brain data, and `OUTPUT_PATH` is where you
would like to save your reconstructions. Note that the script expects `CHECKPOINT` to be
saved in PyTorch Lightning format - you may need to modify how it processes the model
`state_dict` is if you used a differrent training framework.

To run inference on the Transfer data split, run the following command:

```bash
python run_brain_challenge_transfer_inference.py \
    --checkpoint CHECKPOINT \
    --data-path DATA_PATH \
    --output-path OUTPUT_PATH
```

where `CHECKPOINT` and `OUTPUT_PATH` are as above. `DATA_PATH` should point to the 
`brain_multicoil_challenge_transfer` split of the brain data.
