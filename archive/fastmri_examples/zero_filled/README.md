# Zero-filled Solution

The `run_zero_filled.py` script creates a submission with the zero-filled
solution. This script exists as a simple baseline, only to demonstrate how to
create a submission.

To generate the zero-filled submissions, run:

```bash
python run_zero_filled.py \
    --challenge CHALLENGE \
    --data_path DATA \
    --output_path RECONS
```

where `CHALLENGE` is either `singlecoil` or `multicoil`.

For cropping the script infers the target image dimensions from the ISMRMRD
header. Note that there are some issues with files that have "FLAIR_203" in
their file name - there are three of these files in the test set, and they are
ignored for the brain leaderboard.
