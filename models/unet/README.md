## U-Net Model for MRI Reconstruction

This directory contains a reference U-Net implementation for MRI reconstruction 
in PyTorch.

To start training the model, run:
```bash
python models/unet/train_unet.py --mode train --challenge CHALLENGE --data-path DATA --exp unet --mask-type MASK_TYPE
```
where `CHALLENGE` is either `singlecoil` or `multicoil`. And `MASK_TYPE` is either `random` (for knee)
or `equispaced` (for brain). Training logs and checkpoints are saved in `experiments/unet` directory. 

To run the model on test data:
```bash
python models/unet/train_unet.py --mode test --challenge CHALLENGE --data-path DATA --exp unet --out-dir reconstructions --checkpoint MODEL 
```
where `MODEL` is the path to the model checkpoint from `experiments/unet/version_0/checkpoints/`.

The outputs will be saved to `reconstructions` directory which can be uploaded for submission.
