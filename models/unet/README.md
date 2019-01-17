## U-Net Model for MRI Reconstruction

This directory contains a reference U-Net implementation for MRI reconstruction 
in PyTorch.

To start training the model, run:
```bash
python models/unet/train_unet.py --challenge CHALLENGE --data-path DATA --exp-dir checkpoint
```
where `CHALLENGE` is either `singlecoil` or `multicoil`.

To run the model on validation data:
```bash
python models/unet/run_unet.py --data-path DATA --data-split val --checkpoint checkpoint/best_model.pt --challenge CHALLENGE --out-dir reconstructions_val --mask-kspace
```
The outputs will be saved to `reconstructions_val`. To evaluate the results, run:
```bash
python common/evaluate.py --target-path TARGET_DATA --predictions-path reconstructions_val --challenge CHALLENGE
```

To run the model on test data:
```bash
python models/unet/run_unet.py --data-path DATA --data-split test --checkpoint checkpoint/best_model.pt --challenge CHALLENGE --out-dir reconstructions_test
```
The outputs will be saved to `reconstructions_test` directory which can be uploaded for submission.
