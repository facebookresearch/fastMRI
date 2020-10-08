## End-to-End Variational Networks for Accelerated MRI Reconstruction Model

This directory contains a PyTorch implementation for the method described in [End-to-End Variational Networks for Accelerated MRI Reconstruction Model](https://arxiv.org/abs/2004.06688).

To start training the model, run:
```bash
python models/varnet/varnet.py --mode train --challenge multicoil --data-path DATA --exp var_net --mask-type MASK_TYPE
```
where `CHALLENGE` is either `singlecoil` or `multicoil`. And `MASK_TYPE` is either `random` (for knee)
or `equispaced` (for brain). Training logs and checkpoints are saved in `experiments/unet` directory.

To run the model on test data:
```bash
python models/varnet/varnet.py --mode test --challenge multicoil --data-path DATA --exp var_net --mask-type MASK_TYPE --out-dir reconstructions --checkpoint MODEL
```
where `MODEL` is the path to the model checkpoint from `experiments/var_net/version_0/checkpoints/`.

The outputs will be saved to `reconstructions` directory which can be uploaded for submission.

## Citing
If you use this this code in your research, please cite the corresponding paper:
```
@inproceedings{sriram2020endtoend,
    title={End-to-End Variational Networks for Accelerated MRI Reconstruction},
    author={Anuroop Sriram and Jure Zbontar and Tullie Murrell and Aaron Defazio and C. Lawrence Zitnick and Nafissa Yakubova and Florian Knoll and Patricia Johnson},
    year={2020},
    eprint={2004.06688},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
