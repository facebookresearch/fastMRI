# End-to-End Variational Networks for Accelerated MRI Reconstruction Model

This directory contains a PyTorch implementation for the method described in [End-to-End Variational Networks for Accelerated MRI Reconstruction Model](https://arxiv.org/abs/2004.06688).

To start training the model, run:

```bash
python train_varnet_demo.py
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

## Citing

If you use this this code in your research, please cite the corresponding paper:

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

## Implementaiton Notes

There are a few differences between this implementation and the [paper](https://arxiv.org/abs/2004.06688).

- The paper model used a fixed number of center lines, whereas this model uses the `center_fractions` variable that might change depending on image size.
- The paper model trained separately on 4x and 8x, whereas this model trains on both of them together.
- The paper model trained on the `train` and `val` splits together before leaderboard submission, whereas this model only trains on `val`.

These differences have been left partly for backwards compatibility and partly due to the number of areas in the code base that would have go be tweaked and tested to get them working. A full matching implementation may be added to the repository in the future.
