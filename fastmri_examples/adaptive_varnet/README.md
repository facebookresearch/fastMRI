# End-to-End Variational Networks for Accelerated MRI Reconstruction Model

This directory contains a PyTorch implementation for reproducing the following paper, to be published at MIDL 2022.

[On learning adaptive acquisition policies for undersampled multi-coil MRI reconstruction (T. Bakker, et al., 2022).][adaptive_varnet]

## Installation
We **strongly** recommend creating a separate conda environment for this example, as the
PyTorch Lightning versions required differs from that of the base `fastmri` installation.

Before installing dependencies, first install PyTorch according to the directions at the 
PyTorch Website for your operating system and CUDA setup 
(we used `torch` version 1.7.0 for our experiments). Then run

```bash
pip install -r fastmri_examples/adaptive_varnet/requirements.txt
```


## Example commands:

The following commands are for the acceleration 4, budget 22 setting. For an MR image of default size (128, 128) this corresponds to 32 sampled lines, starting from 10 low-frequency lines.

#### Policy (sigmoid with slope 10, 5 cascades):
> python ./fastmri_examples/adaptive_varnet/train_adaptive_varnet_demo.py --data_path PATH_TO_DATA --default_root_dir PATH_TO_OUT --seed SEED --batch_size 16 --num_cascades 5 --accelerations 4 --center_fractions 0.08 --num_sense_lines 10 --budget 22 --learn_acquisition True --loupe_mask False --use_softplus False --slope 10 --cascades_per_policy 5

#### LOUPE (sigmoid with slope 10, 5 cascades):
> python ./fastmri_examples/adaptive_varnet/train_adaptive_varnet_demo.py --data_path PATH_TO_DATA --default_root_dir PATH_TO_OUT --seed SEED --batch_size 16 --num_cascades 5 --accelerations 4 --center_fractions 0.08 --num_sense_lines 10 --budget 22 --learn_acquisition True --loupe_mask True --use_softplus False --slope 10

#### Equispaced (5 cascades):
> python ./fastmri_examples/adaptive_varnet/train_adaptive_varnet_demo.py --data_path PATH_TO_DATA --default_root_dir PATH_TO_OUT --seed SEED --batch_size 16 --num_cascades 5 --accelerations 4 --center_fractions 0.08 --num_sense_lines 10

For logging with [wandb][wandb], add the following arguments.
> --wandb True --project WANDB_PROJECT --wandb_entity WANDB_ENTITY

For acceleration 8, change the following arguments where relevant. For an MR image of default size (128, 128) this corresponds to 16 sampled lines, starting from 4 low-frequency lines.
> --accelerations 8 --center_fractions 0.04 --num_sense_lines 4 --budget 12

See `train_adaptive_varnet_demo.py` for additional arguments.


## Citing

If you use this this code in your research, please cite the corresponding
paper:

```BibTeX
@article{bakker2022adaptive,
    title={On learning adaptive acquisition policies for undersampled multi-coil {MRI} reconstruction},
    author={Tim Bakker and Matthew Muckley and Adriana Romero-Soriano and Michal Drozdzal and Luis Pineda},
    journal={Proceedings of Machine Learning Research (MIDL)},
    pages={to appear},
    year={2022},
}
```

[adaptive_varnet]: https://arxiv.org/abs/2203.16392
[wandb]: https://wandb.ai/site