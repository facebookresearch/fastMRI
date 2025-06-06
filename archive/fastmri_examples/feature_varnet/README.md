# Accelerated MRI reconstructions via variational network and feature domain learning

This directory contains a PyTorch implementation for reproducing the following paper, to be published at MIDL 2022.

[Accelerated MRI reconstructions via variational network and feature domain learning (I. Giannakopoulos, et al., 2024).][feature_varnet]

## Installation
We **strongly** recommend creating a separate conda environment for this example, as the
PyTorch Lightning versions required differs from that of the base `fastmri` installation.

Before installing dependencies, first install PyTorch according to the directions at the
PyTorch Website for your operating system and CUDA setup
(we used `torch` version 1.7.0 for our experiments). Then run

```bash
pip install -r fastmri_examples/feature_varnet/requirements.txt
```


## Example training commands:

This code provides a few ablations of the end-to-end variational network, namely, feature varnet with weight sharing, feature varnet without weight sharing, attention feature varnet with weight sharing, feature-image varnet, and image-feature varnet. Train and test each model with the same commands as the end-to-end variational network and include an additional input argument to your input file:
For the end-to-end varnet
> --varnet_type e2e_varnet

For the feature varnet with weight sharing
> --varnet_type feature_varnet_sh_w

For the feature varnet without weight sharing
> --varnet_type feature_varnet_n_sh_w

For the attention feature varnet with weight sharing
> --varnet_type attention_feature_varnet_sh_w

For the feature-image varnet
> --varnet_type fi_varnet

For the image-feature varnet
> --varnet_type if_varnet

See `train_feature_varnet.py` for additional arguments.


## Example evaluation commands:

Evaluate the model as the end-to-end varnet


## Paths:

Data and log paths are defined the fastmri_dirs.yaml


## Citing

If you use this this code in your research, please cite the corresponding
paper:

```BibTeX
@article{giannakopoulos2024accelerated,
  title={Accelerated MRI reconstructions via variational network and feature domain learning},
  author={Giannakopoulos, Ilias I and Muckley, Matthew J and Kim, Jesi and Breen, Matthew and Johnson, Patricia M and Lui, Yvonne W and Lattanzi, Riccardo},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={10991},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

[feature_varnet]: https://www.nature.com/articles/s41598-024-59705-0
