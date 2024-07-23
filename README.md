# fastMRI

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md)
[![Build and Test](https://github.com/facebookresearch/fastMRI/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/facebookresearch/fastMRI/actions/workflows/build-and-test.yml)

[Website](https://fastMRI.org) |
[Dataset](https://fastmri.med.nyu.edu/) |
[GitHub](https://github.com/facebookresearch/fastMRI) |
[Publications](#list-of-papers)

Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements
has the potential to reduce medical costs, minimize stress to patients and make
MR imaging possible in applications where it is currently prohibitively slow or
expensive.

[fastMRI](https://fastMRI.org) is a collaborative research project from
Facebook AI Research (FAIR) and NYU Langone Health to investigate the use of AI
to make MRI scans faster. NYU Langone Health has released fully anonymized knee
and brain MRI datasets that can be downloaded from
[the fastMRI dataset page](https://fastmri.med.nyu.edu/). Publications
associated with the fastMRI project can be found
[at the end of this README](#list-of-papers).

This repository contains convenient PyTorch data loaders, subsampling
functions, evaluation metrics, and reference implementations of simple baseline
methods. It also contains implementations for methods in some of the
publications of the fastMRI project.

## Documentation

### The fastMRI Dataset

There are multiple publications describing different subcomponents of the data
(e.g., brain vs. knee) and associated baselines. All of the fastMRI data can be
downloaded from the [fastMRI dataset page](https://fastmri.med.nyu.edu/).

* **Project Summary, Datasets, Baselines:** [fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)

* **Knee Data:** [fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning ({F. Knoll*, J. Zbontar*} et al., 2020)](https://doi.org/10.1148/ryai.2020190007)

* **Brain Dataset Properties:** [Supplemental Material](https://ieeexplore.ieee.org/ielx7/42/9526230/9420272/supp1-3075856.pdf?arnumber=9420272) of [Results of the 2020 fastMRI Challenge for Machine Learning MR Image Reconstruction ({M. Muckley*, B. Riemenschneider*} et al., 2021)](https://doi.org/10.1109/TMI.2021.3075856)

* **Prostate Data:** [FastMRI Prostate: A Publicly Available, Biparametric MRI Dataset to Advance Machine Learning for Prostate Cancer Imaging (Tibrewala et al., 2023)](https://arxiv.org/abs/2304.09254)

### Code Repository

For code documentation, most functions and classes have accompanying docstrings
that you can access via the `help` function in IPython. For example:

```python
from fastmri.data import SliceDataset

help(SliceDataset)
```

## Dependencies and Installation

**Note:** Contributions to the code are continuously tested via GitHub actions.
If you encounter an issue, the best first thing to do is to try to match the
`tests` environment in `setup.cfg`, e.g., `pip install --editable ".[tests]"`
when installing from source.

**Note:** As documented in [Issue 215](https://github.com/facebookresearch/fastMRI/issues/215),
there is currently a memory leak when using `h5py` installed from `pip` and
converting to a `torch.Tensor`. To avoid the leak, you need to use `h5py` with
a version of HDF5 before 1.12.1. As of February 16, 2022, the `conda` version
of `h5py` 3.6.0 used HDF5 1.10.6, which avoids the leak.

First install PyTorch according to the directions at the
[PyTorch Website](https://pytorch.org/get-started/) for your operating system
and CUDA setup. Then, run

```bash
pip install fastmri
```

`pip` will handle all package dependencies. After this you should be able to
run most of the code in the repository.

### Installing Directly from Source

If you want to install directly from the GitHub source, clone the repository,
navigate to the `fastmri` root directory and run

```bash
pip install -e .
```

## Package Structure & Usage

The repository is centered around the `fastmri` module. The following breaks
down the basic structure:

`fastmri`: Contains a number of basic tools for complex number math, coil
combinations, etc.

* `fastmri.data`: Contains data utility functions from original `data` folder
that can be used to create sampling masks and submission files.
* `fastmri.models`: Contains reconstruction models, such as the U-Net and
VarNet.
* `fastmri.pl_modules`: PyTorch Lightning modules for data loading, training,
and logging.

## Examples and Reproducibility

The `fastmri_examples` and `banding_removal` folders include code for
reproducibility. The baseline models were used in the [arXiv paper](https://arxiv.org/abs/1811.08839).

A brief summary of implementions based on papers with links to code follows.
For completeness we also mention work on active acquisition, which is hosted
in another repository.

* **Baseline Models**

  * [Zero-filled examples for saving images for leaderboard submission](https://github.com/facebookresearch/fastMRI/tree/master/fastmri_examples/zero_filled/)
  * [ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA (M. Uecker et al., 2013)](https://github.com/facebookresearch/fastMRI/tree/master/fastmri_examples/cs/)
  * [U-Net: Convolutional networks for biomedical image segmentation (O. Ronneberger et al., 2015)](https://github.com/facebookresearch/fastMRI/tree/master/fastmri_examples/unet/)

* **Sampling, Reconstruction and Artifact Correction**

  * [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry (A. Defazio, 2019)](https://github.com/facebookresearch/fastMRI/blob/8abe6eaeeb3d4504f26dc77adffb02a4be41d6f4/fastmri/data/subsample.py#L344-L475)
  * [End-to-End Variational Networks for Accelerated MRI Reconstruction ({A. Sriram*, J. Zbontar*} et al., 2020)](https://github.com/facebookresearch/fastMRI/tree/master/fastmri_examples/varnet/)
  * [MRI Banding Removal via Adversarial Training (A. Defazio, et al., 2020)](https://github.com/facebookresearch/fastMRI/tree/master/banding_removal)
  * [Deep Learning Reconstruction Enables Prospectively Accelerated Clinical Knee MRI (P. Johnson et al., 2023)](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/RadiologyJohnson2022)
  * [Accelerated MRI reconstructions via variational network and feature domain learning (I. Giannakopoulos et al., 2024)](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/feature_varnet)

* **Active Acquisition**
  * (external repository) [Reducing uncertainty in undersampled MRI reconstruction with active acquisition (Z. Zhang et al., 2019)](https://github.com/facebookresearch/active-mri-acquisition/tree/master/activemri/experimental/cvpr19_models)
  * (external repository) [Active MR k-space Sampling with Reinforcement Learning (L. Pineda et al., 2020)](https://github.com/facebookresearch/active-mri-acquisition)
  * [On learning adaptive acquisition policies for undersampled multi-coil MRI reconstruction (T. Bakker et al., 2022)](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/adaptive_varnet/)

* **Prostate Data**
  * (external respository) [FastMRI Prostate: A Publicly Available, Biparametric MRI Dataset to Advance Machine Learning for Prostate Cancer Imaging (Tibrewala et al., 2023)](https://github.com/cai2r/fastMRI_prostate)

## Testing

Run `pytest tests`. By default integration tests that use the fastMRI data are
skipped. If you would like to run these tests, set `SKIP_INTEGRATIONS` to
`False` in the [conftest](https://github.com/facebookresearch/fastMRI/tree/master/tests/conftest.py).

## Training a model

The [data README](https://github.com/facebookresearch/fastMRI/tree/master/fastmri/data/README.md) has a bare-bones example for how to
load data and incorporate data transforms. This
[jupyter notebook](https://github.com/facebookresearch/fastMRI/tree/master/fastMRI_tutorial.ipynb) contains a simple tutorial
explaining how to get started working with the data.

Please look at
[this U-Net demo script](https://github.com/facebookresearch/fastMRI/tree/master/fastmri_examples/unet/train_unet_demo.py) for an
example of how to train a model using the PyTorch Lightning framework.

## Submitting to the Leaderboard

**NOTICE:** As documented in [Discussion 293](https://github.com/facebookresearch/fastMRI/discussions/293),
the fastmri.org domain was transferred from Meta ownership to NYU ownership on
2023-04-17, and NYU has not yet rebuilt the site. Until the site and
leaderbaords are rebuilt by NYU, leaderboards will be unavailable. Mitigations
are presented in [Discussion 293](https://github.com/facebookresearch/fastMRI/discussions/293).

## License

fastMRI is MIT licensed, as found in the [LICENSE file](https://github.com/facebookresearch/fastMRI/tree/master/LICENSE.md).

## Cite

If you use the fastMRI data or code in your project, please cite the arXiv
paper:

```BibTeX
@misc{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```

If you use the fastMRI prostate data or code in your project, please cite that
paper:

```BibTeX
@misc{tibrewala2023fastmri,
  title={{FastMRI Prostate}: A Publicly Available, Biparametric {MRI} Dataset to Advance Machine Learning for Prostate Cancer Imaging},
  author={Tibrewala, Radhika and Dutt, Tarun and Tong, Angela and Ginocchio, Luke and Keerthivasan, Mahesh B and Baete, Steven H and Chopra, Sumit and Lui, Yvonne W and Sodickson, Daniel K and Chandarana, Hersh and Johnson, Patricia M},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint={2304.09254},
  year={2023}
}
```

## List of Papers

The following lists titles of papers from the fastMRI project. The
corresponding abstracts, as well as links to preprints and code can be found
[here](https://github.com/facebookresearch/fastMRI/tree/master/LIST_OF_PAPERS.md).

1. Zbontar, J.\*, Knoll, F.\*, Sriram, A.\*, Murrell, T., Huang, Z., Muckley, M. J., ... & Lui, Y. W. (2018). [fastMRI: An Open Dataset and Benchmarks for Accelerated MRI](https://arxiv.org/abs/1811.08839). *arXiv preprint arXiv:1811.08839*.
2. Zhang, Z., Romero, A., Muckley, M. J., Vincent, P., Yang, L., & Drozdzal, M. (2019). [Reducing uncertainty in undersampled MRI reconstruction with active acquisition](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.html). In *CVPR*, pages 2049-2058.
3. Defazio, A. (2019). [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry](https://arxiv.org/abs/1912.01101). *arXiv preprint, arXiv:1912.01101*.
4. Knoll, F.\*, Zbontar, J.\*, Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., ... & Lui, Y. W. (2020). [fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning](https://doi.org/10.1148/ryai.2020190007). *Radiology: Artificial Intelligence*, 2(1), page e190007.
5. Knoll, F.\*, Murrell, T.\*, Sriram, A.\*, Yakubova, N., Zbontar, J., Rabbat, M., ... & Recht, M. P. (2020). [Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge](https://doi.org/10.1002/mrm.28338). *Magnetic Resonance in Medicine*, 84(6), pages 3054-3070.
6. Sriram, A., Zbontar, J., Murrell, T., Zitnick, C. L., Defazio, A., & Sodickson, D. K. (2020). [GrappaNet: Combining parallel imaging with deep learning for multi-coil MRI reconstruction](https://openaccess.thecvf.com/content_CVPR_2020/html/Sriram_GrappaNet_Combining_Parallel_Imaging_With_Deep_Learning_for_Multi-Coil_MRI_CVPR_2020_paper.html). In *CVPR*, pages 14315-14322.
7. Recht, M. P., Zbontar, J., Sodickson, D. K., Knoll, F., Yakubova, N., Sriram, A., ... & Zitnick, C. L. (2020). [Using Deep Learning to Accelerate Knee MRI at 3T: Results of an Interchangeability Study](https://doi.org/10.2214/AJR.20.23313). *American Journal of Roentgenology*, 215(6), pages 1421-1429.
8. Pineda, L., Basu, S., Romero, A., Calandra, R., & Drozdzal, M. (2020). [Active MR k-space Sampling with Reinforcement Learning](https://doi.org/10.1007/978-3-030-59713-9_3). In *MICCAI*, pages 23-33.
9. Sriram, A.\*, Zbontar, J.\*, Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., ... & Johnson, P. (2020). [End-to-End Variational Networks for Accelerated MRI Reconstruction](https://doi.org/10.1007/978-3-030-59713-9_7). In *MICCAI*, pages 64-73.
10. Defazio, A., Murrell, T., & Recht, M. P. (2020). [MRI Banding Removal via Adversarial Training](https://papers.nips.cc/paper/2020/hash/567b8f5f423af15818a068235807edc0-Abstract.html). In *Advances in Neural Information Processing Systems*, 33, pages 7660-7670.
11. Muckley, M. J.\*, Riemenschneider, B.\*, Radmanesh, A., Kim, S., Jeong, G., Ko, J., ... & Knoll, F. (2021). [Results of the 2020 fastMRI Challenge for Machine Learning MR Image Reconstruction](https://doi.org/10.1109/TMI.2021.3075856). *IEEE Transactions on Medical Imaging*, 40(9), pages 2306-2317.
12. Johnson, P. M., Jeong, G., Hammernik, K., Schlemper, J., Qin, C., Duan, J., ..., & Knoll, F. (2021). [Evaluation of the Robustness of Learned MR Image Reconstruction to Systematic Deviations Between Training and Test Data for the Models from the fastMRI Challenge](https://doi.org/10.1007/978-3-030-88552-6_3). In *MICCAI MLMIR Workshop*, pages 25–34,
13. Bakker, T., Muckley, M.J., Romero-Soriano, A., Drozdzal, M. & Pineda, L. (2022). [On learning adaptive acquisition policies for undersampled multi-coil MRI reconstruction](https://proceedings.mlr.press/v172/bakker22a). In *MIDL*, pages 63-85.
14. Radmanesh, A.\*, Muckley, M. J.\*, Murrell, T., Lindsey, E., Sriram, A., Knoll, F., ... & Lui, Y. W. (2022). [Exploring the Acceleration Limits of Deep Learning VarNet-based Two-dimensional Brain MRI](https://doi.org/10.1148/ryai.210313). *Radiology: Artificial Intelligence*, 4(6), page e210313.
15. Johnson, P.M., Lin, D.J., Zbontar, J., Zitnick, C.L., Sriram, A., Muckley, M., Babb, J.S., Kline, M., Ciavarra, G., Alaia, E., ..., & Knoll, F. (2023). [Deep Learning Reconstruction Enables Prospectively Accelerated Clinical Knee MRI](https://doi.org/10.1148/radiol.220425). *Radiology*, 307(2), page e220425.
16. Tibrewala, R., Dutt, T., Tong, A., Ginocchio, L., Keerthivasan, M.B., Baete, S.H., Lui, Y.W., Sodickson, D.K., Chandarana, H., Johnson, P.M. (2023). [FastMRI Prostate: A Publicly Available, Biparametric MRI Dataset to Advance Machine Learning for Prostate Cancer Imaging](https://arxiv.org/abs/2304.09254). *arXiv preprint, arXiv:2034.09254*.
16. Giannakopoulos, I. I., Muckley, M. J., Kim, J., Breen, M., Johnson, P. M., Lui, Y. W., Lattanzi, R. (2024). [Accelerated MRI reconstructions via variational network and feature domain learning. Scientific Reports](https://www.nature.com/articles/s41598-024-59705-0). *Scientific Reports, 14(1), 10991*.

