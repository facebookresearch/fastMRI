# fastMRI

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md)
[![CircleCI](https://circleci.com/gh/facebookresearch/fastMRI.svg?style=svg)](https://app.circleci.com/pipelines/github/facebookresearch/fastMRI)

[Website and Leaderboards](https://fastMRI.org) |
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

Documentation for the fastMRI dataset and baseline reconstruction performance
can be found in [our paper on arXiv](https://arxiv.org/abs/1811.08839). The
paper is updated on an ongoing basis for dataset additions and new baselines.

For code documentation, most functions and classes have accompanying docstrings
that you can access via the `help` function in IPython. For example:

```python
from fastmri.data import SliceDataset

help(SliceDataset)
```

## Dependencies and Installation

We have tested this code using:

* Ubuntu 18.04
* Python 3.8
* CUDA 10.1
* CUDNN 7.6.5

First install PyTorch according to the directions at the
[PyTorch Website](https://pytorch.org/get-started/) for your operating system
and CUDA setup.

Then, navigate to the `fastmri` root directory and run

```bash
pip install -e .
```

`pip` will handle all package dependencies. After this you should be able to
run most of the code in the repository.

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
reproducibility. The baseline models were used in the arXiv paper:

[fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)

A brief summary of implementions based on papers with links to code follows.
For completeness we also mention work on active acquisition, which is hosted
in another repository.

* **Baseline Models**

  * [Zero-filled examples for saving images for leaderboard submission](fastmri_examples/zero_filled/)
  * [ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA (M. Uecker et al., 2013)](fastmri_examples/cs/)
  * [U-Net: Convolutional networks for biomedical image segmentation (O. Ronneberger et al., 2015)](fastmri_examples/unet/)

* **Sampling, Reconstruction and Artifact Correction**

  * [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry (A. Defazio, 2019)](banding_removal/fastmri/common/subsample.py#L126-L198)
  * [End-to-End Variational Networks for Accelerated MRI Reconstruction ({A. Sriram*, J. Zbontar*} et al., 2020)](fastmri_examples/varnet/)
  * [MRI Banding Removal via Adversarial Training (A. Defazio, et al., 2020)](banding_removal)

* **Active Acquisition** (external repository)
  * [Reducing uncertainty in undersampled MRI reconstruction with active acquisition (Z. Zhang et al., 2019)](https://github.com/facebookresearch/active-mri-acquisition/tree/master/activemri/experimental/cvpr19_models)
  * [Active MR k-space Sampling with Reinforcement Learning (L. Pineda et al., 2020)](https://github.com/facebookresearch/active-mri-acquisition)

## Testing

Run `pytest tests`. By default integration tests that use the fastMRI data are
skipped. If you would like to run these tests, set `SKIP_INTEGRATIONS` to
`False` in the [conftest](tests/conftest.py).

## Training a model

The [data README](fastmri/data/README.md) has a bare-bones example for how to
load data and incorporate data transforms. This
[jupyter notebook](fastMRI_tutorial.ipynb) contains a simple tutorial
explaining how to get started working with the data.

Please look at
[this U-Net demo script](fastmri_examples/unet/train_unet_demo.py) for an
example of how to train a model using the PyTorch Lightning framework.

## Submitting to the Leaderboard

Run your model on the provided test data and create a zip file containing your
predictions. `fastmri` has a `save_reconstructions` function that saves the
data in the correct format.

Upload the zip file to any publicly accessible cloud storage (e.g. Amazon S3,
Dropbox etc). Submit a link to the zip file on the
[challenge website](https://fastmri.org/submit). You will need to create an
account before submitting.

## License

fastMRI is MIT licensed, as found in the [LICENSE file](LICENSE.md).

## Cite

If you use the fastMRI data or code in your project, please cite the arXiv
paper:

```BibTeX
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```

## List of Papers

The following lists titles of papers from the fastMRI project. The
corresponding abstracts, as well as links to preprints and code can be found
[here](LIST_OF_PAPERS.md).

1. Zbontar, J., Knoll, F., Sriram, A., Murrell, T., Huang, Z., Muckley, M. J., ... & Lui, Y. W. (2018). [fastMRI: An open dataset and benchmarks for accelerated MRI](https://arxiv.org/abs/1811.08839). *arXiv preprint arXiv:1811.08839*.
2. Zhang, Z., Romero, A., Muckley, M. J., Vincent, P., Yang, L., & Drozdzal, M. (2019). [Reducing uncertainty in undersampled MRI reconstruction with active acquisition](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.html). In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2049-2058.
3. Defazio, A. (2019). [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry](https://arxiv.org/abs/1912.01101). *arXiv preprint, arXiv:1912.01101*.
4. Knoll, F., Zbontar, J., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., ... & Lui, Y. W. (2020). [fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning](https://doi.org/10.1148/ryai.2020190007). *Radiology: Artificial Intelligence*, 2(1), page e190007.
5. Knoll, F., Murrell, T., Sriram, A., Yakubova, N., Zbontar, J., Rabbat, M., ... & Recht, M. P. (2020). [Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge](https://doi.org/10.1002/mrm.28338). *Magnetic Resonance in Medicine*, 84(6), pages 3054-3070.
6. Sriram, A., Zbontar, J., Murrell, T., Zitnick, C. L., Defazio, A., & Sodickson, D. K. (2020). [GrappaNet: Combining parallel imaging with deep learning for multi-coil MRI reconstruction](https://openaccess.thecvf.com/content_CVPR_2020/html/Sriram_GrappaNet_Combining_Parallel_Imaging_With_Deep_Learning_for_Multi-Coil_MRI_CVPR_2020_paper.html). In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14315-14322.
7. Recht, M. P., Zbontar, J., Sodickson, D. K., Knoll, F., Yakubova, N., Sriram, A., ... & Zitnick, C. L. (2020). [Using Deep Learning to Accelerate Knee MRI at 3T: Results of an Interchangeability Study](https://doi.org/10.2214/AJR.20.23313). *American Journal of Roentgenology*, 215(6), pages 1421-1429.
8. Pineda, L., Basu, S., Romero, A., Calandra, R., & Drozdzal, M. (2020). [Active MR k-space Sampling with Reinforcement Learning](https://doi.org/10.1007/978-3-030-59713-9_3). In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pages 23-33.
9. Sriram, A., Zbontar, J., Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., ... & Johnson, P. (2020). [End-to-End Variational Networks for Accelerated MRI Reconstruction](https://doi.org/10.1007/978-3-030-59713-9_7). In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pages 64-73.
10. Defazio, A., Murrell, T., & Recht, M. P. (2020). [MRI Banding Removal via Adversarial Training](https://papers.nips.cc/paper/2020/hash/567b8f5f423af15818a068235807edc0-Abstract.html). In *Advances in Neural Information Processing Systems*.
11. Muckley, M. J., Riemenschneider, B., Radmanesh, A., Kim, S., Jeong, G., Ko, J., ... & Knoll, F. (2020). [State-of-the-art Machine Learning MRI Reconstruction in 2020: Results of the Second fastMRI Challenge](https://arxiv.org/abs/2012.06318). *arXiv preprint arXiv:2012.06318*.
