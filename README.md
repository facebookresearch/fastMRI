# fastMRI

[Website and Leaderboards](https://fastMRI.org) | [Dataset](https://fastmri.med.nyu.edu/) | [GitHub](https://github.com/facebookresearch/fastMRI) | [Publications](https://github.com/facebookresearch/fastMRI/blob/master/LIST_OF_PAPERS.md)

Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements has the potential to reduce medical costs, minimize stress to patients and make MR imaging possible in applications where it is currently prohibitively slow or expensive.

[fastMRI](https://fastMRI.org) is a collaborative research project from Facebook AI Research (FAIR) and NYU Langone Health to investigate the use of AI to make MRI scans faster. NYU Langone Health has released fully anonymized knee and brain MRI datasets that can be downloaded from [the fastMRI dataset page](https://fastmri.med.nyu.edu/). Publications associated with the fastMRI project can be found on our [list of papers](#list-of-papers).

This repository contains convenient PyTorch data loaders, subsampling functions, evaluation metrics, and reference implementations of simple baseline methods. It also contains implementations for methods in some of the publications of the fastMRI project.

## Outline

1. [Citing](#citing)
2. [Dependencies and Installation](#Dependencies-and-Installation)
3. [Directory Structure & Usage](#directory-structure--usage)
4. [Testing](#testing)
5. [Training a model](#training-a-model)
6. [Submitting to the Leaderboard](#submitting-to-the-leaderboard)
7. [License](#license)
8. [List of Papers](#list-of-papers)

## Citing

Documentation for the fastMRI dataset and baseline reconstruction performance can be found in [our paper on arXiv](https://arxiv.org/abs/1811.08839). The paper is updated on an ongoing basis for dataset additions and new baselines. If you use the fastMRI data or code in your project, please consider citing the dataset paper:

```BibTeX
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Matthew J. Muckley and Mary Bruno and Aaron Defazio and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and James Pinkerton and Duo Wang and Nafissa Yakubova and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```

For code documentation, most functions and classes have accompanying docstrings that you can access via the `help` function in IPython. For example:

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

You can find the full list of Python packages needed to run the code in the `requirements.txt` file. Most people already have their own PyTorch environment configured with Anaconda, and based on `requirements.txt` you can install the final packages as needed.

If you want to install with `pip`, first delete the `git+https://github.com/ismrmrd/ismrmrd-python.git` line from `requirements.txt`. Then, run

```bash
pip install -r requirements.txt
```

Finally, run

```bash
pip install git+https://github.com/ismrmrd/ismrmrd-python.git
```

Then you should have all the packages.

## Directory Structure & Usage

Since August 2020, the repository has been refactored to operate as a package centered around the `fastmri` module, while configurations and scripts for reproducibility are now hosted in `experimental`. Other folders are in the process of being adapted to the new structure and then deprecated.

`fastmri`: Contains a number of basic tools for complex number math, coil combinations, etc.

* `fastmri/data`: Contains data utility functions from original `data` folder that can be used to create sampling masks and submission files.
* `fastmri/models`: Contains baseline models, including the U-Net and the End-to-end Variational Network.

`experimental`: Folders intended to aid reproducibility of baselines. We currently have code for reproducing the following papers:

* fastMRI: An open dataset and benchmarks for accelerated MRI (Zbontar, J. et al., 2018)
  * [U-Net Baseline](https://github.com/facebookresearch/fastMRI/tree/master/experimental/unet)
  * [Zero-filled Baseline](https://github.com/facebookresearch/fastMRI/tree/master/experimental/zero_filled)
* [End-to-End Variational Networks for Accelerated MRI Reconstruction (Sriram, A. et al. 2020)](https://github.com/facebookresearch/fastMRI/tree/master/experimental/varnet)
* [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry (Defazio, A., 2019)](https://github.com/facebookresearch/fastMRI/blob/master/banding_removal/fastmri/common/subsample.py)
* [MRI Banding Removal via Adversarial Training (Defazio, A. et al., 2020)](https://github.com/facebookresearch/fastMRI/tree/master/banding_removal)

## Testing

Run `python -m pytest tests`.

## Training a model

The [data README](https://github.com/facebookresearch/fastMRI/tree/master/fastmri/data/README.md) has a bare-bones example for how to load data and incorporate data transforms. This [jupyter notebook](https://github.com/facebookresearch/fastMRI/blob/master/fastMRI_tutorial.ipynb) contains a simple tutorial explaining how to get started working with the data.

Please look at [this U-Net demo script](https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/train_unet_demo.py) for an example of how to train a model using the PyTorch Lightning framework included with the package.

## Submitting to the Leaderboard

Run your model on the provided test data and create a zip file containing your predictions. `fastmri` has a `save_reconstructions` function that saves the data in the correct format.

Upload the zip file to any publicly accessible cloud storage (e.g. Amazon S3, Dropbox etc). Submit a link to the zip file on the [challenge website](https://fastmri.org/submit). You will need to create an account before submitting.

## License

fastMRI is MIT licensed, as found in the [LICENSE file](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md).

## List of Papers

The following lists titles of papers from the fastMRI project. A more complete list of papers with abstracts and links to code can be found [here](LIST_OF_PAPERS.md).

1. [fastMRI: An open dataset and benchmarks for accelerated MRI (Zbontar, J. et al., 2018)](https://arxiv.org/abs/1811.08839).
2. [Reducing uncertainty in undersampled MRI reconstruction with active acquisition (Zhang, Z. et al., 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.html).
3. [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry (Defazio, A., 2019)](https://arxiv.org/abs/1912.01101).
4. [MRI Banding Removal via Adversarial Training (Defazio, A. et al., 2020)](https://arxiv.org/abs/2001.08699).
5. [fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning (Knoll, F. et al., 2020)](https://doi.org/10.1148/ryai.2020190007).
6. [Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge (Knoll, F. et al., 2020)](https://doi.org/10.1002/mrm.28338).
7. [GrappaNet: Combining parallel imaging with deep learning for multi-coil MRI reconstruction (Sriram, A. et al., 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Sriram_GrappaNet_Combining_Parallel_Imaging_With_Deep_Learning_for_Multi-Coil_MRI_CVPR_2020_paper.html).
8. [Using Deep Learning to Accelerate Knee MRI at 3T: Results of an Interchangeability Study (Recht, M. P. et al., 2020)](https://www.ajronline.org/doi/abs/10.2214/AJR.20.23313).
9. [Active MR k-space Sampling with Reinforcement Learning (Pineda, L. et al., 2020)](https://arxiv.org/abs/2007.10469).
10. [End-to-End Variational Networks for Accelerated MRI Reconstruction (Sriram, A. et al. 2020)](https://arxiv.org/abs/2004.06688).
