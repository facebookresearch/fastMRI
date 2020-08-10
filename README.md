# fastMRI

Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements has the potential to reduce medical costs, minimize stress to patients and make MR imaging possible in applications where it is currently prohibitively slow or expensive.

[fastMRI](https://fastMRI.org) is a collaborative research project from Facebook AI Research (FAIR) and NYU Langone Health to investigate the use of AI to make MRI scans faster. NYU Langone Health has released fully anonymized knee and brain MRI datasets that can be downloaded from [the fastMRI dataset page](https://fastmri.med.nyu.edu/). Publications associated with the fastMRI project can be found on our [list of papers](#list-of-papers).

This repository contains convenient PyTorch data loaders, subsampling functions, evaluation metrics, and reference implementations of simple baseline methods.

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

If you use the fastMRI data or this code in your research, please consider citing
the fastMRI dataset paper:

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

For other publications from the fastMRI project please see our [list of papers](#list-of-papers).

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

* `fastmri.data`: Contains data utility functions from original `data` folder that can be used to create sampling masks and submission files.
* `fastmri.models`: Contains baseline models, including the U-Net and the End-to-end Variational Network.

`experimental`: Folders intended to aid reproducibility of baselines.

## Testing

Run `python -m pytest tests`.

## Training a model

The [data README](https://github.com/facebookresearch/fastMRI/tree/master/fastmri/data/README.md) has a bare-bones example for how to load data and incorporate data transforms. This [jupyter notebook](https://github.com/facebookresearch/fastMRI/blob/master/fastMRI_tutorial.ipynb) contains a simple tutorial explaining how to get started working with the data.

Please look at [this U-Net demo script](https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/train_unet_demo.py) for an example of how to train a model using the PyTorch Lightning framework included with the package.

## Submitting to the Leaderboard

Run your model on the provided test data and create a zip file containing your predictions. `fastmri` has a `save_reconstructions` function that saves the data in the correct format.

Upload the zip file to any publicly accessible cloud storage (e.g. Amazon S3, Dropbox etc). Submit a link to the zip file on the [challenge website](https://fastmri.org/submit). You will need to create an account before submitting.

## License

fastMRI is MIT licensed, as found in the [LICENSE file](LICENSE.md).

## List of Papers

The following lists titles of papers from the fastMRI project. A more complete list of papers with abstracts and links to code can be found [here](LIST_OF_PAPERS.md).

1. Zbontar, J., Knoll, F., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., ... & Zhang, Z. (2018). [fastMRI: An open dataset and benchmarks for accelerated MRI](https://arxiv.org/abs/1811.08839). *arXiv preprint arXiv:1811.08839*.
2. Zhang, Z., Romero, A., Muckley, M. J., Vincent, P., Yang, L., & Drozdzal, M. (2019). [Reducing uncertainty in undersampled MRI reconstruction with active acquisition](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.html). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2049-2058).
3. Defazio, A. (2019). [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry](https://arxiv.org/abs/1912.01101). *arXiv preprint, arXiv:1912.01101*.
4. Defazio, A., Murrell, T., & Recht, M. P. (2020). [MRI Banding Removal via Adversarial Training](https://arxiv.org/abs/2001.08699). *arXiv preprint arXiv:2001.08699*.
5. Knoll, F., Zbontar, J., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., ... & Zhang, Z. (2020). [fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning](https://doi.org/10.1148/ryai.2020190007). *Radiology: Artificial Intelligence*, 2(1), e190007.
6. Knoll, F., Murrell, T., Sriram, A., Yakubova, N., Zbontar, J., Rabbat, M., ... & Recht, M. P. (2020). [Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge](https://doi.org/10.1002/mrm.28338). *Magnetic Resonance in Medicine*.
7. Sriram, A., Zbontar, J., Murrell, T., Zitnick, C. L., Defazio, A., & Sodickson, D. K. (2020). [GrappaNet: Combining parallel imaging with deep learning for multi-coil MRI reconstruction](https://openaccess.thecvf.com/content_CVPR_2020/html/Sriram_GrappaNet_Combining_Parallel_Imaging_With_Deep_Learning_for_Multi-Coil_MRI_CVPR_2020_paper.html). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 14315-14322).
8. Recht, M. P., Zbontar, J., Sodickson, D. K., Knoll, F., Yakubova, N., Sriram, A., ... & Kline, M. (2020). [Using Deep Learning to Accelerate Knee MRI at 3T: Results of an Interchangeability Study](https://www.ajronline.org/doi/abs/10.2214/AJR.20.23313). *American Journal of Roentgenology*.
9. Pineda, L., Basu, S., Romero, A., Calandra, R., & Drozdzal, M. (2020). [Active MR k-space Sampling with Reinforcement Learning](https://arxiv.org/abs/2007.10469). In *International Conference on Medical Image Computing and Computer-Assisted Intervention*.
10. Sriram, A., Zbontar, J., Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., ... & Johnson, P. (2020). [End-to-End Variational Networks for Accelerated MRI Reconstruction](https://arxiv.org/abs/2004.06688). In *International Conference on Medical Image Computing and Computer-Assisted Intervention*.
