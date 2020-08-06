# fastMRI

Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements has the potential to reduce medical costs, minimize stress to patients and make MR imaging possible in applications where it is currently prohibitively slow or expensive.

[fastMRI](http://fastMRI.org) is a collaborative research project from Facebook AI Research (FAIR) and NYU Langone Health to investigate the use of AI to make MRI scans faster. NYU Langone Health has released fully anonymized knee and brain MRI datasets that can be downloaded from [the fastMRI dataset page](https://fastmri.med.nyu.edu/).

This repository contains convenient PyTorch data loaders, subsampling functions, evaluation metrics, and reference implementations of simple baseline methods.

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

For other publications from the fastMRI project please see our [list of papers](https://github.com/facebookresearch/fastMRI/blob/master/LIST_OF_PAPERS.md).

## Dependencies

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

Since August 2020, the repository has been refactored to operate as a package centered around the `fastmri` module, while configurations and scripts for reproducibility are now hosted in `experimental`.

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

Upload the zip file to any publicly accessible cloud storage (e.g. Amazon S3, Dropbox etc). Submit a link to the zip file on the [challenge website](http://fastmri.org/submit). You will need to create an account before submitting.

## License

fastMRI is MIT licensed, as found in the LICENSE file.
