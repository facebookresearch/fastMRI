# fastMRI


Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements has the
potential to reduce medical costs, minimize stress to patients and make MR imaging
possible in applications where it is currently prohibitively slow or expensive.

[fastMRI](http://fastMRI.org) is collaborative research project from Facebook AI Research (FAIR)
and NYU Langone Health to investigate the use of AI to make MRI scans faster.
NYU Langone Health has released fully anonymized Knee MRI datasets that can
be downloaded from [the fastMRI dataset page](https://fastmri.med.nyu.edu/).


This repository contains convenient PyTorch data loaders, subsampling functions, evaluation
metrics, and reference implementations of simple baseline methods.


## Citing
If you use the fastMRI data or this code in your research, please consider citing
the fastMRI dataset paper:
```
@inproceedings{zbontar2018fastMRI,
  title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
  author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Matthew J. Muckley and Mary Bruno and Aaron Defazio and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and James Pinkerton and Duo Wang and Nafissa Yakubova and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1811.08839},
  year={2018}
}
```

## Dependencies
We have tested this code using:
* Ubuntu 18.04
* Python 3.6
* CUDA 9.0
* CUDNN 7.0

You can find the full list of Python packages needed to run the code in the
`requirements.txt` file. These can be installed using:
```bash
pip install -r requirements.txt
```

## Directory Structure & Usage
* `common`: Contains several utility functions and classes that can be used to
create subsampling masks, evaluate results and create submission files.
* `data`: Contains PyTorch data loaders for loading the fastMRI data and PyTorch
data transforms useful for working with MRI data. See `data/README.md` for more
information about using the data loaders.
* `models`: Contains the baseline models.

## Testing
Run `pytest`.

## Training a model
This [jupyter notebook](https://github.com/facebookresearch/fastMRI/blob/master/fastMRI_tutorial.ipynb) contains a simple tutorial explaining how to get started working with the data.

The following is explains how to work with the provided PyTorch data loaders and transforms and training your models. Please look at https://github.com/facebookresearch/fastMRI/master/models/unet/train_unet.py for a more concrete example.
```
from common import transforms, mri_data as data

# Define the data transform class
class DataTransform:
   def __call__(self, kspace, seed, target):
        # Preprocess the data here
        masked_kspace = transforms.apply_mask(kspace)
        image = transforms.ifft2(masked_kspace)
        cropped_image = transforms.center_crop(transforms.complex_abs(image))
        return cropped_image, target

# Create the dataset (either single-coil or multi-coil)
dataset = data.Slice(
    root='path_to_data', # Change based on your setup
    transform=DataTransform()
)
data_loader = DataLoader(dataset, batch_size)

# Create pytorch model and optimizer
my_model = build_pytorch_model()
my_optim = build_pytorch_optimizer()

# Train the model
for epoch in range(num_epochs):
    for masked_kspace, target in data_loader:
        reconstruction = my_model(masked_kspace)
        loss = mse_loss(reconstruction, target)
        my_optim.zero_grad()
        loss.backward()
        optimizer.step()
```

## Submitting to Leaderboard
Run your model on the provided test data and create a zip file containing your
predictions. The `common/utils.py` file has a `save_reconstructions` function
that saves the data in the correct format.

Upload the zip file to any publicly accessible cloud storage (e.g. Amazon S3,
Dropbox etc). Submit a link to the zip file on the [challenge website](http://fastmri.org/submit).
You will need to create an account before submitting.

## License
fastMRI is MIT licensed, as found in the LICENSE file.
