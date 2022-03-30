# fastMRI banding removal

This folder contains minimal code to train an orientation adversary model, as described in the paper [MRI Banding Removal via Adversarial Training](https://arxiv.org/abs/2001.08699).

# Usage

The banding removal models were trained on a system with 8 V100 GPUs and 512GB ram over 2-4 days. We don't officially support training on single GPU systems, it's not practical for models this large. It would be possible to train with 4 GPUs and less RAM at the expense of a longer training time.

Training consists of a pretraining phase, using the pretrain.py script, followed by a training stage from the train.py script. The training stage is slower and requires significantly more GPU memory. Both scripts need to be modified to point to the directory containing the fastMRI knee dataset before running, and the train script requires the path to the model file output by the train.py script.

The recommended way of running training jobs is using ipython, for example you can run `ipython scripts/train_dev.py` from within the `banding_removal` directory.

## Memory Issues
There is a memory leak with versions of HDF5 newer than 1.12.1 or h5py newer than 3.4.0 (https://github.com/facebookresearch/fastMRI/issues/215), this can be worked around by using `'short_epochs': True` to train on only 20% of the dataset each epoch, however using a older version of h5py is an easier fix.


# Monitoring progress

The folder `logs/run/grids` will contain PNG images of a fixed set of test images at regular epoch intervals. Since none of the loss metrics indicate the quality of the banding removal, the only real way to monitor progress is by visual inspection. 

# Limitations

The banding removal model is unstable to train. Retraining with different seeds can give quite different results, including failed runs and heavily artifacted images.
Even with careful training, artifacts can still occur. Our best trained models still exhibit artifacts on slices with little or no anatomy, such as the first or last slice.

This sort of training instability is common with GAN/Adversarial training, and we expect that the use of additional regularization or GAN specific optimization techniques could help. 

# Adversarial architectures

The `model/classifiers` folder contains a few models that we tried for the adversary. GPU memory becomes an issue when using larger architectures on 16GB gpus. The space of potential architectures was not heavily explored, and there may be potential for even better results.
