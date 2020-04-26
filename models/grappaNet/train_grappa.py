"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from torch.nn import functional as F
from torch.optim import RMSprop
import torch.utils.data as data

from common.args import Args
from common.subsample import create_mask_for_mask_type
from data import transforms
from models.mri_model import MRIModel
from models.grappaNet.grappa_unet_model import UnetModel

import h5py
from matplotlib import pyplot as plt
import glob


# READ DATA
multicoil_train = glob.glob('data_knee/multicoil_train/*.h5')
train_data = []
for f in multicoil_train:
    hf = h5py.File(multicoil_train)
    volume_kspace = hf['kspace'][()]
    train_data.append(volume_kspace)

multicoil_val = glob.glob('data_knee/multicoil_val/*.h5')
val_data = []
for f in multicoil_val:
    hf = h5py.File(multicoil_val)
    volume_kspace = hf['kspace'][()]
    val_data.append(volume_kspace)

multicoil_test = glob.glob('data_knee/multicoil_test_v2/*.h5')
train_test = []
for f in multicoil_test:
    hf = h5py.File(multicoil_test)
    volume_kspace = hf['kspace'][()]
    train_data.append(volume_kspace)


train_data = transforms.to_tensor(np.array(train_data))
val_data = transforms.to_tensor(np.array(val_data))
test_data = transforms.to_tensor(np.array(test_data))


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
  
# Load data
train_loader = data.DataLoader(train_data, batch_size = 16, shuffle = True, num_workers = 6)
val_loader = data.DataLoader(val_data, batch_size = 16, shuffle = True, num_workers = 6)
test_loader = data.DataLoader(test_data, batch_size=16, shuffle = True, num_workers = 6)
