"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch.nn import functional as F
import torchvision
import logging
import sys
import pdb
import numpy as np
from torch.nn import functional as F
from PIL import Image
from collections import defaultdict

from fastmri.data import transforms
from fastmri.trainer import Trainer
from fastmri.common import evaluate
from fastmri.common import image_grid
from fastmri.common import utils
from fastmri.data import transforms


class VarNetTrainer(Trainer):
    def predict(self, batch):
        batch = self.preprocess_data(batch)
        input_ksp = batch.input
        num_lf = batch.num_lf
        target = batch.target_im

        if input_ksp.dim() == 5:
            input_ksp = input_ksp.unsqueeze(1)
            target = target.unsqueeze(1)

        output = self.model(dict(
            kspace=input_ksp.transpose(1, 2),
            mask=batch.mask.transpose(1, 2).byte(),
            num_lf=num_lf,
            sens_maps=batch.sens_map))

        return output, target

    def unnorm(self, output, batch):
        batch = self.preprocess_data(batch)
        return output
