"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

class Orientation(object):
    """
        50% of the time rotates the image 90 degrees.
    """
    def __init__(self, after, args):
        self.args = args
        self.after = after

    def __call__(self, kspace, target, attrs, fname, slice):
        alpha = torch.tensor(0.0).uniform_(0, 1).item()

        if alpha > 0.5:
            if isinstance(kspace, torch.Tensor):
                kspace = kspace.transpose(-2, -3) # Width/ Height, last dim is imag
                if target is not None:
                    target = target.transpose(-2, -1)
            else:
                kspace = kspace.transpose(0, 2, 1) # Permutes unlike transpose in pytorch
                if target is not None:
                    target = target.transpose(1, 0)
            attrs['rotated'] = True
        else:
            attrs['rotated'] = False
        return self.after(kspace, target, attrs, fname, slice)
