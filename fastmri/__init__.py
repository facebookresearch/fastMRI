"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from packaging import version

from .coil_combine import rss, rss_complex
from .fftc import fftshift, ifftshift, roll
from .losses import SSIMLoss
from .math import (
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)
from .utils import convert_fnames_to_v2, save_reconstructions

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    from .fftc import fft2c_new as fft2c
    from .fftc import ifft2c_new as ifft2c
else:
    from .fftc import fft2c_old as fft2c
    from .fftc import ifft2c_old as ifft2c
