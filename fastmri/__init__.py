"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fastmri")
except PackageNotFoundError:
    # package is not installed
    import warnings

    warnings.warn("Could not retrieve fastmri version!")


from .coil_combine import rss, rss_complex
from .fftc import fft2c_new as fft2c
from .fftc import fftshift
from .fftc import ifft2c_new as ifft2c
from .fftc import ifftshift, roll
from .losses import SSIMLoss
from .math import (
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)
from .utils import convert_fnames_to_v2, save_reconstructions
