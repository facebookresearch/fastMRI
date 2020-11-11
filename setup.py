"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

install_requires = [
    "numpy>=1.18.5",
    "scikit_image>=0.16.2",
    "torchvision>=0.6.0",
    "torch>=1.6",
    "runstats>=1.8.0",
    "pytorch_lightning",
    "h5py",
    "PyYAML",
]

setup(
    name="fastmri",
    author="Facebook/NYU fastMRI Team",
    author_email="fastmri@fb.com",
    version="0.1",
    packages=find_packages(
        exclude=[
            "tests",
            "fastmri_examples",
            "data",
            "common",
            "banding_removal",
            "models",
        ]
    ),
    setup_requires=["wheel"],
    install_requires=install_requires,
)
