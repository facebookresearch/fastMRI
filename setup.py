"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import re

from setuptools import find_packages, setup

# from https://github.com/facebookresearch/ClassyVision/blob/master/setup.py
# get version string from module
with open(os.path.join(os.path.dirname(__file__), "fastmri/__init__.py"), "r") as f:
    readval = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if readval is None:
        raise RuntimeError("Version not found.")
    version = readval.group(1)
    print("-- Building version " + version)

with open("README.md", encoding="utf8") as f:
    readme = f.read()

install_requires = [
    "numpy>=1.18.5",
    "scikit_image>=0.16.2",
    "torchvision>=0.8.1",
    "torch>=1.8.0",
    "runstats>=1.8.0",
    "pytorch_lightning>=1.4",
    "h5py>=2.10.0",
    "PyYAML>=5.3.1",
    "torchmetrics>=0.5.1",
    "pandas>=1.3.4",
]

setup(
    name="fastmri",
    author="Facebook/NYU fastMRI Team",
    author_email="fastmri@fb.com",
    version=version,
    license="MIT",
    description="A large-scale dataset of both raw MRI measurements and clinical MRI images.",
    long_description_content_type="text/markdown",
    long_description=readme,
    project_urls={
        "Homepage": "https://fastmri.org/",
        "Source": "https://github.com/facebookresearch/fastMRI",
    },
    python_requires=">=3.6",
    packages=find_packages(
        exclude=[
            "tests",
            "fastmri_examples*",
            "banding_removal",
        ]
    ),
    setup_requires=["wheel"],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
