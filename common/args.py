"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--resolution', default=384, type=int, help='Resolution of images (brain \
                challenge expects resolution of 384, knee resolution expcts resolution of 320')

        # Data parameters
        self.add_argument('--challenge', choices=['singlecoil', 'multicoil'], help='Which challenge')
        self.add_argument('--data-path', type=pathlib.Path, help='Path to the dataset')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')

        # Mask parameters
        self.add_argument('--mask-type', choices=['random', 'equispaced'], default='random',
                          help='The type of mask function to use')
        self.add_argument('--accelerations', nargs='+', default=[4], type=int,
                          help='Ratio of k-space columns to be sampled. If multiple values are '
                               'provided, then one of those is chosen uniformly at random for '
                               'each volume.')
        self.add_argument('--center-fractions', nargs='+', default=[0.08], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)
