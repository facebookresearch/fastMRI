"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import pdb
from fastmri.data import transforms
import random

def mask_factory(name, num_low_frequencies, accelerations):
    if name == "random":
        return RandomMask(num_low_frequencies, accelerations)
    if name == "random_fraction":
        return RandomMaskFraction(num_low_frequencies, accelerations)
    elif name == "equispaced":
        return EquiSpacedMask(num_low_frequencies, accelerations)
    elif name == "magic":
         return MagicMask(num_low_frequencies, accelerations)
    elif name == "magic_fraction":
         return MagicMaskFraction(num_low_frequencies, accelerations)
    elif name == "equispaced_v2":
        return EquiSpacedMaskV2(num_low_frequencies, accelerations)
    elif name == "equispaced_fraction":
        return EquiSpacedMaskFraction(num_low_frequencies, accelerations)
    elif name == "magic_v2":
         return MagicMaskV2(num_low_frequencies, accelerations)
    else:
        raise Exception(f"Unknown mask type: {name}")


class MaskFunc:
    def __init__(self, num_low_frequencies, accelerations):
        """
        Args:
            num_low_frequencies (List[float]): Number of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as num_low_frequencies. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        self.num_low_frequencies = num_low_frequencies
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed, offset=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        center_mask, accel_mask, num_low_freqs = self.sample_masks(shape, seed, offset)
        combined_mask = torch.max(center_mask, accel_mask)

        return combined_mask, num_low_freqs

    def reshape_mask(self, mask, shape):
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        return mask

    def center_mask(self, shape, num_low_freqs):
        """
            Produces the mask for the central Low-frequence region only.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs
        return mask

    def accel_mask(self, n, acceleration, offset, num_low_frequencies):
        """
            Produces the mask for the non-central lines. Will be maxed with the center mask,
            so center lines are allowed.
        """
        raise Exception("Implement in a subclass")

    def sample_masks(self, shape, seed, offset=None):
        n = shape[-2]
        num_low_freqs, acceleration = self.choose_acceleration(seed)
        center_mask = self.reshape_mask(self.center_mask(shape, num_low_freqs), shape)
        accel_mask = self.reshape_mask(self.accel_mask(n, acceleration, offset, num_low_freqs), shape)
        return center_mask, accel_mask, num_low_freqs

    def choose_acceleration(self, seed):
        self.rng.seed(seed)
        low_freq_choice = self.rng.randint(0, len(self.num_low_frequencies))
        num_low_freqs = self.num_low_frequencies[low_freq_choice]
        acceleration_choice = self.rng.randint(0, len(self.accelerations))
        acceleration = self.accelerations[acceleration_choice]
        return num_low_freqs, acceleration


class EquiSpacedMask(MaskFunc):
    """
    This mask selects a subset of columsn form the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. num_low_frequencies = columns in the center corresponding to low-frequencies
        2. The other columns are selected at evenly spaced intervals corresponding to the given acceleration factor.
    """
    def accel_mask(self, n, acceleration, offset, num_low_frequencies):
        if offset == None:
            offset = random.randrange(acceleration)

        mask = np.zeros(n, dtype=np.float32)
        mask[offset::acceleration] = 1
        return mask

class MagicMask(MaskFunc):
    """
    Like EquiSpacedMask, except the masking is applied before the fft-shift transform.
    A mask offset is used that ensures that the aliasing patterns are shifted in phase.
    """

    def accel_mask(self, n, acceleration, offset, num_low_frequencies):
        if offset == None:
            offset = random.randrange(acceleration)
        if offset % 2 == 0:
            offset_pos = 1
            offset_neg = 2
        else:
            offset_pos = 3
            offset_neg = 0

        #pdb.set_trace()

        poslen = (n+1)//2
        neglen = n - (n+1)//2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        mask_positive[offset_pos::acceleration] = 1
        mask_negative[offset_neg::acceleration] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        shifted_mask = np.fft.fftshift(mask) # Shift it
        return shifted_mask

class MagicMaskFraction(MaskFunc):
    def sample_masks(self, shape, seed, offset=None):
        n = shape[-2]
        fraction_low_freqs, acceleration = self.choose_acceleration(seed)
        num_cols = shape[-2]
        num_low_freqs = int(round(num_cols * fraction_low_freqs))
        center_mask = self.reshape_mask(self.center_mask(shape, num_low_freqs), shape)
        accel_mask = self.reshape_mask(self.accel_mask(n, acceleration, offset, num_low_freqs), shape)
        return center_mask, accel_mask, num_low_freqs

    def accel_mask(self, n, acceleration, offset, num_low_frequencies):
        if offset == None:
            offset = random.randrange(acceleration)

        offset = offset % acceleration
        if offset % 2 == 0:
            offset_pos = 1
            offset_neg = 2
        else:
            offset_pos = 3
            offset_neg = 0

        poslen = (n+1)//2
        neglen = n - (n+1)//2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        adjusted_accel = (acceleration * (num_low_frequencies - n)) / (num_low_frequencies * acceleration - n)
        pos_accel_samples = np.arange(offset_pos, poslen - 1, adjusted_accel)
        pos_accel_samples = np.around(pos_accel_samples).astype(np.uint)
        mask_positive[pos_accel_samples] = 1

        neg_accel_samples = np.arange(offset_neg, neglen - 1, adjusted_accel)
        neg_accel_samples = np.around(neg_accel_samples).astype(np.uint)
        mask_negative[neg_accel_samples] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        shifted_mask = np.fft.fftshift(mask) # Shift it
        return shifted_mask


class EquiSpacedMaskV2(MaskFunc):
    """
    Modified equispaced mask that takes offset modulo acceleration
    """
    def accel_mask(self, n, acceleration, offset, num_low_frequencies):
        if offset == None:
            offset = random.randrange(acceleration)

        offset = offset % acceleration
        mask = np.zeros(n, dtype=np.float32)
        mask[offset::acceleration] = 1
        return mask

class MagicMaskV2(MaskFunc):
    """
    Applies magic mask with the offset taken modulo acceleration
    """
    def accel_mask(self, n, acceleration, offset, num_low_frequencies):

        if offset == None:
            offset = random.randrange(acceleration)

        offset = offset % acceleration
        if offset % 2 == 0:
            offset_pos = 1
            offset_neg = 2
        else:
            offset_pos = 3
            offset_neg = 0

        poslen = (n+1)//2
        neglen = n - (n+1)//2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        mask_positive[offset_pos::acceleration] = 1
        mask_negative[offset_neg::acceleration] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        shifted_mask = np.fft.fftshift(mask) # Shift it

        #assert shifted_mask[offset] == 1
        # Clamp offset region on both sides
        #shifted_mask[:offset] = 0
        #shifted_mask[-offset:] = 0
        return shifted_mask


class RandomMask(MaskFunc):
    """
    This mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. num_low_frequencies = columns in the center corresponding to low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - num_low_frequencies) / (N - num_low_frequencies).
    This ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple num_low_frequencies and accelerations, in which case one possible
    (num_low_frequencies, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and num_low_frequencies = [28, 14], then there
    is a 50% probability that 4-fold acceleration with 28 num low frequencies is selected and a 50%
    probability that 8-fold acceleration with 14 low frequencies is selected.
    """
    def accel_mask(self, n, acceleration, offset, num_low_frequencies):
        # Create the mask
        prob = (n / acceleration - num_low_frequencies) / (n - num_low_frequencies)
        mask = self.rng.uniform(size=n) < prob
        return mask


class RandomMaskFraction(RandomMask):
    def __call__(self, shape, seed, offset=None):
       fraction_low_freqs, acceleration = self.choose_acceleration(seed)
       num_cols = shape[-2]
       num_low_freqs = int(round(num_cols * fraction_low_freqs))

       # Create the mask
       prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
       mask = self.rng.uniform(size=num_cols) < prob
       pad = (num_cols - num_low_freqs + 1) // 2
       mask[pad:pad + num_low_freqs] = True

       # Reshape the mask
       mask_shape = [1 for _ in shape]
       mask_shape[-2] = num_cols
       mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

       return mask, num_low_freqs


class EquiSpacedMaskFraction(MaskFunc):
    def __call__(self, shape, seed, offset=None):
       fraction_low_freqs, acceleration = self.choose_acceleration(seed)
       num_cols = shape[-2]
       num_low_freqs = int(round(num_cols * fraction_low_freqs))

       # Create the mask
       mask = np.zeros(num_cols, dtype=np.float32)
       pad = (num_cols - num_low_freqs + 1) // 2
       mask[pad:pad + num_low_freqs] = True

       # Determine acceleration rate by adjusting for the number of low frequencies
       adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
       if offset == None:
           offset = random.randrange(round(adjusted_accel))

       accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
       accel_samples = np.around(accel_samples).astype(np.uint)
       mask[accel_samples] = True

       # Reshape the mask
       mask_shape = [1 for _ in shape]
       mask_shape[-2] = num_cols
       mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

       return mask, num_low_freqs
