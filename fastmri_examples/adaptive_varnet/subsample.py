"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional, Sequence

import numpy as np

from fastmri.data.subsample import MaskFunc, RandomMaskFunc


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    skip_low_freqs: bool,
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
        skip_low_freqs: Whether to skip already sampled low-frequency lines
            for the purposes of determining where equispaced lines should be.
            Set this `True` to guarantee the same number of sampled lines for
            all masks with a given (acceleration, center_fraction) setting.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "adaptive_equispaced_fraction":
        return EquispacedMaskFractionFunc(
            center_fractions, accelerations, skip_low_freqs
        )

    else:
        raise ValueError(f"{mask_type_str} not supported")


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with strictly exact acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        skip_low_freqs: bool = False,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            skip_low_freqs: Whether to skip already sampled low-frequency lines
                for the purposes of determining where equispaced lines should
                be. Set this `True` to guarantee the same number of sampled
                lines for all masks with a given (acceleration,
                center_fraction) setting.
        """
        super().__init__(center_fractions, accelerations)
        self.skip_low_freqs = skip_low_freqs

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        mask = np.zeros(num_cols)
        pad = (num_cols - num_low_frequencies + 1) // 2

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        offset = self.rng.randint(0, round(adjusted_accel) - 1)

        # Select samples from the remaining columns
        accel_samples = np.arange(
            offset, num_cols - num_low_frequencies - 1, adjusted_accel
        )
        accel_samples = np.around(accel_samples).astype(int)

        skip = (
            num_low_frequencies  # Skip low freq AND optionally lines right next to it
        )
        for sample in accel_samples:
            if sample < pad:
                mask[sample] = True
            else:  # sample is further than center, so skip low_freqs
                mask[int(sample + skip)] = True

        return mask
