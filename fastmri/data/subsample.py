"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Create the mask.

        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            mask = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
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
        skip_low_freqs: Optional[bool] = False,
        skip_around_low_freqs: Optional[bool] = False,
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
                for the purposes of determining where equispaced lines should be.
                Set this `True` to guarantee the same number of sampled lines for
                all masks with a given (acceleration, center_fraction) setting.
            skip_around_low_freqs: Whether to also skip the two k-space lines right
                next to the already sampled low-frequency region. Used to guarantee
                that equispaced sampling doesn't extend the low-frequency region.
                This is mostly useful for VarNet, since it guarantees the same number
                of low-frequency lines are used for the sensitivity map calculation
                for all masks with a given (acceleration, center_fraction) setting.
                This argument has no effect when `skip_low_freqs` is `False`.
        """

        super().__init__(center_fractions, accelerations)
        self.skip_low_freqs = skip_low_freqs
        self.skip_around_low_freqs = skip_around_low_freqs

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # If everything has been sampled in the center: we don't need to sample anything else.
            if num_low_freqs * acceleration <= num_cols:
                if self.skip_low_freqs:
                    buffer = 0
                    if self.skip_around_low_freqs:
                        buffer = 2
                    # Compute the adjusted acceleration according to having
                    #  (num_low_freqs + buffer) center lines.
                    adjusted_accel = (
                        acceleration * (num_low_freqs + buffer - num_cols)
                    ) / (num_low_freqs * acceleration - num_cols)
                    offset = self.rng.randint(0, round(adjusted_accel) - 1)

                    # Select samples from the remaining columns
                    accel_samples = np.arange(
                        offset, num_cols - num_low_freqs - buffer - 1, adjusted_accel
                    )
                    accel_samples = np.around(accel_samples).astype(np.uint)

                    skip = (
                        num_low_freqs + buffer
                    )  # Skip low freq AND optionally lines right next to it
                    for sample in accel_samples:
                        if sample < pad - buffer // 2:
                            mask[sample] = True
                        else:  # sample is further than center, so skip low_freqs
                            mask[int(sample + skip)] = True
                else:  # Default behaviour
                    # determine acceleration rate by adjusting for the number of low frequencies
                    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                        num_low_freqs * acceleration - num_cols
                    )
                    offset = self.rng.randint(0, round(adjusted_accel))

                    accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
                    accel_samples = np.around(accel_samples).astype(np.uint)
                    mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    skip_low_freqs: Optional[bool] = False,
    skip_around_low_freqs: Optional[bool] = False,
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
        skip_low_freqs: Only used for EquispacedMaskFunc.
        skip_around_low_freqs: Only used for EquispacedMaskFunc.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquispacedMaskFunc(
            center_fractions, accelerations, skip_low_freqs, skip_around_low_freqs
        )
    else:
        raise Exception(f"{mask_type_str} not supported")
