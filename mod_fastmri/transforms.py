"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import fastmri


def _central_idx_finder(height, 
                        width, 
                        region_fraction=0.5) -> Tuple[int, int, int, int]:
    # Compute central region indices
    center_h, center_w = height // 2, width // 2
    region_h = int(height * region_fraction // 2)
    region_w = int(width * region_fraction // 2)

    h_start, h_end = center_h - region_h, center_h + region_h
    w_start, w_end = center_w - region_w, center_w + region_w
    return h_start, h_end, w_start, w_end


def batched_central_weight_mask(shape, 
                                region_fraction=0.5, 
                                central_weight=2.0, 
                                outer_weight=1.0) -> torch.Tensor:
    """
    Create a batched central weight mask for the spatial dimensions of a tensor, emphasizing the central region.

    Parameters:
    - shape: The shape of the tensor (e.g., torch.Size([40, 640, 368, 2])).
    - region_fraction: Fraction of the spatial dimensions to consider as the central region.
    - central_weight: Weight assigned to the central region.
    - outer_weight: Weight assigned to the outer region.

    Returns:
    - weight_mask: A tensor of shape [40, 640, 368] with weights applied to the central region.
    """
    # Extract spatial dimensions
    if shape.ndim < 4:
        return create_central_weight_mask(shape, region_fraction, central_weight, outer_weight)
    
    _, slices, height, width, _ = shape

    # Initialize the weight mask with the outer weight
    weight_mask = torch.full((slices, height, width), outer_weight)

    # Compute central region indices
    h_start, h_end, w_start, w_end = _central_idx_finder(height, width, region_fraction)

    # Apply central weight to the central region of the batches
    weight_mask[:, :, h_start:h_end, w_start:w_end] = central_weight

    return weight_mask

def create_central_weight_mask(shape, 
                               region_fraction=0.5, 
                               central_weight=2.0, 
                               outer_weight=1.0)-> torch.Tensor:
    """
    Create a weight mask for the spatial dimensions of a tensor, emphasizing the central region.

    Parameters:
    - shape: The shape of the tensor (e.g., torch.Size([40, 640, 368, 2])).
    - region_fraction: Fraction of the spatial dimensions to consider as the central region.
    - central_weight: Weight assigned to the central region.
    - outer_weight: Weight assigned to the outer region.

    Returns:
    - weight_mask: A tensor of shape [40, 640, 368] with weights applied to the central region.
    """
    # Extract spatial dimensions
    slices, height, width, _ = shape

    # Initialize the weight mask with the outer weight
    weight_mask = torch.full((slices, height, width), outer_weight)

    # Compute central region indices
    h_start, h_end, w_start, w_end = _central_idx_finder(height, width, region_fraction)

    # Apply central weight to the central region
    weight_mask[:, h_start:h_end, w_start:w_end] = central_weight

    return weight_mask


