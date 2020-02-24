import sys
sys.path.append(sys.path[0] + "/../../..")
import numpy as np
import cmath
import pytest
import torch
import pdb

from fastmri.data.transforms import *
from fastmri.common.subsample import *

@pytest.mark.parametrize("num_low_frequencies, accelerations, batch_size, dim", [
    ([round(0.2 * 320)], [4], 4, 320),
    ([round(0.2 * 368), round(0.4 * 368)], [4, 8], 2, 368),
])
def test_mask_reuse(num_low_frequencies, accelerations, batch_size, dim):
    mask_func = RandomMask(num_low_frequencies, accelerations)
    shape = (batch_size, dim, dim, 2)
    mask1, _ = mask_func(shape, seed=123)
    mask2, _ = mask_func(shape, seed=123)
    mask3, _ = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize("num_low_frequencies, accelerations, batch_size, dim", [
    ([round(0.2 * 320)], [4], 4, 320),
    ([round(0.2 * 368), round(0.4 * 368)], [4, 8], 2, 368),
])
def test_mask_low_freqs(num_low_frequencies, accelerations, batch_size, dim):
    mask_func = RandomMask(num_low_frequencies, accelerations)
    shape = (batch_size, dim, dim, 2)
    mask, _ = mask_func(shape, seed=123)
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = dim
    assert list(mask.shape) == mask_shape

    num_low_freqs_matched = False
    for num_low_freqs in num_low_frequencies:
####        num_low_freqs = round(dim * center_frac)
        pad = (dim - num_low_freqs + 1) // 2
        if np.all(mask[pad:(pad + num_low_freqs)].numpy() == 1):
            num_low_freqs_matched = True
    assert num_low_freqs_matched

@pytest.mark.parametrize("n", range(12, 20, 4))
def test_magic_mask(n):
    """
        It's hard to test the behavior for widths that are non-multiples of accel.
    """
    offset = None
    accel = 4
    mask_func = MagicMask([0], [4])
    mask = mask_func.accel_mask(n, accel, offset=0, num_low_frequencies=0)

    ### Apply mask in fft space to a random image then ifft back

    original = np.random.normal(size=n)
    #original = np.zeros(n)
    #original[3] = 1.0

    data = np.fft.ifftshift(original)
    data = np.fft.fft(data, norm="ortho")
    data = np.fft.fftshift(data)

    #pdb.set_trace()
    data = data * mask + 0.0

    data = np.fft.ifftshift(data)
    data = np.fft.ifft(data, norm="ortho")
    image = np.fft.fftshift(data)

    image = to_tensor(image)
    ### Check
    image_padded = torch.cat((image, image), dim=0)
    for r in range(accel):
        shift = (r*n)//accel
        remainder = (r*n) % accel
        delta = remainder/accel
        omega = cmath.exp(-2j*cmath.pi*r/accel)
        omega_tensor = transforms.complex_scalar_to_tensor(omega)
        shifted_raw = image_padded[shift:(n+shift)]
        # if remainder != 0:
        #     shifted_raw_next = image_padded[(shift-1):(n+shift-1)]
        #     shifted_out = (1-delta)*shifted_raw + delta * shifted_raw_next
        # pdb.set_trace()
        shifted_mult = transforms.complex_mult(omega_tensor, shifted_raw)
        if n % accel == 0:
            assert torch.allclose(image, shifted_mult, rtol=1e-03, atol=1e-06) 
        else:
            assert torch.allclose(image, shifted_mult, rtol=1e-01, atol=1e-02)
