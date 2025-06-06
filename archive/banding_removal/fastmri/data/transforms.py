"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
import numpy as np
import torch
import traceback
import pdb
from torch.nn import functional as F
from scipy.sparse.linalg import svds
from scipy.linalg import svd

# Renamed in later versions of pytorch
try:
    fft = torch.fft
    ifft = torch.ifft   
    rfft = torch.rfft
    irfft = torch.irfft
except AttributeError:
    # Forwards compatibility for new pytorch versions
    def fft(input, signal_ndim, normalized=True):
        return torch.view_as_real(torch.fft.fft2(
            torch.view_as_complex(input),
            norm="ortho" if normalized else "backward"))
            
    def ifft(input, signal_ndim, normalized=True):
        return torch.view_as_real(torch.fft.ifft2(
            torch.view_as_complex(input),
            norm="ortho" if normalized else "backward"))

    def rfft(input, signal_ndim, normalized=True, onesided=False):
        raise Exception("Real handling rfft")
        return torch.fft.rfft2(input, signal_ndim, 
            norm="ortho" if normalized else "backward", 
            onesided=onesided)
    def irfft(input, signal_ndim, normalized=True, onesided=False):
        raise Exception("Real handling irfft")
        return torch.fft.irfft2(input, signal_ndim, 
            norm="ortho" if normalized else "backward", 
            onesided=onesided)

def complex_to_chans(data):
    batch, chans, rows, cols, dims = data.shape
    result = data.permute(0, 1, 4, 2, 3).contiguous().view([batch, chans * dims, rows, cols])
    return result


def chans_to_complex(data):
    batch, chans, rows, cols = data.shape
    assert chans % 2 == 0
    result = data.view([batch, chans // 2, 2, rows, cols]).permute(0, 1, 3, 4, 2).contiguous()
    return result


def apply_complex_model(model, data, norm=False, mask=None):
    if norm:
        b, c, h, w, d = data.shape
        std = data.view(b, c * h * w, d).std(dim=1).view(b, 1, 1, 1, d)
        std = std.expand(b, c, 1, 1, d).contiguous()
    else:
        std = 1
    return chans_to_complex(model(complex_to_chans(data / std), mask)) * std


def kspace_dc(pred_kspace, ref_kspace, mask):
    return (1 - mask) * pred_kspace + mask * ref_kspace


def image_dc(pred_image, ref_kspace, mask):
    return T.ifft2(kspace_dc(T.fft2(pred_image), ref_kspace, mask))

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension. Noop if data is already a Pytorch tensor

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if isinstance(data, torch.Tensor):
        return data
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None, offset=None, padding=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask, num_low_frequencies = mask_func(shape, seed, offset)
    mask = mask.to(data.device)

    if padding is not None:
        mask[:, :, :padding[0]] = 0
        mask[:, :, padding[1]:] = 0 # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0 # The + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies

def apply_mask_tensor(data, mask):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask (torch.Tensor): Mask tensor produced by a masking function
    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    return data * mask + 0.0


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def rfft2(data):
    data = ifftshift(data, dim=(-2, -1))
    data = rfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-3, -2))
    return data


def irfft2(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = irfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-2, -1))
    return data

def fft2_np(data):
    """
    Numpy version of fft2
    """
    data = np.fft.ifftshift(data, axes=(-2, -1))
    data = np.fft.fft2(data, norm="ortho")
    data = np.fft.fftshift(data, axes=(-2, -1))
    return data


def ifft2_np(data):
    """
    Numpy version of ifft2
    """
    data = np.fft.ifftshift(data, axes=(-2, -1))
    data = np.fft.ifft2(data, norm="ortho")
    data = np.fft.fftshift(data, axes=(-2, -1))
    return data

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return ((data ** 2).sum(dim=-1) + 0.0).sqrt()

def complex_abs_sq(data):
    """
    Compute the squared absolute value of a complex tensor
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1)

def complex_conj(x):
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    if data.shape[dim] == 1:
        raise Exception("BUG: RSS called on a dimension of size 1")
    #return torch.sqrt((data ** 2).sum(dim))
    return torch.norm(data, dim=dim)

def root_sum_of_squares_complex(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    if data.shape[dim] == 1:
        raise Exception("BUG: RSS called on a dimension of size 1")
    # Pytorches norm is more numerically stable for float16 than squaring/sqrt ops
    #return data.norm(dim=dim).norm(dim=-1)
    return torch.sqrt(complex_abs_sq(data).sum(dim))
    # a = data.abs().max()
    # xda = data.div(a)
    # return torch.sqrt(xda.pow(2).sum(dim=(-1, dim))).mul(a)
    #return data.add(1e-4).norm(dim=dim).norm(dim=-1)
    # shape = list(data.shape)[:-1]
    # shape[dim] *= 2
    # data_viewed = data.unsqueeze(dim+1).transpose(dim+1, -1).view(shape)
    # a = data_viewed.abs(dim=dim).max(dim=dim, keep_dim=True)
    # xda = data_viewed.div(a)
    # return xda.norm(dim=dim, keep_dim=True).div(a).squeeze(dim=dim)
    #return torch.sqrt(complex_abs_sq(data.float()).sum(dim)).type_as(data)
    # def hook(g):
    #     if torch.any(torch.isnan(g)):
    #         pdb.set_trace()
    #         print(g.shape)
        #     rss.register_hook(hook)

    #rss = data.float().norm(dim=dim).norm(dim=-1)
    #rss_half = rss.type_as(data)

    #if rss.requires_grad:
    #    print(f"in min: {data.min().item()} max: {data.max().item()} var:{data.var().item()}")
    #    print(f"out min: {rss_half.min().item()} max: {rss_half.max().item()} var:{rss_half.var().item()}")

    #return rss_half
    # xda = data.div(a)

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def center_crop_to_smallest(x, y):
    """
    Apply a center crop on the larger image to the size of the smaller image.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))
    return x, y


def center_crop_or_pad(data, shape):
    """
    Apply a center crop on pad (inferred from shape) on the data.
    """
    new_shape = list(data.shape)
    new_shape[-2:] = shape

    def crop_or_pad_indices(old_length, new_length):
        # Returns indices of size new_length or the max/min size of the data they're
        # accessing for both the resulting and data array.

        result_from = max((new_length - old_length) // 2, 0)
        result_to = min((new_length + old_length) // 2, new_length)
        data_from = max((old_length - new_length) // 2, 0)
        data_to = min((old_length + new_length) // 2, old_length)
        return result_from, result_to, data_from, data_to

    h_indices = crop_or_pad_indices(data.shape[-2], shape[-2])
    w_indices = crop_or_pad_indices(data.shape[-1], shape[-1])

    result = torch.zeros(new_shape).to(data.device)
    result[..., h_indices[0]:h_indices[1], w_indices[0]:w_indices[1]] = data[..., h_indices[2]:h_indices[3], w_indices[2]:w_indices[3]]
    return result


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

def complex_scalar_to_tensor(omega):
    """
        Converts pythons imaginary type to a pytorch tensor
    """
    return torch.tensor([omega.real, omega.imag])

def complex_conj(data):
    return torch.stack([data[..., 0], -data[..., 1]], dim=-1)


def complex_mult(a, b):
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    ], dim=-1)

def complex_mult_real(a, b):
    """ b matrix is real component only """
    return torch.stack([
        a[..., 0] * b,
        a[..., 1] * b
    ], dim=-1)

def complex_conj_mult_real(a, b):
    """ a * conj(b) but returns real channel only """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def complex_conj_mult(a, b):
    """ a * conj(b) """
    return torch.stack([
        a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1],
        - a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    ], dim=-1)

def complex_div(a,b):
    """" a / b """
    return complex_conj_mult(a,b)/(complex_abs_sq(b)[..., None])

def complex_pack(x):
    """
        Expects shape: b x c x height x width x 2
        Returns shape: b x 2c x height x width
    """
    b, c, h, w, im = x.shape
    assert im == 2
    return x.permute(0, 1, 4, 2, 3).contiguous().view(b, c * im, h, w)

def complex_unpack(x):
    """
        Expects shape: b x 2c x height x width
        Returns shape: b x c x height x width x 2
    """
    b, c, h, w = x.shape
    assert c % 2 == 0
    newc = c//2
    return x.view(b, newc, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()


def complex_packed_to_planar(data):
    """
        Expects shape: ... x height x width x 2
        Returns shape: ... x 2 x height x width
    """
    assert data.shape[-1] == 2
    real = data[..., 0]
    imaginary = data[..., 1]
    return torch.stack([real, imaginary], dim=-3)


def complex_planar_to_packed(data):
    """
        Expects shape: ... x 2 x height x width
        Returns shape: height x width x 2
    """
    assert data.shape[-3] == 2
    real = data[..., 0, :, :]
    imaginary = data[..., 1, :, :]
    return torch.stack([real, imaginary], dim=-1)

# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim=dim, start=0, length=x.size(dim) - shift)
    right = x.narrow(dim=dim, start=x.size(dim) - shift, length=shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def mask_center(x, num_lf):
    b, c, h, w, two = x.shape
    mask = torch.zeros_like(x)
    pad = (w - num_lf + 1) // 2
    mask[:, :, :, pad:pad + num_lf] = x[:, :, :, pad:pad + num_lf]
    return mask


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape] #TODO: looks wrong
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def topolar(x):
    dim = x.dim()-1
    modulus = torch.norm(x, dim=dim)
    argument = torch.atan2(x[..., 1], x[..., 0]) # Img, real.
    return torch.stack((modulus, argument), dim=dim)

def frompolar(x):
    dim = x.dim()-1
    real = x[...,0]*torch.cos(x[...,1])
    imag = x[...,0]*torch.sin(x[...,1])
    return torch.stack((real, imag), dim=dim)


def subsample(input_ksp, accel_factor):
    for n in range(1, accel_factor):
        input_ksp[:, :, :, n::accel_factor, :] = 0
    return input_ksp


def apply_grappa(input_ksp, kernel, ref_ksp, mask, sample_accel=None):
    batch = input_ksp.dim() == 5
    if not batch:
        input_ksp = input_ksp.unsqueeze(0)
        kernel = kernel.unsqueeze(0)

    kernel = kernel.to(input_ksp.device)
    if sample_accel is not None:
        input_ksp = subsample(input_ksp, sample_accel)

    input_ksp_ = complex_to_chans(input_ksp)
    pad = (kernel.shape[-2] // 2, kernel.shape[-2] // 2, kernel.shape[-1] // 2, kernel.shape[-1] // 2)
    input_ksp_ = F.pad(input_ksp_, pad, mode='reflect')
    # input_ksp_ = F.pad(input_ksp_, pad, mode='constant')
    result_ksp = [
        F.conv2d(input_ksp_[b].unsqueeze(0), kernel[b])
        for b in range(input_ksp_.shape[0])
    ]
    result_ksp = torch.cat(result_ksp)
    result_ksp = chans_to_complex(result_ksp)
    result_ksp = kspace_dc(result_ksp, ref_ksp, mask)

    if not batch:
        result_ksp = result_ksp.squeeze(0)
    return result_ksp


def coil_compress(kspace, out_coils):
    if kspace.shape[0] <= out_coils:
        return kspace

    kspace = kspace.numpy()
    kspace = kspace[..., 0] + 1j * kspace[..., 1]

    start_shape = tuple(kspace.shape)
    in_coils = start_shape[0]
    kspace = kspace.reshape(in_coils, -1)
    try:
        if in_coils == 5:
            u, _, _ = svd(kspace, full_matrices=False)
        else:
            u, _, _ = svds(kspace, k=out_coils)
    except Exception as e:
        print("SVD failed: ", kspace.shape)
        traceback.print_exc(file=sys.stdout)
        raise e

    u = np.transpose(np.conj(u[:, :out_coils]))
    new_shape = (out_coils, ) + start_shape[1:]
    new_kspace = u @ kspace
    kspace = np.reshape(new_kspace, new_shape)

    kspace = torch.stack((torch.Tensor(np.real(kspace)), torch.Tensor(np.imag(kspace))), dim=-1)
    return kspace
