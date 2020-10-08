"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


def complex_mul(x, y):
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x (torch.Tensor): A PyTorch tensor with the last dimension of size 2.
        y (torch.Tensor): A PyTorch tensor with the last dimension of size 2.

    Returns:
        torch.Tensor: A PyTorch tensor with the last dimension of size 2.
    """
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x):
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x (torch.Tensor): A PyTorch tensor with the last dimension of size 2.
        y (torch.Tensor): A PyTorch tensor with the last dimension of size 2.

    Returns:
        torch.Tensor: A PyTorch tensor with the last dimension of size 2.
    """
    assert x.shape[-1] == 2

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def fft2c(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3
            dimensions: dimensions -3 & -2 are spatial dimensions and dimension
            -1 has size 2. All other dimensions are assumed to be batch
            dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))

    return data


def ifft2c(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3
            dimensions: dimensions -3 & -2 are spatial dimensions and dimension
            -1 has size 2. All other dimensions are assumed to be batch
            dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))

    return data


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the
            final dimension should be 2.

    Returns:
        torch.Tensor: Absolute value of data.
    """
    assert data.size(-1) == 2

    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_sq(data):
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the
            final dimension should be 2.

    Returns:
        torch.Tensor: Squared absolute value of data.
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1)


# Helper functions


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x (torch.Tensor): A PyTorch tensor.
        shift (int): Amount to roll.
        dim (int): Which dimension to roll.

    Returns:
        torch.Tensor: Rolled version of x.
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to fftshift.

    Returns:
        torch.Tensor: fftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]

    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to ifftshift.

    Returns:
        torch.Tensor: ifftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]

    return roll(x, shift, dim)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]
