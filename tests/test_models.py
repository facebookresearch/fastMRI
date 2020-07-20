import pytest

from fastmri.models import Unet

from .conftest import create_input


@pytest.mark.parametrize(
    "shape,out_chans,chans",
    [
        ([1, 1, 32, 16], 5, 1),
        ([5, 1, 15, 12], 10, 32),
        ([3, 2, 13, 18], 1, 16),
        ([1, 2, 17, 19], 3, 8),
    ],
)
def test_unet(shape, out_chans, chans):
    x = create_input(shape)

    num_chans = x.shape[1]

    unet = Unet(in_chans=num_chans, out_chans=out_chans, chans=chans, num_pool_layers=2)

    y = unet(x)

    assert y.shape[1] == out_chans
