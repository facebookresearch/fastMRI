import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, variant=None, ks=3, pad=1, dil=1, num_group=1):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
            variant (string): Variant of Convolutional Block ('dense', 'res', or None).
            ks (int): Kernel size.
            pad (int): Padding.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans1 = out_chans
        self.out_chans2 = out_chans
        self.drop_prob = drop_prob
        self.variant = variant

        if variant == 'dense':
            self.out_chans1 = out_chans//2
            self.out_chans2 = out_chans - out_chans//2
        elif variant == 'res':
            self.out_chans2 = out_chans - min(in_chans, out_chans//2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, self.out_chans1, kernel_size=ks, padding=pad, dilation=dil, groups=num_group),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.out_chans1, self.out_chans2, kernel_size=ks, padding=pad, dilation=dil, groups=num_group),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        x1 = self.layer1(input)
        x2 = self.layer2(x1)

        if self.variant == 'dense':
            return torch.cat((x2, x1), 1)
        elif self.variant == 'res':
            return torch.cat((x2, input[:, :min(self.in_chans, self.out_chans1//2)]), 1)
        else:
            return x2

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans1={self.out_chans1}, out_chans2={self.out_chans2} ' \
               f'drop_prob={self.drop_prob}, variant={self.variant})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, variant=None, kernel_size=3, padding=1, dilation=1, groups=1):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input, *args):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

        return self.conv2(output)


class Push(nn.Module):
    pass


class Pop(nn.Module):
    pass


def conv(in_channels, out_channels, transpose=False, kernel_size=3):
    if transpose:
        yield nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False)
    else:
        yield nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False)
    yield nn.InstanceNorm2d(out_channels)
    yield nn.LeakyReLU(0.2, True)


class UnetModel2(nn.Module):
    def __init__(self, in_chans, out_chans, chans):
        super().__init__()
        c = chans
        self.layers = nn.ModuleList([
            *conv(in_chans, 1 * c), *conv(1 * c, 1 * c), Push(), nn.AvgPool2d(2, 2),
            *conv(1 * c, 2 * c), *conv(2 * c, 2 * c), Push(), nn.AvgPool2d(2, 2),
            *conv(2 * c, 4 * c), *conv(4 * c, 4 * c), Push(), nn.AvgPool2d(2, 2),
            *conv(4 * c, 8 * c), *conv(8 * c, 8 * c), Push(), nn.AvgPool2d(2, 2),
            *conv(8 * c, 16 * c), *conv(16 * c, 16 * c), *conv(16 * c, 8 * c, transpose=True), Pop(),
            *conv(16 * c, 8 * c), *conv(8 * c, 8 * c), *conv(8 * c, 4 * c, transpose=True), Pop(),
            *conv(8 * c, 4 * c), *conv(4 * c, 4 * c), *conv(4 * c, 2 * c, transpose=True), Pop(),
            *conv(4 * c, 2 * c), *conv(2 * c, 2 * c), *conv(2 * c, 1 * c, transpose=True), Pop(),
            *conv(2 * c, 1 * c), *conv(1 * c, 1 * c), nn.Conv2d(1 * c, out_chans, 1),
        ])

    def forward(self, input, *args):
        self.stack = []
        x = input
        for lyr in self.layers:
            if isinstance(lyr, Push):
                self.stack.append(x)
            elif isinstance(lyr, Pop):
                x = torch.cat([x, self.stack.pop()], dim=1)
            else:
                x = lyr(x)
        return x


def unet(args):
    return UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob)
