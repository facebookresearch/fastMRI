import torch
import torch.nn as nn


class RAKI(torch.nn.Module):
    def __init__(self, kx_1, ky_1, kx_2, ky_2, kx_3, ky_3, acceleration_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(30, 32, (kx_1, ky_1), dilation=(1, acceleration_rate)),
            nn.ReLU(),
            nn.Conv2d(32, 8, (kx_2, ky_2), dilation=(1, acceleration_rate)),
            nn.ReLU(),
            nn.Conv2d(8, acceleration_rate - 1, (kx_3, ky_3), dilation=(1, acceleration_rate))
        )

    def forward(self, x):
        return self.model(x)
