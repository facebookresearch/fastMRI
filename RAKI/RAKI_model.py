import torch
import torch.nn as nn


class RAKI(torch.nn.Module):
    def __init__(self, acceleration_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(30, 32, (5, 2), dilation=(1, acceleration_rate)),
            nn.ReLU(),
            nn.Conv2d(32, 8, (1, 1), dilation=(1, acceleration_rate)),
            nn.ReLU(),
            nn.Conv2d(8, acceleration_rate - 1, (3, 2), dilation=(1, acceleration_rate))
        )

    def forward(self, x):
        return self.model(x)
