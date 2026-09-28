"""Batch-independent critic used by WGAN variants."""
from __future__ import annotations

import torch.nn as nn


class WGANCritic(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 8, 1, 4, 1, 0),
        )

    def forward(self, image):
        return self.net(image).flatten()
