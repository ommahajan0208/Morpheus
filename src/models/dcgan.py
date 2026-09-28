"""Compact DCGAN generator and discriminator for 64x64 RGB faces."""
from __future__ import annotations

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, c * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(c * 8), nn.ReLU(True),
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 4), nn.ReLU(True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.ReLU(True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 8, 1, 4, 1, 0),
        )

    def forward(self, image):
        return self.net(image).flatten()
