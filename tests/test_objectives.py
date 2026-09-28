import torch

from src.models.critics import WGANCritic
from src.training.objectives import dcgan_losses, gradient_penalty, wgan_losses


def test_objectives_are_finite():
    real, fake = torch.randn(2), torch.randn(2)
    assert all(torch.isfinite(value) for value in (*dcgan_losses(real, fake), *wgan_losses(real, fake)))
    critic = WGANCritic(base_channels=2)
    penalty, norms = gradient_penalty(critic, torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64), 10)
    assert torch.isfinite(penalty) and norms.shape == (2,)
