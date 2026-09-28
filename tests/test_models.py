import torch

from src.models.critics import WGANCritic
from src.models.dcgan import Discriminator, Generator


def test_models_produce_64px_images_and_scalar_scores():
    generator = Generator(latent_dim=8, base_channels=4)
    images = generator(torch.randn(2, 8, 1, 1))
    assert images.shape == (2, 3, 64, 64)
    assert Discriminator(base_channels=4)(images).shape == (2,)
    assert WGANCritic(base_channels=4)(images).shape == (2,)
