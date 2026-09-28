"""Objectives shared by the three GAN variants."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dcgan_losses(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    discriminator = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    generator = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    return discriminator, generator


def wgan_losses(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return fake_scores.mean() - real_scores.mean(), -fake_scores.mean()


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor, lambda_gp: float) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon = torch.rand(real.size(0), 1, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = (epsilon * real + (1 - epsilon) * fake.detach()).requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(scores, interpolated, grad_outputs=torch.ones_like(scores), create_graph=True, only_inputs=True)[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return lambda_gp * ((norms - 1) ** 2).mean(), norms.detach()


def critic_input_gradient_stats(critic, real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    with torch.enable_grad():
        epsilon = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        inputs = (epsilon * real + (1 - epsilon) * fake.detach()).requires_grad_(True)
        scores = critic(inputs)
        gradients = torch.autograd.grad(scores, inputs, grad_outputs=torch.ones_like(scores), create_graph=False)[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return {"grad_mean": norms.mean().item(), "grad_std": norms.std(unbiased=False).item(),
            "grad_p05": norms.quantile(0.05).item(), "grad_p50": norms.quantile(0.5).item(), "grad_p95": norms.quantile(0.95).item()}
