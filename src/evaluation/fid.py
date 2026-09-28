"""In-memory FID evaluation; real Inception features stay in the metric instance."""
from __future__ import annotations

import torch


class CachedFIDEvaluator:
    def __init__(self, real_loader, device: torch.device, batch_size: int):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError as error:
            raise RuntimeError("Install torchmetrics[image] and torch-fidelity before computing FID") from error
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
        self.device, self.batch_size = device, batch_size
        with torch.inference_mode():
            for batch in real_loader:
                images = ((batch.to(device, non_blocking=True) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                self.metric.update(images, real=True)

    @torch.inference_mode()
    def compute(self, generator, latent_dim: int, count: int) -> float:
        was_training = generator.training
        generator.eval()
        remaining = count
        while remaining:
            size = min(self.batch_size, remaining)
            noise = torch.randn(size, latent_dim, 1, 1, device=self.device)
            images = ((generator(noise) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.metric.update(images, real=False)
            remaining -= size
        value = float(self.metric.compute().item())
        self.metric.reset()
        if was_training:
            generator.train()
        return value
