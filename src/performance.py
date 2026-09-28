"""Small, measurable CUDA and DataLoader performance controls."""
from __future__ import annotations

from time import perf_counter
import torch


def configure_cuda(tf32: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high")
    return device


def benchmark_dataloader(loader_factory, candidates: list[int], batches: int = 100) -> int:
    """Return the worker count with the lowest measured host loading time."""
    scores: dict[int, float] = {}
    for workers in candidates:
        loader = loader_factory(workers)
        start = perf_counter()
        for _, _batch in zip(range(batches), loader):
            pass
        scores[workers] = perf_counter() - start
    return min(scores, key=scores.get)
