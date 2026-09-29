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


def maybe_compile(module: torch.nn.Module, enabled: bool = True) -> torch.nn.Module:
    """Wrap a module with torch.compile when CUDA is available and enabled.

    torch.compile fuses CUDA kernel graphs after a one-time warm-up (~20-30 s
    on the first batch) and then runs noticeably faster for every subsequent
    forward/backward pass.  We guard behind a flag so smoke tests can skip
    the warm-up cost.
    """
    if enabled and torch.cuda.is_available():
        try:
            return torch.compile(module, mode="reduce-overhead")
        except Exception:
            # torch.compile can fail on some Windows / driver combinations;
            # fall back to eager silently so training still works.
            pass
    return module


def benchmark_dataloader(loader_factory, candidates: list[int], batches: int = 30) -> int:
    """Return the worker count with the lowest measured host loading time.

    Reduced from 100 to 30 batches: 30 is enough to amortise the DataLoader
    start-up cost while avoiding the 3x overhead of the original value.
    """
    scores: dict[int, float] = {}
    for workers in candidates:
        loader = loader_factory(workers)
        start = perf_counter()
        for _, _batch in zip(range(batches), loader):
            pass
        scores[workers] = perf_counter() - start
    return min(scores, key=scores.get)
