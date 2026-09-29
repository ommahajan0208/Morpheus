"""Deterministic CelebA manifests and DataLoaders backed by the image cache."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import RunConfig
from src.data.cache import build_or_open_uint8_memmap_cache


def _find_images(root: Path) -> list[Path]:
    candidates = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not candidates:
        raise FileNotFoundError(f"No CelebA images found below {root}")
    return candidates


def manifest_path(config: RunConfig) -> Path:
    return config.path("data/splits") / f"celeba_n{config.num_images}_seed{config.seed}.csv"


def build_manifest(config: RunConfig) -> pd.DataFrame:
    """Create/load a stable manifest specific to the configured image count."""
    output = manifest_path(config)
    if output.exists():
        manifest = pd.read_csv(output)
        if len(manifest) == config.num_images:
            return manifest
    images = _find_images(config.path(config.get("dataset.root")))
    if len(images) < config.num_images:
        raise ValueError(f"Requested {config.num_images} images but found only {len(images)}")
    choices = list(range(len(images)))
    random.Random(config.seed).shuffle(choices)
    selected = [images[index] for index in choices[:config.num_images]]
    partitions = np.array(["train"] * config.num_images, dtype=object)
    holdout_indices = list(range(config.num_images))
    random.Random(config.seed + 1).shuffle(holdout_indices)
    partitions[holdout_indices[:config.get("dataset.eval_images")]] = "eval"
    manifest = pd.DataFrame({"filename": [path.name for path in selected], "path": [str(path.resolve()) for path in selected],
                             "partition": partitions})
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return manifest


class CachedCelebADataset(Dataset[torch.Tensor]):
    def __init__(self, cache_path: Path, indices: np.ndarray, image_count: int, resolution: int):
        self.images = np.memmap(cache_path, mode="r", dtype=np.uint8, shape=(image_count, resolution, resolution, 3))
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Slice into a contiguous uint8 ndarray, then convert to float in one op.
        # Avoids a second np.array allocation (copy=True) that the original used.
        image = np.ascontiguousarray(self.images[self.indices[index]])
        return torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32).div_(127.5).sub_(1.0)


def build_dataloaders(config: RunConfig, workers: int | None = None) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    manifest = build_manifest(config)
    cache_path, _ = build_or_open_uint8_memmap_cache(config, manifest)
    worker_count = config.get("performance.num_workers") if workers is None else workers
    train_indices = np.flatnonzero(manifest["partition"].to_numpy() == "train")
    eval_indices = np.flatnonzero(manifest["partition"].to_numpy() == "eval")
    options: dict = {
        "batch_size": config.get("training.batch_size"),
        "num_workers": worker_count,
        "pin_memory": config.get("performance.pin_memory"),
        "persistent_workers": worker_count > 0,
    }
    if worker_count > 0:
        options["prefetch_factor"] = config.get("performance.prefetch_factor")
        # 'spawn' avoids inherited CUDA contexts that cause deadlocks with
        # persistent_workers on Windows (the default 'fork' is not available).
        options["multiprocessing_context"] = "spawn"
    train = DataLoader(
        CachedCelebADataset(cache_path, train_indices, len(manifest), config.resolution),
        shuffle=True, drop_last=True, **options,
    )
    evaluate = DataLoader(
        CachedCelebADataset(cache_path, eval_indices, len(manifest), config.resolution),
        shuffle=False, drop_last=False, **options,
    )
    return train, evaluate, manifest
