"""One-time 64x64 uint8 cache keyed by the immutable split manifest."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import RunConfig


def manifest_hash(manifest: pd.DataFrame) -> str:
    text = manifest[["filename", "path", "partition"]].to_csv(index=False, lineterminator="\n")
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_dir(config: RunConfig, manifest: pd.DataFrame) -> Path:
    token = manifest_hash(manifest)
    return config.path("data/cache") / f"celeba{config.resolution}_n{config.num_images}_seed{config.seed}_{token}"


def _crop_resize(image: Image.Image, resolution: int) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((resolution, resolution), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def build_or_open_uint8_memmap_cache(config: RunConfig, manifest: pd.DataFrame) -> tuple[Path, Path]:
    """Return `(images.dat, metadata.json)`, rebuilding only when metadata differs."""
    target = cache_dir(config, manifest)
    data_path, metadata_path = target / "images.dat", target / "metadata.json"
    expected = {"manifest_hash": manifest_hash(manifest), "image_count": len(manifest),
                "resolution": config.resolution, "transform_version": config.get("dataset.transform_version")}
    if data_path.exists() and metadata_path.exists():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual == expected:
            return data_path, metadata_path

    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "images.tmp.dat"
    pixels = np.memmap(temporary, mode="w+", dtype=np.uint8,
                       shape=(len(manifest), config.resolution, config.resolution, 3))
    for index, image_path in enumerate(manifest["path"]):
        with Image.open(image_path) as image:
            pixels[index] = _crop_resize(image, config.resolution)
    pixels.flush()
    del pixels
    temporary.replace(data_path)
    metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return data_path, metadata_path
