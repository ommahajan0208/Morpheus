"""Loading, validating, and snapshotting the single run configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json

import yaml


REQUIRED_SECTIONS = {"dataset", "models", "training", "evaluation", "performance", "experiments", "output"}


@dataclass(frozen=True)
class RunConfig:
    data: dict[str, Any]
    root: Path

    def get(self, dotted_key: str) -> Any:
        value: Any = self.data
        for part in dotted_key.split("."):
            value = value[part]
        return value

    @property
    def num_images(self) -> int:
        return int(self.get("dataset.num_images"))

    @property
    def seed(self) -> int:
        return int(self.get("dataset.seed"))

    @property
    def resolution(self) -> int:
        return int(self.get("dataset.resolution"))

    def path(self, relative: str | Path) -> Path:
        return (self.root / relative).resolve()

    def run_root(self) -> Path:
        return self.path(self.get("output.runs_root")) / f"n{self.num_images}"


def _require(data: dict[str, Any], dotted_key: str, kind: type) -> None:
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Missing required config key: {dotted_key}")
        value = value[part]
    if kind is int and isinstance(value, bool):
        raise ValueError(f"Config key {dotted_key} must be an integer")
    if not isinstance(value, kind):
        raise ValueError(f"Config key {dotted_key} must be a {kind.__name__}")


def load_run_config(path: str | Path = "config/run.yaml", root: str | Path | None = None) -> RunConfig:
    config_path = Path(path).resolve()
    project_root = Path(root).resolve() if root else config_path.parent.parent.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or set(data) != REQUIRED_SECTIONS:
        raise ValueError(f"Config must contain exactly these sections: {sorted(REQUIRED_SECTIONS)}")
    for key, kind in (("dataset.num_images", int), ("dataset.eval_images", int), ("dataset.seed", int),
                      ("dataset.resolution", int), ("training.batch_size", int)):
        _require(data, key, kind)
    if data["dataset"]["num_images"] < 501:
        raise ValueError("dataset.num_images must be at least 501")
    if data["dataset"]["num_images"] <= data["dataset"]["eval_images"]:
        raise ValueError("dataset.num_images must be greater than dataset.eval_images")
    if data["dataset"]["resolution"] != 64:
        raise ValueError("This project intentionally supports only 64x64 images")
    return RunConfig(copy.deepcopy(data), project_root)


def write_config_snapshot(config: RunConfig, destination: str | Path, extra: dict[str, Any] | None = None) -> Path:
    snapshot = copy.deepcopy(config.data)
    if extra:
        snapshot["resolved"] = extra
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    return destination


def config_summary(config: RunConfig) -> str:
    return json.dumps({"num_images": config.num_images, "eval_images": config.get("dataset.eval_images"),
                       "resolution": config.resolution, "seed": config.seed,
                       "profile": config.get("performance.profile")}, indent=2)
