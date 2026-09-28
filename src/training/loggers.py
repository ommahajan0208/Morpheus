"""Buffered CSV metrics and phase timings."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any


class RunLogger:
    def __init__(self, run_dir: Path, flush_every: int = 100):
        self.run_dir, self.flush_every = run_dir, flush_every
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.csv"
        self.rows: list[dict[str, Any]] = []
        self.fields: list[str] = []

    def log(self, row: dict[str, Any]) -> None:
        self.fields = list(dict.fromkeys(self.fields + list(row)))
        self.rows.append(row)
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        write_header = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(self.rows)
        self.rows.clear()

    def close(self) -> None:
        self.flush()


class PhaseTimer:
    def __init__(self):
        self.timings: dict[str, float] = {}

    def measure(self, name: str):
        timer = self
        class Scope:
            def __enter__(self):
                self.start = perf_counter()
            def __exit__(self, *_):
                timer.timings[name] = perf_counter() - self.start
        return Scope()

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.timings, indent=2), encoding="utf-8")
