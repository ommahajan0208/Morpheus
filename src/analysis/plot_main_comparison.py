"""Create core training figures from main experiment logs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_main_comparison(run_root: Path, output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    for column, filename, ylabel in (("fid", "fid_vs_updates.png", "FID"), ("training_seconds", "training_time.png", "Training seconds")):
        plt.figure(figsize=(7, 4))
        for model_dir in sorted(run_root.glob("main/*/seed_42")):
            metrics = pd.read_csv(model_dir / "metrics.csv")
            if column in metrics and metrics[column].notna().any():
                plt.plot(metrics["gen_step"], metrics[column], label=model_dir.parent.name)
        plt.xlabel("Generator updates")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = output_root / filename
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(path)
    return figures
