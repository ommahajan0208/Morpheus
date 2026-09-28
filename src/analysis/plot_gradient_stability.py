"""Plot recorded critic input-gradient diagnostics."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_gradient_stability(run_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for model_dir in sorted(run_root.glob("main/wgan*/seed_42")):
        metrics = pd.read_csv(model_dir / "metrics.csv").dropna(subset=["grad_p50"])
        if len(metrics):
            plt.plot(metrics["gen_step"], metrics["grad_p50"], label=f"{model_dir.parent.name} median")
            plt.fill_between(metrics["gen_step"], metrics["grad_p05"], metrics["grad_p95"], alpha=0.15)
    plt.xlabel("Generator updates"); plt.ylabel("Critic input-gradient norm"); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=160); plt.close()
    return output_path
