"""Summarize the intentionally small WGAN-GP lambda ablation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def build_lambda_ablation(run_root: Path, output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    main = run_root / "main" / "wgan_gp" / "seed_42" / "metrics.csv"
    sources = [(10.0, main)] + [(float(path.parents[1].name.split("_")[-1]), path) for path in run_root.glob("lambda_ablation/lambda_*/seed_42/metrics.csv")]
    for value, path in sources:
        if not path.exists():
            continue
        metrics = pd.read_csv(path); metrics = metrics[metrics["gen_step"] <= 3000]
        if len(metrics):
            final = metrics.iloc[-1]
            rows.append({"lambda": value, "fid": final.get("fid"), "training_seconds": final.get("training_seconds"),
                         "gp_loss": final.get("gp_loss"), "grad_p50": final.get("grad_p50"), "grad_p95": final.get("grad_p95")})
    table = output_root / "lambda_ablation.csv"; pd.DataFrame(rows).sort_values("lambda").to_csv(table, index=False)
    frame = pd.DataFrame(rows).sort_values("lambda")
    figure = output_root / "lambda_ablation.png"
    plt.figure(figsize=(6, 4)); plt.plot(frame["lambda"], frame["fid"], marker="o"); plt.xscale("log"); plt.xlabel("lambda"); plt.ylabel("FID at 3,000 updates"); plt.tight_layout(); plt.savefig(figure, dpi=160); plt.close()
    return figure, table
