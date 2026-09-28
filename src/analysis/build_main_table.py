"""Build a compact model comparison table from final metric rows."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_main_table(run_root: Path, output_path: Path) -> Path:
    rows = []
    for metrics_path in run_root.glob("main/*/seed_42/metrics.csv"):
        metrics = pd.read_csv(metrics_path); final = metrics.iloc[-1]
        rows.append({"model": metrics_path.parent.parent.name, "final_fid": final.get("fid"),
                     "training_seconds": final.get("training_seconds"), "status": "stable" if pd.isna(final.get("fid_error")) else "FID unavailable"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
