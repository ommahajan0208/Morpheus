"""Rebuild all report artifacts from one configuration-specific run root."""
from __future__ import annotations

from src.config import RunConfig
from src.analysis.build_main_table import build_main_table
from src.analysis.plot_gradient_stability import plot_gradient_stability
from src.analysis.plot_lambda_ablation import build_lambda_ablation
from src.analysis.plot_main_comparison import plot_main_comparison


def rebuild_all(config: RunConfig):
    figures = config.path(config.get("output.figures_root")) / f"n{config.num_images}"
    tables = config.path(config.get("output.tables_root")) / f"n{config.num_images}"
    paths = plot_main_comparison(config.run_root(), figures)
    paths.append(plot_gradient_stability(config.run_root(), figures / "critic_gradient_norms.png"))
    paths.extend(build_lambda_ablation(config.run_root(), tables))
    paths.append(build_main_table(config.run_root(), tables / "main_comparison.csv"))
    return paths
