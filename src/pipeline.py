"""Public, notebook-callable end-to-end pipeline stages."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import copy
import json

from src.config import RunConfig, config_summary
from src.data.celeba import build_dataloaders, build_manifest
from src.performance import benchmark_dataloader, configure_cuda
from src.training.trainer import GANTrainer
from src.analysis.plot_main_comparison import plot_main_comparison
from src.analysis.plot_gradient_stability import plot_gradient_stability
from src.analysis.rebuild_all import rebuild_all


def preflight(config: RunConfig) -> dict:
    device = configure_cuda(config.get("performance.tf32"))
    return {"device": str(device), "cuda_available": device.type == "cuda", "config": json.loads(config_summary(config))}


def prepare_data(config: RunConfig) -> dict:
    """Build manifest, benchmark workers, and return DataLoaders - without the
    extra build_dataloaders call the original had after the benchmark."""
    start = perf_counter()
    manifest = build_manifest(config)
    candidates = config.get("performance.worker_candidates")

    # Cache the last-built loader pair inside a dict so we can reuse the winner
    # instead of calling build_dataloaders a second time.
    _cache: dict = {}

    def loader_factory(workers: int):
        train_loader, eval_loader, _ = build_dataloaders(config, workers)
        _cache["train"] = train_loader
        _cache["eval"] = eval_loader
        _cache["workers"] = workers
        return train_loader

    selected = benchmark_dataloader(loader_factory, candidates)

    if _cache.get("workers") == selected:
        train_loader = _cache["train"]
        eval_loader = _cache["eval"]
    else:
        train_loader, eval_loader, manifest = build_dataloaders(config, selected)

    return {"train_loader": train_loader, "eval_loader": eval_loader, "manifest": manifest, "workers": selected,
            "prepare_seconds": perf_counter() - start}


def run_smoke_tests(config: RunConfig, prepared: dict, steps: int = 2) -> list[dict]:
    # compile_models=False: skip the torch.compile warm-up for quick sanity checks.
    device = configure_cuda(config.get("performance.tf32")); results = []
    for model in config.get("experiments.main_models"):
        directory = config.run_root() / "smoke" / model / f"seed_{config.seed}"
        results.append(
            GANTrainer(config, model, directory, device, compile_models=False)
            .train(prepared["train_loader"], None, steps, resume=False)
        )
    return results


def run_main_comparison(config: RunConfig, prepared: dict, resume: bool = True) -> list[dict]:
    """Train all main models.  Pass resume=True (default) to auto-resume from
    a crashed run; pass resume=False to force a fresh start."""
    device = configure_cuda(config.get("performance.tf32")); results = []
    timings: dict[str, float] = {"data_prepare_seconds": prepared["prepare_seconds"]}
    for model in config.get("experiments.main_models"):
        directory = config.run_root() / "main" / model / f"seed_{config.seed}"
        result = GANTrainer(config, model, directory, device).train(
            prepared["train_loader"], prepared["eval_loader"], resume=resume)
        results.append(result); timings[f"{model}_training_seconds"] = float(result["training_seconds"])
    timing_path = config.run_root() / "pipeline_timings.json"; timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return results


def _with_lambda(config: RunConfig, value: float) -> RunConfig:
    data = copy.deepcopy(config.data); data["training"]["wgan_gp"]["lambda_gp"] = float(value)
    return RunConfig(data, config.root)


def run_lambda_ablation(config: RunConfig, prepared: dict, resume: bool = True) -> list[dict]:
    """Run the WGAN-GP lambda ablation.  Pass resume=True (default) to
    auto-resume from a crashed run; pass resume=False to force a fresh start."""
    device = configure_cuda(config.get("performance.tf32")); results = []
    for value in config.get("experiments.lambda_values"):
        if float(value) == float(config.get("training.wgan_gp.lambda_gp")):
            continue
        active = _with_lambda(config, value)
        directory = config.run_root() / "lambda_ablation" / f"lambda_{value:g}" / f"seed_{config.seed}"
        results.append(
            GANTrainer(active, "wgan_gp", directory, device)
            .train(prepared["train_loader"], prepared["eval_loader"],
                   active.get("training.ablation_steps"), resume=resume)
        )
    return results


def build_report(config: RunConfig) -> list[Path]:
    figure_root = config.path(config.get("output.figures_root")) / f"n{config.num_images}"
    figures = plot_main_comparison(config.run_root(), figure_root)
    figures.append(plot_gradient_stability(config.run_root(), figure_root / "critic_gradient_norms.png"))
    report = config.path("reports/celeba_wgan_gp_results.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# CelebA GAN Results\n\nThis report is generated from the active configuration. FID-500 is a comparative small-sample metric.\n", encoding="utf-8")
    return figures + [report]


def summarize_artifacts(config: RunConfig) -> dict:
    root = config.run_root()
    return {"active_run_root": str(root), "exists": root.exists(), "files": [str(path) for path in root.rglob("*") if path.is_file()]}
