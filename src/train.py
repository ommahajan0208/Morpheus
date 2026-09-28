"""Optional command-line entry point; the notebook calls the same pipeline functions."""
from __future__ import annotations

import argparse

from src.config import load_run_config
from src.pipeline import prepare_data, run_lambda_ablation, run_main_comparison, run_smoke_tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/run.yaml")
    parser.add_argument("--mode", choices=["smoke", "main", "ablation"], default="smoke")
    arguments = parser.parse_args()
    config = load_run_config(arguments.config)
    prepared = prepare_data(config)
    if arguments.mode == "smoke": print(run_smoke_tests(config, prepared))
    elif arguments.mode == "main": print(run_main_comparison(config, prepared))
    else: print(run_lambda_ablation(config, prepared))


if __name__ == "__main__": main()
