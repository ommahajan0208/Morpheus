from pathlib import Path
import yaml

from src.config import load_run_config


def test_config_image_count_changes_derived_paths(tmp_path: Path):
    base = Path("config/run.yaml")
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    first = load_run_config(path, tmp_path)
    data["dataset"]["num_images"] = 10000
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    second = load_run_config(path, tmp_path)
    assert first.run_root() != second.run_root()
    assert "n10000" in str(second.run_root())


def test_config_rejects_too_few_images(tmp_path: Path):
    data = yaml.safe_load(Path("config/run.yaml").read_text(encoding="utf-8")); data["dataset"]["num_images"] = 500
    path = tmp_path / "run.yaml"; path.write_text(yaml.safe_dump(data), encoding="utf-8")
    try: load_run_config(path, tmp_path)
    except ValueError as error: assert "at least 501" in str(error)
    else: raise AssertionError("Invalid configuration was accepted")
