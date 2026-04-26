from __future__ import annotations

from pathlib import Path

import yaml

from lora_gpt2.config import load_config, resolve_path


def test_nested_rank_sweep_config_resolves_paths_from_project_root(tmp_path) -> None:
    project_root = tmp_path / "project"
    (project_root / "src" / "lora_gpt2").mkdir(parents=True)
    (project_root / "scripts").mkdir()
    config_path = project_root / "configs" / "rank_sweep" / "e2e_lora_r1.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({"data": {"processed_dir": "data/processed/e2e_gpt2"}}))

    config = load_config(config_path)

    assert Path(config["_project_root"]) == project_root
    assert resolve_path(config, config["data"]["processed_dir"]) == project_root / "data" / "processed" / "e2e_gpt2"
