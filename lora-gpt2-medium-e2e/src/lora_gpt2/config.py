"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _infer_project_root(config_path: Path) -> Path:
    """Find the repository-local project root for configs at any depth."""
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parent.parents):
        if (candidate / "src" / "lora_gpt2").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return resolved.parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(_infer_project_root(config_path))
    return config


def project_root(config: dict[str, Any]) -> Path:
    """Return the project root associated with a loaded config."""
    if "_project_root" in config:
        return Path(config["_project_root"])
    return Path.cwd()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root(config) / path


def nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Read a nested config value without raising on missing keys."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
