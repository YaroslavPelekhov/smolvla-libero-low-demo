from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repository_root() / config_path
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected a mapping in {config_path}")
    return cfg


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root() / value
