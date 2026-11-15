"""Configuration loading. All paths in config.yaml are resolved relative to the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """dict with attribute access and resolved paths."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    for k in ("cache", "artifacts"):
        cfg.path(k).mkdir(parents=True, exist_ok=True)
    return cfg
