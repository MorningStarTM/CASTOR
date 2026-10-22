"""Checkpoint writing and resumption.

Writes to a temporary file and renames, so an interrupted save cannot leave a
truncated checkpoint where a valid one used to be. The Kaggle route depends on
this: a 100M arm will be interrupted by the session limit several times.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from .config import PretrainConfig


def save(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: PretrainConfig,
    step: int,
    tokens_seen: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": dataclasses.asdict(cfg),
        "step": step,
        "tokens_seen": tokens_seen,
        **(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load(path: str | Path, model: torch.nn.Module, optimizer=None, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("step", 0), payload.get("tokens_seen", 0)


def write_manifest(out_dir: str | Path, cfg: PretrainConfig, extra: dict[str, Any]) -> Path:
    """The run record: config, hash, and whatever the caller knows about data."""
    from .config import config_hash

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": dataclasses.asdict(cfg),
        "config_hash": config_hash(cfg),
        **extra,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
