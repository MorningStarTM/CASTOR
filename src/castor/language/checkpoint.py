"""Checkpoint writing and resumption, in safetensors.

A checkpoint is a directory, not a file:

    latest/
      model.safetensors   weights, plus a little metadata
      optimizer.pt        AdamW state
      state.json          step, tokens seen, the full config

Weights are safetensors because a checkpoint is the thing most likely to be
moved between machines, attached to a Kaggle notebook, or published beside a
paper, and torch.save is a pickle: loading one executes whatever it contains.
safetensors is also mmap-able, so loading touches only what it needs.

Optimizer state stays a pickle. safetensors holds a flat map of tensors, while
AdamW's state is nested dicts keyed by parameter index with non-tensor entries
for the param groups. Splitting the two is what the HuggingFace Trainer does
and for the same reason. It costs nothing: the optimizer file is only read when
resuming your own run, never when loading a published model.

Tied embeddings need care. With tie_embeddings the language-model head and the
token embedding are one tensor under two names, and safetensors refuses to
serialise aliased storage. The duplicate name is dropped on save and recorded
in the metadata; on load the model is already tied by build_model, so filling
the embedding fills the head too.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import PretrainConfig

MODEL_FILE = "model.safetensors"
OPTIM_FILE = "optimizer.pt"
STATE_FILE = "state.json"


def _storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except AttributeError:  # torch < 2.0
        return tensor.storage().data_ptr()


def _dedupe_shared(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop names that alias a tensor already kept.

    state_dict preserves module registration order, and gpt.py registers the
    token embedding before the head, so the embedding is the name that
    survives and the head is the alias that is dropped.
    """
    kept: dict[int, str] = {}
    out: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for name, tensor in state.items():
        ptr = _storage_ptr(tensor)
        if ptr in kept:
            dropped.append(name)
            continue
        kept[ptr] = name
        out[name] = tensor.detach().cpu().contiguous()
    return out, dropped


def save(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: PretrainConfig,
    step: int,
    tokens_seen: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint directory. Each file lands by rename, never truncated."""
    path = Path(path)
    if path.suffix:  # tolerate a caller still passing "latest.pt"
        path = path.with_suffix("")
    path.mkdir(parents=True, exist_ok=True)

    weights, dropped = _dedupe_shared(model.state_dict())
    metadata = {
        "format": "pt",
        "arm": str(cfg.arm),
        "step": str(step),
        "tokens_seen": str(tokens_seen),
        "tied_keys": json.dumps(dropped),
    }
    tmp = path / (MODEL_FILE + ".tmp")
    save_file(weights, str(tmp), metadata=metadata)
    tmp.replace(path / MODEL_FILE)

    if optimizer is not None:
        tmp = path / (OPTIM_FILE + ".tmp")
        torch.save(optimizer.state_dict(), tmp)
        tmp.replace(path / OPTIM_FILE)

    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "config": dataclasses.asdict(cfg),
        "tied_keys": dropped,
        **(extra or {}),
    }
    tmp = path / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(path / STATE_FILE)
    return path


def load(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> tuple[int, int]:
    """Restore weights and optimizer state. Returns (step, tokens_seen)."""
    path = Path(path)
    if path.is_file():
        return _load_legacy_pickle(path, model, optimizer, map_location)

    weights = load_file(str(path / MODEL_FILE), device="cpu")
    missing, unexpected = model.load_state_dict(weights, strict=False)

    state_path = path / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    tied = set(state.get("tied_keys", []))

    # The only names allowed to be absent are the ones tying dropped, and they
    # are already populated because build_model tied them to a tensor we just
    # filled. Anything else means the checkpoint does not match the model.
    unresolved = set(missing) - tied
    if unresolved:
        raise ValueError(f"checkpoint {path} is missing {sorted(unresolved)}")
    if unexpected:
        raise ValueError(f"checkpoint {path} has unexpected keys {sorted(unexpected)}")

    optim_path = path / OPTIM_FILE
    if optimizer is not None and optim_path.exists():
        optimizer.load_state_dict(torch.load(optim_path, map_location=map_location))

    return int(state.get("step", 0)), int(state.get("tokens_seen", 0))


def _load_legacy_pickle(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    map_location: str,
) -> tuple[int, int]:
    """Read a checkpoint written before the move to safetensors."""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0)), int(payload.get("tokens_seen", 0))


def load_weights_only(path: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Just the tensors, for a later phase that wants the arm but not its optimizer."""
    path = Path(path)
    return load_file(str(path / MODEL_FILE), device=device)


def read_metadata(path: str | Path) -> dict[str, str]:
    """Header metadata without reading any weights."""
    from safetensors import safe_open

    with safe_open(str(Path(path) / MODEL_FILE), framework="pt") as handle:
        return handle.metadata() or {}


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
