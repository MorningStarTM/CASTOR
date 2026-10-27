"""Configuration objects for the language arms.

Sizes are chosen so the embedding table never dominates. At a 32k vocabulary
with tied embeddings:

    50M arm   d=512, L=8    25.2M non-embedding + 16.4M embedding =  42M
    100M arm  d=768, L=10   70.8M non-embedding + 24.6M embedding =  96M

That is the reason the vocabulary is 32k rather than 50k: at this scale a
50k vocabulary would put roughly half the 50M arm into the embedding table,
and the ablation would be comparing lookup tables rather than transformers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GPTConfig:
    """Architecture. One instance per ablation arm."""

    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_emb: int = 512
    dropout: float = 0.0          # 0.0 for pretraining; a single epoch cannot overfit 6B tokens
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_emb % self.n_head:
            raise ValueError(f"n_emb {self.n_emb} not divisible by n_head {self.n_head}")

    def parameter_estimate(self) -> dict[str, int]:
        """Closed-form parameter count, for sizing an arm before building it."""
        non_embedding = 12 * self.n_layer * self.n_emb**2
        position = self.block_size * self.n_emb
        token = self.vocab_size * self.n_emb
        total = non_embedding + position + token * (1 if self.tie_embeddings else 2)
        return {
            "non_embedding": non_embedding,
            "position_embedding": position,
            "token_embedding": token,
            "total": total,
        }


@dataclass
class OptimConfig:
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1     # cosine floor, as a fraction of peak
    # Warmup is a fraction of the run, not a fixed step count. Steps are
    # already derived from the token budget and the world size, so a literal
    # warmup_steps silently becomes a different fraction of training every
    # time either changes: 2000 steps is 10% of a 2.5B single-GPU run and 52%
    # of the same arm at 1B tokens on two GPUs. Set warmup_steps only to
    # override, and it wins where present.
    warmup_ratio: float = 0.05
    warmup_steps: int | None = None
    weight_decay: float = 0.1     # applied to matmul weights only, not biases or norms
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    def warmup_for(self, max_steps: int) -> int:
        if self.warmup_steps is not None:
            return self.warmup_steps
        return max(1, int(max_steps * self.warmup_ratio))


@dataclass
class PretrainConfig:
    """A full pretraining run: one arm, one corpus, one budget."""

    arm: str = "50m"
    model: GPTConfig = field(default_factory=GPTConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # Budget. target_tokens is the contract; max_steps is derived from it.
    target_tokens: int = 2_500_000_000
    batch_size: int = 16          # sequences per forward pass
    grad_accum_steps: int = 8     # effective batch = batch_size * grad_accum * block_size

    data_dir: str = "data/text_corpus/tokenized"
    # Which training shards to read. None means all. "0-9" or "0,3,5" selects
    # a subset, which is how consecutive sessions cover a corpus exactly once
    # instead of resampling the whole of it at random. Validation always reads
    # every shard, so the held-out number stays comparable between sessions.
    shards: str | None = None
    out_dir: str = "experiments/runs"
    seed: int = 1337

    eval_interval: int = 1000
    eval_batches: int = 100
    checkpoint_interval: int = 2000
    log_interval: int = 20

    compile_model: bool = True
    amp_dtype: str = "bfloat16"   # falls back to float16 where bf16 is unsupported

    # Off-machine checkpointing. A Kaggle session's disk disappears with the
    # session, so the only copy of a 10 hour run is whatever left the box.
    hf_repo: str | None = None    # e.g. "ernestbeckham/castor-50m"
    hf_private: bool = True       # publishing a half-trained arm is rarely wanted
    push_best: bool = True        # upload whenever held-out loss improves
    push_latest: bool = False     # upload resumable state too, optimizer included

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum_steps * self.model.block_size

    @property
    def max_steps(self) -> int:
        return max(1, self.target_tokens // self.tokens_per_step)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    """Build a PretrainConfig from YAML, honouring a `defaults:` chain."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    for parent in raw.pop("defaults", []):
        parent_raw = yaml.safe_load((path.parent / parent).read_text(encoding="utf-8")) or {}
        parent_raw.pop("defaults", None)
        raw = _merge(parent_raw, raw)

    model = GPTConfig(**raw.pop("model", {}))
    optim = OptimConfig(**raw.pop("optim", {}))
    known = {f.name for f in dataclasses.fields(PretrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return PretrainConfig(model=model, optim=optim, **raw)


# Fields that determine what the model becomes. Everything outside this set
# describes where things are read from, written to or reported, and two runs
# differing only in those are the same run.
_IDENTITY_FIELDS = (
    "arm",
    "model",
    "optim",
    "target_tokens",
    "batch_size",
    "grad_accum_steps",
    "amp_dtype",
    "seed",
)


def config_hash(cfg: PretrainConfig) -> str:
    """Stable short hash of the training-determining config.

    The hash names the run directory, so it has to survive the things that
    change between sessions without changing the run: the dataset mounting at
    a different path, an output directory, a Hub repo being added, a different
    logging interval. Hashing the whole config made adding --hf-repo produce a
    new directory and orphan the checkpoints being resumed from.
    """
    import hashlib
    import json

    identity = {name: getattr(cfg, name) for name in _IDENTITY_FIELDS}
    identity["model"] = dataclasses.asdict(cfg.model)
    identity["optim"] = dataclasses.asdict(cfg.optim)
    blob = json.dumps(identity, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
