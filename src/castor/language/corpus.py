"""Download, tokenize and pack the pretraining corpus into flat shards.

Streams from the Hub rather than downloading parquet, because the mixture
discards most of what it reads and 28GB of SlimPajama is not worth storing to
keep its ArXiv slice.

Output is the dumbest format that works: a directory of `shard_XXXX.bin`,
each a flat uint16 array of token ids with an EOS between documents, plus a
`meta.json`. uint16 is sound only while the vocabulary is under 65536, which
`pack` asserts rather than assumes.

Token budget is the contract. The writer stops at target_tokens and records
how many it actually wrote, so a short download is visible rather than
silently producing a smaller corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

SHARD_TOKENS = 100_000_000  # 200 MB per shard at uint16


@dataclass
class Source:
    """One dataset in the mixture."""

    name: str
    hf_path: str
    weight: float
    hf_name: str | None = None
    split: str = "train"
    text_column: str = "text"
    # Optional metadata filter, e.g. {"field": "meta.redpajama_set_name",
    # "keep": ["RedPajamaArXiv", "RedPajamaStackExchange"]}
    filter: dict[str, Any] | None = None

    def matches(self, row: dict[str, Any]) -> bool:
        if not self.filter:
            return True
        value: Any = row
        for part in self.filter["field"].split("."):
            if not isinstance(value, dict) or part not in value:
                return False
            value = value[part]
        return value in self.filter["keep"]


@dataclass
class MixConfig:
    sources: list[Source] = field(default_factory=list)
    target_tokens: int = 6_000_000_000
    holdout_tokens: int = 10_000_000
    seed: int = 1337

    @staticmethod
    def from_yaml(path: str | Path) -> "MixConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sources = [Source(**s) for s in raw.pop("sources", [])]
        return MixConfig(sources=sources, **raw)


def stream_mixture(cfg: MixConfig) -> Iterator[str]:
    """Yield documents from the weighted mixture of sources, indefinitely."""
    from datasets import load_dataset

    streams = []
    for source in cfg.sources:
        ds = load_dataset(
            source.hf_path, name=source.hf_name, split=source.split, streaming=True
        ).shuffle(seed=cfg.seed, buffer_size=10_000)
        streams.append((source, iter(ds)))

    weights = np.array([s.weight for s, _ in streams], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(cfg.seed)

    exhausted: set[int] = set()
    while len(exhausted) < len(streams):
        idx = int(rng.choice(len(streams), p=weights))
        if idx in exhausted:
            continue
        source, it = streams[idx]
        try:
            row = next(it)
        except StopIteration:
            exhausted.add(idx)
            continue
        if not source.matches(row):
            continue
        text = row.get(source.text_column)
        if text:
            yield text


def pack(
    cfg: MixConfig,
    tokenizer,
    out_dir: str | Path,
    split: str = "train",
    target_tokens: int | None = None,
) -> dict[str, Any]:
    """Tokenize the mixture into flat uint16 shards. Returns the written meta."""
    vocab = len(tokenizer)
    if vocab >= 2**16:
        raise ValueError(f"vocab {vocab} does not fit uint16; widen the shard dtype")

    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = target_tokens if target_tokens is not None else cfg.target_tokens
    eos = tokenizer.eos_token_id

    buffer: list[int] = []
    written = 0
    shard_index = 0
    shards: list[dict[str, Any]] = []

    def flush(final: bool = False) -> None:
        nonlocal buffer, shard_index
        while len(buffer) >= SHARD_TOKENS or (final and buffer):
            take = buffer[:SHARD_TOKENS]
            buffer = buffer[SHARD_TOKENS:]
            path = out_dir / f"shard_{shard_index:04d}.bin"
            np.asarray(take, dtype=np.uint16).tofile(path)
            shards.append({"file": path.name, "tokens": len(take)})
            shard_index += 1

    batch: list[str] = []
    for document in stream_mixture(cfg):
        batch.append(document)
        if len(batch) < 1000:
            continue
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            buffer.extend(ids)
            buffer.append(eos)
        batch = []
        flush()
        written = sum(s["tokens"] for s in shards) + len(buffer)
        if written >= budget:
            break

    if batch and written < budget:
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            buffer.extend(ids)
            buffer.append(eos)
    flush(final=True)

    total = sum(s["tokens"] for s in shards)
    meta = {
        "split": split,
        "tokens": total,
        "target_tokens": budget,
        "complete": total >= budget,
        "vocab_size": vocab,
        "dtype": "uint16",
        "eos_token_id": eos,
        "tokenizer": tokenizer.name_or_path,
        "shards": shards,
        "sources": [
            {"name": s.name, "hf_path": s.hf_path, "weight": s.weight} for s in cfg.sources
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
