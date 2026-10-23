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

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

SHARD_TOKENS = 100_000_000  # 200 MB per shard at uint16


class _ShardWriter:
    """Append token ids to a rolling series of flat uint16 shards.

    Tokens go to disk as they arrive rather than accumulating in a list. A
    Python list of 100M ids costs roughly 3.6GB, because ids above 256 are not
    interned and each is a separate object behind a pointer, and flushing one
    means an O(n) slice and an element-by-element conversion to numpy. Writing
    through keeps memory at the size of one batch.
    """

    def __init__(self, out_dir: Path, shard_tokens: int = SHARD_TOKENS) -> None:
        self.dir = out_dir
        self.shard_tokens = shard_tokens
        self.shards: list[dict[str, Any]] = []
        self._handle = None
        self._count = 0
        self._total = 0

    @property
    def total(self) -> int:
        return self._total

    def _open(self) -> None:
        path = self.dir / f"shard_{len(self.shards):04d}.bin"
        self._handle = open(path, "wb")
        self._current = path
        self._count = 0

    def write(self, ids: np.ndarray) -> None:
        view = ids
        while len(view):
            if self._handle is None:  # first write, or the previous shard rolled
                self._open()
            room = self.shard_tokens - self._count
            chunk, view = view[:room], view[room:]
            self._handle.write(chunk.tobytes())
            self._count += len(chunk)
            self._total += len(chunk)
            if self._count >= self.shard_tokens:
                self._roll()

    def _roll(self) -> None:
        self._handle.close()
        self.shards.append({"file": self._current.name, "tokens": self._count})
        self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            if self._count:
                self._roll()
            else:  # opened but never written to
                self._handle.close()
                self._current.unlink(missing_ok=True)
                self._handle = None


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


HOLDOUT_BUCKETS = 10_000
HOLDOUT_SHARE = 50  # 50/10000 = 0.5%, so a 1B token train split yields ~5M for val


def is_holdout(text: str) -> bool:
    """Assign a document to validation by a hash of its content.

    Both splits stream the same shuffled mixture, so splitting by position
    would put the validation documents at the front of the training stream and
    the held-out loss would be training loss. Hashing the text decides the
    split per document, identically on every machine and every rerun, so the
    two splits cannot overlap however many times either is regenerated.
    """
    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HOLDOUT_BUCKETS < HOLDOUT_SHARE


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
    target_tokens: int | None = None,
    batch_documents: int = 1000,
    progress_every: int = 50_000_000,
) -> dict[str, dict[str, Any]]:
    """Tokenize the mixture into flat uint16 shards. Returns meta per split.

    Both splits are written in a single pass over the stream, each document
    routed by its content hash. Packing them separately would mean streaming
    the whole corpus twice, and worse: the validation share is a small
    fraction of documents, so filling a 10M token validation split on its own
    would require streaming several times the training budget to find enough
    held-out documents. One pass fills both, and validation ends up at roughly
    HOLDOUT_SHARE/HOLDOUT_BUCKETS of the training size.
    """
    vocab = len(tokenizer)
    if vocab >= 2**16:
        raise ValueError(f"vocab {vocab} does not fit uint16; widen the shard dtype")

    root = Path(out_dir)
    budget = target_tokens if target_tokens is not None else cfg.target_tokens
    eos = tokenizer.eos_token_id

    writers: dict[str, _ShardWriter] = {}
    for split in ("train", "val"):
        (root / split).mkdir(parents=True, exist_ok=True)
        writers[split] = _ShardWriter(root / split)

    def encode(documents: list[str]) -> np.ndarray:
        """Tokenize a batch and flatten it, one EOS after each document."""
        encoded = tokenizer(documents, add_special_tokens=False)["input_ids"]
        flat: list[np.ndarray] = []
        for ids in encoded:
            flat.append(np.asarray(ids, dtype=np.uint16))
            flat.append(np.asarray([eos], dtype=np.uint16))
        return np.concatenate(flat) if flat else np.empty(0, dtype=np.uint16)

    batches: dict[str, list[str]] = {"train": [], "val": []}
    next_report = progress_every

    for document in stream_mixture(cfg):
        split = "val" if is_holdout(document) else "train"
        batch = batches[split]
        batch.append(document)
        if len(batch) < batch_documents:
            continue
        writers[split].write(encode(batch))
        batches[split] = []

        if writers["train"].total >= next_report:
            print(
                f"  {writers['train'].total / 1e6:.0f}M train tokens, "
                f"{writers['val'].total / 1e6:.1f}M val",
                flush=True,
            )
            next_report += progress_every
        if writers["train"].total >= budget:
            break

    for split, batch in batches.items():
        if batch:
            writers[split].write(encode(batch))

    meta: dict[str, dict[str, Any]] = {}
    for split, writer in writers.items():
        writer.close()
        split_meta = {
            "split": split,
            "tokens": writer.total,
            "target_tokens": budget if split == "train" else None,
            "complete": writer.total >= budget if split == "train" else writer.total > 0,
            "vocab_size": vocab,
            "dtype": "uint16",
            "eos_token_id": eos,
            "tokenizer": tokenizer.name_or_path,
            "holdout_share": f"{HOLDOUT_SHARE}/{HOLDOUT_BUCKETS}",
            # How documents sit in the shard. "contiguous" means they are
            # concatenated with an EOS between them and a sampled window may
            # straddle a boundary. Correct for pretraining, wrong for
            # supervised examples that must not bleed into each other, so the
            # mode is recorded rather than assumed by whoever reads this later.
            "packing": "contiguous",
            "shards": writer.shards,
            "sources": [
                {"name": s.name, "hf_path": s.hf_path, "weight": s.weight} for s in cfg.sources
            ],
        }
        (root / split / "meta.json").write_text(
            json.dumps(split_meta, indent=2), encoding="utf-8"
        )
        meta[split] = split_meta
    return meta
