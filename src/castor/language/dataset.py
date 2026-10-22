"""Batching over the packed shards.

The shards are memory mapped, never loaded. A 12GB corpus does not fit in RAM
on the machines these arms are meant to train on, and does not need to.

Sampling is random crops rather than a sequential sweep. At one epoch over
6B tokens the difference is immaterial, and random crops make a resumed run
statistically identical to an uninterrupted one without tracking a cursor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class PackedShards:
    """Read-only view over a split written by corpus.pack."""

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.dir = Path(data_dir) / split
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing. Run scripts/prepare_pretrain_data.py first."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.shards = [
            np.memmap(self.dir / s["file"], dtype=np.uint16, mode="r")
            for s in self.meta["shards"]
        ]
        if not self.shards:
            raise ValueError(f"{self.dir} contains no shards")
        self.lengths = np.array([len(s) for s in self.shards])
        self.total_tokens = int(self.lengths.sum())

    def __len__(self) -> int:
        return self.total_tokens

    def batch(
        self,
        batch_size: int,
        block_size: int,
        rng: np.random.Generator,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch of (input, target) pairs, targets shifted by one."""
        # Pick shards in proportion to length so every token is equally likely.
        probs = self.lengths / self.lengths.sum()
        picks = rng.choice(len(self.shards), size=batch_size, p=probs)

        xs = np.empty((batch_size, block_size), dtype=np.int64)
        ys = np.empty((batch_size, block_size), dtype=np.int64)
        for row, shard_idx in enumerate(picks):
            shard = self.shards[shard_idx]
            high = len(shard) - block_size - 1
            if high <= 0:
                raise ValueError(f"shard {shard_idx} shorter than block_size {block_size}")
            start = int(rng.integers(0, high))
            window = np.asarray(shard[start : start + block_size + 1], dtype=np.int64)
            xs[row] = window[:-1]
            ys[row] = window[1:]

        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if device != "cpu":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y
