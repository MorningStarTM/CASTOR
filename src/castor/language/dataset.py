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


def _locate_split(root: Path, split: str) -> tuple[Path, str]:
    """Find a split's files, nested or flattened. Returns (directory, prefix).

    corpus.pack writes `<root>/train/meta.json` and `<root>/train/shard_0000.bin`.
    Some upload paths, Kaggle datasets among them, flatten a directory tree into
    one directory and fold the separators into the names, so the same split
    arrives as `<root>/train_meta.json` and `<root>/train_shard_0000.bin`. The
    contents are identical and re-uploading a 2GB corpus to restore two
    directories would be a waste, so both layouts are read.
    """
    if (root / split / "meta.json").exists():
        return root / split, ""
    if (root / f"{split}_meta.json").exists():
        return root, f"{split}_"

    # A dataset may also sit one level down, under its own folder.
    for candidate in sorted(p for p in root.glob("*") if p.is_dir()):
        if (candidate / split / "meta.json").exists():
            return candidate / split, ""
        if (candidate / f"{split}_meta.json").exists():
            return candidate, f"{split}_"

    raise FileNotFoundError(
        f"no {split} split under {root}: expected {split}/meta.json or "
        f"{split}_meta.json. Run scripts/prepare_pretrain_data.py first."
    )


class PackedShards:
    """Read-only view over a split written by corpus.pack."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        expect_packing: str = "contiguous",
    ) -> None:
        self.dir, self._prefix = _locate_split(Path(data_dir), split)
        self.meta = json.loads((self.dir / f"{self._prefix}meta.json").read_text(encoding="utf-8"))

        # Corpora written before this field existed are contiguous.
        self.packing = self.meta.get("packing", "contiguous")
        if expect_packing and self.packing != expect_packing:
            raise ValueError(
                f"{self.dir} is packed {self.packing!r} but the caller expects "
                f"{expect_packing!r}. A contiguous corpus lets a sampled window "
                f"cross a document boundary, which is right for pretraining and "
                f"wrong for supervised examples."
            )
        self.shards = [
            # meta.json records the name as written; a flattened upload carries
            # the split prefix, so it goes back on here.
            np.memmap(self.dir / f"{self._prefix}{s['file']}", dtype=np.uint16, mode="r")
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
