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
from typing import Sequence

import numpy as np
import torch


def _select_shards(spec: str | Sequence[int] | None, available: int) -> list[int]:
    """Resolve which shards to read. None means all of them.

    Accepts indices or a compact string: "0-9", "0,3,5", "0-9,20". Selecting a
    subset does not reduce memory, since shards are memory mapped and cost only
    the pages actually touched. What it gives is control over coverage: windows
    are drawn at random with replacement, so a budget equal to the corpus sees
    only about 63% of it. Training consecutive sessions on consecutive shards
    covers everything exactly once instead.
    """
    if spec is None:
        return list(range(available))

    if isinstance(spec, str):
        indices: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:
                low, high = part.split("-", 1)
                indices.extend(range(int(low), int(high) + 1))
            else:
                indices.append(int(part))
    else:
        indices = [int(i) for i in spec]

    chosen = sorted(set(indices))
    out_of_range = [i for i in chosen if not 0 <= i < available]
    if out_of_range:
        raise ValueError(
            f"shard(s) {out_of_range} requested but the split has {available} "
            f"(valid range 0-{available - 1})"
        )
    if not chosen:
        raise ValueError(f"shard selection {spec!r} matched nothing")
    return chosen


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

    # Otherwise search. Mount points nest unpredictably: a Kaggle dataset can
    # land under its slug, its title, or a versioned folder, and guessing the
    # depth has not worked. Look for the file instead of the directory.
    for found in sorted(root.rglob("meta.json")):
        if found.parent.name == split:
            return found.parent, ""
    for found in sorted(root.rglob(f"{split}_meta.json")):
        return found.parent, f"{split}_"

    listing = "\n".join(f"  {p}" for p in sorted(root.rglob("*"))[:40]) or "  (empty)"
    raise FileNotFoundError(
        f"no {split} split under {root}: expected {split}/meta.json or "
        f"{split}_meta.json anywhere beneath it.\nWhat is actually there:\n{listing}"
    )


class PackedShards:
    """Read-only view over a split written by corpus.pack."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        expect_packing: str = "contiguous",
        shards: str | Sequence[int] | None = None,
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
        entries = self.meta["shards"]
        self.selected = _select_shards(shards, len(entries))
        self.shards = [
            # meta.json records the name as written; a flattened upload carries
            # the split prefix, so it goes back on here.
            np.memmap(self.dir / f"{self._prefix}{entries[i]['file']}", dtype=np.uint16, mode="r")
            for i in self.selected
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
