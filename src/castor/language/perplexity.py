"""Held-out loss and perplexity.

Phase 2's exit test is a perplexity drop on held-out domain text without
wrecking general ability. During pretraining this is the general-ability half:
the number the domain adaptation later has to avoid destroying.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .dataset import PackedShards


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    shards: PackedShards,
    batch_size: int,
    block_size: int,
    batches: int,
    device: str,
    seed: int = 0,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)  # fixed seed: the same windows every eval

    losses = []
    for _ in range(batches):
        x, y = shards.batch(batch_size, block_size, rng, device=device)
        if autocast_dtype is not None and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train(was_training)
    mean = float(np.mean(losses))
    return {"loss": mean, "perplexity": math.exp(min(mean, 20.0))}
