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
    correct = 0
    counted = 0
    for _ in range(batches):
        x, y = shards.batch(batch_size, block_size, rng, device=device)
        if autocast_dtype is not None and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        losses.append(loss.item())

        # gpt.py flattens logits to (batch * block, vocab) when targets are
        # given, so the targets flatten the same way. This is top-1 next-token
        # accuracy: the share of positions where the most likely token was the
        # right one. It is not a measure of quality the way perplexity is, but
        # it moves early and reads more plainly than a log loss.
        predicted = logits.argmax(dim=-1)
        target = y.reshape(-1)
        correct += int((predicted == target).sum().item())
        counted += target.numel()

    model.train(was_training)
    mean = float(np.mean(losses))
    return {
        "loss": mean,
        "perplexity": math.exp(min(mean, 20.0)),
        "accuracy": correct / counted if counted else float("nan"),
    }
