"""Learning rate schedule: linear warmup, then cosine decay to a floor.

Warmup matters more here than it would at larger scale, because gpt.py applies
its layer norm after the residual add. Post-norm blocks are less forgiving of
a large early learning rate than the pre-norm arrangement most current models
use, so the warmup is not decoration.
"""

from __future__ import annotations

import math


def lr_at(step: int, peak_lr: float, warmup_steps: int, max_steps: int, min_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return peak_lr * min_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_ratio + (1.0 - min_ratio) * cosine)
