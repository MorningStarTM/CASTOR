"""Builds a configured GPTLanguageModel out of gpt.py.

gpt.py keeps its hyperparameters in module-level globals (`n_emb`,
`block_size`, `n_layers`, `n_head`, `dropout`, `device`) which are read at
construction time and, for `device`, inside forward. That makes it impossible
to hold a 50M and a 100M arm in one process, which is precisely what the
ablation asks for.

`build_model` therefore sets those globals immediately before constructing the
model. This is a shim, it is not a good pattern, and it is not thread safe:
two arms built concurrently in one process would race. It is written this way
so gpt.py stays your file. The proper fix is for GPTLanguageModel to take a
GPTConfig, which would delete this function's body. See the note in
castor.language.__init__.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import gpt
from .config import GPTConfig


def build_model(cfg: GPTConfig, device: str = "cpu") -> nn.Module:
    """Construct the GPT with the given architecture, on the given device."""
    gpt.n_emb = cfg.n_emb
    gpt.n_layers = cfg.n_layer
    gpt.n_head = cfg.n_head
    gpt.block_size = cfg.block_size
    gpt.dropout = cfg.dropout
    gpt.device = device  # read inside forward for the position indices

    model = gpt.GPTLanguageModel(cfg.vocab_size)

    if cfg.tie_embeddings:
        # Saves vocab_size * n_emb parameters, which is a third of the 50M arm.
        model.lm_head.weight = model.token_embeding_table.weight

    _scale_residual_init(model, cfg.n_layer)
    return model.to(device)


def _scale_residual_init(model: nn.Module, n_layer: int) -> None:
    """GPT-2's 1/sqrt(2N) scaling on the layers that write to the residual.

    Without it the residual stream variance grows with depth and the early
    steps are needlessly fragile. gpt.py initialises every Linear at std=0.02;
    this narrows the two projections per block that feed the residual add.
    """
    std = 0.02 / math.sqrt(2 * n_layer)
    for block in model.blocks:
        nn.init.normal_(block.selfAttention.proj.weight, mean=0.0, std=std)
        nn.init.normal_(block.ffwd.net[2].weight, mean=0.0, std=std)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    embedding = (
        model.token_embeding_table.weight.numel()
        + model.position_embedding_table.weight.numel()
    )
    # Tied heads share storage, so lm_head is already counted once.
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}


def configure_optimizer(
    model: nn.Module, weight_decay: float, learning_rate: float, betas: tuple[float, float]
) -> torch.optim.Optimizer:
    """AdamW with decay on matmul weights only.

    Biases and LayerNorm gains are excluded: decaying them regularises the
    scale of the normalisation itself, which is not the intent.
    """
    decay, no_decay = [], []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # Decide from where the parameters actually live, not from cuda availability:
    # fused AdamW rejects CPU tensors, and a model can sit on CPU on a CUDA box.
    fused = bool(decay) and all(p.is_cuda for p in decay + no_decay)
    try:
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)
    except (TypeError, RuntimeError):  # torch too old for fused, or refused it
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)
