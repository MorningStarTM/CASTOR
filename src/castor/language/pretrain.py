"""The pretraining loop, as a class.

One Pretrainer owns one arm: its data, model, optimizer and run directory.
The budget is in tokens, not steps. Steps are derived from the token target,
so changing the batch size to fit a smaller card does not quietly change how
much data the arm sees. That is what makes the ablation's "same data, same
budget" claim true rather than aspirational.

    trainer = Pretrainer(cfg).setup()
    trainer.train()

Every stage is a method, so an arm can be driven piecewise when that is more
useful than train():

    trainer = Pretrainer(cfg, resume="experiments/runs/.../latest.pt").setup()
    trainer.evaluate()
    trainer.train_step(trainer.step)
    trainer.save("latest.pt")
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import checkpoint as ckpt
from .config import PretrainConfig, config_hash
from .dataset import PackedShards
from .model import build_model, configure_optimizer, count_parameters
from .perplexity import evaluate as evaluate_split
from .schedule import lr_at


class Pretrainer:
    """Pretrains one language arm from a PretrainConfig."""

    def __init__(
        self,
        cfg: PretrainConfig,
        resume: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.resume_from = Path(resume) if resume else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: torch.nn.Module | None = None
        self.raw_model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.train_shards: PackedShards | None = None
        self.val_shards: PackedShards | None = None
        self.parameters: dict[str, int] = {}

        self.step = 0
        self.tokens_seen = 0
        self.history: list[dict[str, Any]] = []

        self._rng = np.random.default_rng(cfg.seed)
        self._amp_dtype: torch.dtype | None = None
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._is_setup = False
        self._t0 = 0.0
        self._tokens_at_start = 0

    # ---------------------------------------------------------------- setup

    def setup(self) -> "Pretrainer":
        """Build data, model and optimizer. Idempotent."""
        if self._is_setup:
            return self

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._load_data()
        self._build_model()
        self._configure_amp()

        if self.resume_from:
            self.load(self.resume_from)

        self._maybe_compile()
        self._write_manifest()
        self._is_setup = True
        print(self.describe())
        return self

    def _load_data(self) -> None:
        self.train_shards = PackedShards(self.cfg.data_dir, "train")
        self.val_shards = PackedShards(self.cfg.data_dir, "val")
        corpus_vocab = self.train_shards.meta["vocab_size"]
        if corpus_vocab != self.cfg.model.vocab_size:
            raise ValueError(
                f"corpus was tokenized with vocab {corpus_vocab} but the model is "
                f"configured for {self.cfg.model.vocab_size}"
            )

    def _build_model(self) -> None:
        self.raw_model = build_model(self.cfg.model, self.device)
        self.model = self.raw_model
        self.parameters = count_parameters(self.raw_model)
        self.optimizer = configure_optimizer(
            self.raw_model,
            self.cfg.optim.weight_decay,
            self.cfg.optim.learning_rate,
            (self.cfg.optim.beta1, self.cfg.optim.beta2),
        )

    def _configure_amp(self) -> None:
        """Pick an autocast dtype the card actually supports.

        Turing and earlier have no bfloat16, so a bf16 request falls back to
        float16, which needs the gradient scaler that bf16 does not.
        """
        name = self.cfg.amp_dtype
        if not self.device.startswith("cuda") or name == "float32":
            self._amp_dtype = None
        elif name == "bfloat16" and torch.cuda.is_bf16_supported():
            self._amp_dtype = torch.bfloat16
        elif name in ("bfloat16", "float16"):
            self._amp_dtype = torch.float16
        else:
            self._amp_dtype = None
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._amp_dtype is torch.float16)

    def _maybe_compile(self) -> None:
        """Compile if asked and available. Never fatal: Triton is absent on Windows."""
        if not (self.cfg.compile_model and self.device.startswith("cuda")):
            return
        if not hasattr(torch, "compile"):
            return
        try:
            # raw_model stays uncompiled, so checkpoints carry plain keys rather
            # than torch.compile's _orig_mod. prefix.
            self.model = torch.compile(self.raw_model)
        except Exception as exc:
            print(f"torch.compile unavailable, continuing eager: {exc}")
            self.model = self.raw_model

    # -------------------------------------------------------------- reporting

    @property
    def run_dir(self) -> Path:
        return Path(self.cfg.out_dir) / f"pretrain_{self.cfg.arm}_{config_hash(self.cfg)}"

    def describe(self) -> str:
        p = self.parameters
        return (
            f"arm={self.cfg.arm} "
            f"params={p['total'] / 1e6:.1f}M (non-embedding {p['non_embedding'] / 1e6:.1f}M) "
            f"steps={self.cfg.max_steps} tokens/step={self.cfg.tokens_per_step:,} "
            f"corpus={self.train_shards.total_tokens:,} "
            f"device={self.device} amp={self._amp_dtype} "
            f"compiled={self.model is not self.raw_model}"
        )

    def _write_manifest(self) -> None:
        ckpt.write_manifest(
            self.run_dir,
            self.cfg,
            {
                "parameters": self.parameters,
                "device": self.device,
                "amp_dtype": str(self._amp_dtype),
                "max_steps": self.cfg.max_steps,
                "tokens_per_step": self.cfg.tokens_per_step,
                "corpus": {
                    k: self.train_shards.meta[k] for k in ("tokens", "tokenizer", "sources")
                },
            },
        )

    # ---------------------------------------------------------------- training

    def learning_rate(self, step: int) -> float:
        return lr_at(
            step,
            self.cfg.optim.learning_rate,
            self.cfg.optim.warmup_steps,
            self.cfg.max_steps,
            self.cfg.optim.min_lr_ratio,
        )

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                _, loss = self.model(x, y)
        else:
            _, loss = self.model(x, y)
        return loss

    def train_step(self, step: int) -> float:
        """One optimizer step, including gradient accumulation. Returns the loss."""
        lr = self.learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(self.cfg.grad_accum_steps):
            x, y = self.train_shards.batch(
                self.cfg.batch_size, self.cfg.model.block_size, self._rng, device=self.device
            )
            loss = self._forward(x, y) / self.cfg.grad_accum_steps
            self._scaler.scale(loss).backward()
            total += loss.item()

        if self.cfg.optim.grad_clip > 0:
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        self._scaler.step(self.optimizer)
        self._scaler.update()

        self.step = step + 1
        self.tokens_seen += self.cfg.tokens_per_step
        return total

    def train(self) -> dict[str, Any]:
        """Run to the token budget. Returns the final metrics."""
        self.setup()
        self.model.train()
        self._t0 = time.time()
        self._tokens_at_start = self.tokens_seen
        start = self.step

        for step in range(start, self.cfg.max_steps):
            loss = self.train_step(step)

            if step % self.cfg.log_interval == 0:
                self._log_progress(step, loss)
            if step > start and step % self.cfg.eval_interval == 0:
                metrics = self.evaluate()
                print(f"  val loss {metrics['loss']:.4f} ppl {metrics['perplexity']:.2f}")
            if step > start and step % self.cfg.checkpoint_interval == 0:
                self.save("latest.pt")

        metrics = self.evaluate()
        print(f"final val loss {metrics['loss']:.4f} ppl {metrics['perplexity']:.2f}")
        self.save("final.pt", extra={"final_metrics": metrics, "parameters": self.parameters})
        return metrics

    def _log_progress(self, step: int, loss: float) -> None:
        elapsed = max(time.time() - self._t0, 1e-6)
        done = self.tokens_seen - self._tokens_at_start
        rate = done / elapsed if done else 0.0
        remaining = (self.cfg.max_steps - step) * self.cfg.tokens_per_step
        eta_h = remaining / rate / 3600 if rate else float("nan")
        record = {
            "step": step,
            "loss": loss,
            "lr": self.learning_rate(step),
            "tokens": self.tokens_seen,
            "tokens_per_s": rate,
        }
        self.history.append(record)
        print(
            f"step {step:>6}/{self.cfg.max_steps} loss {loss:.4f} "
            f"lr {record['lr']:.2e} tok {self.tokens_seen / 1e9:.3f}B "
            f"{rate / 1e3:.1f}k tok/s eta {eta_h:.1f}h"
        )

    # -------------------------------------------------------------- evaluation

    def evaluate(self) -> dict[str, float]:
        return evaluate_split(
            self.model,
            self.val_shards,
            self.cfg.batch_size,
            self.cfg.model.block_size,
            self.cfg.eval_batches,
            self.device,
            autocast_dtype=self._amp_dtype,
        )

    # ------------------------------------------------------------ persistence

    def save(self, name: str = "latest.pt", extra: dict[str, Any] | None = None) -> Path:
        return ckpt.save(
            self.run_dir / name,
            self.raw_model,
            self.optimizer,
            self.cfg,
            self.step,
            self.tokens_seen,
            extra=extra,
        )

    def load(self, path: str | Path) -> "Pretrainer":
        self.step, self.tokens_seen = ckpt.load(
            path, self.raw_model, self.optimizer, map_location=self.device
        )
        # Advance the sampler so a resumed run does not replay the same windows.
        self._rng = np.random.default_rng(self.cfg.seed + self.step)
        return self

    # --------------------------------------------------------------- estimates

    def estimate_runtime(self, measured_tokens_per_s: float | None = None) -> dict[str, float]:
        """Rough wall clock for the configured budget.

        6ND against a measured throughput. Without a measurement the hours are
        nan rather than a guess, because on a card with no tensor cores the
        usual assumptions are wrong by more than they are right. Take the rate
        from the first few hundred steps and pass it in.
        """
        flops = 6 * self.parameters["total"] * self.cfg.target_tokens
        hours = (
            self.cfg.target_tokens / measured_tokens_per_s / 3600
            if measured_tokens_per_s
            else float("nan")
        )
        return {"total_flops": flops, "hours": hours, "log10_flops": math.log10(flops)}
