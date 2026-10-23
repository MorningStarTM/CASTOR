"""The pretraining loop, as a class, single GPU or DDP.

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
    trainer.save("latest")

Distributed
-----------
Data parallelism via DistributedDataParallel: every rank holds the whole model
and reads a different slice of the corpus. Model parallelism is not offered
because it is not needed. The largest arm here is 96M parameters, roughly 1.5GB
with optimizer state, against 16GB on a T4.

Two ways in. Under torchrun, Pretrainer reads the environment and configures
itself, so nothing changes at the call site:

    torchrun --nproc_per_node=2 scripts/pretrain_language.py --config ...

From a notebook, where there is no torchrun, spawn the ranks:

    train_ddp(cfg, world_size=2)

Note what DDP does to the budget. Each rank consumes cfg.tokens_per_step, so
the global rate is that times world_size and the step count falls by the same
factor. Pretrainer.max_steps accounts for this; cfg.max_steps does not, and is
only correct for the single process case.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import checkpoint as ckpt
from .config import PretrainConfig, config_hash
from .dataset import PackedShards
from .model import build_model, configure_optimizer, count_parameters
from .perplexity import evaluate as evaluate_split
from .schedule import lr_at


# --------------------------------------------------------------- DDP helpers


def ddp_env() -> tuple[int, int, int] | None:
    """(rank, local_rank, world_size) if launched under torchrun, else None."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None
    return (
        int(os.environ["RANK"]),
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ["WORLD_SIZE"]),
    )


def setup_ddp(rank: int, world_size: int, backend: str | None = None) -> None:
    """Initialise the process group. nccl on CUDA, gloo everywhere else."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # device_count can be 0 while is_available() is True, e.g. under an empty
    # CUDA_VISIBLE_DEVICES, so check the count rather than availability.
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class Pretrainer:
    """Pretrains one language arm from a PretrainConfig."""

    def __init__(
        self,
        cfg: PretrainConfig,
        resume: str | Path | None = None,
        device: str | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.resume_from = Path(resume) if resume else None

        # Explicit arguments win, then torchrun's environment, then single process.
        env = ddp_env()
        if rank is not None and world_size is not None:
            self.rank, self.world_size = rank, world_size
            self.local_rank = local_rank if local_rank is not None else rank
        elif env is not None:
            self.rank, self.local_rank, self.world_size = env
        else:
            self.rank, self.local_rank, self.world_size = 0, 0, 1

        if device is not None:
            self.device = device
        elif torch.cuda.device_count() > 0:
            # device_count, not is_available: the latter can report True with
            # no usable devices, e.g. under an empty CUDA_VISIBLE_DEVICES.
            self.device = f"cuda:{self.local_rank}" if self.distributed else "cuda"
        else:
            self.device = "cpu"

        self.model: torch.nn.Module | None = None
        self.raw_model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.train_shards: PackedShards | None = None
        self.val_shards: PackedShards | None = None
        self.parameters: dict[str, int] = {}

        self.step = 0
        self.tokens_seen = 0
        self.history: list[dict[str, Any]] = []

        # Each rank draws different windows, or DDP would average identical
        # gradients and the extra GPU would buy nothing.
        self._rng = np.random.default_rng(cfg.seed + 7919 * self.rank)
        self._amp_dtype: torch.dtype | None = None
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._is_setup = False
        self._compiled = False
        self._t0 = 0.0
        self._tokens_at_start = 0

    # ------------------------------------------------------------- properties

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def global_tokens_per_step(self) -> int:
        """Tokens consumed per optimizer step across all ranks."""
        return self.cfg.tokens_per_step * self.world_size

    @property
    def max_steps(self) -> int:
        """Steps to reach the token budget, at the current world size."""
        return max(1, self.cfg.target_tokens // self.global_tokens_per_step)

    @property
    def run_dir(self) -> Path:
        return Path(self.cfg.out_dir) / f"pretrain_{self.cfg.arm}_{config_hash(self.cfg)}"

    # ------------------------------------------------------------------ setup

    def setup(self) -> "Pretrainer":
        """Build data, model and optimizer. Idempotent."""
        if self._is_setup:
            return self

        torch.manual_seed(self.cfg.seed + self.rank)
        np.random.seed(self.cfg.seed + self.rank)
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._load_data()
        self._build_model()
        self._configure_amp()

        if self.resume_from:
            self.load(self.resume_from)

        self._maybe_compile()
        self._wrap_ddp()

        if self.is_master:
            self._write_manifest()
            print(self.describe())
        self._is_setup = True
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
            self._compiled = True
        except Exception as exc:
            if self.is_master:
                print(f"torch.compile unavailable, continuing eager: {exc}")
            self.model = self.raw_model

    def _wrap_ddp(self) -> None:
        if not self.distributed:
            return
        if not dist.is_initialized():
            raise RuntimeError(
                "world_size > 1 but no process group. Launch with torchrun, or use "
                "train_ddp(), which calls setup_ddp() in each spawned rank."
            )
        device_ids = [self.local_rank] if self.device.startswith("cuda") else None
        self.model = DistributedDataParallel(self.model, device_ids=device_ids)

    # -------------------------------------------------------------- reporting

    def describe(self) -> str:
        p = self.parameters
        return (
            f"arm={self.cfg.arm} "
            f"params={p['total'] / 1e6:.1f}M (non-embedding {p['non_embedding'] / 1e6:.1f}M) "
            f"world={self.world_size} steps={self.max_steps} warmup={self.warmup_steps} "
            f"tokens/step={self.global_tokens_per_step:,} "
            f"corpus={self.train_shards.total_tokens:,} "
            f"device={self.device} amp={self._amp_dtype} compiled={self._compiled}"
        )

    def _write_manifest(self) -> None:
        ckpt.write_manifest(
            self.run_dir,
            self.cfg,
            {
                "parameters": self.parameters,
                "device": self.device,
                "amp_dtype": str(self._amp_dtype),
                "world_size": self.world_size,
                "max_steps": self.max_steps,
                "tokens_per_step": self.global_tokens_per_step,
                "corpus": {
                    k: self.train_shards.meta[k] for k in ("tokens", "tokenizer", "sources")
                },
            },
        )

    # ---------------------------------------------------------------- training

    @property
    def warmup_steps(self) -> int:
        """Warmup resolved against this run's derived step count."""
        return self.cfg.optim.warmup_for(self.max_steps)

    def learning_rate(self, step: int) -> float:
        return lr_at(
            step,
            self.cfg.optim.learning_rate,
            self.warmup_steps,
            self.max_steps,
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
        accum = self.cfg.grad_accum_steps
        total = 0.0

        for micro in range(accum):
            x, y = self.train_shards.batch(
                self.cfg.batch_size, self.cfg.model.block_size, self._rng, device=self.device
            )
            # Suppress the all-reduce on every micro-step but the last: the
            # gradients are still accumulating and syncing them is wasted
            # bandwidth. This is most of DDP's efficiency at high accumulation.
            if self.distributed and micro < accum - 1:
                with self.model.no_sync():
                    loss = self._forward(x, y) / accum
                    self._scaler.scale(loss).backward()
            else:
                loss = self._forward(x, y) / accum
                self._scaler.scale(loss).backward()
            total += loss.item()

        if self.cfg.optim.grad_clip > 0:
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.cfg.optim.grad_clip)
        self._scaler.step(self.optimizer)
        self._scaler.update()

        self.step = step + 1
        self.tokens_seen += self.global_tokens_per_step
        return total

    def train(self) -> dict[str, Any]:
        """Run to the token budget. Returns the final metrics."""
        self.setup()
        self.model.train()
        self._t0 = time.time()
        self._tokens_at_start = self.tokens_seen
        start = self.step

        try:
            for step in range(start, self.max_steps):
                loss = self.train_step(step)

                if step % self.cfg.log_interval == 0:
                    self._log_progress(step, self._mean_across_ranks(loss))
                if step > start and step % self.cfg.eval_interval == 0:
                    metrics = self.evaluate()
                    if self.is_master:
                        print(f"  val loss {metrics['loss']:.4f} ppl {metrics['perplexity']:.2f}")
                if step > start and step % self.cfg.checkpoint_interval == 0:
                    self.save("latest")

            metrics = self.evaluate()
            if self.is_master:
                print(f"final val loss {metrics['loss']:.4f} ppl {metrics['perplexity']:.2f}")
            self.save("final", extra={"final_metrics": metrics, "parameters": self.parameters})
        finally:
            if self.distributed:
                dist.barrier()
        return metrics

    def _mean_across_ranks(self, value: float) -> float:
        """Average a scalar over ranks, so the logged loss is the global one."""
        if not self.distributed:
            return value
        tensor = torch.tensor([value], device=self.device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item() / self.world_size)

    def _log_progress(self, step: int, loss: float) -> None:
        if not self.is_master:
            return
        elapsed = max(time.time() - self._t0, 1e-6)
        done = self.tokens_seen - self._tokens_at_start
        rate = done / elapsed if done else 0.0
        remaining = (self.max_steps - step) * self.global_tokens_per_step
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
            f"step {step:>6}/{self.max_steps} loss {loss:.4f} "
            f"lr {record['lr']:.2e} tok {self.tokens_seen / 1e9:.3f}B "
            f"{rate / 1e3:.1f}k tok/s eta {eta_h:.1f}h"
        )

    # -------------------------------------------------------------- evaluation

    def evaluate(self) -> dict[str, float]:
        """Held-out loss, on the master only.

        Uses raw_model rather than the DDP wrapper: a forward with no backward
        confuses DDP's reducer, and a second rank evaluating the same fixed
        windows would only duplicate the work.
        """
        if not self.is_master:
            return {"loss": float("nan"), "perplexity": float("nan")}
        return evaluate_split(
            self.raw_model,
            self.val_shards,
            self.cfg.batch_size,
            self.cfg.model.block_size,
            self.cfg.eval_batches,
            self.device,
            autocast_dtype=self._amp_dtype,
        )

    # ------------------------------------------------------------ persistence

    def save(self, name: str = "latest", extra: dict[str, Any] | None = None) -> Path | None:
        """Write a checkpoint from the master. Other ranks hold identical weights."""
        if not self.is_master:
            return None
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
        # Advance the sampler so a resumed run does not replay the same windows,
        # keeping the per-rank offset.
        self._rng = np.random.default_rng(self.cfg.seed + 7919 * self.rank + self.step)
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


# ------------------------------------------------------------- DDP entry point


def _ddp_worker(
    rank: int,
    world_size: int,
    cfg: PretrainConfig,
    resume: str | None,
    backend: str | None,
) -> None:
    """One spawned rank: join the group, train, leave."""
    setup_ddp(rank, world_size, backend=backend)
    try:
        Pretrainer(cfg, resume=resume, rank=rank, world_size=world_size, local_rank=rank).train()
    finally:
        cleanup_ddp()


def train_ddp(
    cfg: PretrainConfig,
    resume: str | Path | None = None,
    world_size: int | None = None,
    backend: str | None = None,
) -> int:
    """Spawn one process per GPU and pretrain the arm across them.

    For notebooks, where torchrun is not available. Under torchrun, construct
    Pretrainer directly: it reads RANK and WORLD_SIZE from the environment.

    Returns the world size actually used. A world size of 1 trains in this
    process rather than paying for a pointless process group.
    """
    import torch.multiprocessing as mp

    available = torch.cuda.device_count()
    if world_size is None:
        world_size = available or 1
    elif available and world_size > available:
        # Otherwise rank N simply fails on cuda:N with "invalid device ordinal",
        # which on Kaggle usually means the accelerator is set to one GPU.
        raise ValueError(
            f"world_size {world_size} exceeds the {available} visible GPU(s). "
            "Check the notebook accelerator is set to 2 x T4."
        )

    if world_size <= 1:
        Pretrainer(cfg, resume=resume).train()
        return 1

    mp.spawn(
        _ddp_worker,
        args=(world_size, cfg, str(resume) if resume else None, backend),
        nprocs=world_size,
        join=True,
    )
    return world_size
