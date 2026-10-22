#!/usr/bin/env python
"""Pretrain one language arm.

    python scripts/pretrain_language.py --config configs/language/pretrain_50m.yaml
    python scripts/pretrain_language.py --config ... --resume experiments/runs/.../latest.pt

The random-initialisation arm of the ablation needs no pretraining: it is the
same GPTConfig handed to build_model and trained from scratch in Phase 4. Only
the language arms are pretrained here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from castor.language.config import load_pretrain_config  # noqa: E402
from castor.language.pretrain import Pretrainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--data-dir", default=None, help="override the config's data_dir")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--device", default=None, help="cuda or cpu; default is whatever exists")
    args = parser.parse_args()

    cfg = load_pretrain_config(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.no_compile:
        cfg.compile_model = False

    trainer = Pretrainer(cfg, resume=args.resume, device=args.device)
    trainer.train()
    print(f"\nrun written to {trainer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
