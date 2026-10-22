#!/usr/bin/env python
"""Tokenize the pretraining mixture into packed shards.

    python scripts/prepare_pretrain_data.py \
        --mix configs/language/data_mix.yaml \
        --tokenizer configs/language/tokenizer.yaml \
        --out data/text_corpus/tokenized

Runs the tokenizer audit first and refuses to proceed if element names come
out badly, because discovering that after 6B tokens have been written means
tokenizing them all again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from castor.language.corpus import MixConfig, pack  # noqa: E402
from castor.language.tokenizer import (  # noqa: E402
    audit_tokenizer,
    load_tokenizer,
    load_tokenizer_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default="configs/language/data_mix.yaml")
    parser.add_argument("--tokenizer", default="configs/language/tokenizer.yaml")
    parser.add_argument("--out", default="data/text_corpus/tokenized")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="pack even if the tokenizer audit fails"
    )
    args = parser.parse_args()

    tok_cfg = load_tokenizer_config(args.tokenizer)
    tokenizer = load_tokenizer(tok_cfg.get("name"), cache_dir=args.cache_dir)

    audit = audit_tokenizer(tokenizer)
    limit = tok_cfg.get("audit", {}).get("max_tokens_per_name", 4)
    print(f"tokenizer: {tokenizer.name_or_path}")
    print(audit.report())

    if not audit.ok(limit) and not args.force:
        print(
            "\nAudit failed. Element names or digits tokenize badly, which will cost "
            "you in Phase 3 grounding. Pick another tokenizer or pass --force.",
            file=sys.stderr,
        )
        return 1
    if args.audit_only:
        return 0

    mix = MixConfig.from_yaml(args.mix)

    print(f"\npacking validation split ({mix.holdout_tokens:,} tokens)")
    val_meta = pack(mix, tokenizer, args.out, "val", target_tokens=mix.holdout_tokens)
    print(f"  wrote {val_meta['tokens']:,} tokens in {len(val_meta['shards'])} shard(s)")

    print(f"\npacking train split ({mix.target_tokens:,} tokens)")
    train_meta = pack(mix, tokenizer, args.out, "train")
    print(f"  wrote {train_meta['tokens']:,} tokens in {len(train_meta['shards'])} shard(s)")
    if not train_meta["complete"]:
        print("  WARNING: sources exhausted before the token target was met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
