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
from castor.utils.kaggle import push_dataset  # noqa: E402
from castor.language.tokenizer import (  # noqa: E402
    DEFAULT_MAX_TOKENS_PER_NAME,
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
        "--target-tokens", type=int, default=None, help="override the mix's token target"
    )
    parser.add_argument(
        "--force", action="store_true", help="pack even if the tokenizer audit fails"
    )
    parser.add_argument("--kaggle-dataset", default=None,
                        help="publish the shards here on completion, e.g. user/castor-tokenized")
    parser.add_argument("--kaggle-public", action="store_true")
    args = parser.parse_args()

    tok_cfg = load_tokenizer_config(args.tokenizer)
    tokenizer = load_tokenizer(tok_cfg.get("name"), cache_dir=args.cache_dir)

    limit = tok_cfg.get("audit", {}).get("max_tokens_per_name", DEFAULT_MAX_TOKENS_PER_NAME)
    audit = audit_tokenizer(tokenizer, max_tokens_per_name=limit)
    print(f"tokenizer: {tokenizer.name_or_path}")
    print(audit.report())

    if not audit.ok() and not args.force:
        print(
            "\nAudit failed. Element names or digits tokenize badly, which will cost "
            "you in Phase 3 grounding. Pick another tokenizer or pass --force.",
            file=sys.stderr,
        )
        return 1
    if args.audit_only:
        return 0

    mix = MixConfig.from_yaml(args.mix)
    if args.target_tokens:
        mix.target_tokens = args.target_tokens

    print(f"\npacking {mix.target_tokens:,} train tokens (val fills in the same pass)")
    meta = pack(mix, tokenizer, args.out)
    for split in ("train", "val"):
        m = meta[split]
        print(
            f"  {split}: {m['tokens']:,} tokens in {len(m['shards'])} shard(s) "
            f"-> {Path(args.out) / split}"
        )
    if not meta["train"]["complete"]:
        print("  WARNING: sources exhausted before the token target was met")
    if meta["val"]["tokens"] == 0:
        print("  WARNING: validation split is empty", file=sys.stderr)
        return 1

    if args.kaggle_dataset:
        print()
        push_dataset(
            args.out,
            args.kaggle_dataset,
            title="CASTOR tokenized corpus",
            public=args.kaggle_public,
            notes=f"{meta['train']['tokens']:,} train / {meta['val']['tokens']:,} val tokens",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
