#!/usr/bin/env python
"""Build the Phase 2 domain adaptation corpus.

    # grid2op documentation only, from the installed package
    python scripts/build_domain_corpus.py --out data/text_corpus/domain

    # with the narrative pages too, from a clone of the project
    git clone --depth 1 https://github.com/Grid2Op/grid2op /tmp/grid2op-src
    python scripts/build_domain_corpus.py --out data/text_corpus/domain \
        --rst-dir /tmp/grid2op-src/docs \
        --reports-dir data/raw/blackout_reports \
        --gridcodes-dir data/raw/grid_codes

Reports and grid codes are read as .txt/.md. Convert PDFs beforehand: a bad
conversion should be visible as a bad text file, not as strange tokens
discovered in month six.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from castor.language.domain import (  # noqa: E402
    from_package,
    from_rst,
    from_text_dir,
    pack_domain,
)
from castor.utils.kaggle import push_dataset  # noqa: E402
from castor.language.tokenizer import (  # noqa: E402
    DEFAULT_MAX_TOKENS_PER_NAME,
    audit_tokenizer,
    load_tokenizer,
    load_tokenizer_config,
)


def find_grid2op() -> Path | None:
    try:
        import grid2op
    except ImportError:
        return None
    return Path(grid2op.__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/text_corpus/domain")
    parser.add_argument("--tokenizer", default="configs/language/tokenizer.yaml")
    parser.add_argument("--package", default=None, help="path to the installed grid2op")
    parser.add_argument("--rst-dir", default=None, help="docs/ from a grid2op clone")
    parser.add_argument("--reports-dir", default=None, help="blackout investigation reports")
    parser.add_argument("--gridcodes-dir", default=None, help="grid codes")
    parser.add_argument("--literature-dir", default=None, help="topology control literature")
    parser.add_argument("--jsonl", default=None, help="also write the records, with provenance")
    parser.add_argument("--holdout-share", type=float, default=0.05,
                        help="fraction of documents held out for the Phase 2 exit test")
    parser.add_argument("--kaggle-dataset", default=None,
                        help="publish the shards here on completion, e.g. user/castor-domain")
    parser.add_argument("--kaggle-public", action="store_true")
    args = parser.parse_args()

    tok_cfg = load_tokenizer_config(args.tokenizer)
    tokenizer = load_tokenizer(tok_cfg.get("name"))
    limit = tok_cfg.get("audit", {}).get("max_tokens_per_name", DEFAULT_MAX_TOKENS_PER_NAME)
    audit = audit_tokenizer(tokenizer, max_tokens_per_name=limit)
    print(f"tokenizer: {tokenizer.name_or_path}")
    print(audit.report())
    if not audit.ok():
        print("\nTokenizer audit failed; the domain corpus must use the same "
              "tokenizer as pretraining.", file=sys.stderr)
        return 1

    records = []
    package = Path(args.package) if args.package else find_grid2op()
    if package and package.exists():
        found = list(from_package(package))
        print(f"\ngrid2op API docstrings: {len(found)} from {package}")
        records += found
    else:
        print("\ngrid2op not importable and no --package given; skipping API docs")

    if args.rst_dir:
        found = list(from_rst(args.rst_dir))
        print(f"grid2op narrative pages: {len(found)}")
        records += found
    else:
        print("no --rst-dir; narrative documentation skipped (clone the repo for it)")

    for flag, source in (
        (args.reports_dir, "report"),
        (args.gridcodes_dir, "gridcode"),
        (args.literature_dir, "literature"),
    ):
        if flag:
            found = list(from_text_dir(flag, source))
            print(f"{source}: {len(found)} documents")
            records += found

    if not records:
        print("\nnothing to pack", file=sys.stderr)
        return 1

    meta = pack_domain(records, tokenizer, args.out, jsonl_path=args.jsonl,
                       holdout_share=args.holdout_share)

    print(f"\npacked to {args.out}")
    total = meta["train"]["tokens"] + meta["val"]["tokens"]
    for split in ("train", "val"):
        m = meta[split]
        print(f"  {split}: {m['tokens']:>9,} tokens in {len(m['shards'])} shard(s)")
    print("\nby source:")
    for source, stats in sorted(meta["train"]["by_source"].items()):
        share = 100 * stats["tokens"] / total if total else 0
        print(f"  {source:16} {stats['records']:>5} records  "
              f"{stats['tokens']:>9,} tokens  {share:5.1f}%")

    if meta["val"]["tokens"] == 0:
        print("\nWARNING: validation split is empty, the Phase 2 exit test needs it",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
