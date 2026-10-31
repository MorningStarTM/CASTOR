#!/usr/bin/env python
"""Build the grid2op QA corpus: paginate, retrieve, generate, verify, publish.

    # what the run would do, and what it costs in requests. No API calls.
    python scripts/build_rag_corpus.py --dry-run

    # the run
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/build_rag_corpus.py

    # a small slice first, to see the output before committing the allowance
    python scripts/build_rag_corpus.py --limit 5

    # simulator-grounded rows only; no API key needed
    python scripts/build_rag_corpus.py --executed-only

Every setting lives in configs/rag/rag.yaml. The flags below override single
values for one run; they do not replace the file.

The run is resumable. Rows append per request with an fsync, and a restart
skips every page already recorded in pages.jsonl, so a dropped connection or
an exhausted daily allowance costs one request rather than the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from castor.rag import dotenv  # noqa: E402
from castor.rag import hub  # noqa: E402
from castor.rag import load_config, load_prompts  # noqa: E402
from castor.rag import pipeline  # noqa: E402
from castor.rag.audit import LeakError  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "rag" / "rag.yaml"
DEFAULT_PROMPTS = ROOT / "configs" / "rag" / "prompts.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))

    p.add_argument("--dry-run", action="store_true",
                   help="paginate and index, report the request budget, send nothing")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N generate requests")
    p.add_argument("--executed-only", action="store_true",
                   help="only the simulator-grounded rows; no API key needed")
    p.add_argument("--no-executed", action="store_true",
                   help="skip the simulator-grounded rows. The teacher path needs only "
                        "the grid2op source, so this avoids downloading an environment")
    p.add_argument("--reverify", action="store_true",
                   help="re-run verification over rows left unchecked, without "
                        "regenerating them. One request per page, not two")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-attempt pages whose reply would not parse. Pages that "
                        "parsed but had nothing usable stay done: those are thin, "
                        "not broken")
    p.add_argument("--finalise-only", action="store_true",
                   help="re-run dedup and the leak audit over the corpus on disk")

    # Single-value overrides. Each mirrors a key in rag.yaml.
    p.add_argument("--generator", default=None, help="override models.generator.id")
    p.add_argument("--verifier", default=None, help="override models.verifier.id")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="override generation.max_pairs_per_page")
    p.add_argument("--bundle-under", type=int, default=None,
                   help="override pages.bundle_under; 0 is one page per request")
    p.add_argument("--no-verify", action="store_true", help="disable stage 5")
    p.add_argument("--no-dedup", action="store_true", help="disable stage 6")
    p.add_argument("--held-out", action="append", default=None,
                   help="environment the leak audit must never find; repeatable")
    p.add_argument("--out-dir", default=None, help="override output.dir")
    p.add_argument("--hf-repo", default=None, help="override output.hf_repo")
    p.add_argument("--hf-public", action="store_true")
    p.add_argument("--push-every", type=int, default=None,
                   help="upload the partial corpus every N requests; 0 disables")
    p.add_argument("--rpm", type=int, default=None,
                   help="override api.requests_per_minute")
    p.add_argument("--per-day", type=int, default=1000,
                   help="daily allowance assumed when reporting days (default 1000)")
    return p.parse_args()


def apply_overrides(cfg, args) -> None:
    if args.generator:
        cfg.models.generator.id = args.generator
    if args.verifier:
        cfg.models.verifier.id = args.verifier
    if args.max_pairs is not None:
        cfg.generation.max_pairs_per_page = args.max_pairs
    if args.bundle_under is not None:
        cfg.pages.bundle_under = args.bundle_under
    if args.no_verify:
        cfg.verification.enabled = False
    if args.no_dedup:
        cfg.dedup.enabled = False
    if args.no_executed:
        cfg.executed.enabled = False
    if args.held_out:
        cfg.audit.held_out_environments = list(args.held_out)
    if args.out_dir:
        cfg.output.dir = args.out_dir
    if args.hf_repo:
        cfg.output.hf_repo = args.hf_repo
    if args.hf_public:
        cfg.output.hf_private = False
    if args.push_every is not None:
        cfg.output.push_every = args.push_every
    if args.rpm is not None:
        cfg.api.requests_per_minute = args.rpm


def load_env(cfg) -> None:
    """Pull .env into the environment before anything asks for a key.

    Only the names are printed. A key echoed into a terminal ends up in shell
    history, in a screenshot, or in a pasted log.
    """
    if not cfg.api.env_file:
        return
    found = dotenv.find_upwards(cfg.api.env_file)
    if not found:
        return
    names = dotenv.load(found)
    if names:
        print(f"env        {found}  ->  {', '.join(sorted(names))}")


def banner(cfg) -> None:
    print(f"generator  {cfg.models.generator.id}  T={cfg.models.generator.temperature}")
    print(f"verifier   {cfg.models.verifier.id}  T={cfg.models.verifier.temperature}"
          f"{'' if cfg.verification.enabled else '  (DISABLED)'}")
    import os as _os
    key = _os.environ.get(cfg.api.api_key_env, "")
    print(f"output     {cfg.output.corpus_path}")
    if cfg.output.hf_repo:
        every = cfg.output.push_every
        print(f"hub        {cfg.output.hf_repo}  "
              f"{'every ' + str(every) + ' requests + at the end' if every else 'AT THE END ONLY'}")
    print(f"api key    {cfg.api.api_key_env}: "
          f"{'set (' + str(len(key)) + ' chars)' if key else 'MISSING'}")

    # What the allowance actually is, before the run spends any of it. A 429
    # is counted before it is refused, so discovering an exhausted cap by
    # running into it costs real requests.
    if key:
        from castor.rag.client import key_status
        status = key_status(cfg.api, key)
        free = (status or {}).get("free_model_daily_requests") or {}
        if free:
            used, limit = free.get("used", "?"), free.get("limit", "?")
            left = free.get("remaining", "?")
            print(f"allowance  {left} of {limit} free requests left today "
                  f"({used} used)"
                  f"{'   <-- NOTHING LEFT, the run will stop immediately' if left == 0 else ''}")
        elif status:
            print(f"allowance  paid tier, ${status.get('usage', 0):.4f} used")

    for warning in cfg.validate():
        print(f"  warning: {warning}")
    print()


def report_budget(the_plan, cfg, per_day: int) -> None:
    verify = cfg.verification.enabled
    print(f"\nrequests   {the_plan.n_generate:,} generate"
          f"{f' + {the_plan.n_generate:,} verify' if verify else ''}"
          f" = {the_plan.n_requests(verify):,}")
    print(f"ceiling    {len(the_plan.pages) * cfg.generation.max_pairs_per_page:,} pairs "
          f"({cfg.generation.max_pairs_per_page} x {len(the_plan.pages):,} pages)")
    print(f"days       {the_plan.days(verify, per_day):.1f} at {per_day}/day, "
          f"{the_plan.n_requests(verify) / max(1, cfg.api.requests_per_minute) / 60:.1f} h "
          f"at {cfg.api.requests_per_minute} rpm")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    templates = load_prompts(args.prompts)
    apply_overrides(cfg, args)
    load_env(cfg)
    banner(cfg)

    if args.reverify:
        try:
            pipeline.reverify(cfg, templates, limit=args.limit)
        except LeakError as exc:
            print(f"\nLEAK AUDIT FAILED\n{exc}")
            return 2
        return 0

    if args.finalise_only:
        try:
            pipeline.finalise(cfg)
        except LeakError as exc:
            print(f"\nLEAK AUDIT FAILED\n{exc}")
            return 2
        return 0

    if args.dry_run:
        the_plan = pipeline.plan(cfg, retry_failed=args.retry_failed)
        report_budget(the_plan, cfg, args.per_day)
        if the_plan.pending:
            sample = the_plan.pending[0][0]
            print(f"\nfirst request would send: {sample.id}  ({sample.chars:,} chars)")
            print("-" * 70)
            print(sample.text[:600])
        return 0

    if args.executed_only:
        from castor.rag import executed as executed_stage
        from castor.rag.schema import PageRecord, append_pages, append_rows, next_row_number

        rows = executed_stage.build(cfg.executed.environment,
                                    next_row_number(cfg.output.corpus_path))
        append_rows(cfg.output.corpus_path, rows)
        append_pages(cfg.output.pages_path, [PageRecord(
            id=f"{cfg.executed.environment}.executed", chars=0,
            n_pairs=len(rows), n_pass=len(rows), note="simulator-grounded")])
        print(f"{len(rows)} executed rows -> {cfg.output.corpus_path}")
        return 0

    try:
        totals = pipeline.run(cfg, templates, limit=args.limit,
                              retry_failed=args.retry_failed)
    except LeakError as exc:
        print(f"\nLEAK AUDIT FAILED\n{exc}")
        return 2

    print("\n" + json.dumps(totals, indent=2, default=str))
    if cfg.output.hf_repo:
        hub.push(cfg, totals, partial=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
