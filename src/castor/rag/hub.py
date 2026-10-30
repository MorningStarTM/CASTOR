"""Uploading the corpus to the Hub, during the run and not only after it.

A Kaggle session's disk disappears with the session, so the only copy of a
long run is whatever left the box before it died. Pushing once at the end
means a session killed at the 12 hour wall, or a browser tab closed, or an
OOM, costs every row generated up to that point.

So the corpus is uploaded periodically while the run is in progress, and the
dataset card says plainly that it is partial. A later push overwrites it with
a complete one. Uploading a few extra megabytes is cheap; regenerating a
day's worth of requests against a capped daily allowance is not.

Nothing here raises. A failed upload must never destroy a run whose output is
already safely on local disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RagConfig


def _card(cfg: RagConfig, totals: dict[str, Any], partial: bool) -> str:
    status = (
        "> **Run in progress.** This is a partial upload written while generation "
        "was still running. It will be overwritten when the run completes.\n\n"
        if partial else ""
    )
    return (
        "# CASTOR grid2op QA\n\n"
        f"{status}"
        "Question, reasoning and answer triples over the grid2op API, for Phase 2\n"
        "domain adaptation of the CASTOR language arm.\n\n"
        f"- rows: {totals.get('rows', 0):,}\n"
        f"- pages: {totals.get('pages', 0):,}\n"
        f"- pass rate: {totals.get('pass_rate', 0):.1%}\n"
        f"- verdicts: {totals.get('verdict', {})}\n"
        f"- grounding: {totals.get('grounding', {})}\n"
        f"- reasoning depth: {totals.get('depth', {})}\n"
        f"- generator: `{cfg.models.generator.id}`\n"
        f"- verifier: `{cfg.models.verifier.id}`\n\n"
        "## Schema\n\n"
        "One JSON object per line. `support` is a verbatim span from the source page,\n"
        "`reasoning` is the derivation from that span to the answer, and `checks`\n"
        "records the outcome of the span, entailment, reasoning and execution checks.\n\n"
        "Rows that failed verification ship with their verdict rather than being\n"
        "dropped, so the corpus states its own error rate. **Train on\n"
        "`verdict == \"pass\"` only.**\n"
    )


def push(cfg: RagConfig, totals: dict[str, Any], partial: bool = False,
         report=print) -> bool:
    """Upload corpus, pages and card. Returns success; never raises."""
    repo = cfg.output.hf_repo
    if not repo:
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, repo_type="dataset", private=cfg.output.hf_private,
                        exist_ok=True)
        api.upload_file(
            path_or_fileobj=_card(cfg, totals, partial).encode(),
            path_in_repo="README.md", repo_id=repo, repo_type="dataset",
        )
        for path in (cfg.output.corpus_path, cfg.output.pages_path):
            if Path(path).exists():
                api.upload_file(path_or_fileobj=str(path), path_in_repo=Path(path).name,
                                repo_id=repo, repo_type="dataset")
        label = "partial" if partial else "final"
        report(f"  [hub] {label} push: {totals.get('rows', 0):,} rows -> "
               f"https://huggingface.co/datasets/{repo}")
        return True
    except Exception as exc:  # noqa: BLE001 - the corpus is on disk either way
        report(f"  [hub] push failed ({type(exc).__name__}: {exc}); "
               f"corpus is at {cfg.output.corpus_path}")
        return False


def quick_totals(path: str | Path) -> dict[str, Any]:
    """Row and page counts without a full summarise pass.

    An in-progress push happens every few requests, and walking the whole file
    to build a verdict histogram each time would grow quadratic over a long
    run. The complete figures go on the card at the end.
    """
    from .schema import read_rows

    rows = pages = 0
    seen: set[str] = set()
    for row in read_rows(path):
        rows += 1
        seen.add(row.page)
    pages = len(seen)
    return {"rows": rows, "pages": pages}
