"""The seven stages, wired together.

    1 paginate   pages.build
    2 index      index.build
    3 augment    augment.augment          <- retrieval
    4 generate   generate.generate
    5 verify     verify.verify
    6 dedup      dedup.deduplicate
    7 audit      audit.audit

Stages 1-5 run per request and checkpoint to disk as they go, so a dropped
connection or an exhausted daily allowance costs one request rather than the
run. Stages 6 and 7 are corpus-wide and run at the end, over everything on
disk including rows written by earlier sessions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import audit as audit_stage
from . import augment as augment_stage
from . import dedup as dedup_stage
from . import executed as executed_stage
from . import generate as generate_stage
from . import hub
from . import index as index_stage
from . import pages as pages_stage
from . import verify as verify_stage
from .client import Client, QuotaExhausted, TeacherError
from .config import RagConfig
from .schema import (
    FAIL_PARSE, PASS, UNCHECKED, Page, PageRecord, QARow, append_pages,
    append_rows, done_pages, make_id, next_row_number, read_rows, summarise,
)

Reporter = Callable[[str], None]


@dataclass
class Plan:
    """What a run would do, before it does any of it."""

    pages: list[Page]
    groups: list[list[Page]]
    index_size: int
    pending: list[list[Page]] = field(default_factory=list)

    @property
    def n_generate(self) -> int:
        return len(self.pending)

    def n_requests(self, verify: bool) -> int:
        return self.n_generate * (2 if verify else 1)

    def days(self, verify: bool, per_day: int) -> float:
        return self.n_requests(verify) / max(1, per_day)


def plan(cfg: RagConfig, report: Reporter = print) -> Plan:
    """Stages 1 and 2, plus whatever a resume has already done. No API calls."""
    root = pages_stage.locate_package(cfg.source.package)
    all_pages = pages_stage.build(cfg.pages, cfg.source)
    report(f"stage 1  {len(all_pages):,} pages from {root}")

    symbol_index = index_stage.build(cfg.index, cfg.source, root)
    report(f"stage 2  {symbol_index.n_symbols:,} symbols indexed")

    groups = pages_stage.bundle(all_pages, cfg.pages)
    done = done_pages(cfg.output.pages_path)
    pending = [g for g in groups if not all(p.id in done for p in g)]
    if done:
        report(f"resume   {len(done):,} pages already done, {len(pending):,} requests left")

    return Plan(pages=all_pages, groups=groups, index_size=symbol_index.n_symbols,
                pending=pending)


def _to_row(number: int, raw: generate_stage.RawPair, verdict: verify_stage.Verdict,
            refs: list[str], generator_id: str, verifier_id: str) -> QARow:
    checks = QARow.new_checks()
    checks["span"] = verdict.span
    checks["entailment"] = verdict.entailment
    checks["reasoning"] = verdict.reasoning
    return QARow(
        id=make_id(number),
        question=raw.question,
        reasoning=raw.reasoning,
        answer=raw.answer,
        support=raw.support,
        grounding=raw.grounding,
        reasoning_depth=raw.reasoning_depth,
        verdict=verdict.verdict,
        checks=checks,
        reason=verdict.reason,
        page=raw.page.id,
        refs=refs,
        source=raw.page.source,
        model=generator_id,
        # The model that actually judged, which may be a fallback.
        verifier=verdict.model or (verifier_id if verdict.verdict != UNCHECKED else ""),
    )


def run(cfg: RagConfig, templates: dict[str, str], limit: int | None = None,
        report: Reporter = print) -> dict:
    """Stages 1 to 5 over every pending request, then 6 and 7 over the corpus."""
    the_plan = plan(cfg, report)
    root = pages_stage.locate_package(cfg.source.package)
    symbol_index = index_stage.build(cfg.index, cfg.source, root)

    number = next_row_number(cfg.output.corpus_path)
    done = done_pages(cfg.output.pages_path)

    # Executed rows first: free, decisive, and they need no API key.
    if cfg.executed.enabled and f"{cfg.executed.environment}.executed" not in done:
        try:
            rows = executed_stage.build(cfg.executed.environment, number)
            append_rows(cfg.output.corpus_path, rows)
            append_pages(cfg.output.pages_path, [PageRecord(
                id=f"{cfg.executed.environment}.executed", chars=0,
                n_pairs=len(rows), n_pass=len(rows), note="simulator-grounded",
            )])
            number += len(rows)
            report(f"stage 0  {len(rows)} executed rows from {cfg.executed.environment}")
        except Exception as exc:  # noqa: BLE001 - no simulator must not stop the teacher
            report(f"stage 0  skipped ({type(exc).__name__}: {exc})")

    client = Client(cfg.api)
    pending = the_plan.pending[:limit] if limit else the_plan.pending
    started = time.monotonic()
    written = failed = 0

    for i, group in enumerate(pending, 1):
        items = [augment_stage.augment(p, symbol_index, cfg.augment, templates,
                                   cfg.index.bare_fallback_for_qualified)
                 for p in group]
        refs = sorted({r for it in items for r in it.refs})
        head = items[0]

        try:
            result = generate_stage.generate(
                client, cfg.models.generator, items, cfg.generation, templates)
        except QuotaExhausted as exc:
            report(f"\nDAILY ALLOWANCE SPENT at {head.page.id}: {exc}")
            report("  the corpus on disk is intact; rerun tomorrow and it "
                   "resumes from the next unseen page")
            break
        except TeacherError as exc:
            # A hard API error is the daily cap or a bad key. Stopping keeps
            # the partial corpus intact and resumable; carrying on would burn
            # every remaining page against the same error.
            report(f"\nstopped at {head.page.id}: {exc}")
            break

        if not result.parsed:
            failed += 1
            if result.raw:
                try:
                    cfg.output.rejects_dir.mkdir(parents=True, exist_ok=True)
                    safe = head.page.id.replace("/", "_").replace("\\", "_")
                    header = f"# {result.note}\n# page {head.page.id}\n\n"
                    (cfg.output.rejects_dir / f"{safe}.txt").write_text(
                        header + result.raw, encoding="utf-8")
                except OSError:
                    pass  # a diagnostic that cannot be written must not stop a run
            append_pages(cfg.output.pages_path, [
                PageRecord(id=p.id, chars=p.chars, refs=refs, note=result.note)
                for p in group
            ])
            continue

        # A spent allowance here must not discard the pairs: the generator
        # was already paid for them. They are written unchecked and the loop
        # stops, so --reverify can judge them later without regenerating.
        spent = ""
        try:
            verdicts = verify_stage.verify(
                client, cfg.models.verifier, result.pairs,
                head.page.id, head.page.text, cfg.verification, templates)
        except QuotaExhausted as exc:
            spent = str(exc)
            verdicts = [
                verify_stage.Verdict(
                    UNCHECKED,
                    verify_stage.span_ok(p.support, head.page.text, cfg.verification),
                    None, None, "daily allowance spent before verification")
                for p in result.pairs
            ]

        rows = [
            _to_row(number + n, raw, verdict, refs,
                    cfg.models.generator.id, cfg.models.verifier.id)
            for n, (raw, verdict) in enumerate(zip(result.pairs, verdicts))
        ]
        number += len(rows)
        written += append_rows(cfg.output.corpus_path, rows)

        by_page: dict[str, list[QARow]] = {}
        for row in rows:
            by_page.setdefault(row.page, []).append(row)
        append_pages(cfg.output.pages_path, [
            PageRecord(
                id=p.id, chars=p.chars, refs=refs,
                n_pairs=len(by_page.get(p.id, [])),
                n_pass=sum(1 for r in by_page.get(p.id, []) if r.verdict == PASS),
                n_fail=sum(1 for r in by_page.get(p.id, []) if r.verdict not in (PASS, UNCHECKED)),
            ) for p in group
        ])

        if spent:
            report(f"\nDAILY ALLOWANCE SPENT verifying {head.page.id}: {spent}")
            report(f"  {len(rows)} rows written unchecked; "
                   f"recover them with --reverify")
            break

        if cfg.output.hf_repo and cfg.output.push_every and i % cfg.output.push_every == 0:
            hub.push(cfg, hub.quick_totals(cfg.output.corpus_path), partial=True,
                     report=report)

        if i % 5 == 0 or i == len(pending):
            rate = i / max(1e-9, (time.monotonic() - started) / 60)
            report(f"[{i:>4}/{len(pending)}] {written:,} rows  {failed} unparsed  "
                   f"{client.calls} calls  {rate:.1f} req/min  {head.page.id[:48]}")

    report(f"\nteacher: {client.stats()}")
    return finalise(cfg, report)


def finalise(cfg: RagConfig, report: Reporter = print) -> dict:
    """Stages 6 and 7 over the whole corpus, then the summary.

    Rewrites the corpus in place: deduplication and the audit both change
    verdicts, and a reader of the file should see the verdict that stands, not
    the one a single page's worth of context produced.
    """
    rows = list(read_rows(cfg.output.corpus_path))
    if not rows:
        report("nothing to finalise")
        return {}

    deduped = dedup_stage.deduplicate(rows, cfg.dedup)
    report(f"stage 6  {deduped.n_dropped:,} near-duplicate questions dropped")

    audited = audit_stage.audit(deduped.kept, cfg.audit)
    if cfg.audit.terms:
        report(f"stage 7  {len(audited.violations):,} leak violation(s) over "
               f"{len(cfg.audit.terms)} forbidden term(s)")
    else:
        report("stage 7  SKIPPED: audit.held_out_environments is empty")

    path = Path(cfg.output.corpus_path)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    append_rows(tmp, audited.clean + [r for r, _ in audited.violations])
    tmp.replace(path)

    totals = summarise(path)
    report(f"\ncorpus   {totals['rows']:,} rows over {totals['pages']:,} pages, "
           f"pass rate {totals['pass_rate']:.1%}")
    report(f"verdict  {totals['verdict']}")
    report(f"depth    {totals['depth']}")

    audit_stage.enforce(audited, cfg.audit)
    return totals


def reverify(cfg: RagConfig, templates: dict[str, str], limit: int | None = None,
             report: Reporter = print) -> dict:
    """Re-run stage 5 over rows left unchecked, without regenerating them.

    An unchecked row is not a wrong row. Its question, reasoning, answer and
    support span are all intact; only the judge was unreachable, usually
    because a :free pool was saturated at that moment. Regenerating it would
    pay the generator again for work already done and throw away a pair that
    may well be fine.

    So this re-reads the corpus, rebuilds the pages the unchecked rows came
    from, and asks the verifier again. One request per page instead of two,
    and it can run days later when the pools are quiet.
    """
    rows = list(read_rows(cfg.output.corpus_path))
    if not rows:
        report("nothing to re-verify")
        return {}

    stale = [r for r in rows if r.verdict == UNCHECKED]
    if not stale:
        report("no unchecked rows")
        return summarise(cfg.output.corpus_path)

    by_page: dict[str, list[QARow]] = {}
    for row in stale:
        by_page.setdefault(row.page, []).append(row)
    report(f"{len(stale):,} unchecked rows across {len(by_page):,} pages")

    pages = {p.id: p for p in pages_stage.build(cfg.pages, cfg.source)}
    client = Client(cfg.api)
    index = {r.id: r for r in rows}
    targets = list(by_page.items())[:limit] if limit else list(by_page.items())

    recovered = 0
    for i, (page_id, page_rows) in enumerate(targets, 1):
        page = pages.get(page_id)
        if page is None:
            # Pagination is deterministic, so a missing page means the source
            # package changed since the row was written. Judging against a
            # page the row did not come from would be worse than leaving it.
            report(f"  skip {page_id}: no longer produced by pagination")
            continue

        pairs = [generate_stage.RawPair(
            question=r.question, reasoning=r.reasoning, answer=r.answer,
            support=r.support, grounding=r.grounding,
            reasoning_depth=r.reasoning_depth, page=page) for r in page_rows]

        verdicts = verify_stage.verify(client, cfg.models.verifier, pairs,
                                       page.id, page.text, cfg.verification, templates)
        for row, verdict in zip(page_rows, verdicts):
            if verdict.verdict == UNCHECKED:
                continue
            row.verdict = verdict.verdict
            row.checks = {"span": verdict.span, "entailment": verdict.entailment,
                          "reasoning": verdict.reasoning, "execution": None}
            row.reason = verdict.reason
            row.verifier = verdict.model or cfg.models.verifier.id
            index[row.id] = row
            recovered += 1

        if i % 5 == 0 or i == len(targets):
            report(f"[{i:>4}/{len(targets)}] {recovered:,} rows now judged")

    path = Path(cfg.output.corpus_path)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    append_rows(tmp, [index[r.id] for r in rows])
    tmp.replace(path)

    report(f"\nrecovered {recovered:,} of {len(stale):,} unchecked rows")
    report(f"teacher: {client.stats()}")
    return finalise(cfg, report)
