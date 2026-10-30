"""Stage 7: the leak audit.

The QA corpus is text the student trains on. A held-out environment named in a
question or an answer reaches the model through the language channel, and
every transfer number in the paper becomes unsupportable. The audit is short
and non-negotiable, and it runs as an assertion rather than a warning.

It checks the question, the answer, the reasoning and the support span. The
page id is deliberately included too: a page named after a held-out
environment should never have been paginated in the first place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import AuditConfig
from .schema import FAIL_AUDIT, QARow


class LeakError(RuntimeError):
    """A held-out environment reached the corpus."""


@dataclass
class AuditResult:
    clean: list[QARow] = field(default_factory=list)
    violations: list[tuple[QARow, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def _patterns(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    # Word boundaries so "case14" does not fire on "case140", but the terms
    # carry underscores and digits, so \b alone is not enough on its own.
    return [(t, re.compile(rf"(?<![\w]){re.escape(t)}(?![\w])", re.IGNORECASE))
            for t in terms]


def audit(rows: list[QARow], cfg: AuditConfig) -> AuditResult:
    """Split rows into clean and leaking. Marks leakers, never deletes them."""
    terms = cfg.terms
    if not terms:
        return AuditResult(clean=list(rows))

    compiled = _patterns(terms)
    result = AuditResult()
    for row in rows:
        haystack = " ".join((row.question, row.answer, row.reasoning, row.support, row.page))
        hit = next((term for term, pattern in compiled if pattern.search(haystack)), None)
        if hit is None:
            result.clean.append(row)
            continue
        row.verdict = FAIL_AUDIT
        row.reason = f"names held-out environment {hit!r}"
        result.violations.append((row, hit))
    return result


def enforce(result: AuditResult, cfg: AuditConfig) -> None:
    """Raise when configured to, after the rows have been written.

    Writing first and raising second is intentional: the operator should be
    able to read exactly which rows leaked, not just be told that some did.
    """
    if result.ok or not cfg.fail_on_violation:
        return
    sample = "\n".join(f"  {row.id}  [{term}]  {row.question[:70]}"
                       for row, term in result.violations[:10])
    raise LeakError(
        f"{len(result.violations)} row(s) name a held-out environment. They are "
        f"written with verdict={FAIL_AUDIT} and must not be trained on:\n{sample}"
    )
