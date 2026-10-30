"""The dataset row, and the JSON lines it is stored as.

One row is one training example plus everything needed to audit it later: the
span it rests on, the page it came from, the symbols that were retrieved for
that page, and the verdict of every check that applied.

Rows that fail are written with their verdict rather than dropped. The corpus
then states its own error rate, which is a number the paper can quote instead
of a claim it has to make. Training filters on verdict == PASS.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# Verdicts
PASS = "pass"
FAIL_SPAN = "fail_span"
FAIL_ENTAILMENT = "fail_entailment"
FAIL_REASONING = "fail_reasoning"
FAIL_EXEC = "fail_exec"
FAIL_PARSE = "fail_parse"
FAIL_AUDIT = "fail_audit"
DUPLICATE = "duplicate"
UNCHECKED = "unchecked"

# Grounding
EXECUTED = "executed"
EXTRACTIVE = "extractive"
GENERATIVE = "generative"

# Source
API = "grid2op.api"
DOCS = "grid2op.docs"
EXEC = "grid2op.exec"


@dataclass
class Page:
    """One unit of work: a class with its methods, or a module's loose parts."""

    id: str
    text: str
    source: str = API
    refs: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class PageRecord:
    """What happened to one page. Written to pages.jsonl.

    Exists so a page that produced nothing can be told apart from a page that
    was never attempted: the difference between a thin docstring and a run
    that was interrupted.
    """

    id: str
    chars: int
    refs: list[str] = field(default_factory=list)
    n_pairs: int = 0
    n_pass: int = 0
    n_fail: int = 0
    note: str = ""

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QARow:
    """One dataset row. Field order here is the field order on disk."""

    id: str
    question: str
    reasoning: str
    answer: str
    support: str
    grounding: str
    reasoning_depth: int
    verdict: str
    checks: dict[str, bool | None]
    reason: str
    page: str
    refs: list[str]
    source: str
    model: str
    verifier: str

    @staticmethod
    def new_checks() -> dict[str, bool | None]:
        return {"span": None, "entailment": None, "reasoning": None, "execution": None}

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return self.verdict == PASS

    def as_chat(self, system: str, reasoning_open: str = "<reasoning>",
                reasoning_close: str = "</reasoning>") -> dict[str, Any]:
        """The training view. Applied at load time, never stored.

        A row with no reasoning trains as a plain question and answer with no
        tags. Mixing the two is deliberate: a student trained only on
        derivations learns to produce a paragraph of deliberation before
        answering "186", which is the wrong behaviour for a lookup.
        """
        if self.reasoning:
            content = f"{reasoning_open}\n{self.reasoning}\n{reasoning_close}\n\n{self.answer}"
        else:
            content = self.answer
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": content},
            ]
        }


def make_id(n: int, prefix: str = "g2o") -> str:
    return f"{prefix}-{n:06d}"


# --------------------------------------------------------------------- JSONL


def append_rows(path: str | Path, rows: Iterable[QARow]) -> int:
    """Append and fsync, so the file on disk is always complete to a line."""
    return _append(path, (r.as_json() for r in rows))


def append_pages(path: str | Path, records: Iterable[PageRecord]) -> int:
    return _append(path, (r.as_json() for r in records))


def _append(path: str | Path, payloads: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def read_rows(path: str | Path) -> Iterator[QARow]:
    """Rows from disk. A corrupt line is skipped, not fatal."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield QARow(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue


def done_pages(path: str | Path) -> set[str]:
    """Page ids already written, so a resumed run does not pay for them twice."""
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def next_row_number(path: str | Path) -> int:
    """How many rows are already on disk, so ids continue rather than collide."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def summarise(path: str | Path) -> dict[str, Any]:
    """Counts the dataset card needs, and that a human should look at."""
    totals: dict[str, Any] = {
        "rows": 0, "pages": set(), "verdict": {}, "grounding": {},
        "source": {}, "depth": {},
    }
    for row in read_rows(path):
        totals["rows"] += 1
        totals["pages"].add(row.page)
        for key, value in (("verdict", row.verdict), ("grounding", row.grounding),
                           ("source", row.source), ("depth", row.reasoning_depth)):
            totals[key][value] = totals[key].get(value, 0) + 1
    totals["pages"] = len(totals["pages"])
    passed = totals["verdict"].get(PASS, 0)
    totals["pass_rate"] = round(passed / totals["rows"], 4) if totals["rows"] else 0.0
    return totals
