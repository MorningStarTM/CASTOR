"""Stage 6: deduplication.

272 pages describing similar methods produce similar questions. "What does
BaseAction.set_bus do" and "What does PlayableAction.set_bus do" may differ by
one word and carry the same answer.

MinHash over character shingles, implemented here rather than taken from
datasketch: the whole algorithm is twenty lines and the repository already
avoids a dependency for a single call site.

A surviving row keeps the better verdict, so deduplication never silently
demotes a verified pair in favour of an unchecked near-duplicate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .config import DedupConfig
from .schema import PASS, QARow

# Verdict preference when two near-duplicates collide. Higher wins.
_RANK = {PASS: 3}
_DEFAULT_RANK = 1
_UNCHECKED_RANK = 2


def _rank(verdict: str) -> int:
    if verdict in _RANK:
        return _RANK[verdict]
    return _UNCHECKED_RANK if verdict == "unchecked" else _DEFAULT_RANK


def shingles(text: str, size: int) -> set[str]:
    """Character n-grams of the normalised question."""
    normalised = " ".join(text.lower().split())
    if len(normalised) <= size:
        return {normalised} if normalised else set()
    return {normalised[i : i + size] for i in range(len(normalised) - size + 1)}


def signature(items: set[str], num_permutations: int) -> tuple[int, ...]:
    """MinHash signature.

    Each permutation is a distinct hash of the same shingle, taken by salting
    with the permutation index. The minimum over the set is the sketch, and
    the fraction of matching minima estimates Jaccard similarity.
    """
    if not items:
        return tuple([0] * num_permutations)
    out = []
    for perm in range(num_permutations):
        salt = str(perm).encode()
        out.append(min(
            int.from_bytes(hashlib.blake2b(salt + s.encode("utf-8", "ignore"),
                                           digest_size=8).digest(), "big")
            for s in items
        ))
    return tuple(out)


def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


@dataclass
class DedupResult:
    kept: list[QARow]
    dropped: list[QARow]

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)


def deduplicate(rows: list[QARow], cfg: DedupConfig) -> DedupResult:
    """Collapse near-duplicate questions, keeping the better verdict.

    Banding is used so this stays linear in practice: signatures are bucketed
    on fixed slices, and only rows sharing a bucket are compared. Comparing
    every pair would be quadratic and, at a few thousand rows, pointless work.
    """
    if not cfg.enabled or len(rows) < 2:
        return DedupResult(kept=list(rows), dropped=[])

    sigs = [signature(shingles(r.question, cfg.shingle_size), cfg.num_permutations)
            for r in rows]
    # Answers are sketched too. Questions that differ by a few characters can
    # carry entirely different facts: "the shape of obs.rho" and "the shape of
    # obs.gen_p" are 93% similar as strings and have different answers.
    # Collapsing those would silently delete one of them, so a pair counts as
    # duplicate only when the question AND the answer both match.
    answer_sigs = [signature(shingles(r.answer, cfg.shingle_size), cfg.num_permutations)
                   for r in rows]

    band_size = max(1, cfg.num_permutations // 16)
    buckets: dict[tuple, list[int]] = {}
    for i, sig in enumerate(sigs):
        for start in range(0, len(sig), band_size):
            buckets.setdefault((start, sig[start : start + band_size]), []).append(i)

    dropped: set[int] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for pos, i in enumerate(members):
            if i in dropped:
                continue
            for j in members[pos + 1 :]:
                if j in dropped:
                    continue
                if similarity(sigs[i], sigs[j]) < cfg.jaccard_threshold:
                    continue
                if similarity(answer_sigs[i], answer_sigs[j]) < cfg.jaccard_threshold:
                    continue
                loser = j if _rank(rows[i].verdict) >= _rank(rows[j].verdict) else i
                dropped.add(loser)
                if loser == i:
                    break

    kept = [r for i, r in enumerate(rows) if i not in dropped]
    removed = [r for i, r in enumerate(rows) if i in dropped]
    return DedupResult(kept=kept, dropped=removed)
