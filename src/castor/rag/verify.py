"""Stage 5: verification.

Three checks run here, cheapest first, and a fourth (execution) lives in the
executed generator where the simulator is already open.

    span        does the quoted span appear in the evidence?    no request
    entailment  is the answer supported by the evidence?        shares one call
    reasoning   is every step of the derivation supported?      shares one call

"Evidence" is everything the generator was shown: the page, and the
definitions of the symbols that page refers to. It used to mean the page
alone, which was wrong in a way that quietly failed good rows. The generator
is told an answer may rest on a referenced definition, so a pair that quotes
one is doing as it was asked; checking that quote against the page only, and
showing the judge only the page, failed it twice over. Of 49 span failures in
one trial, 22 were verbatim quotes of definitions that had been supplied.

Both gates therefore see the same evidence, and the span check records which
part of it matched, so a row grounded in a reference is distinguishable from
one grounded in the page itself rather than merely accepted.

The answer and the reasoning are judged separately and on purpose. A correct
answer reached by an unsupported step is worse than a wrong answer: a wrong
answer is one bad fact, while a wrong derivation is a bad rule the student
will apply again to inputs it has never seen. Those rows fail on reasoning
despite being right.

The verifier is a different model family from the generator. A model asked to
check its own output agrees with itself, so the check would otherwise measure
nothing. What stays hidden from the judge is how the pair was produced - the
generator's prompt and its grounding claim - not the text the answer rests on,
which it cannot judge without.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass

from .client import Client, QuotaExhausted, extract_json
from .config import ModelConfig, VerificationConfig
from .generate import RawPair
from .schema import (
    FAIL_ENTAILMENT, FAIL_REASONING, FAIL_SPAN, PASS, SPAN_NONE, SPAN_PAGE,
    SPAN_REFERENCE, UNCHECKED,
)

_WS = re.compile(r"\s+")


@dataclass
class Verdict:
    verdict: str = UNCHECKED
    span: bool | None = None
    entailment: bool | None = None
    reasoning: bool | None = None
    reason: str = ""
    # Which model actually judged this pair. Not the configured primary: when
    # that one is rate-limited the chain falls through to another vendor, and
    # a row claiming to have been judged by a model that never saw it would
    # make the whole corpus unauditable.
    model: str = ""
    # Where the span was found: SPAN_PAGE, SPAN_REFERENCE or SPAN_NONE. A
    # corpus that accepts reference-grounded rows without saying so cannot
    # report how much of itself rests on the page it names.
    span_source: str = SPAN_NONE


def _normalise(text: str, enabled: bool) -> str:
    text = text.strip().lower()
    return _WS.sub(" ", text) if enabled else text


def span_ok(support: str, text: str, cfg: VerificationConfig) -> bool:
    """Is the quoted span really in this text?

    An exact containment test first. Models trim articles and collapse
    newlines when copying, so a near miss falls back to the longest matching
    block as a fraction of the quote's length. A fabricated quote scores far
    below the threshold; a slightly trimmed real one scores just under 1.0.
    """
    if not support:
        return False
    needle = _normalise(support, cfg.normalise_whitespace)
    haystack = _normalise(text, cfg.normalise_whitespace)
    if needle in haystack:
        return True
    if len(needle) < 12:          # too short to match meaningfully
        return False
    matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
    match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    return (match.size / len(needle)) >= cfg.span_min_ratio


def span_match(support: str, page_text: str, references: str,
               cfg: VerificationConfig) -> tuple[bool, str]:
    """(found, which part of the evidence it was found in).

    The page is tried first, so a span present in both is attributed to the
    page. That ordering matters for the corpus statistics: a reference block
    repeats text from the pages it was drawn from, and counting those rows as
    reference-grounded would overstate how much of the corpus depends on the
    retrieval layer.
    """
    if span_ok(support, page_text, cfg):
        return True, SPAN_PAGE
    if references and span_ok(support, references, cfg):
        return True, SPAN_REFERENCE
    return False, SPAN_NONE


_ENTRY = re.compile(r"^--- (.+?) ---", re.MULTILINE)


def trim_references(references: str, pairs: list[RawPair],
                    cfg: VerificationConfig) -> str:
    """Only the definitions this chunk of pairs actually touches.

    The generator is given every reference a page resolves, because it cannot
    know in advance which one a good question will need. The judge is given
    the pairs, so it can: a definition no pair quotes or names is prompt
    weight that buys nothing.

    This is not an optimisation for its own sake. Verification of the four
    largest pages failed outright when the prompt was the whole page plus
    forty pairs, and adding the full reference block back would have made the
    prompt larger than the one that failed - 37.7k characters against the 32k
    that broke. Chunking alone did not protect against that, because the page
    and the references are per-page, not per-chunk.
    """
    if not references:
        return ""
    bounds = [m.start() for m in _ENTRY.finditer(references)]
    if not bounds:
        return references[:cfg.max_reference_chars]

    bounds.append(len(references))
    entries = [references[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

    # A definition is in play if a pair quotes it, or names it anywhere in the
    # question, the derivation or the answer.
    haystack = _normalise(
        " ".join(f"{p.question} {p.reasoning} {p.answer} {p.support}" for p in pairs),
        cfg.normalise_whitespace)
    # Two tiers, and the order matters. An entry a pair actually quotes is
    # the evidence for that pair's span: drop it and the judge is asked to
    # rule on a quote it cannot see, which is the bug this whole change
    # exists to fix. An entry merely named is context, and is the first thing
    # to give up when the budget runs out.
    quoted: list[tuple[int, str]] = []
    named: list[tuple[int, str]] = []
    for i, entry in enumerate(entries):
        match = _ENTRY.match(entry)
        name = _normalise(match.group(1), cfg.normalise_whitespace) if match else ""
        if _overlaps(entry, pairs, cfg):
            quoted.append((i, entry))
        elif name and name.split(".")[-1] in haystack:
            named.append((i, entry))

    kept = list(quoted)
    total = sum(len(e) for _, e in kept)
    for i, entry in named:
        # continue, not break: one oversized entry must not shut out the
        # smaller ones behind it.
        if total + len(entry) > cfg.max_reference_chars:
            continue
        kept.append((i, entry))
        total += len(entry)

    if not kept:
        return "(no referenced definition is quoted or named by these pairs)"
    # Back into the order the generator saw them in.
    return "".join(entry for _, entry in sorted(kept))


def _overlaps(entry: str, pairs: list[RawPair], cfg: VerificationConfig) -> bool:
    """Does any pair's span come from this entry?"""
    return any(p.support and span_ok(p.support, entry, cfg) for p in pairs)


def build_prompt(page_id: str, page_text: str, pairs: list[RawPair],
                 templates: dict[str, str], references: str = "",
                 cfg: VerificationConfig | None = None) -> str:
    """The evidence and the pairs. Not how the pairs were produced.

    The judge sees what the generator saw, because an answer resting on a
    referenced definition cannot be judged against the page alone. It does not
    see the generator's prompt, its grounding label, or its reasoning about
    why it wrote what it wrote; withholding those is what stops the verifier
    being led down the path that produced the error.
    """
    payload = [
        {"index": i, "question": p.question, "reasoning": p.reasoning, "answer": p.answer}
        for i, p in enumerate(pairs)
    ]
    shown = trim_references(references, pairs, cfg) if cfg else references
    return templates["verify"].format(
        page_id=page_id,
        page_text=page_text,
        references=shown or "(none supplied for this page)",
        pairs_json=json.dumps(payload, ensure_ascii=False, indent=1),
    )


def _decide(span: bool, answer_ok: bool, reasoning_ok: bool, reason: str,
            model: str = "", span_source: str = SPAN_NONE) -> Verdict:
    if not span:
        return Verdict(FAIL_SPAN, span, answer_ok, reasoning_ok,
                       reason or "quoted span not found in the page or its references",
                       model, span_source)
    if not answer_ok:
        return Verdict(FAIL_ENTAILMENT, span, answer_ok, reasoning_ok,
                       reason or "answer not supported by the evidence",
                       model, span_source)
    if not reasoning_ok:
        return Verdict(FAIL_REASONING, span, answer_ok, reasoning_ok,
                       reason or "a reasoning step is not supported by the evidence",
                       model, span_source)
    return Verdict(PASS, span, answer_ok, reasoning_ok, "", model, span_source)


def _judge_chunk(client: Client, model: ModelConfig, chunk: list[RawPair],
                 page_id: str, page_text: str, cfg: VerificationConfig,
                 templates: dict[str, str],
                 references: str = "") -> tuple[dict[int, dict], str, str]:
    """({position within chunk: verdict item}, model used, failure reason)."""
    reply, used = client.complete_any(
        model, build_prompt(page_id, page_text, chunk, templates, references, cfg))
    payload = extract_json(reply)
    if payload is None:
        # Record which model produced the unusable reply: a fallback that
        # reliably cannot emit JSON should be visible in the corpus rather
        # than hidden behind an anonymous "unchecked".
        return {}, used, "verifier reply did not parse"

    by_index: dict[int, dict] = {}
    for item in payload.get("verdicts", []):
        if isinstance(item, dict):
            try:
                by_index[int(item.get("index", -1))] = item
            except (TypeError, ValueError):
                continue
    return by_index, used, ""


def verify(client: Client, model: ModelConfig, pairs: list[RawPair],
           page_id: str, page_text: str, cfg: VerificationConfig,
           templates: dict[str, str], references: str = "") -> list[Verdict]:
    """A verdict per pair, in the order given.

    The pairs are judged in chunks. A large page asked for forty pairs is a
    prompt of the whole page plus forty items to rule on, and the free
    endpoints silently fail to answer at that size: every row from the four
    biggest pages of a trial run came back unchecked while every small page
    was judged. Chunking spends more requests on exactly the pages whose
    rows are worth the most.

    Never raises on a judge failure, with one exception. A spent allowance
    propagates, because marking the rows unchecked and returning would let
    the caller walk to the next page and spend the whole fallback chain again
    on an allowance that is already gone. If some chunks were judged before
    that happened, their verdicts are kept and the run stops at the next
    page's generate call instead.
    """
    matches = [span_match(p.support, p.page.text or page_text, references, cfg)
               for p in pairs]

    if not cfg.enabled:
        return [Verdict(UNCHECKED, ok, None, None, "verification disabled",
                        "", src) for ok, src in matches]

    size = max(1, cfg.max_pairs_per_request)
    items: dict[int, dict] = {}
    reasons: dict[int, str] = {}
    used = ""
    judged_any = False

    for start in range(0, len(pairs), size):
        chunk = pairs[start:start + size]
        try:
            found, used_now, reason = _judge_chunk(
                client, model, chunk, page_id, page_text, cfg, templates, references)
        except QuotaExhausted:
            if not judged_any:
                raise
            for n in range(start, len(pairs)):
                reasons[n] = "daily allowance spent before this chunk was judged"
            break
        except Exception as exc:   # noqa: BLE001 - a judge failing must not stop the run
            for n in range(start, min(start + size, len(pairs))):
                reasons[n] = f"verifier unavailable: {type(exc).__name__}"
            continue

        used = used_now or used
        if reason:
            for n in range(start, start + len(chunk)):
                reasons[n] = reason
            continue

        judged_any = True
        # Indices are relative to the chunk the judge was shown, so they are
        # offset back onto the page's own numbering before being merged.
        for local, item in found.items():
            if 0 <= local < len(chunk):
                items[start + local] = item

    out: list[Verdict] = []
    for i, (span, source) in enumerate(matches):
        item = items.get(i)
        if item is None:
            out.append(Verdict(UNCHECKED, span, None, None,
                               reasons.get(i, "verifier returned no verdict"),
                               used, source))
            continue
        answer_ok = bool(item.get("answer_supported", False))
        # An empty derivation has nothing to check, so it cannot fail.
        reasoning_ok = (not pairs[i].reasoning) or bool(item.get("reasoning_supported", False))
        out.append(_decide(span, answer_ok, reasoning_ok,
                           str(item.get("reason", "")).strip(), used, source))
    return out
