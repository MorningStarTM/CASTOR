"""Stage 5: verification.

Three checks run here, cheapest first, and a fourth (execution) lives in the
executed generator where the simulator is already open.

    span        does the quoted span appear on the page?      no request
    entailment  is the answer supported by the page?          shares one call
    reasoning   is every step of the derivation supported?    shares one call

The answer and the reasoning are judged separately and on purpose. A correct
answer reached by an unsupported step is worse than a wrong answer: a wrong
answer is one bad fact, while a wrong derivation is a bad rule the student
will apply again to inputs it has never seen. Those rows fail on reasoning
despite being right.

The verifier is a different model family from the generator. A model asked to
check its own output agrees with itself, so the check would otherwise measure
nothing.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass

from .client import Client, extract_json
from .config import ModelConfig, VerificationConfig
from .generate import RawPair
from .schema import (
    FAIL_ENTAILMENT, FAIL_REASONING, FAIL_SPAN, PASS, UNCHECKED,
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


def _normalise(text: str, enabled: bool) -> str:
    text = text.strip().lower()
    return _WS.sub(" ", text) if enabled else text


def span_ok(support: str, page_text: str, cfg: VerificationConfig) -> bool:
    """Is the quoted span really on the page?

    An exact containment test first. Models trim articles and collapse
    newlines when copying, so a near miss falls back to the longest matching
    block as a fraction of the quote's length. A fabricated quote scores far
    below the threshold; a slightly trimmed real one scores just under 1.0.
    """
    if not support:
        return False
    needle = _normalise(support, cfg.normalise_whitespace)
    haystack = _normalise(page_text, cfg.normalise_whitespace)
    if needle in haystack:
        return True
    if len(needle) < 12:          # too short to match meaningfully
        return False
    matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
    match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    return (match.size / len(needle)) >= cfg.span_min_ratio


def build_prompt(page_id: str, page_text: str, pairs: list[RawPair],
                 templates: dict[str, str]) -> str:
    """The verifier sees the page and the pairs. Nothing else.

    Not the augmented references, not the generator's prompt, not its
    reasoning about why it wrote what it wrote. Withholding those is what
    stops the verifier being led down the path that produced the error.
    """
    payload = [
        {"index": i, "question": p.question, "reasoning": p.reasoning, "answer": p.answer}
        for i, p in enumerate(pairs)
    ]
    return templates["verify"].format(
        page_id=page_id,
        page_text=page_text,
        pairs_json=json.dumps(payload, ensure_ascii=False, indent=1),
    )


def _decide(span: bool, answer_ok: bool, reasoning_ok: bool, reason: str,
            model: str = "") -> Verdict:
    if not span:
        return Verdict(FAIL_SPAN, span, answer_ok, reasoning_ok,
                       reason or "quoted span not found on the page", model)
    if not answer_ok:
        return Verdict(FAIL_ENTAILMENT, span, answer_ok, reasoning_ok,
                       reason or "answer not supported by the page", model)
    if not reasoning_ok:
        return Verdict(FAIL_REASONING, span, answer_ok, reasoning_ok,
                       reason or "a reasoning step is not supported by the page", model)
    return Verdict(PASS, span, answer_ok, reasoning_ok, "", model)


def verify(client: Client, model: ModelConfig, pairs: list[RawPair],
           page_id: str, page_text: str, cfg: VerificationConfig,
           templates: dict[str, str]) -> list[Verdict]:
    """A verdict per pair, in the order given.

    Never raises on a verifier failure. If the judge cannot be reached or its
    reply will not parse, the span check still stands and the rest is recorded
    as unchecked, which is honest. Discarding the pairs would throw away work
    the generator was already paid for.
    """
    spans = [span_ok(p.support, p.page.text or page_text, cfg) for p in pairs]

    if not cfg.enabled:
        return [Verdict(UNCHECKED, s, None, None, "verification disabled") for s in spans]

    try:
        reply, used = client.complete_any(
            model, build_prompt(page_id, page_text, pairs, templates))
        payload = extract_json(reply)
    except Exception as exc:   # noqa: BLE001 - a judge failing must not stop the run
        return [Verdict(UNCHECKED, s, None, None, f"verifier unavailable: {type(exc).__name__}")
                for s in spans]

    if payload is None:
        # Still record which model produced the unusable reply: a fallback
        # that reliably cannot emit JSON should be visible in the corpus
        # rather than hidden behind an anonymous "unchecked".
        return [Verdict(UNCHECKED, s, None, None, "verifier reply did not parse", used)
                for s in spans]

    by_index: dict[int, dict] = {}
    for item in payload.get("verdicts", []):
        if isinstance(item, dict):
            try:
                by_index[int(item.get("index", -1))] = item
            except (TypeError, ValueError):
                continue

    out: list[Verdict] = []
    for i, span in enumerate(spans):
        item = by_index.get(i)
        if item is None:
            out.append(Verdict(UNCHECKED, span, None, None,
                               "verifier returned no verdict", used))
            continue
        answer_ok = bool(item.get("answer_supported", False))
        # An empty derivation has nothing to check, so it cannot fail.
        reasoning_ok = (not pairs[i].reasoning) or bool(item.get("reasoning_supported", False))
        out.append(_decide(span, answer_ok, reasoning_ok,
                           str(item.get("reason", "")).strip(), used))
    return out
