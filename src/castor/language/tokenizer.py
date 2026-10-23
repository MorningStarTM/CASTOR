"""The pretrained tokenizer, and the audit that justifies the choice.

Default: mistralai/Mistral-7B-v0.1. A 32k byte-fallback BPE, Apache 2.0.

Three reasons, in order of how much they matter to CASTOR:

  1. Digits tokenize one per token. The model will spend its life reading
     `line_12`, `sub_4` and `rho=0.97`. GPT-2 style BPE merges digit runs
     inconsistently, so 118 and 119 can land in unrelated token spaces,
     which is exactly the wrong property for a grid with numbered elements.
  2. 32k vocabulary. See config.GPTConfig: at 50M parameters a 50k vocabulary
     puts half the model in the embedding table.
  3. Byte fallback. No UNK token ever, so an em dash in a blackout report or
     an unusual glyph in a grid code cannot silently become a hole.

The audit below is not ceremony. If element names tokenize into five pieces
each, Phase 3 grounding burns its context on fragments and the cross attention
has to relearn that those fragments denote one object. Run it before
tokenizing 6B tokens, not after.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Repositories carrying an equivalent 32k sentencepiece BPE, in fallback order.
# Used when the primary requires accepting terms the run cannot accept
# unattended.
FALLBACKS = (
    "mistralai/Mistral-7B-v0.1",
    "mistral-community/Mistral-7B-v0.1",
    "NousResearch/Llama-2-7b-hf",
)


@dataclass
class TokenizerAudit:
    """What the tokenizer does to the names castor.naming produces."""

    worst_name: str
    worst_token_count: int
    mean_token_count: float
    digits_are_atomic: bool
    stable_width: bool
    vocab_size: int
    phrases: dict[str, int] = None  # reported for context, never gated

    def ok(self, max_tokens_per_name: int = 6) -> bool:
        return (
            self.digits_are_atomic
            and self.stable_width
            and self.worst_token_count <= max_tokens_per_name
        )

    def report(self, max_tokens_per_name: int = 6) -> str:
        verdict = "PASS" if self.ok(max_tokens_per_name) else "FAIL"
        lines = [
            f"[{verdict}] vocab={self.vocab_size} "
            f"digits_atomic={self.digits_are_atomic} "
            f"stable_width={self.stable_width} "
            f"mean_tokens_per_name={self.mean_token_count:.2f} "
            f"worst={self.worst_name!r} at {self.worst_token_count} "
            f"(limit {max_tokens_per_name})"
        ]
        for phrase, count in (self.phrases or {}).items():
            lines.append(f"    {phrase!r}: {count} tokens")
        return "\n".join(lines)


def load_tokenizer(name: str | None = None, cache_dir: str | None = None):
    """Load the pretrained tokenizer, trying fallbacks on access failure."""
    from transformers import AutoTokenizer

    candidates = [name] if name else list(FALLBACKS)
    last: Exception | None = None
    for candidate in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(candidate, cache_dir=cache_dir)
        except Exception as exc:  # gated repo, offline, or renamed
            last = exc
            continue
        if tok.eos_token_id is None:
            raise ValueError(f"{candidate} has no eos token; cannot delimit documents")
        return tok
    raise RuntimeError(f"no tokenizer could be loaded from {candidates}: {last}")


def _grid_name_samples() -> list[str]:
    """Single element names, in the shape castor.naming emits.

    Only single names are gated. A phrase mentioning two elements is longer
    because it says more, and holding it to a per-name budget would be
    measuring the wrong thing.
    """
    names: list[str] = []
    for i in (0, 7, 12, 59, 118, 186):
        names += [f"line_{i}", f"gen_{i}", f"load_{i}", f"sub_{i}"]
    return names


def _phrase_samples() -> list[str]:
    return ["line_12 to line_13", "rho=0.97", "rho 1.04", "bus 2 at sub_4"]


def _count(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def _digits_are_atomic(tok) -> bool:
    """One token per digit, tested by growth rather than absolute length.

    A sentencepiece tokenizer prefixes a word-start marker, so encoding "12"
    gives three tokens, not two. Comparing an absolute count against the digit
    count therefore fails for every sentencepiece tokenizer regardless of how
    it actually treats digits. Measuring the increment from adding one digit
    sidesteps the marker entirely: if each extra digit costs exactly one more
    token, digits are atomic.
    """
    lengths = [_count(tok, f"line_{n}") for n in (1, 12, 123, 1234)]
    return all(b - a == 1 for a, b in zip(lengths, lengths[1:]))


def _width_is_stable(tok) -> bool:
    """Element ids of equal digit length cost equally many tokens.

    This is the property that matters for a grid: line_118 and line_119 must
    not land in token spaces of different shape, or the model has to learn
    each id as its own special case.
    """
    for group in ((10, 11, 42, 99), (100, 118, 119, 186)):
        counts = {_count(tok, f"line_{n}") for n in group}
        if len(counts) != 1:
            return False
    return True


def audit_tokenizer(tok, names: list[str] | None = None) -> TokenizerAudit:
    """Measure what the tokenizer does to grid element names and rho values."""
    names = names or _grid_name_samples()

    counts = {n: _count(tok, n) for n in names}
    worst = max(counts, key=counts.get)

    return TokenizerAudit(
        worst_name=worst,
        worst_token_count=counts[worst],
        mean_token_count=sum(counts.values()) / len(counts),
        digits_are_atomic=_digits_are_atomic(tok),
        stable_width=_width_is_stable(tok),
        vocab_size=len(tok),
        phrases={p: _count(tok, p) for p in _phrase_samples()},
    )


def load_tokenizer_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
