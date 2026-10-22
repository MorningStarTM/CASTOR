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
    vocab_size: int

    def ok(self, max_tokens_per_name: int = 4) -> bool:
        return self.digits_are_atomic and self.worst_token_count <= max_tokens_per_name

    def report(self) -> str:
        verdict = "PASS" if self.ok() else "FAIL"
        return (
            f"[{verdict}] vocab={self.vocab_size} "
            f"digits_atomic={self.digits_are_atomic} "
            f"mean_tokens_per_name={self.mean_token_count:.2f} "
            f"worst={self.worst_name!r} at {self.worst_token_count} tokens"
        )


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
    """Names in the shape castor.naming emits, spanning the three grid sizes."""
    names: list[str] = []
    for i in (0, 7, 12, 59, 118, 186):
        names += [f"line_{i}", f"gen_{i}", f"load_{i}", f"sub_{i}"]
    names += ["line_12 to line_13", "rho=0.97", "rho 1.04", "bus 2 at sub_4"]
    return names


def audit_tokenizer(tok, names: list[str] | None = None) -> TokenizerAudit:
    """Measure what the tokenizer does to grid element names and rho values."""
    names = names or _grid_name_samples()

    counts = {n: len(tok.encode(n, add_special_tokens=False)) for n in names}
    worst = max(counts, key=counts.get)

    # Atomic digits: every digit run tokenizes to exactly one token per digit,
    # so numerically adjacent element ids stay in adjacent token space.
    digits_atomic = all(
        len(tok.encode(str(n), add_special_tokens=False)) == len(str(n))
        for n in (7, 12, 42, 118, 1234)
    )

    return TokenizerAudit(
        worst_name=worst,
        worst_token_count=counts[worst],
        mean_token_count=sum(counts.values()) / len(counts),
        digits_are_atomic=digits_atomic,
        vocab_size=len(tok),
    )


def load_tokenizer_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
