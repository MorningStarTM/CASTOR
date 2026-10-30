"""Stage 3: augmentation. The retrieval step.

Measured on grid2op 1.12.1: 60% of docstrings contain a cross-reference, 2.4
of them on average, 2,201 in total. Each is a pointer to a definition the
teacher was never given. A page saying "see BaseAction.set_bus" without
carrying that definition forces the teacher to reconstruct it, and whatever it
reconstructs becomes an answer stated as confidently as something it read.

This module resolves those pointers and attaches the definitions. A name that
cannot be resolved is labelled external and the teacher is told not to
describe it, rather than being left to guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import AugmentConfig
from .index import SymbolIndex
from .schema import Page


@dataclass
class Augmented:
    """A page plus the definitions it points at."""

    page: Page
    references: str                       # rendered block for the prompt
    resolved: list[str] = field(default_factory=list)
    external: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def refs(self) -> list[str]:
        return self.resolved


def extract_refs(text: str, cfg: AugmentConfig) -> list[str]:
    """Referenced names from raw, uncleaned text, most frequent first.

    Pagination harvests these before clean() removes the markup and stores
    them on the Page, so this is a fallback for callers holding text that
    still carries its roles. A page whose refs are already populated does not
    come back through here.
    """
    pattern = re.compile(cfg.role_pattern)
    counts: dict[str, int] = {}
    for raw in pattern.findall(text):
        name = raw.strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return sorted(counts, key=lambda n: (-counts[n], n))


def is_external(name: str, cfg: AugmentConfig) -> bool:
    head = name.split(".")[0].lower()
    return head in {p.lower() for p in cfg.external_prefixes}


def augment(page: Page, index: SymbolIndex, cfg: AugmentConfig,
            templates: dict[str, str], bare_fallback: bool = False) -> Augmented:
    """Resolve the page's references and render them for the prompt."""
    # Pagination collected these from the markup before it was stripped. Only
    # fall back to scanning if a caller built the Page by hand.
    names = page.refs or extract_refs(page.text, cfg)

    blocks: list[str] = []
    resolved: list[str] = []
    external: list[str] = []
    unresolved: list[str] = []
    total = 0

    for name in names:
        if len(resolved) + len(external) >= cfg.max_refs_per_page:
            break
        if total >= cfg.max_total_ref_chars:
            break

        if is_external(name, cfg):
            blocks.append(templates["reference_external"].format(name=name))
            external.append(name)
            continue

        hit = index.lookup(name, bare_fallback_for_qualified=bare_fallback)
        if hit is None:
            unresolved.append(name)
            continue

        resolved_name, doc = hit
        # Two different reference spellings can resolve to the same symbol.
        if resolved_name in resolved:
            continue
        # Do not re-send what the page already contains.
        if doc[:200] and doc[:200] in page.text:
            continue
        budget = min(cfg.max_chars_per_ref, cfg.max_total_ref_chars - total)
        snippet = doc[:budget].rstrip()
        if len(doc) > budget:
            snippet += " ..."
        blocks.append(templates["reference_entry"].format(name=resolved_name, text=snippet))
        resolved.append(resolved_name)
        total += len(snippet)

    rendered = "\n".join(blocks) if blocks else templates["reference_none"]
    return Augmented(page=page, references=rendered, resolved=resolved,
                     external=external, unresolved=unresolved)
