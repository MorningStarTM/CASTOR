"""Stage 4: generation.

One request per page, or per bundle of small pages. At most max_pairs_per_page
pairs, and the prompt states that fewer is correct when the page is thin: a
model asked for ten pairs from four sentences will manufacture six.

Two things make the contract checkable. Each pair quotes a verbatim span from
its page, and each pair carries its derivation separately from its conclusion.
A span can be looked for; a derivation can be judged step by step. A
self-assigned grounding label can be neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .augment import Augmented
from .client import Client, extract_json
from .config import GenerationConfig, ModelConfig
from .schema import EXTRACTIVE, GENERATIVE, Page

_GROUNDINGS = {EXTRACTIVE, GENERATIVE}


@dataclass
class RawPair:
    """What the generator claimed, before anything checked it."""

    question: str
    reasoning: str
    answer: str
    support: str
    grounding: str
    reasoning_depth: int
    page: Page


@dataclass
class GenerateResult:
    pairs: list[RawPair] = field(default_factory=list)
    parsed: bool = True
    note: str = ""


def build_prompt(items: list[Augmented], cfg: GenerationConfig,
                 templates: dict[str, str]) -> str:
    """One prompt for one request. Several pages are delimited, not merged."""
    targets = cfg.depth_targets
    if len(items) == 1:
        item = items[0]
        return templates["generate"].format(
            n_pairs=cfg.max_pairs_per_page,
            page_id=item.page.id,
            page_text=item.page.text,
            references=item.references,
            depth_lookup=targets.lookup,
            depth_single=targets.single,
            depth_multi=targets.multi,
        )

    # A bundle. Each page keeps its own id and its own reference block, so a
    # pair can still be attributed to the page it came from.
    merged_text = "\n\n".join(
        f"[PAGE {i + 1}: {it.page.id}]\n{it.page.text}" for i, it in enumerate(items)
    )
    merged_refs = "\n\n".join(
        f"[FOR PAGE {i + 1}: {it.page.id}]\n{it.references}" for i, it in enumerate(items)
    )
    ids = ", ".join(it.page.id for it in items)
    prompt = templates["generate"].format(
        n_pairs=cfg.max_pairs_per_page * len(items),
        page_id=ids,
        page_text=merged_text,
        references=merged_refs,
        depth_lookup=targets.lookup,
        depth_single=targets.single,
        depth_multi=targets.multi,
    )
    return prompt + (
        '\n\nThis request carries several pages. Add a "page" field to each pair '
        'holding the id of the page it came from, exactly as written in its '
        '[PAGE n: id] header.\n'
    )


def parse(payload: dict, items: list[Augmented], cfg: GenerationConfig) -> list[RawPair]:
    """Turn a parsed reply into pairs, discarding anything malformed."""
    by_id = {it.page.id: it.page for it in items}
    default_page = items[0].page

    out: list[RawPair] = []
    limit = cfg.max_pairs_per_page * len(items)
    for item in payload.get("pairs", []):
        if not isinstance(item, dict) or len(out) >= limit:
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        grounding = str(item.get("grounding", GENERATIVE)).strip().lower()
        if grounding not in _GROUNDINGS:
            grounding = GENERATIVE

        try:
            depth = int(item.get("reasoning_depth", 0))
        except (TypeError, ValueError):
            depth = 0
        reasoning = str(item.get("reasoning", "")).strip()
        # A depth the reasoning does not support is a claim, not a fact.
        depth = 0 if not reasoning else max(1, depth)

        page = by_id.get(str(item.get("page", "")).strip(), default_page)
        out.append(RawPair(
            question=question,
            reasoning=reasoning,
            answer=answer,
            support=str(item.get("support", "")).strip(),
            grounding=grounding,
            reasoning_depth=depth,
            page=page,
        ))
    return out


def generate(client: Client, model: ModelConfig, items: list[Augmented],
             cfg: GenerationConfig, templates: dict[str, str]) -> GenerateResult:
    """Pairs for one request. Raises only on an API error worth stopping for."""
    prompt = build_prompt(items, cfg, templates)
    reply = client.complete(model, prompt)
    payload = extract_json(reply)
    if payload is None:
        return GenerateResult(parsed=False, note="reply did not contain parseable JSON")
    pairs = parse(payload, items, cfg)
    if not pairs:
        return GenerateResult(parsed=False, note="reply parsed but carried no usable pairs")
    return GenerateResult(pairs=pairs)
