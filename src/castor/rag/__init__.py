"""Phase 2 QA corpus: retrieval-augmented generation with verification.

Seven stages, each its own module, each driven by configs/rag/rag.yaml:

    pages      stage 1   class-level pagination
    index      stage 2   symbol index, exact-name lookup
    augment    stage 3   cross-reference resolution
    generate   stage 4   the teacher writes pairs
    verify     stage 5   span, entailment and reasoning checks
    dedup      stage 6   MinHash over question shingles
    audit      stage 7   held-out environment leak check

    executed             simulator-grounded rows, no teacher
    pipeline             the seven wired together
    client               OpenRouter
    schema               the dataset row
    config               YAML to dataclasses

Nothing here hard-codes a model, a threshold, a prompt or a path.
"""

from .config import RagConfig, load_config, load_prompts
from .pipeline import Plan, finalise, plan, run
from .schema import QARow, Page, PageRecord, read_rows, summarise

__all__ = [
    "RagConfig", "load_config", "load_prompts",
    "Plan", "plan", "run", "finalise",
    "QARow", "Page", "PageRecord", "read_rows", "summarise",
]
