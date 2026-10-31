"""Configuration for the QA pipeline.

Every literal the pipeline depends on lives in configs/rag/rag.yaml and
configs/rag/prompts.yaml. The dataclasses here exist to give those files a
shape that fails loudly when a key is misspelled, rather than silently
defaulting and producing a corpus nobody can explain afterwards.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceConfig:
    package: str | None = None
    rst_dir: str | None = None
    exclude_path_substrings: list[str] = field(default_factory=lambda: ["test"])


@dataclass
class PagesConfig:
    max_chars: int = 24_000
    min_chars: int = 120
    bundle_under: int = 0
    bundle_max_chars: int = 40_000
    skip_internal: bool = True
    internal_markers: list[str] = field(default_factory=list)


@dataclass
class IndexConfig:
    include_bare_names: bool = True
    include_short_qualified: bool = True
    min_doc_chars: int = 40
    bare_fallback_for_qualified: bool = False


@dataclass
class AugmentConfig:
    role_pattern: str = r':[a-z:]+:`~?([^`<]+?)(?:\s*<[^>]+>)?`'
    max_refs_per_page: int = 12
    max_chars_per_ref: int = 1_200
    max_total_ref_chars: int = 10_000
    external_prefixes: list[str] = field(default_factory=list)


@dataclass
class ApiConfig:
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    env_file: str = ".env"
    requests_per_minute: int = 20
    timeout_s: float = 180.0
    max_retries: int = 4
    # A 429 is counted against the free daily allowance before it is
    # refused, so it gets a smaller budget than a timeout or a 503. Those
    # are free to retry; a 429 is not.
    max_retries_rate_limited: int = 2
    backoff_base_s: float = 5.0
    backoff_max_s: float = 60.0
    retry_status: list[int] = field(default_factory=lambda: [408, 429, 500, 502, 503, 504])
    referer: str = ""
    title: str = "CASTOR"


@dataclass
class ModelConfig:
    id: str = ""
    temperature: float = 0.3
    max_tokens: int = 8_192
    # Tried in order when id is rate-limited upstream. A :free endpoint is a
    # shared pool: saturation there is somebody else's traffic, not your
    # quota, and waiting it out can take hours. A second model from a
    # different provider is usually available immediately.
    fallbacks: list[str] = field(default_factory=list)


@dataclass
class ModelsConfig:
    generator: ModelConfig = field(default_factory=ModelConfig)
    verifier: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if isinstance(self.generator, dict):
            self.generator = ModelConfig(**self.generator)
        if isinstance(self.verifier, dict):
            self.verifier = ModelConfig(**self.verifier)


@dataclass
class DepthTargets:
    lookup: float = 0.35
    single: float = 0.40
    multi: float = 0.25


@dataclass
class GenerationConfig:
    max_pairs_per_page: int = 10
    depth_targets: DepthTargets = field(default_factory=DepthTargets)

    def __post_init__(self) -> None:
        if isinstance(self.depth_targets, dict):
            self.depth_targets = DepthTargets(**self.depth_targets)


@dataclass
class VerificationConfig:
    enabled: bool = True
    normalise_whitespace: bool = True
    span_min_ratio: float = 0.90


@dataclass
class DedupConfig:
    enabled: bool = True
    shingle_size: int = 4
    num_permutations: int = 64
    jaccard_threshold: float = 0.80


@dataclass
class AuditConfig:
    held_out_environments: list[str] = field(default_factory=list)
    extra_forbidden_terms: list[str] = field(default_factory=list)
    fail_on_violation: bool = True

    @property
    def terms(self) -> list[str]:
        return [t for t in (*self.held_out_environments, *self.extra_forbidden_terms) if t]


@dataclass
class ExecutedConfig:
    enabled: bool = True
    environment: str = "l2rpn_wcci_2022"


@dataclass
class OutputConfig:
    dir: str = "data/rag"
    corpus_file: str = "grid2op_qa.jsonl"
    pages_file: str = "pages.jsonl"
    hf_repo: str | None = None
    hf_private: bool = True
    push_every: int = 20

    @property
    def corpus_path(self) -> Path:
        return Path(self.dir) / self.corpus_file

    @property
    def pages_path(self) -> Path:
        return Path(self.dir) / self.pages_file

    @property
    def rejects_dir(self) -> Path:
        """Where a reply that would not parse is kept.

        Without this the only record of a parse failure is the counter, which
        says how often it happened and nothing about why. One file per failed
        page costs nothing and makes the cause readable.
        """
        return Path(self.dir) / "rejected"


@dataclass
class RagConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    pages: PagesConfig = field(default_factory=PagesConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    executed: ExecutedConfig = field(default_factory=ExecutedConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> list[str]:
        """Problems worth telling the operator about before a run starts.

        Returns warnings rather than raising. A missing held-out list is not a
        reason to refuse to run a dry run, but it is a reason to say so.
        """
        warnings: list[str] = []
        if not self.models.generator.id:
            warnings.append("models.generator.id is empty")
        if not self.models.verifier.id:
            warnings.append("models.verifier.id is empty")
        if self.models.generator.id == self.models.verifier.id:
            warnings.append(
                "generator and verifier are the same model: a model asked to check "
                "its own output agrees with itself, so the entailment check is weak"
            )
        if self.verification.enabled and self.models.verifier.temperature > 0.2:
            warnings.append(
                f"verifier temperature is {self.models.verifier.temperature}; a judge "
                f"should be near 0 so the same pair gets the same verdict twice"
            )
        if not self.audit.terms:
            warnings.append(
                "audit.held_out_environments is empty: the leak audit will pass "
                "everything. Set it once the held-out split is decided"
            )
        if self.pages.min_chars >= self.pages.max_chars:
            warnings.append("pages.min_chars >= pages.max_chars")
        return warnings


_SECTIONS = {f.name: f.type for f in dataclasses.fields(RagConfig)}
_CLASSES = {
    "source": SourceConfig, "pages": PagesConfig, "index": IndexConfig,
    "augment": AugmentConfig, "api": ApiConfig, "models": ModelsConfig,
    "generation": GenerationConfig, "verification": VerificationConfig,
    "dedup": DedupConfig, "audit": AuditConfig, "executed": ExecutedConfig,
    "output": OutputConfig,
}


def _build(name: str, raw: dict[str, Any]) -> Any:
    cls = _CLASSES[name]
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"rag config: unknown key(s) {sorted(unknown)} under '{name}'")
    return cls(**raw)


def load_config(path: str | Path) -> RagConfig:
    """Read rag.yaml into a RagConfig, rejecting keys the code does not read."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"rag config: unknown section(s) {sorted(unknown)} in {path}")
    return RagConfig(**{name: _build(name, raw[name] or {}) for name in raw})


def load_prompts(path: str | Path) -> dict[str, str]:
    """Read prompts.yaml. Every template the pipeline needs must be present."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = ("generate", "verify", "reference_entry", "reference_external", "reference_none")
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise ValueError(f"{path}: missing prompt template(s) {missing}")
    return {k: str(v) for k, v in data.items()}
