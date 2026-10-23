"""Phase 2: the domain corpus for adaptation.

The plan's sources are blackout investigation reports, grid codes and topology
control literature. grid2op's own documentation joins them: it is the
authoritative definition of the vocabulary both channels have to share, where
rho is a ratio of flow to thermal limit and a cooldown is a number of steps.

Nothing here is scraped. The documentation exists in two better forms than the
rendered HTML:

  * docstrings in the installed package, reachable by parsing the source, with
    their structure intact
  * the Sphinx .rst sources in the project repository, which carry the
    narrative pages

Scraping readthedocs would mean reconstructing from HTML the structure that
Sphinx was given in the first place, and inheriting navigation chrome,
permalink markers and duplicated sidebars along the way.

Output is the uint16 shard format corpus.py writes, so castor.language.dataset
reads a domain corpus exactly as it reads the pretraining one.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .corpus import HOLDOUT_BUCKETS, _ShardWriter

# Sphinx roles: :class:`grid2op.Action.BaseAction` -> grid2op.Action.BaseAction
_ROLE = re.compile(r":[a-z:]+:`~?([^`<]+?)(?:\s*<[^>]+>)?`")
_LITERAL = re.compile(r"``([^`]+)``")
_DIRECTIVE = re.compile(r"^\s*\.\.\s+\w+::.*$", re.MULTILINE)
_RST_UNDERLINE = re.compile(r"^[=\-~^\"'`#*+]{3,}\s*$", re.MULTILINE)


def _held_out(text: str, share: float) -> bool:
    """Route a document to validation by a hash of its content.

    corpus.is_holdout fixes the share at the fraction that suits a billion
    token corpus. The domain corpus is three orders of magnitude smaller, so
    the same fraction yields a validation split of a few hundred tokens, and
    the Phase 2 exit test is a perplexity measurement on that split. The share
    has to scale with the corpus, so it is a parameter here.
    """
    import hashlib

    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HOLDOUT_BUCKETS < share * HOLDOUT_BUCKETS


@dataclass
class Record:
    """One piece of domain text, with where it came from."""

    text: str
    source: str  # grid2op.api | grid2op.docs | report | gridcode | literature
    ref: str     # qualified name, or path

    def as_json(self) -> dict[str, str]:
        return {"text": self.text, "source": self.source, "ref": self.ref}


def clean(text: str) -> str:
    """Strip reStructuredText markup, keeping the words.

    Role markup and double backticks are noise to a language model: the model
    should learn that rho is a ratio, not that someone wrote :attr:`rho`.
    """
    text = _ROLE.sub(r"\1", text)
    text = _LITERAL.sub(r"\1", text)
    text = _DIRECTIVE.sub("", text)
    text = _RST_UNDERLINE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def from_package(root: str | Path, min_chars: int = 120) -> Iterator[Record]:
    """Docstrings from an installed package, read with ast rather than import.

    Importing grid2op module by module executes package side effects and pulls
    in optional dependencies. Parsing the source has neither problem and sees
    exactly the same docstrings.
    """
    root = Path(root)
    for path in sorted(root.rglob("*.py")):
        if "test" in str(path).lower():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        module = path.relative_to(root.parent).with_suffix("").as_posix().replace("/", ".")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                continue
            doc = ast.get_docstring(node)
            if not doc or len(doc) < min_chars:
                continue
            name = getattr(node, "name", "")
            text = clean(doc)
            if len(text) < min_chars:
                continue
            yield Record(text, "grid2op.api", f"{module}.{name}" if name else module)


def from_rst(docs_dir: str | Path, min_chars: int = 200) -> Iterator[Record]:
    """Narrative pages from a Sphinx docs/ directory.

    Whole files rather than sections: the prose is continuous and splitting on
    headings would cut the explanatory passages that are the point of using
    these at all.
    """
    docs_dir = Path(docs_dir)
    for path in sorted(docs_dir.rglob("*.rst")):
        text = clean(path.read_text(encoding="utf-8", errors="ignore"))
        if len(text) >= min_chars:
            yield Record(text, "grid2op.docs", str(path.relative_to(docs_dir)))


def from_text_dir(
    directory: str | Path, source: str, patterns: Iterable[str] = ("*.txt", "*.md")
) -> Iterator[Record]:
    """Plain text from a directory: blackout reports, grid codes, literature.

    PDFs are converted before they reach here. Extraction quality varies far
    too much to bury inside a corpus builder, and a bad conversion should be
    visible as a bad .txt rather than as mysterious tokens.
    """
    directory = Path(directory)
    for pattern in patterns:
        for path in sorted(directory.rglob(pattern)):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) >= 200:
                yield Record(text, source, str(path.relative_to(directory)))


def pack_domain(
    records: Iterable[Record],
    tokenizer,
    out_dir: str | Path,
    jsonl_path: str | Path | None = None,
    holdout_share: float = 0.05,
) -> dict[str, Any]:
    """Tokenize records into train/val shards, split by content hash.

    The same hash split as the pretraining corpus, for the same reason: the
    Phase 2 exit test is a perplexity drop on held-out domain text, and that
    number means nothing if the held-out text was also trained on.
    """
    vocab = len(tokenizer)
    if vocab >= 2**16:
        raise ValueError(f"vocab {vocab} does not fit uint16")
    eos = tokenizer.eos_token_id

    root = Path(out_dir)
    writers = {}
    for split in ("train", "val"):
        (root / split).mkdir(parents=True, exist_ok=True)
        writers[split] = _ShardWriter(root / split)

    by_source: dict[str, dict[str, int]] = {}
    sink = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None
    try:
        for record in records:
            split = "val" if _held_out(record.text, holdout_share) else "train"
            ids = tokenizer(record.text, add_special_tokens=False)["input_ids"]
            writers[split].write(np.asarray(list(ids) + [eos], dtype=np.uint16))

            stats = by_source.setdefault(record.source, {"records": 0, "tokens": 0})
            stats["records"] += 1
            stats["tokens"] += len(ids) + 1
            if sink:
                sink.write(json.dumps(record.as_json()) + "\n")
    finally:
        if sink:
            sink.close()

    meta: dict[str, Any] = {}
    for split, writer in writers.items():
        writer.close()
        split_meta = {
            "split": split,
            "tokens": writer.total,
            "vocab_size": vocab,
            "dtype": "uint16",
            "eos_token_id": eos,
            "tokenizer": tokenizer.name_or_path,
            "packing": "contiguous",
            "corpus": "domain",
            "holdout_share": holdout_share,
            "by_source": by_source,
            "shards": writer.shards,
        }
        (root / split / "meta.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
        meta[split] = split_meta
    return meta
