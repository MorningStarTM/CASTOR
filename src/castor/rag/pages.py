"""Stage 1: pagination.

A page is a coherent unit of documentation, which for a Python library means a
class together with its own methods. Not a module, because grid2op's largest
reaches 95,743 characters. Not a bare docstring, because a method docstring
arrives without the class it belongs to and the teacher has to infer the
missing context, which is where invention starts.

Oversized classes split into numbered parts, each carrying the class docstring
so part 3 still knows what it is part of.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator

from .config import PagesConfig, SourceConfig
from .schema import API, DOCS, Page

# Sphinx markup that is noise to a language model: it should learn that rho is
# a ratio, not that someone wrote :attr:`rho`.
_ROLE = re.compile(r":[a-z:]+:`~?([^`<]+?)(?:\s*<[^>]+>)?`")
_LITERAL = re.compile(r"``([^`]+)``")
_DIRECTIVE = re.compile(r"^\s*\.\.\s+\w+::.*$", re.MULTILINE)
_RST_UNDERLINE = re.compile(r"^[=\-~^\"'`#*+]{3,}\s*$", re.MULTILINE)


def clean(text: str) -> str:
    """Strip reStructuredText markup, keeping the words and the names."""
    text = _ROLE.sub(r"\1", text)
    text = _LITERAL.sub(r"\1", text)
    text = _DIRECTIVE.sub("", text)
    text = _RST_UNDERLINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def locate_package(configured: str | None) -> Path:
    """The grid2op package directory, configured or discovered."""
    if configured:
        return Path(configured)
    import grid2op

    return Path(grid2op.__file__).parent


def is_internal(ref: str, text: str, cfg: PagesConfig) -> bool:
    """True for private paths and for what grid2op declares internal itself.

    These docstrings are warnings rather than descriptions. Pairs drawn from
    them teach the student about machinery it will never call, in the voice of
    a maintainer telling the reader to go away.
    """
    if any(part.startswith("_") for part in ref.split(".")):
        return True
    head = text[:400].lower()
    return any(marker.lower() in head for marker in cfg.internal_markers)


def _split(text: str, max_chars: int) -> list[str]:
    """Break an oversized page at paragraph boundaries where possible."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 2:          # no usable boundary; cut hard
            cut = max_chars
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def raw_refs(text: str) -> list[str]:
    """Referenced names, most frequent first, read from the UNCLEANED text.

    This has to happen before clean(), which rewrites :class:`grid2op.X` to
    plain `grid2op.X` and leaves the retrieval stage nothing to match on. The
    markup is noise to a language model and must go, but it is the only signal
    that says which names a page is pointing at, so it is harvested first.

    Frequency order matters because the reference budget is finite: when a
    page points at more symbols than fit, the ones it leans on repeatedly are
    the ones worth carrying.
    """
    counts: dict[str, int] = {}
    for raw in _ROLE.findall(text):
        name = raw.strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return sorted(counts, key=lambda n: (-counts[n], n))


def _emit(page_id: str, text: str, source: str, cfg: PagesConfig) -> Iterator[Page]:
    refs = raw_refs(text)
    text = clean(text)
    if len(text) < cfg.min_chars:
        return
    if cfg.skip_internal and is_internal(page_id, text, cfg):
        return
    parts = _split(text, cfg.max_chars)
    for i, part in enumerate(parts, 1):
        if len(part) < cfg.min_chars:
            continue
        suffix = f"#{i}" if len(parts) > 1 else ""
        # A split class shares its parent's references: part 3 still points at
        # whatever the class as a whole pointed at.
        yield Page(id=f"{page_id}{suffix}", text=part, source=source, refs=list(refs))


def from_package(root: Path, cfg: PagesConfig, source_cfg: SourceConfig) -> Iterator[Page]:
    """Pages from an installed package, read with ast rather than imported.

    Importing module by module executes package side effects and pulls in
    optional dependencies. Parsing the source has neither problem and sees
    exactly the same docstrings.
    """
    for path in sorted(root.rglob("*.py")):
        lowered = str(path).lower()
        if any(s.lower() in lowered for s in source_cfg.exclude_path_substrings):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

        module = path.relative_to(root.parent).with_suffix("").as_posix().replace("/", ".")

        # The module page: its own docstring plus any top-level functions.
        loose = [ast.get_docstring(n) or "" for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        module_text = "\n\n".join(p for p in [ast.get_docstring(tree) or "", *loose] if p)
        yield from _emit(module, module_text, API, cfg)

        # One page per class: the class docstring, then each method's.
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            class_doc = ast.get_docstring(node) or ""
            methods = []
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(member)
                    if doc:
                        methods.append(f"{member.name}:\n{doc}")
            text = "\n\n".join(p for p in [class_doc, *methods] if p)
            yield from _emit(f"{module}.{node.name}", text, API, cfg)


def from_rst(docs_dir: Path, cfg: PagesConfig) -> Iterator[Page]:
    """Narrative pages from a Sphinx docs/ directory, one page per file."""
    for path in sorted(docs_dir.rglob("*.rst")):
        page_id = path.relative_to(docs_dir).with_suffix("").as_posix().replace("/", ".")
        yield from _emit(f"docs.{page_id}", path.read_text(encoding="utf-8", errors="ignore"),
                         DOCS, cfg)


def build(cfg: PagesConfig, source_cfg: SourceConfig) -> list[Page]:
    """Every page, in a stable order."""
    pages = list(from_package(locate_package(source_cfg.package), cfg, source_cfg))
    if source_cfg.rst_dir:
        pages.extend(from_rst(Path(source_cfg.rst_dir), cfg))
    return pages


def bundle(pages: list[Page], cfg: PagesConfig) -> list[list[Page]]:
    """Group pages into requests.

    bundle_under = 0 means one page per request, which is the default and the
    better default. Above zero, pages below that size share a request: the
    median page is under a thousand characters, and spending a whole request
    from a capped daily allowance on four sentences is waste. The trade is
    attention, not correctness, so raise the threshold only as far as the
    allowance forces.
    """
    if cfg.bundle_under <= 0:
        return [[page] for page in pages]

    groups: list[list[Page]] = []
    current: list[Page] = []
    current_chars = 0
    for page in pages:
        if page.chars >= cfg.bundle_under:
            groups.append([page])
            continue
        if current and current_chars + page.chars > cfg.bundle_max_chars:
            groups.append(current)
            current, current_chars = [], 0
        current.append(page)
        current_chars += page.chars
    if current:
        groups.append(current)
    return groups
