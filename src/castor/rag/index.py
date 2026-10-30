"""Stage 2: the symbol index.

Every documented name in the package mapped to its docstring. This is the
retrieval corpus, and it is deliberately an exact-name lookup rather than an
embedding store: a cross-reference is an exact name, so approximate nearest
neighbours would be a worse way to answer an exact question, and a wrong
neighbour here produces a confidently wrong answer downstream.

Built offline in a few seconds. No embeddings, no vector database, no
dependency beyond the standard library.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .config import IndexConfig, SourceConfig
from .pages import clean


class SymbolIndex:
    """Qualified name, short name or bare name to docstring."""

    def __init__(self) -> None:
        self._exact: dict[str, str] = {}      # grid2op.Action.baseAction.BaseAction
        self._short: dict[str, str] = {}      # baseAction.BaseAction
        self._bare: dict[str, str] = {}       # BaseAction
        # Class name -> its declared base class names, for inherited methods.
        self._bases: dict[str, list[str]] = {}
        # Class name -> {method name: doc}, so Class.method can walk the bases.
        self._methods: dict[str, dict[str, str]] = {}

    def add_class(self, class_name: str, bases: list[str]) -> None:
        self._bases.setdefault(class_name, [b.split(".")[-1] for b in bases])

    def add_method(self, class_name: str, method: str, doc: str, cfg: IndexConfig) -> None:
        if len(doc) >= cfg.min_doc_chars:
            self._methods.setdefault(class_name, {}).setdefault(method, doc)

    def _inherited(self, class_name: str, method: str,
                   seen: set[str] | None = None) -> tuple[str, str] | None:
        """Find method on class_name or on one of its ancestors.

        grid2op documents step, reset and simulate once on a base class and
        inherits them everywhere. Without this, a reference to Environment.step
        is unresolvable even though the definition is right there on BaseEnv,
        and the teacher is left to invent what step does.
        """
        seen = seen or set()
        if class_name in seen:
            return None
        seen.add(class_name)
        own = self._methods.get(class_name, {})
        if method in own:
            return f"{class_name}.{method}", own[method]
        for base in self._bases.get(class_name, []):
            found = self._inherited(base, method, seen)
            if found:
                return found
        return None

    def __len__(self) -> int:
        return len(self._exact) + len(self._short) + len(self._bare)

    @property
    def n_symbols(self) -> int:
        return len(self._exact)

    def add(self, qualified: str, doc: str, cfg: IndexConfig) -> None:
        if len(doc) < cfg.min_doc_chars:
            return
        self._exact.setdefault(qualified, doc)
        parts = qualified.split(".")
        if cfg.include_short_qualified and len(parts) >= 2:
            self._short.setdefault(".".join(parts[-2:]), doc)
        if cfg.include_bare_names:
            self._bare.setdefault(parts[-1], doc)

    def lookup(self, name: str, bare_fallback_for_qualified: bool = False
               ) -> tuple[str, str] | None:
        """Resolve a referenced name. Returns (resolved_name, doc) or None.

        Tried most specific first. grid2op re-exports heavily, so a reference
        written grid2op.Action.BaseAction never matches the real module path
        grid2op.Action.baseAction.BaseAction exactly; falling back to the last
        two components is what makes those resolve.

        Whether the last component alone may be used depends on what it is,
        and the split is measured rather than assumed. Across grid2op, 1.1% of
        class names are ambiguous against 20.6% of method names: reset is
        defined on 24 classes and __init__ on 25, while almost every class
        name occurs once. So a CapWords last component falls back to the bare
        name, which is how the re-export aliases grid2op.Action.BaseAction and
        grid2op.Environment.Environment resolve, and a lowercase one does not,
        because reducing Environment.step to step would hand back whichever
        step was indexed first.

        A wrongly resolved reference hands the teacher a confident definition
        for the wrong object, which is the failure this stage exists to
        prevent. Where the two conflict, missing beats guessing.
        """
        name = name.strip().rstrip("()")
        if not name:
            return None
        for table in (self._exact, self._short, self._bare):
            if name in table:
                return name, table[name]

        parts = name.split(".")
        if len(parts) >= 2:
            short = ".".join(parts[-2:])
            if short in self._short:
                return short, self._short[short]
            # Class.method where the method is inherited rather than declared.
            found = self._inherited(parts[-2], parts[-1])
            if found:
                return found
            looks_like_class = parts[-1][:1].isupper()
            if not (looks_like_class or bare_fallback_for_qualified):
                return None
        if parts[-1] in self._bare:
            return parts[-1], self._bare[parts[-1]]
        return None


def _base_name(node: ast.AST) -> str:
    """The dotted name of a base class expression.

    ast.unparse would do this in one call but arrived in Python 3.9, and the
    interpreter this runs under is not always the newest one on the machine.
    Only the three forms a base class actually takes are handled; anything
    stranger yields "" and is simply not recorded as a base, which costs one
    inheritance edge rather than the whole index.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _base_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):      # Generic[T]
        return _base_name(node.value)
    return ""


def _walk(body: list, prefix: str, index: SymbolIndex, cfg: IndexConfig) -> None:
    """Index a scope, carrying the enclosing class into each method's name.

    ast.walk would flatten this and record a method as module.method, losing
    the class it belongs to. Sphinx references are written Class.method, and
    several classes in grid2op define a method of the same name, so a
    flattened index resolves BaseObservation.simulate to whichever simulate
    happened to be seen first. That is worse than not resolving it: the
    teacher is handed a definition for the wrong object and has no way to
    tell.
    """

    for node in body:
        if isinstance(node, ast.ClassDef):
            qualified = f"{prefix}.{node.name}"
            doc = ast.get_docstring(node)
            if doc:
                index.add(qualified, clean(doc), cfg)
            index.add_class(node.name, [_base_name(b) for b in node.bases])
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    member_doc = ast.get_docstring(member)
                    if member_doc:
                        index.add_method(node.name, member.name, clean(member_doc), cfg)
            _walk(node.body, qualified, index, cfg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                index.add(f"{prefix}.{node.name}", clean(doc), cfg)


def build(cfg: IndexConfig, source_cfg: SourceConfig, root: Path) -> SymbolIndex:
    """Index every class and method docstring in the package."""
    index = SymbolIndex()
    for path in sorted(root.rglob("*.py")):
        lowered = str(path).lower()
        if any(s.lower() in lowered for s in source_cfg.exclude_path_substrings):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        module = path.relative_to(root.parent).with_suffix("").as_posix().replace("/", ".")
        _walk(tree.body, module, index, cfg)
    return index
