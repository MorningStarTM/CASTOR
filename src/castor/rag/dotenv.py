"""Minimal .env reader.

python-dotenv would do this, but it is one file format and twenty lines, and
the rest of this repository already avoids a dependency for a single call
site.

An existing environment variable always wins. A key exported in the shell, or
injected by Kaggle Secrets, is a deliberate act for this run; a .env file is a
default sitting on disk from last time. Letting the file override the shell
would make `OPENROUTER_API_KEY=... python script.py` silently do the wrong
thing, which is the kind of surprise that costs an afternoon.
"""

from __future__ import annotations

import os
from pathlib import Path

_QUOTES = ("'", '"')


def parse(text: str) -> dict[str, str]:
    """KEY=value lines. Comments, blanks, `export` and quotes handled."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        else:
            # An unquoted trailing comment is not part of the value.
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def load(path: str | Path = ".env", override: bool = False) -> list[str]:
    """Load path into os.environ. Returns the names set, never the values.

    Returns [] when the file does not exist: running with the key exported
    instead is a perfectly good way to run, so a missing .env is not an error.
    """
    path = Path(path)
    if not path.is_file():
        return []
    applied: list[str] = []
    for key, value in parse(path.read_text(encoding="utf-8", errors="ignore")).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied


def find_upwards(name: str = ".env", start: str | Path | None = None) -> Path | None:
    """Nearest .env at or above start, so the script works from any directory."""
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None
