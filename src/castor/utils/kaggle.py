"""Publish a built corpus to a Kaggle dataset.

A Kaggle session's disk disappears with the session, and Quick Save is a
manual step that is easy to forget after a job that ran for half an hour. This
pushes the result as soon as it exists.

Two things about Kaggle datasets shape this module:

  * Uploads do not reliably preserve a directory tree. The API's dir_mode
    either zips subdirectories or skips them, and the web upload folds the
    separators into the filenames, which is how castor-tokenized arrived as
    train_shard_0000.bin rather than train/shard_0000.bin. Rather than hope,
    files are flattened deliberately with that same naming, which
    castor.language.dataset already reads.
  * A dataset is created once and versioned afterwards, through two different
    calls. Which one applies is not knowable without asking, so creation is
    tried first and a failure falls through to versioning.

Credentials come from KAGGLE_USERNAME and KAGGLE_KEY, or ~/.kaggle/kaggle.json.
Inside a Kaggle notebook neither is set by default; see push_dataset's docstring.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def flatten_for_upload(source: str | Path, destination: str | Path | None = None) -> Path:
    """Stage a tree as one directory, folding separators into names.

    train/shard_0000.bin becomes train_shard_0000.bin, which is exactly the
    layout Kaggle produces on its own and which _locate_split understands.
    Doing it here makes the result the same whichever upload path is used.

    Files are hard linked rather than copied. A 6B token corpus is about 12GB
    and Kaggle's working directory caps at 20GB, so copying it to stage the
    upload would run the session out of disk. A hard link is a second name for
    the same bytes and costs nothing. Falls back to copying when the staging
    directory is on another filesystem, which is why it defaults to a sibling
    of the source rather than the system temp directory.
    """
    source = Path(source)
    destination = Path(destination or (source.parent / f"{source.name}_upload"))
    destination.mkdir(parents=True, exist_ok=True)

    linked = copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source).as_posix().replace("/", "_")
        if target.exists():
            target.unlink()
        try:
            os.link(path, target)
            linked += 1
        except OSError:  # cross-device, or a filesystem without hard links
            shutil.copy2(path, target)
            copied += 1

    size = sum(f.stat().st_size for f in destination.iterdir() if f.is_file())
    note = f"{linked} linked" + (f", {copied} copied" if copied else "")
    print(f"  [kaggle] staged {linked + copied} files ({size / 1e9:.2f} GB, {note})")
    return destination


def write_metadata(folder: str | Path, slug: str, title: str | None = None) -> Path:
    """The dataset-metadata.json the Kaggle API requires beside the files."""
    folder = Path(folder)
    if "/" not in slug:
        raise ValueError(f"dataset slug must be 'username/name', got {slug!r}")
    meta = {
        "title": title or slug.split("/", 1)[1].replace("-", " ").title(),
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    path = folder / "dataset-metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def push_dataset(
    folder: str | Path,
    slug: str,
    title: str | None = None,
    public: bool = False,
    notes: str = "automated upload",
    flatten: bool = True,
) -> str | None:
    """Create or version a Kaggle dataset from a folder. Never raises.

    An upload failing must not fail the job that produced the data, so every
    error here is reported and swallowed. The files are still on disk and
    Quick Save remains available.

    Inside a Kaggle notebook, authenticate first:

        from kaggle_secrets import UserSecretsClient
        import os
        os.environ["KAGGLE_USERNAME"] = "ernestbeckham"
        os.environ["KAGGLE_KEY"] = UserSecretsClient().get_secret("KAGGLE_KEY")
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  [kaggle] kaggle package not installed, skipping upload")
        return None

    try:
        staged = flatten_for_upload(folder) if flatten else Path(folder)
        write_metadata(staged, slug, title)

        api = KaggleApi()
        api.authenticate()

        try:
            api.dataset_create_new(folder=str(staged), public=public, quiet=True, dir_mode="skip")
            action = "created"
        except Exception:
            # Already exists, almost always. Versioning is the other half of
            # the same operation and the API gives no clean way to ask first.
            api.dataset_create_version(
                folder=str(staged), version_notes=notes, quiet=True, dir_mode="skip"
            )
            action = "versioned"

        url = f"https://www.kaggle.com/datasets/{slug}"
        print(f"  [kaggle] {action} {slug} -> {url}")
        return url
    except Exception as exc:
        print(f"  [kaggle] upload failed, files are still on disk: {exc}")
        return None
