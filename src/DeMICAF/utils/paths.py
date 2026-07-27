"""Path resolution shared across the DeMICAF scripts.

The repository root holds ``annotations/`` (the released compliance annotations),
``data/`` (figure palette, dataset statistics), and the importable library under
``src/``. Its location is resolved automatically from this file; override it with
the ``DEMICAF_ROOT`` environment variable when running from an installed copy or a
relocated checkout.

Three further roots are resolved relative to the repository root unless overridden:

- ``get_annotations_root()`` — the released compliance annotations (``annotations/``).
- ``get_cxr_root()`` — the raw chest-radiograph image data (``$CXR_ROOT``), which is
  never committed and is registration-gated (see ``docs/DATASET_CARD.md``).
- ``get_results_root()`` — where every script writes its outputs (``results/``),
  which is git-ignored.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

#: Repository root, inferred as the parent of ``src/`` (…/src/DeMICAF/utils/paths.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Per-dataset top-level folder names that may prefix an image path.
DATASET_PREFIXES = {
    "CheXpert",
    "ChestX-ray8",
    "MIMIC-CXR",
    "PadChest",
    "All",
}


def get_repo_root() -> Path:
    """Return the repository root: ``$DEMICAF_ROOT`` if set, else inferred from this file."""
    env_root = os.environ.get("DEMICAF_ROOT")
    if env_root:
        return Path(env_root)
    return _REPO_ROOT


def get_annotations_root() -> Path:
    """Return the released annotations root: ``<repo_root>/annotations``."""
    return get_repo_root() / "annotations"


def get_cxr_root() -> Path:
    """Return the raw image data root: ``$CXR_ROOT`` if set, else ``<repo_root>/data/chest_xray``."""
    env_root = os.environ.get("CXR_ROOT")
    if env_root:
        return Path(env_root)
    return get_repo_root() / "data" / "chest_xray"


def get_results_root() -> Path:
    """Return the results root all scripts write under: ``<repo_root>/results``."""
    return get_repo_root() / "results"


def normalize_path(value: Any) -> str:
    """Normalize image path strings so train and score tables can be joined safely.

    Strips a leading dataset-name component (e.g. ``CheXpert/``) and collapses
    separators, so paths recorded relative to different roots compare equal.
    """
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return raw
    parts = [part for part in PurePosixPath(raw).parts if part not in {"", "."}]
    if parts and parts[0] in DATASET_PREFIXES:
        parts = parts[1:]
    return "/".join(parts)


def resolve_path(path_like: str, *, workspace_root: Path | None = None) -> Path:
    """Resolve a path as absolute or relative to ``workspace_root`` (default: repo root)."""
    path = Path(path_like)
    root = workspace_root if workspace_root is not None else get_repo_root()
    return path if path.is_absolute() else root / path
