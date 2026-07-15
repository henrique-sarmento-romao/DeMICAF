"""Path resolution shared across the DeCaF scripts.

The repository root holds ``data/`` (annotations, taxonomy, figure palette) and the
importable library under ``src/``. Its location is resolved automatically from this
file; override it with the ``DECAF_ROOT`` environment variable when running from an
installed copy or a relocated checkout.
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
    """Return the repository root: ``$DECAF_ROOT`` if set, else inferred from this file."""
    env_root = os.environ.get("DECAF_ROOT")
    if env_root:
        return Path(env_root)
    return _REPO_ROOT


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
