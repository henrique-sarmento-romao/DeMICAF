"""Feature and score persistence shared by the scoring scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(image_names, feature_matrix)`` from a compressed ``.npz`` feature file.

    The array layout is the one written by :func:`DeMICAF.caf.encoder.encode`:
    column 0 holds the image name, the remaining columns the feature vector.
    """
    arr = np.load(path, allow_pickle=True)["arr_0"]
    return arr[:, 0].astype(str), arr[:, 1:].astype(np.float32)


def upsert_scores_csv(
    csv_path: Path,
    tag: str,
    image_names: np.ndarray,
    scores: np.ndarray,
    dataset_name: str,
) -> None:
    """Insert or update a score column in the merged scores CSV.

    The CSV is indexed by ``(image_path, dataset)``. Existing rows keep their
    other score columns; incoming values overwrite previous values of ``tag``
    for matching rows and new rows are appended.
    """
    incoming = pd.DataFrame(
        {
            "image_path": image_names.astype(str),
            "dataset": dataset_name,
            tag: scores,
        }
    ).set_index(["image_path", "dataset"])
    incoming = incoming[~incoming.index.duplicated(keep="last")]

    if csv_path.exists() and csv_path.stat().st_size > 0:
        base = pd.read_csv(csv_path).set_index(["image_path", "dataset"])
        base = base[~base.index.duplicated(keep="last")]
        all_index = base.index.union(incoming.index)
        base = base.reindex(all_index)
        incoming = incoming.reindex(all_index)
        if tag in base.columns:
            base[tag] = incoming[tag].combine_first(base[tag])
        else:
            base[tag] = incoming[tag]
        out = base.reset_index()
    else:
        out = incoming.reset_index()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
