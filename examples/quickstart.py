"""End-to-end quickstart for the DeMICAF compliance dataset.

Loads the bundled annotations, builds a :class:`DatasetCXR` over locally-available
source images, extracts CAF encoder features, scores each image with the Mahalanobis
ranking function, and reports the AUC against the manual compliance annotations.

Prerequisites
-------------
* The four source datasets downloaded and laid out under ``$CXR_ROOT`` as
  ``<Dataset>/...`` (see ``docs/DATASET_CARD.md`` for download instructions). The
  annotation paths use the same convention as ``scripts/preprocessing/draw_subsets.py``.
* Optionally, a trained CAF encoder checkpoint (``--checkpoint``). Without one, a
  randomly-initialised encoder is used and the AUC is only illustrative.

Run::

    CXR_ROOT=/path/to/CXR python -m examples.quickstart --checkpoint path/to/encoder.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DeMICAF.caf.networks import ResNet2D
from DeMICAF.caf.scorer import mahalanobis
from DeMICAF.data.datasets import DatasetCXR
from DeMICAF.evaluation.auc import compute_auc
from DeMICAF.utils.paths import get_annotations_root, get_cxr_root

ANNOTATIONS_CSV = get_annotations_root() / "annotated_causes.csv"


def _strip_dataset_prefix(image_path: str) -> str:
    """``CheXpert/CheXpert-v1.0/...`` -> ``CheXpert-v1.0/...`` so it joins with $CXR_ROOT/<Dataset>."""
    return image_path.split("/", 1)[1] if "/" in image_path else image_path


def extract_features(dataset: DatasetCXR, encoder: torch.nn.Module, batch_size: int = 32) -> tuple[list[str], np.ndarray]:
    """Run the encoder over the dataset and return ``(filenames, features)``."""
    loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    device = next(encoder.parameters()).device

    names: list[str] = []
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for fnames, inputs, _ in loader:
            outputs = encoder(inputs.to(device).float(), head=False)
            feats.append(outputs.cpu().numpy())
            names.extend(fnames)
    return names, np.vstack(feats)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeMICAF compliance-scoring quickstart.")
    parser.add_argument(
        "--cxr-root",
        type=Path,
        default=get_cxr_root(),
        help="Root containing one subfolder per source dataset.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional CAF encoder checkpoint (.pt).")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    annotations = pd.read_csv(ANNOTATIONS_CSV)
    annotations["annotation"] = annotations["compliance"]  # evaluation.auc expects an 'annotation' column
    print(f"Loaded {len(annotations)} annotations ({(annotations['compliance'] == 'compliant').sum()} compliant).")

    # Build one dataset over all annotated images that exist locally.
    frame = annotations[["image_path"]].copy()
    frame["_resolved"] = frame["image_path"].map(_strip_dataset_prefix)
    dataset = DatasetCXR(
        root_dir=args.cxr_root,
        dataframe=frame,
        image_column="_resolved",
        output_mode="label",
    )
    if len(dataset) == 0:
        raise SystemExit(
            f"No annotated images found under {args.cxr_root}. Download the source datasets first "
            "(see docs/DATASET_CARD.md) and set $CXR_ROOT."
        )
    print(f"{len(dataset)} annotated images resolved on disk.")

    encoder = ResNet2D(bn_head=True, normalize_f=True, out_dim=128)
    if args.checkpoint is not None:
        encoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded encoder weights from {args.checkpoint}.")
    else:
        print("No checkpoint given; using a randomly-initialised encoder (AUC is illustrative only).")

    names, features = extract_features(dataset, encoder, batch_size=args.batch_size)

    # Reference distribution: the compliant annotated images (self-reference).
    compliant_paths = set(annotations.loc[annotations["compliance"] == "compliant", "image_path"].map(_strip_dataset_prefix))
    is_ref = np.array([name in compliant_paths for name in names])
    scores, _ = mahalanobis(features[is_ref], features, k=1)

    # Higher Mahalanobis distance = more anomalous; negate so higher score = more compliant.
    scores_df = pd.DataFrame({"image_path": [n for n in names], "score": -scores})
    # Re-map resolved names back to the original dataset-prefixed image_path for the join.
    resolved_to_original = dict(zip(frame["_resolved"], frame["image_path"], strict=False))
    scores_df["image_path"] = scores_df["image_path"].map(resolved_to_original)

    auc = compute_auc(annotations, scores_df, score="score", percentage=True, invert_labels=True)
    print(f"Compliance AUC (positive class = compliant): {auc:.1f}%")


if __name__ == "__main__":
    main()
