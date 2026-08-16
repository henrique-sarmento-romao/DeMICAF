r"""Extract penultimate-layer features and per-label probabilities from the classifier.

One-time pass over the 3 client datasets' annotated images
(``annotations/annotated_causes.csv`` restricted to ChestX-ray8/
MIMIC-CXR/PadChest). A forward hook on the pooling layer captures the
penultimate (pre-``fc``) feature in the same forward pass that produces the
final logits, so both outputs come from a single inference pass per image:

- ``classifier_features_{client}.npz`` — penultimate features, feeding the
  ``knn``/``maha_pp`` scorers via ``scripts/benchmarking.py``'s ``FeatureBank``.
- ``classifier_probs_{client}.npz`` — sigmoid per-label probabilities, feeding
  the ``gen`` (generalized entropy) scorer.

Both use the same ``{name, value...}`` layout :func:`DeMICAF.utils.io.load_features`
expects.

Uses :class:`DeMICAF.data.datasets.DatasetCXR` with ``DeMICAF.classifier.val_tfm``
— the exact same preprocessing pipeline the classifier saw during
validation/test evaluation while training — so the extracted outputs are
consistent with what the model was actually trained/evaluated on.

Uses the FedAvg classifier's checkpoint by default (see ``DEFAULT_CHECKPOINT`` below)
— pass ``--checkpoint`` to point at a different one.

Typical usage::

    uv run python -m scripts.extract_classifier_features
    uv run python -m scripts.extract_classifier_features --limit 20  # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from DeMICAF.classifier import build_model, val_tfm
from DeMICAF.data.datasets import DatasetCXR
from DeMICAF.utils.paths import get_annotations_root, get_cxr_root, get_results_root, resolve_path

CLIENTS = ["ChestX-ray8", "MIMIC-CXR", "PadChest"]
LABEL_COLS = ["Cardiomegaly", "Atelectasis", "Nodule", "Alveolar Pattern", "Pleural Effusion", "Pneumothorax"]

#: The `knn`/`maha_pp`/`gen` scorers are defined against the FedAvg classifier, not a
#: centralised one — matches `scripts/federated/train.py`'s own `--out-dir`/
#: `--experiment-name` defaults, which is what actually produces this file (see
#: `FLServer.run_federated`'s `final_model.pt` checkpoint in `DeMICAF.federated.server`).
DEFAULT_CHECKPOINT = get_results_root() / "per_cause" / "classifier" / "fedavg_resnet18" / "seed_0" / "final_model.pt"


def load_model_with_hook(
    checkpoint_path: Path,
    architecture: str,
    dropout_rate: float,
    device: str,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Load the trained classifier with a forward hook capturing its pooled backbone feature.

    Unlike swapping ``model.fc`` for identity, this keeps the classification head
    intact, so a single forward pass yields both the penultimate feature (via the
    hook) and the final per-label logits (the model's normal output).
    """
    model = build_model(
        pretrained=False,
        num_labels=len(LABEL_COLS),
        architecture=architecture,
        dropout_rate=dropout_rate,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def _capture_penultimate(_module: torch.nn.Module, _input: Any, output: torch.Tensor) -> None:
        captured["penultimate"] = output.flatten(1)

    model.avgpool.register_forward_hook(_capture_penultimate)
    return model, captured


def build_loader(df: pd.DataFrame, data_root: Path, *, batch_size: int, num_workers: int) -> DataLoader:
    """Wrap ``df`` in a DatasetCXR using the classifier's own eval-time transform."""
    dataset = DatasetCXR(
        root_dir=data_root,
        dataframe=df,
        transform=val_tfm,
        image_column="image_path",
        output_mode="label",
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def extract_features_and_probs(
    model: torch.nn.Module,
    captured: dict[str, torch.Tensor],
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over every batch; return ``(image_paths, penultimate_features, sigmoid_probs)``."""
    names: list[str] = []
    feats: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for fnames, imgs, _ in tqdm(loader, desc="features", unit="batch"):
            logits = model(imgs.to(device))
            feats.append(captured["penultimate"].cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            names.extend(fnames)

    if not feats:
        empty = np.zeros((0, 0), dtype=np.float32)
        return np.asarray([], dtype=str), empty, empty
    names_arr = np.asarray(names, dtype=str)
    feats_arr = np.concatenate(feats, axis=0).astype(np.float32)
    probs_arr = np.concatenate(probs, axis=0).astype(np.float32)
    return names_arr, feats_arr, probs_arr


def save_npz(path: Path, names: np.ndarray, values: np.ndarray) -> None:
    """Write ``{name, value...}`` rows to a compressed ``.npz``, matching ``utils.io.load_features``."""
    rows = np.empty((len(names), 1 + values.shape[1]), dtype=object)
    rows[:, 0] = names
    rows[:, 1:] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows)


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract penultimate-layer features and per-label probabilities from the trained classifier."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=get_annotations_root() / "annotated_causes.csv",
    )
    parser.add_argument("--data-root", type=str, default=str(get_cxr_root()))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Trained classifier checkpoint (default: the FedAvg model's final_model.pt).",
    )
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet152"])
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=get_results_root() / "per_cause")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images per client (smoke test).")
    return parser.parse_args()


def main() -> None:
    """Load the checkpoint, extract features+probs for each client, and save two .npz per client."""
    args = parse_args()
    data_root = resolve_path(args.data_root)

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model, captured = load_model_with_hook(args.checkpoint, args.architecture, args.dropout_rate, args.device)

    annotations = pd.read_csv(args.annotations)

    for client in CLIENTS:
        df = annotations[annotations["dataset"] == client]
        if args.limit is not None:
            df = df.head(args.limit)

        print(f"Extracting features+probs for {client} ({len(df)} images requested) ...")
        loader = build_loader(df, data_root, batch_size=args.batch_size, num_workers=args.num_workers)
        names, feats, probs = extract_features_and_probs(model, captured, loader, args.device)

        feats_path = args.out_dir / f"classifier_features_{client}.npz"
        save_npz(feats_path, names, feats)
        probs_path = args.out_dir / f"classifier_probs_{client}.npz"
        save_npz(probs_path, names, probs)

        feat_dim = feats.shape[1] if feats.size else 0
        print(f"  Saved {len(names)}/{len(df)} feature vectors (dim={feat_dim}) to {feats_path}")
        print(f"  Saved {len(names)}/{len(df)} probability vectors (dim={probs.shape[1] if probs.size else 0})"
              f" to {probs_path}")


if __name__ == "__main__":
    main()
