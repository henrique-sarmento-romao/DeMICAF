"""Compute no-reference perceptual IQA baselines (BRISQUE, NIQE) over the annotated samples.

For each image in ``annotations/annotated_causes.csv`` the script computes blind
image quality metrics and records them alongside the ground-truth compliance label.

One output CSV is produced per run, written incrementally (every 50 rows) so that
interrupted runs can be continued with --resume, which skips images already present
in the output file.

After the metrics are computed, per-dataset AUC, AP, and FPR@95TPR of each
metric against the compliance labels are printed and saved to a second CSV: for
each dataset both the values on its own samples and on the leave-one-out holdout
(the pooled other datasets). Use --auc-only to recompute only these evaluation
metrics from an existing metrics CSV.

Typical usage:
    uv run python scripts/baseline.py
    uv run python scripts/baseline.py --limit-per-dataset 500 --resume
    uv run python scripts/baseline.py --auc-only
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyiqa
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

from DeMICAF.data.datasets import KNOWN_DATASETS
from DeMICAF.utils.paths import get_annotations_root, get_cxr_root, get_results_root, resolve_path

# ── Constants ─────────────────────────────────────────────────────────────────

COMPLIANCE_MAP = {
    "compliant": 0,
    "non-compliant": 1,
    "non compliant": 1,
    "noncompliant": 1,
}

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute perceptual IQA baselines over the annotated samples.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=get_annotations_root() / "annotated_causes.csv",
        help="Compliance annotation CSV with image_path and compliance columns.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(get_cxr_root()),
        help="Root directory containing one image sub-folder per dataset (absolute or relative to the repo root).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=get_results_root() / "baseline",
        help="Directory in which to save the result CSV.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in the output CSV.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N images per dataset (default: all annotated samples).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --limit-per-dataset is set.",
    )
    parser.add_argument(
        "--auc-only",
        action="store_true",
        help="Skip metric computation and only compute AUCs from the existing output CSV.",
    )
    return parser.parse_args()


# ── Data I/O ──────────────────────────────────────────────────────────────────


def load_annotations(path: Path, limit: int | None, seed: int) -> pd.DataFrame:
    """Read the annotation CSV, tag each row with its dataset, and optionally sample per dataset.

    The dataset is the leading component of ``image_path`` (e.g. ``CheXpert/...``),
    the convention used throughout the released annotation tables.
    """
    df = pd.read_csv(path)
    df["dataset"] = df["image_path"].astype(str).str.split("/", n=1).str[0]
    df = df[df["dataset"].isin(KNOWN_DATASETS)]
    if limit is not None:
        sampled = [g.sample(n=min(limit, len(g)), random_state=seed) for _, g in df.groupby("dataset", sort=False)]
        df = pd.concat(sampled)
    return df.reset_index(drop=True)


def load_image(image_path: Path) -> Image.Image | None:
    """Open an image and return it as RGB; return None if the file is missing or corrupt."""
    with contextlib.suppress(FileNotFoundError, OSError):
        image = Image.open(image_path)
        if image.mode in ("I", "I;16"):
            arr = np.asarray(image, dtype=np.float32) / 65535.0 * 255.0
            image = Image.fromarray(arr.astype(np.uint8))
        return image.convert("RGB")
    return None


def write_rows(path: Path, rows: list[dict[str, Any]], *, write_header: bool) -> None:
    """Append a batch of result rows to the output CSV, creating it on the first write."""
    fieldnames = [
        "dataset",
        "image_path",
        "compliance",
        "label_true",
        "brisque",
        "niqe",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ── Metric computation ────────────────────────────────────────────────────────


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a [1, 3, H, W] float32 tensor in [0, 1]."""
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _safe_metric(fn: Any, tensor: torch.Tensor) -> float | None:
    """Call a pyiqa metric; return None if the image causes the metric to fail."""
    result: float | None = None
    with contextlib.suppress(ValueError, RuntimeError):
        result = fn(tensor).item()
    return result


def create_metric_fns() -> dict[str, Any]:
    """Instantiate the pyiqa metrics once; parameters are downloaded on first use."""
    return {name: pyiqa.create_metric(name) for name in ("brisque", "niqe", "qalign")}


def compute_metrics(image: Image.Image, metric_fns: dict[str, Any]) -> dict[str, float | None]:
    """Return BRISQUE and NIQE scores for one image; None on per-metric failure."""
    tensor = image_to_tensor(image)
    return {name: _safe_metric(fn, tensor) for name, fn in metric_fns.items()}


# ── AUC evaluation ────────────────────────────────────────────────────────────

METRIC_COLUMNS = ("brisque", "niqe")


def _metric_scores(group: pd.DataFrame, metric: str) -> tuple[int, float, float, float]:
    """AUC, AP, and FPR@95TPR (%) of one metric over one group.

    Returns ``(n, auc, ap, fpr95)``; the metrics are NaN when fewer than two
    classes remain.
    """
    valid = group.dropna(subset=["label_true", metric])
    labels = valid["label_true"].to_numpy(dtype=np.int64)
    if len(valid) == 0 or np.all(labels == labels[0]):
        return len(valid), np.nan, np.nan, np.nan

    scores = valid[metric].to_numpy(dtype=np.float64)
    auc = float(roc_auc_score(labels, scores)) * 100
    ap = float(average_precision_score(labels, scores)) * 100
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = float(np.interp(0.95, tpr, fpr)) * 100
    return len(valid), auc, ap, fpr95


def compute_aucs(results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-set AUC, AP, and FPR@95TPR of each metric: each dataset and its holdout.

    Each dataset contributes two rows: one over its own samples (``CheXpert``)
    and one over the pooled samples of all *other* datasets (``CheXpert
    Holdout``, leave-one-out). The positive class is *non-compliant*
    (``label_true`` = 1) and samples are ranked directly by the metric (higher
    BRISQUE/NIQE = worse perceptual quality). All values are percentages; rows
    with a missing label or metric value are dropped per metric.
    """
    rows: list[dict[str, Any]] = []
    for dataset in results["dataset"].unique():
        sets = (
            (str(dataset), results[results["dataset"] == dataset]),
            (f"{dataset} Holdout", results[results["dataset"] != dataset]),
        )
        for name, group in sets:
            row: dict[str, Any] = {"set": name}
            for metric in METRIC_COLUMNS:
                (
                    row[f"{metric}_n"],
                    row[f"{metric}_auc"],
                    row[f"{metric}_ap"],
                    row[f"{metric}_fpr95"],
                ) = _metric_scores(group, metric)
            rows.append(row)

    return pd.DataFrame(rows)


def report_aucs(results_path: Path, auc_path: Path) -> None:
    """Compute the holdout AUCs from the metrics CSV, save them, and print them."""
    results = pd.read_csv(results_path)
    auc_df = compute_aucs(results)

    auc_path.parent.mkdir(parents=True, exist_ok=True)
    auc_df.to_csv(auc_path, index=False)

    print("\nAUC / AP / FPR@95TPR (%) per dataset — own samples and leave-one-out holdout")
    print("(positive class: non-compliant):")
    print(auc_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"Metrics saved to {auc_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def compute_scores(args: argparse.Namespace, output_path: Path) -> None:
    """Orchestrate the full metric loop across all datasets."""
    data_root = resolve_path(args.data_root)

    print("Metrics : BRISQUE, NIQE")
    print(f"Limit   : {args.limit_per_dataset or 'all'} images per dataset")
    print(f"Results : {output_path}")

    existing: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        done_df = pd.read_csv(output_path, usecols=["dataset", "image_path"])
        existing = set(zip(done_df["dataset"].astype(str), done_df["image_path"].astype(str), strict=False))

    write_header = not output_path.exists() or not args.resume

    annotations = load_annotations(args.annotations, args.limit_per_dataset, args.seed)
    if annotations.empty:
        print(f"No annotated samples found in {args.annotations}, nothing to do.")
        return

    metric_fns = create_metric_fns()

    for dataset, df in annotations.groupby("dataset", sort=False):
        buffer: list[dict[str, Any]] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=str(dataset), unit="img"):
            image_path = str(row["image_path"]).strip()
            if (dataset, image_path) in existing:
                continue

            full_path = Path(image_path)
            if not full_path.is_absolute():
                full_path = data_root / full_path

            image = load_image(full_path)
            compliance = str(row.get("compliance", "")).strip()
            label_true = COMPLIANCE_MAP.get(compliance.lower())

            if image is None:
                buffer.append(
                    {
                        "dataset": dataset,
                        "image_path": image_path,
                        "compliance": compliance,
                        "label_true": label_true,
                        "brisque": None,
                        "niqe": None,
                        "status": "missing",
                    }
                )
            else:
                metrics = compute_metrics(image, metric_fns)
                ok = any(value is not None for value in metrics.values())
                buffer.append(
                    {
                        "dataset": dataset,
                        "image_path": image_path,
                        "compliance": compliance,
                        "label_true": label_true,
                        "brisque": metrics["brisque"],
                        "niqe": metrics["niqe"],
                        "status": "ok" if ok else "error",
                    }
                )

            if len(buffer) >= 50:
                write_rows(output_path, buffer, write_header=write_header)
                write_header = False
                buffer.clear()

        if buffer:
            write_rows(output_path, buffer, write_header=write_header)
            write_header = False

    print(f"Done. Results saved to {output_path}")


def main() -> None:
    """Compute the perceptual metrics (unless --auc-only) and report the holdout AUCs."""
    args = parse_args()
    output_path = args.out_dir / "perceptual_iqa.csv"
    auc_path = args.out_dir / "perceptual_iqa_metrics.csv"

    if not args.auc_only:
        compute_scores(args, output_path)

    if not output_path.exists():
        print(f"No metrics CSV found at {output_path}; run without --auc-only first.")
        return
    report_aucs(output_path, auc_path)


if __name__ == "__main__":
    main()
