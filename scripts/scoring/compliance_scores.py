r"""Persist normalized CAF compliance scores for the audit eval set.

Reproduces the compliance score used in ``notebooks/federated/normalization.ipynb``:
raw Mahalanobis distances against the CheXpert reference prior (the
``Average Weighted`` federated scheme) are Gamma-normalised into a ``[0, 1]``
compliance score (higher = more compliant) via a federated fit / aggregate /
transform across the three training clients.

The result is written once to ``Assets/Scores/compliance_normalized.csv`` so the
audit pipeline can join compliance scores by ``image_path`` without re-fitting.

Usage
-----
::

    uv run python -m scripts.federated.compliance_scores

Output
------
``Assets/Scores/compliance_normalized.csv``
    Columns: ``image_path``, ``dataset``, ``mahalanobis`` (raw Average Weighted
    distance against the CheXpert prior), ``compliance`` (Gamma-normalised, [0, 1]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_compliance.caf.normalization import aggregate, fit, transform
from data_compliance.utils.paths import get_repo_root

# The compliance score = Mahalanobis distance to the CheXpert reference prior,
# using the "Average Weighted" federated reference scheme (matches normalization.ipynb).
PRIOR_DATASET = "CheXpert"
SCHEME_COLUMN = "CheXpert_Average_Weighted_mahala"
CLIENTS = ["ChestX-ray8", "MIMIC-CXR", "PadChest"]
NORM_METHOD = "gamma"


def compute_compliance(scores_csv: Path) -> pd.DataFrame:
    """Load raw Mahalanobis scores and return per-image normalised compliance.

    Parameters
    ----------
    scores_csv
        Path to ``Results/FA1/100k_MahalaScores.csv`` (one row per image, with a
        ``CheXpert_Average_Weighted_mahala`` column).

    Returns
    -------
    pandas.DataFrame
        Columns ``image_path``, ``dataset``, ``mahalanobis``, ``compliance`` for
        the three training clients (the CheXpert reference rows are dropped).
    """
    df = pd.read_csv(scores_csv, usecols=["image_path", SCHEME_COLUMN])
    df["dataset"] = df["image_path"].str.split("/").str[0]
    df = df[df["dataset"].isin(CLIENTS)].copy()
    df = df.dropna(subset=[SCHEME_COLUMN]).reset_index(drop=True)
    df = df.rename(columns={SCHEME_COLUMN: "mahalanobis"})

    # Federated Gamma normalisation: fit per client, aggregate (uniform weights),
    # then transform every image's distance with the shared global parameters.
    client_params = [fit(df.loc[df["dataset"] == client, "mahalanobis"], method=NORM_METHOD) for client in CLIENTS]
    global_params = aggregate(client_params)
    df["compliance"] = transform(df["mahalanobis"].to_numpy(), global_params)

    return df[["image_path", "dataset", "mahalanobis", "compliance"]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(
        description="Persist normalized CAF compliance scores for the audit eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=repo_root / "Results" / "FA1" / "100k_MahalaScores.csv",
        help="Raw FA1 Mahalanobis scores CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root / "Assets" / "Scores" / "compliance_normalized.csv",
        help="Destination for the normalised compliance scores.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    if not args.scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {args.scores_csv}")

    out = compute_compliance(args.scores_csv)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Saved {len(out):,} compliance scores → {args.out_csv}")
    for client in CLIENTS:
        sub = out[out["dataset"] == client]["compliance"]
        print(
            f"  {client:14s}: n={len(sub):4d}  compliance mean={sub.mean():.3f}  "
            f"range=[{sub.min():.3f}, {sub.max():.3f}]"
        )


if __name__ == "__main__":
    main()
