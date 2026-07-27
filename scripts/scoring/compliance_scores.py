r"""Persist normalized CAF compliance scores from the Reference Scheme Comparison.

Raw Mahalanobis distances against the CheXpert reference prior (the
Aggregated federated scheme, see ``scripts/scoring/score.py``) are
Gamma-normalised into a ``[0, 1]`` compliance score (higher = more compliant)
via a federated fit / aggregate / transform across the three training clients.

Usage
-----
::

    uv run python -m scripts.scoring.compliance_scores

Output
------
``results/reference_schemes/compliance_normalized.csv``
    Columns: ``image_path``, ``dataset``, ``mahalanobis`` (raw Aggregated
    distance against the CheXpert prior), ``compliance`` (Gamma-normalised, [0, 1]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from DeMICAF.caf.normalization import aggregate, fit, transform
from DeMICAF.utils.paths import get_results_root

# The compliance score = Mahalanobis distance to the CheXpert reference prior,
# using the Aggregated federated reference scheme (see scripts/scoring/score.py).
PRIOR_DATASET = "CheXpert"
SCHEME_COLUMN = "CheXpert_Aggregated_mahala"
CLIENTS = ["ChestX-ray8", "MIMIC-CXR", "PadChest"]
NORM_METHOD = "gamma"


def compute_compliance(scores_csv: Path) -> pd.DataFrame:
    """Load raw Mahalanobis scores and return per-image normalised compliance.

    Parameters
    ----------
    scores_csv
        Path to ``results/reference_schemes/100k_MahalaScores.csv`` (one row per
        image, with a ``CheXpert_Aggregated_mahala`` column).

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
    results_root = get_results_root() / "reference_schemes"
    parser = argparse.ArgumentParser(
        description="Persist normalized CAF compliance scores from the Reference Scheme Comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=results_root / "100k_MahalaScores.csv",
        help="Raw reference-scheme Mahalanobis scores CSV (see scripts/scoring/score.py).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=results_root / "compliance_normalized.csv",
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
