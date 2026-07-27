"""Exploratory data analysis.

Generates the per-dataset demographic figures (position, sex, age) from the raw
metadata CSVs of the four chest X-ray datasets.

Usage (from the repository root)::

    python -m scripts.stats.eda --data-root "$CXR_ROOT" --out <results-dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DeMICAF.utils.colors import COLOR_DICT
from DeMICAF.utils.paths import get_cxr_root, get_results_root
from DeMICAF.utils.plotting import FIG_SIZE, annotate_bars, apply_paper_style, use_thousands_axis

POSITION_MAP = {
    "AP": "Frontal (AP)",
    "PA": "Frontal (PA)",
    "LL": "Lateral",
    "RL": "Lateral",
    "L": "Lateral",
    "LATERAL": "Lateral",
    "AP_horizontal": "Frontal (AP)",
}

ALL_GENDERS = ["Male", "Female", "Unknown"]

#: MIMIC-CXR demographics are not in the public metadata CSV; these are the
#: published cohort statistics (sex split %, age bin %) for 371,858 studies.
MIMIC_TOTAL = 371858
MIMIC_SEX_PERCENTAGES = [52.17, 47.83, 0.0]
MIMIC_AGE_PERCENTAGES = [2.2, 19.51, 37.20, 34.12, 6.96]


def load_metadata(data_root: Path) -> dict[str, pd.DataFrame]:
    """Load and harmonize Sex / Position / Age columns across the four datasets."""
    meta = {
        "CheXpert": pd.read_csv(data_root / "CheXpert" / "train.csv"),
        "ChestX-ray8": pd.read_csv(data_root / "ChestX-ray8" / "Data_Entry_2017_v2020.csv"),
        "MIMIC-CXR": pd.read_csv(data_root / "MIMIC-CXR" / "mimic-cxr-2.0.0-metadata.csv"),
        "PadChest": pd.read_csv(data_root / "PadChest" / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"),
    }

    meta["CheXpert"]["Sex"] = meta["CheXpert"]["Sex"].fillna("Unknown")
    meta["CheXpert"]["Position"] = meta["CheXpert"]["AP/PA"].map(POSITION_MAP).fillna("Lateral")
    meta["CheXpert"]["Age"] = meta["CheXpert"]["Age"].fillna("Unknown")

    meta["ChestX-ray8"]["Sex"] = (
        meta["ChestX-ray8"]["Patient Gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown")
    )
    meta["ChestX-ray8"]["Position"] = meta["ChestX-ray8"]["View Position"].map(POSITION_MAP).fillna("Unknown")
    meta["ChestX-ray8"]["Age"] = meta["ChestX-ray8"]["Patient Age"].fillna("Unknown")

    meta["PadChest"]["Sex"] = meta["PadChest"]["PatientSex_DICOM"].map({"M": "Male", "F": "Female"}).fillna("Unknown")
    meta["PadChest"]["Position"] = meta["PadChest"]["Projection"].map(POSITION_MAP).fillna("Unknown")
    study_year = pd.to_numeric(meta["PadChest"]["StudyDate_DICOM"].astype(str).str[:4], errors="coerce")
    birth_year = pd.to_numeric(meta["PadChest"]["PatientBirth"], errors="coerce")
    meta["PadChest"]["Age"] = study_year - birth_year

    meta["MIMIC-CXR"]["Position"] = meta["MIMIC-CXR"]["ViewPosition"].map(POSITION_MAP).fillna("Unknown")

    return meta


def _save_bar_plot(
    labels: list[str],
    counts: list[float],
    colors: list[tuple[float, float, float]],
    save_path: Path,
    xlabel: str | None = None,
    tick_labels: list[str] | None = None,
) -> None:
    """Save a paper-style annotated bar plot."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(range(len(labels)), counts, color=colors)
    max_count = max(counts) if counts else 1
    ax.set_ylim(0, max_count * 1.10)
    if tick_labels is not None:
        tick_positions = np.arange(len(tick_labels)) - 0.5
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    annotate_bars(ax, bars)
    use_thousands_axis(ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Image saved to: {save_path}")


def plot_demographics(meta: dict[str, pd.DataFrame], save_folder: Path) -> None:
    """Generate position, sex and age figures for every dataset."""
    sex_colors = [COLOR_DICT["Male"], COLOR_DICT["Female"], COLOR_DICT["Unknown"]]

    all_positions: set[str] = set()
    for dataset in ["CheXpert", "ChestX-ray8", "PadChest"]:
        all_positions.update(meta[dataset]["Position"].dropna().unique())
    all_positions.add("Unknown")
    positions = sorted(all_positions)
    position_colors = [COLOR_DICT.get(pos, COLOR_DICT["Unknown"]) for pos in positions]

    # Highest age across datasets, for a consistent final age bin.
    all_ages = [
        pd.to_numeric(meta[ds]["Age"], errors="coerce").dropna() for ds in ["CheXpert", "ChestX-ray8", "PadChest"]
    ]
    max_age = int(np.nanmax(np.concatenate(all_ages))) if all_ages else 100

    for dataset in ["CheXpert", "ChestX-ray8", "PadChest", "MIMIC-CXR"]:
        print(f"Dataset: {dataset}")

        position_counts = meta[dataset]["Position"].fillna("Unknown").value_counts().reindex(positions).fillna(0)
        _save_bar_plot(
            positions,
            list(position_counts.values),
            position_colors,
            save_folder / f"{dataset}_Position.pdf",
        )

        if dataset == "MIMIC-CXR":
            sex_counts = [p / 100 * MIMIC_TOTAL for p in MIMIC_SEX_PERCENTAGES]
        else:
            sex_counts = list(meta[dataset]["Sex"].value_counts().reindex(ALL_GENDERS).fillna(0).values)
        _save_bar_plot(ALL_GENDERS, sex_counts, sex_colors, save_folder / f"{dataset}_Sex.pdf")

        bin_labels = ["0-20", "20-40", "40-60", "60-80", f"80-{max_age}"]
        if dataset == "MIMIC-CXR":
            counts = [int(p / 100 * MIMIC_TOTAL) for p in MIMIC_AGE_PERCENTAGES]
            age_unknown_count = 0
        else:
            age_numeric = pd.to_numeric(meta[dataset]["Age"], errors="coerce")
            age_known = age_numeric.dropna()
            age_unknown_count = int(age_numeric.isna().sum())
            bins = np.array([0, 20, 40, 60, 80, max_age + 1])
            hist, _ = np.histogram(age_known, bins=bins)
            counts = list(hist)

        _save_bar_plot(
            bin_labels,
            [float(c) for c in counts],
            [COLOR_DICT["Age"]] * len(bin_labels),
            save_folder / f"{dataset}_Age.pdf",
            xlabel=f"{age_unknown_count} Unknown" if age_unknown_count > 0 else None,
            tick_labels=["0", "20", "40", "60", "80", str(max_age)],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset statistics and demographic figures.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=get_cxr_root(),
        help="Folder containing the per-dataset metadata CSVs (default: $CXR_ROOT or <repo_root>/data/chest_xray).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder for figures (default: <repo_root>/results/demographics).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    apply_paper_style()
    save_folder = args.out if args.out is not None else get_results_root() / "demographics"
    save_folder.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(args.data_root)
    plot_demographics(meta, save_folder)


if __name__ == "__main__":
    main()
