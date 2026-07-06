"""Compliance non-compliance-cause statistics and figures.

Standalone entry point (``python -m scripts.stats.causes``) regenerates, from the
bundled annotations (``data/annotations/annotated_causes.csv``):

* ``data/stats/Causes_Chart.png`` — grouped bar chart of cause counts per dataset;
* ``data/stats/causes_stats.md`` — the per-dataset cause-count table.

The score-overlay figures, LaTeX percentage tables, and progress video from the paper
require the CAF score CSVs (not part of the dataset release) and are produced only when
``--scores`` / ``--encoder-scores`` are supplied.
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np

from data_compliance.utils.colors import COLOR_DICT
from data_compliance.utils.paths import get_repo_root

if TYPE_CHECKING:
    from collections.abc import Sequence

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13

FIG_SIZE = (13.5, 6.5)
VIDEO_FIG_SIZE = (11.2, 6.5)
BAR_WIDTH = 0.11


CAUSE_COLUMNS: list[tuple[str, str]] = [
    ("corrupted_images", "Corrupted Images"),
    ("not_a_thorax", "Not a Thorax"),
    ("incomplete_thorax", "Incomplete Thorax"),
    ("image_quality_problems", "Image Quality Problems"),
    ("area_not_valid", "Area Not Valid"),
    ("overlaying_objects", "Overlaying Objects"),
    ("non_canonical_positions", "Non-Canonical Positions"),
]

ENCODER_COLUMNS: dict[str, str] = {
    "CheXpert": "Encoder_CheXpert",
    "ChestX-ray8": "Encoder_ChestX-ray8",
    "MIMIC-CXR": "Encoder_MIMIC-CXR",
    "PadChest": "Encoder_PadChest",
    "All": "Encoder_All",
}

SELF_COLUMNS: dict[str, str] = {name: column.replace("Encoder_", "Self_") for name, column in ENCODER_COLUMNS.items()}


def _prefixed_encoder_columns(prefix: str) -> dict[str, str]:
    return {name: column.replace("Encoder_", f"{prefix}_") for name, column in ENCODER_COLUMNS.items()}


def _is_one(value: str) -> bool:
    return str(value).strip() in {"1", "1.0", "True", "true"}


def _normalize_score_path(score_path: str) -> str:
    raw = score_path.strip()
    if not raw:
        return raw

    parts = raw.split("/", 1)
    if len(parts) == 1:
        return raw

    dataset, remainder = parts[0], parts[1]
    for marker in ("train/", "valid/", "test/", "images/", "files/"):
        idx = remainder.find(marker)
        if idx >= 0:
            return f"{dataset}/{remainder[idx:]}"

    return raw


def load_annotated_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def compute_dataset_counts(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    preferred_dataset_order = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
    dataset_counts: dict[str, dict[str, int]] = {
        ds: {display_name: 0 for _, display_name in CAUSE_COLUMNS} | {"Compliant": 0} for ds in preferred_dataset_order
    }
    dataset_totals: dict[str, int] = dict.fromkeys(preferred_dataset_order, 0)

    for row in rows:
        image_path = str(row.get("image_path", "")).strip()
        if "/" not in image_path:
            continue

        dataset = image_path.split("/", 1)[0]
        if dataset not in dataset_counts:
            continue

        compliance = str(row.get("compliance", "")).strip().lower()
        dataset_totals[dataset] += 1
        if compliance == "compliant":
            dataset_counts[dataset]["Compliant"] += 1
            continue

        if compliance != "non-compliant":
            continue

        for column_name, display_name in CAUSE_COLUMNS:
            if _is_one(str(row.get(column_name, ""))):
                dataset_counts[dataset][display_name] += 1

    return dataset_counts, dataset_totals


def make_plot(dataset_counts: dict[str, dict[str, int]], dataset_totals: dict[str, int], output_pdf: Path) -> None:
    preferred_dataset_order = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
    datasets = [ds for ds in preferred_dataset_order if ds in dataset_counts]

    # Keep a deterministic cause order and force "Area Not Valid" to 5th position.
    preferred_order = [
        "Corrupted Images",
        "Not a Thorax",
        "Incomplete Thorax",
        "Image Quality Problems",
        "Area Not Valid",
        "Overlaying Objects",
        "Non-Canonical Positions",
    ]

    all_causes = {cause for counts in dataset_counts.values() for cause in counts}
    causes = [cause for cause in preferred_order if cause in all_causes]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # fig, ax = plt.subplots(figsize=(12,4))
    x = np.arange(len(datasets))
    bar_width = BAR_WIDTH

    for i, cause in enumerate(causes):
        off = (i - (len(causes) - 1) / 2) * bar_width
        values = [dataset_counts[ds].get(cause, 0) for ds in datasets]
        ax.bar(
            x + off,
            values,
            width=bar_width,
            label=cause,
            color=COLOR_DICT.get(cause, (0, 0, 0)),
        )

    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_xticks(x)
    non_compliant_counts = [dataset_totals[ds] - dataset_counts[ds].get("Compliant", 0) for ds in datasets]
    ax.set_xticklabels(
        [f"{ds}\n{count} Non-compliant" for ds, count in zip(datasets, non_compliant_counts, strict=False)]
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        title="Cause",
        ncols=1,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300)
    plt.close(fig)


def select_top_worst_rows_per_dataset(
    annotated_rows: list[dict[str, str]],
    score_csv_path: Path,
    score_column: str,
    n: int,
    score_dataset: str = "All",
) -> dict[str, list[dict[str, str]]]:
    annotated_by_path = {
        str(row.get("image_path", "")).strip(): row for row in annotated_rows if str(row.get("image_path", "")).strip()
    }

    heaps_by_dataset: dict[str, list[tuple[float, str]]] = {}
    with score_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty score CSV: {score_csv_path}")

        required = {"image_path", "dataset", score_column}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Score CSV {score_csv_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            if str(row.get("dataset", "")).strip() != score_dataset:
                continue

            normalized_path = _normalize_score_path(str(row.get("image_path", "")))
            annotated_row = annotated_by_path.get(normalized_path)
            if annotated_row is None:
                continue

            try:
                score_value = float(str(row.get(score_column, "")).strip())
            except ValueError:
                continue

            dataset = normalized_path.split("/", 1)[0]
            heap = heaps_by_dataset.setdefault(dataset, [])
            entry = (score_value, normalized_path)
            if len(heap) < n:
                heapq.heappush(heap, entry)
            elif score_value > heap[0][0]:
                heapq.heapreplace(heap, entry)

    top_rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    for dataset, heap in heaps_by_dataset.items():
        sorted_top = sorted(heap, key=lambda item: item[0], reverse=True)
        top_rows_by_dataset[dataset] = [annotated_by_path[path] for _, path in sorted_top]

    return top_rows_by_dataset


def compute_top_counts_by_dataset(
    datasets: list[str],
    top_rows_by_dataset: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, int]]:
    causes = [display_name for _, display_name in CAUSE_COLUMNS] + ["Compliant"]
    top_counts_by_dataset: dict[str, dict[str, int]] = {ds: dict.fromkeys(causes, 0) for ds in datasets}

    for ds in datasets:
        for row in top_rows_by_dataset.get(ds, []):
            compliance = str(row.get("compliance", "")).strip().lower()
            if compliance == "compliant":
                top_counts_by_dataset[ds]["Compliant"] += 1
                continue

            if compliance != "non-compliant":
                continue

            for col_name, cause_name in CAUSE_COLUMNS:
                if _is_one(str(row.get(col_name, ""))):
                    top_counts_by_dataset[ds][cause_name] += 1

    return top_counts_by_dataset


def load_ranked_rows_per_dataset(
    annotated_rows: list[dict[str, str]],
    score_csv_path: Path,
    score_column: str,
    score_dataset: str = "All",
) -> dict[str, list[dict[str, str]]]:
    annotated_by_path = {
        str(row.get("image_path", "")).strip(): row for row in annotated_rows if str(row.get("image_path", "")).strip()
    }

    score_by_dataset_and_path: dict[str, dict[str, float]] = {}
    with score_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty score CSV: {score_csv_path}")

        required = {"image_path", "dataset", score_column}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Score CSV {score_csv_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            if str(row.get("dataset", "")).strip() != score_dataset:
                continue

            normalized_path = _normalize_score_path(str(row.get("image_path", "")))
            if normalized_path not in annotated_by_path:
                continue

            try:
                score_value = float(str(row.get(score_column, "")).strip())
            except ValueError:
                continue

            dataset = normalized_path.split("/", 1)[0]
            dataset_scores = score_by_dataset_and_path.setdefault(dataset, {})
            previous = dataset_scores.get(normalized_path)
            if previous is None or score_value > previous:
                dataset_scores[normalized_path] = score_value

    ranked_rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    for dataset, scores in score_by_dataset_and_path.items():
        ranked_paths = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_rows_by_dataset[dataset] = [annotated_by_path[path] for path, _ in ranked_paths]

    return ranked_rows_by_dataset


def make_overlay_progress_video(
    dataset_counts: dict[str, dict[str, int]],
    dataset_totals: dict[str, int],
    ranked_rows_by_dataset: dict[str, list[dict[str, str]]],
    output_video: Path,
    max_n: int = 1000,
    fps: int = 30,
) -> None:
    import imageio.v2 as imageio  # noqa: PLC0415 (optional dependency, only needed for the video)

    preferred_dataset_order = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
    datasets = [ds for ds in preferred_dataset_order if ds in dataset_counts]
    causes = [display_name for _, display_name in CAUSE_COLUMNS]
    non_compliant_target = {ds: dataset_totals[ds] - dataset_counts[ds].get("Compliant", 0) for ds in datasets}

    # Precompute cumulative top-N counts so each frame only updates artist heights.
    cumulative: dict[str, dict[str, np.ndarray]] = {
        ds: {cause: np.zeros(max_n + 1, dtype=np.int32) for cause in causes} for ds in datasets
    }
    cumulative_non_compliant_seen: dict[str, np.ndarray] = {ds: np.zeros(max_n + 1, dtype=np.int32) for ds in datasets}

    for ds in datasets:
        rows = ranked_rows_by_dataset.get(ds, [])
        limit = min(max_n, len(rows))
        running = dict.fromkeys(causes, 0)
        running_non_compliant = 0
        for n in range(1, max_n + 1):
            if n <= limit:
                row = rows[n - 1]
                compliance = str(row.get("compliance", "")).strip().lower()
                if compliance == "non-compliant":
                    running_non_compliant += 1
                    for col_name, cause_name in CAUSE_COLUMNS:
                        if _is_one(str(row.get(col_name, ""))):
                            running[cause_name] += 1

            for cause in causes:
                cumulative[ds][cause][n] = running[cause]
            cumulative_non_compliant_seen[ds][n] = running_non_compliant

    x_stop = max_n
    for n in range(max_n + 1):
        if all(cumulative_non_compliant_seen[ds][n] >= non_compliant_target[ds] for ds in datasets):
            x_stop = n
            break

    frame_steps = list(range(x_stop + 1))
    if x_stop < max_n:
        frame_steps.append(max_n)

    base_max = 0
    for ds in datasets:
        for cause in causes:
            base_max = max(base_max, int(dataset_counts[ds].get(cause, 0)))
    y_max = max(1, int(base_max * 1.12))

    fig = plt.figure(figsize=VIDEO_FIG_SIZE)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.12, 1.0], hspace=0.05)
    ax_progress = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    fig.subplots_adjust(right=0.62, top=0.95, bottom=0.08)

    x = np.arange(len(datasets))
    bar_width = BAR_WIDTH

    overlay_bars: dict[str, list] = {cause: [] for cause in causes}
    legend_handles = []
    for i, cause in enumerate(causes):
        off = (i - (len(causes) - 1) / 2) * bar_width
        base_values = [dataset_counts[ds].get(cause, 0) for ds in datasets]
        color = COLOR_DICT.get(cause, (0.0, 0.0, 0.0))

        ax.bar(
            x + off,
            base_values,
            width=bar_width,
            color=color,
            alpha=0.5,
        )

        overlay_container = ax.bar(
            x + off,
            [0] * len(datasets),
            width=bar_width,
            color=color,
            edgecolor="none",
            linewidth=0.0,
        )
        overlay_bars[cause] = list(overlay_container)
        legend_handles.append(overlay_container.patches[0])

    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    fig.legend(
        handles=legend_handles,
        labels=causes,
        title="Cause",
        ncols=1,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(0.64, 0.5),
    )

    ax_progress.barh([0], [max_n], color=(0.8, 0.8, 0.8), alpha=0.35, height=0.55)
    progress_color = COLOR_DICT.get("Compliant", (0.0, 0.5, 0.0))
    progress_bar = ax_progress.barh([0], [0], color=progress_color, alpha=0.85, height=0.55)[0]
    progress_text = ax_progress.text(
        max_n,
        0.48,
        f"0/{max_n}",
        ha="right",
        va="bottom",
        fontsize=12,
    )
    ax_progress.set_xlim(0, max_n)
    ax_progress.set_ylim(-0.6, 0.9)
    ax_progress.set_yticks([])
    ax_progress.set_xticks([])
    ax_progress.set_title("", pad=0)
    for spine in ("left", "right", "top", "bottom"):
        ax_progress.spines[spine].set_visible(False)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_video),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        writer_any = cast("Any", writer)
        canvas_any = cast("Any", fig.canvas)
        for n in frame_steps:
            for cause in causes:
                vals = [int(cumulative[ds][cause][n]) for ds in datasets]
                for j, bar in enumerate(overlay_bars[cause]):
                    bar.set_height(vals[j])

            progress_bar.set_width(n)
            progress_text.set_text(f"{n}/{max_n}")
            ax.set_title("")

            fig.canvas.draw()
            frame = np.asarray(canvas_any.buffer_rgba())[..., :3]
            writer_any.append_data(frame)

    plt.close(fig)


def export_latex_percentage_table(
    dataset_counts: dict[str, dict[str, int]],
    top_counts_by_dataset: dict[str, dict[str, int]],
    output_tex: Path,
) -> None:
    preferred_dataset_order = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
    datasets = [ds for ds in preferred_dataset_order if ds in dataset_counts]
    causes = [display_name for _, display_name in CAUSE_COLUMNS]

    def esc(text: str) -> str:
        return text.replace("_", r"\_")

    header_cols = "l" + "c" * len(causes)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + header_cols + "}")
    lines.append(r"\toprule")
    lines.append("Dataset & " + " & ".join(esc(c) for c in causes) + r" \\")
    lines.append(r"\midrule")

    for ds in datasets:
        values: list[str] = []
        for cause in causes:
            base = float(dataset_counts[ds].get(cause, 0))
            top = float(top_counts_by_dataset.get(ds, {}).get(cause, 0))
            pct = 100.0 * top / base if base > 0 else 0.0
            values.append(f"{pct:.1f}\\%")
        lines.append(esc(ds) + " & " + " & ".join(values) + r" \\")

    all_values: list[str] = []
    for cause in causes:
        base_sum = float(sum(dataset_counts[ds].get(cause, 0) for ds in datasets))
        top_sum = float(sum(top_counts_by_dataset.get(ds, {}).get(cause, 0) for ds in datasets))
        pct = 100.0 * top_sum / base_sum if base_sum > 0 else 0.0
        all_values.append(f"{pct:.1f}\\%")

    lines.append(r"\midrule")
    lines.append("All & " + " & ".join(all_values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_latex_percentage_table_shared_baseline(
    row_names: list[str],
    baseline_counts: dict[str, int],
    top_counts_by_row: dict[str, dict[str, int]],
    output_tex: Path,
    row_label: str,
) -> None:
    causes = [display_name for _, display_name in CAUSE_COLUMNS]

    def esc(text: str) -> str:
        return text.replace("_", r"\_")

    header_cols = "l" + "c" * len(causes)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + header_cols + "}")
    lines.append(r"\toprule")
    lines.append(row_label + " & " + " & ".join(esc(c) for c in causes) + r" \\")
    lines.append(r"\midrule")

    for row_name in row_names:
        values: list[str] = []
        for cause in causes:
            base = float(baseline_counts.get(cause, 0))
            top = float(top_counts_by_row.get(row_name, {}).get(cause, 0))
            pct = 100.0 * top / base if base > 0 else 0.0
            values.append(f"{pct:.1f}\\%")
        lines.append(esc(row_name) + " & " + " & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_overlay_plot(
    dataset_counts: dict[str, dict[str, int]],
    dataset_totals: dict[str, int],
    top_rows_by_dataset: dict[str, list[dict[str, str]]],
    output_pdf: Path,
) -> None:
    preferred_dataset_order = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
    datasets = [ds for ds in preferred_dataset_order if ds in dataset_counts]
    causes = [display_name for _, display_name in CAUSE_COLUMNS] + ["Compliant"]

    top_counts_by_dataset = compute_top_counts_by_dataset(datasets, top_rows_by_dataset)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(datasets))
    bar_width = BAR_WIDTH

    for i, cause in enumerate(causes):
        off = (i - (len(causes) - 1) / 2) * bar_width
        if cause == "Compliant":
            base_values = [0 for _ in datasets]
        else:
            base_values = [dataset_counts[ds].get(cause, 0) for ds in datasets]
        overlay_values = [top_counts_by_dataset[ds].get(cause, 0) for ds in datasets]
        color = COLOR_DICT.get(cause, (0.0, 0.0, 0.0))

        ax.bar(
            x + off,
            base_values,
            width=bar_width,
            color=color,
            alpha=0.33,
        )
        ax.bar(
            x + off,
            overlay_values,
            width=bar_width,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            label=cause,
        )

    ax.set_ylabel("Count")
    ax.set_xticks(x)
    top_compliant_counts = [top_counts_by_dataset[ds].get("Compliant", 0) for ds in datasets]
    ax.set_xticklabels([f"{ds}\n{count} Compliant" for ds, count in zip(datasets, top_compliant_counts, strict=False)])
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        title="Cause",
        ncols=1,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300)
    plt.close(fig)


def compute_global_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {display_name: 0 for _, display_name in CAUSE_COLUMNS} | {"Compliant": 0}

    for row in rows:
        compliance = str(row.get("compliance", "")).strip().lower()
        if compliance == "compliant":
            counts["Compliant"] += 1
            continue

        if compliance != "non-compliant":
            continue

        for col_name, cause_name in CAUSE_COLUMNS:
            if _is_one(str(row.get(col_name, ""))):
                counts[cause_name] += 1

    return counts


def select_top_rows_by_encoder(
    annotated_rows: list[dict[str, str]],
    score_csv_path: Path,
    encoder_columns: dict[str, str],
    n: int,
    score_dataset: str = "All",
) -> dict[str, list[dict[str, str]]]:
    annotated_by_path = {
        str(row.get("image_path", "")).strip(): row for row in annotated_rows if str(row.get("image_path", "")).strip()
    }

    best_score_by_encoder_and_path: dict[str, dict[str, float]] = {encoder: {} for encoder in encoder_columns}

    with score_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty score CSV: {score_csv_path}")

        required = {"image_path", "dataset"} | set(encoder_columns.values())
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Score CSV {score_csv_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            if str(row.get("dataset", "")).strip() != score_dataset:
                continue

            normalized_path = _normalize_score_path(str(row.get("image_path", "")))
            if normalized_path not in annotated_by_path:
                continue

            for encoder, score_column in encoder_columns.items():
                try:
                    score_value = float(str(row.get(score_column, "")).strip())
                except ValueError:
                    continue

                encoder_scores = best_score_by_encoder_and_path[encoder]
                previous = encoder_scores.get(normalized_path)
                if previous is None or score_value > previous:
                    encoder_scores[normalized_path] = score_value

    top_rows_by_encoder: dict[str, list[dict[str, str]]] = {}
    for encoder in encoder_columns:
        ranked = sorted(
            best_score_by_encoder_and_path[encoder].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_paths = [path for path, _ in ranked[:n]]
        top_rows_by_encoder[encoder] = [annotated_by_path[path] for path in top_paths]

    return top_rows_by_encoder


def make_encoder_overlay_plot(
    all_counts: dict[str, int],
    top_rows_by_panel: dict[str, dict[str, list[dict[str, str]]]],
    output_pdf: Path,
) -> None:
    panel_specs = [("Self", SELF_COLUMNS), ("Encoder", ENCODER_COLUMNS)]
    causes = [display_name for _, display_name in CAUSE_COLUMNS] + ["Compliant"]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 11.8), sharex=False)
    bar_width = BAR_WIDTH
    legend_handles = None

    for ax, (panel_title, encoder_columns) in zip(axes, panel_specs, strict=False):
        encoders = list(encoder_columns.keys())
        top_rows_by_encoder = top_rows_by_panel[panel_title]
        top_counts_by_encoder = compute_top_counts_by_dataset(encoders, top_rows_by_encoder)
        x = np.arange(len(encoders))

        panel_handles = []
        for i, cause in enumerate(causes):
            off = (i - (len(causes) - 1) / 2) * bar_width
            if cause == "Compliant":
                base_values = [0 for _ in encoders]
            else:
                base_values = [all_counts.get(cause, 0) for _ in encoders]
            overlay_values = [top_counts_by_encoder[enc].get(cause, 0) for enc in encoders]
            color = COLOR_DICT.get(cause, (0.0, 0.0, 0.0))

            ax.bar(
                x + off,
                base_values,
                width=bar_width,
                color=color,
                alpha=0.5,
            )
            overlay_bar = ax.bar(
                x + off,
                overlay_values,
                width=bar_width,
                color=color,
                edgecolor="none",
                linewidth=0.0,
                label=cause,
            )
            panel_handles.append(overlay_bar.patches[0])

        if legend_handles is None:
            legend_handles = panel_handles

        ax.set_title(panel_title, loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        top_non_compliant_counts = [
            len(top_rows_by_encoder[enc]) - top_counts_by_encoder[enc].get("Compliant", 0) for enc in encoders
        ]
        ax.set_xticklabels(
            [f"{enc}\n{count} Non-compliant" for enc, count in zip(encoders, top_non_compliant_counts, strict=False)]
        )
        ax.set_xlim(-0.6, len(encoders) - 0.4)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("")
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0)
    fig.legend(
        handles=legend_handles,
        labels=causes,
        title="Cause",
        ncols=3,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=600, bbox_inches="tight" if output_pdf.suffix.lower() == ".png" else None)
    plt.close(fig)


def write_stats_markdown(
    dataset_counts: dict[str, dict[str, int]],
    dataset_totals: dict[str, int],
    output_md: Path,
) -> None:
    """Write the per-dataset non-compliance-cause counts to a Markdown table file."""
    causes = [display_name for _, display_name in CAUSE_COLUMNS]
    lines: list[str] = ["# CSV Statistics", "", "# Database Statistics", ""]

    for dataset, counts in dataset_counts.items():
        total = dataset_totals.get(dataset, 0)
        non_compliant = total - counts.get("Compliant", 0)
        lines.append(f"# Dataset: {dataset}")
        lines.append(f"Total non-compliant images: {non_compliant}")
        lines.append("")
        lines.append("## Causes for non-compliant images")
        for cause in causes:
            count = counts.get(cause, 0)
            pct = (count / non_compliant * 100) if non_compliant else 0.0
            lines.append(f"- {cause}: {count} ({pct:.2f}%)")
        no_cause = non_compliant - sum(counts.get(cause, 0) for cause in causes if counts.get(cause, 0))
        lines.append(f"- No cause: {max(no_cause, 0)} ({(max(no_cause, 0) / non_compliant * 100) if non_compliant else 0.0:.2f}%)")
        lines.append("")

    lines.append("![Cause Counts Bar Chart](Causes_Chart.png)")
    lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Regenerate compliance-cause statistics and figures.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repo_root / "data" / "annotations" / "annotated_causes.csv",
        help="Annotations CSV (default: bundled data/annotations/annotated_causes.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "data" / "stats",
        help="Output folder for the chart and stats table (default: data/stats).",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Optional CAF score CSV (column MD_k1) to render the top-N overlay figures and video.",
    )
    parser.add_argument(
        "--encoder-scores",
        type=Path,
        default=None,
        help="Optional per-encoder Mahalanobis score CSV to render the encoder-grouped overlay/table.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Regenerate the cause bar chart + stats table, plus optional score-based overlays."""
    args = parse_args(argv)

    rows = load_annotated_rows(args.annotations)
    if not rows:
        raise ValueError(f"No rows found in {args.annotations}")

    dataset_counts, dataset_totals = compute_dataset_counts(rows)

    base_chart = args.out / "Causes_Chart.png"
    make_plot(dataset_counts, dataset_totals, base_chart)
    print(f"Saved chart to: {base_chart}")

    stats_md = args.out / "causes_stats.md"
    write_stats_markdown(dataset_counts, dataset_totals, stats_md)
    print(f"Saved stats to: {stats_md}")

    if args.scores is None and args.encoder_scores is None:
        print("Score CSVs not supplied; skipping the top-N overlay figures, LaTeX tables, and video.")
        return

    _render_score_overlays(rows, dataset_counts, dataset_totals, args)


def _render_score_overlays(
    rows: list[dict[str, str]],
    dataset_counts: dict[str, dict[str, int]],
    dataset_totals: dict[str, int],
    args: argparse.Namespace,
) -> None:
    """Render the paper's score-dependent overlay figures, tables, and progress video."""
    datasets = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]

    if args.scores is not None:
        top_rows_by_dataset = select_top_worst_rows_per_dataset(
            annotated_rows=rows, score_csv_path=args.scores, score_column="MD_k1", n=200, score_dataset="All"
        )
        make_overlay_plot(dataset_counts, dataset_totals, top_rows_by_dataset, args.out / "Causes_Top200.png")
        top_counts = compute_top_counts_by_dataset(datasets, top_rows_by_dataset)
        export_latex_percentage_table(dataset_counts, top_counts, args.out / "Causes_Top200_Percentages.tex")

        ranked_rows = load_ranked_rows_per_dataset(
            annotated_rows=rows, score_csv_path=args.scores, score_column="MD_k1", score_dataset="All"
        )
        make_overlay_progress_video(
            dataset_counts=dataset_counts,
            dataset_totals=dataset_totals,
            ranked_rows_by_dataset=ranked_rows,
            output_video=args.out / "Causes_TopN_Progress.mp4",
            max_n=1000,
            fps=30,
        )

    if args.encoder_scores is not None:
        encoder_top_rows = select_top_rows_by_encoder(
            annotated_rows=rows, score_csv_path=args.encoder_scores, encoder_columns=ENCODER_COLUMNS, n=400, score_dataset="All"
        )
        self_top_rows = select_top_rows_by_encoder(
            annotated_rows=rows, score_csv_path=args.encoder_scores, encoder_columns=SELF_COLUMNS, n=400, score_dataset="All"
        )
        global_counts = compute_global_counts(rows)
        make_encoder_overlay_plot(
            global_counts, {"Self": self_top_rows, "Encoder": encoder_top_rows}, args.out / "Causes_Top400_ByEncoder.png"
        )
        encoder_top_counts = compute_top_counts_by_dataset(list(ENCODER_COLUMNS.keys()), encoder_top_rows)
        export_latex_percentage_table_shared_baseline(
            row_names=list(ENCODER_COLUMNS.keys()),
            baseline_counts=global_counts,
            top_counts_by_row=encoder_top_counts,
            output_tex=args.out / "Causes_Top400_Percentages.tex",
            row_label="Encoder",
        )


if __name__ == "__main__":
    main()
