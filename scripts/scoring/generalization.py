import argparse
import copy
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from matplotlib.transforms import Bbox
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from DeMICAF.caf.encoder import encode, train
from DeMICAF.caf.scorer import mahalanobis_distance, reference_vector
from DeMICAF.data.base import AugmentedDataset, DatasetBase, multi_view_collate
from DeMICAF.data.datasets import ChestXray, ChestXrayDataset
from DeMICAF.evaluation.auc import compute_auc, compute_fpr_at_tpr
from DeMICAF.utils.colors import COLOR_DICT
from DeMICAF.utils.io import load_features
from DeMICAF.utils.io import upsert_scores_csv as save_scores_csv
from DeMICAF.utils.paths import get_cxr_root, get_results_root
from DeMICAF.utils.plotting import apply_paper_style
from DeMICAF.utils.seeding import seed_worker, set_seed

study = "Generalization"
base_datasets = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
framework = "SupCon"
batch_size = 256
epochs = 100
l_r = 1e-4
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
num_workers = 8
train_encoder = True
get_features = True
specific_models: list[str] = []
percentage = True


class ImageNetResNet18Features(nn.Module):
    """ResNet-18 feature extractor compatible with encode(..., head=False)."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt RGB pretrained stem to grayscale by averaging channel weights.
        conv1_rgb = backbone.conv1
        conv1_gray = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1_gray.weight.copy_(conv1_rgb.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = conv1_gray

        self.feats = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor, head: bool = False) -> torch.Tensor:
        del head
        return self.feats(x).flatten(1)


def export_methods_table(csv_path: Path, caption: str, label: str, percentage=True) -> None:
    """Export a LaTeX table from a CSV file, matching the provided example's structure and formatting."""
    df = pd.read_csv(csv_path)
    # Map 'MIMIC-CXR' to 'MIMIC' in columns and Encoder values
    df = df.rename(columns={"MIMIC-CXR": "MIMIC"})
    if "Encoder" in df.columns:
        df["Encoder"] = df["Encoder"].replace("MIMIC-CXR", "MIMIC")

    col_order = ["Reference", "Encoder", "CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All", "Holdout"]
    ENCODER_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All"]
    SELF_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All", "ImageNet", "Aggregated"]

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        f"    \\caption{{{caption}}} \\label{{{label}}}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{0.6cm}|c|c|c|c|c|c|c}",
        r"        \hline",
        r"        \multicolumn{2}{c|}{Scoring Method} & \multicolumn{6}{c}{Evaluation Dataset} \\",
        r"        \hline",
        r"        {Ref.} & {Encoder} & CheXpert & ChestX-ray8 & MIMIC & PadChest & All & Hold-Out \\",
        r"        \hhline{========}",
    ]

    for ref_type, multirow, row_order in [
        ("Adaptive", r"\multirow[c]{7}{*}{\shortstack{S\\e\\l\\f}}", SELF_ORDER),
        ("Prior", r"\multirow[c]{5}{*}{\shortstack{E\\n\\c\\o\\d\\e\\r}}", ENCODER_ORDER),
    ]:
        ref_rows = df[df["Reference"].str.lower() == ref_type.lower()]
        if ref_rows.empty:
            continue

        if ref_type == "Prior":
            lines.append(r"        \hhline{========}")

        # Hard-enforce vertical order
        ref_rows = ref_rows.set_index("Encoder").reindex(row_order).reset_index()
        total = len(ref_rows)

        for idx, (_, row) in enumerate(ref_rows.iterrows()):
            # Hard-coded separator between "All" (idx=4) and "ImageNet" (idx=5) in Self group
            if ref_type == "Adaptive" and idx == 5:
                lines.append(r"        \hhline{~|=======}")

            row_dict = row.to_dict()
            row_list = []
            row_list.append(multirow if idx == 0 else "")

            encoder = str(row_dict["Encoder"])
            for col in col_order[1:]:
                if col == "Reference":
                    continue
                val = row_dict.get(col, None)
                cell = ""
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    cell = ""
                elif isinstance(val, int | float):
                    v = val * 100 if percentage else val
                    if encoder == col and encoder in ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest"]:
                        cell = r"\cellcolor{gray!10} " + f"{v:.2f}"
                    else:
                        cell = f"{v:.2f}"
                else:
                    cell = str(val)
                row_list.append(cell)

            lines.append("        " + " & ".join(row_list) + r" \\")

            is_last = idx == total - 1
            is_before_hhline = ref_type == "Adaptive" and idx == 4  # suppress \cline before \hhline
            if not is_last and not is_before_hhline:
                lines.append(r"        \cline{2-8}")

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path = csv_path.with_suffix(".tex")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_scaling_table(subset_tables: dict, caption: str, label: str, outdir: Path, percentage=True) -> None:
    """Export a LaTeX table from a dict of DataFrames (keys: '100k', '10k', '1k').

    Focuses on the All and Holdout columns for each subset.
    """
    subset_names = ["100k", "10k", "1k"]
    dfs = []
    for s in subset_names:
        df = subset_tables[s].copy()
        if "Encoder" in df.columns:
            df["Encoder"] = df["Encoder"].replace("MIMIC-CXR", "MIMIC")
        dfs.append(df)

    ENCODER_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All"]
    SELF_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All", "ImageNet", "Aggregated"]

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        f"    \\caption{{{caption}}} \\label{{{label}}}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{0.6cm}|c|>{\centering\arraybackslash}m{1.2cm}"
        r"|c|c|>{\centering\arraybackslash}m{1.2cm}|c|c}",
        r"        \hline",
        r"        \multicolumn{2}{c|}{Scoring Method} & \multicolumn{3}{c|}{All} & \multicolumn{3}{c}{Hold-Out} \\",
        r"        \hline",
        r"        {Ref.} & {Encoder} & 100k & 10k & 1k & 100k & 10k & 1k \\",
        r"        \hhline{========}",
    ]

    for ref_type, multirow, row_order in [
        ("Adaptive", r"\multirow[c]{7}{*}{\shortstack{S\\e\\l\\f}}", SELF_ORDER),
        ("Prior", r"\multirow[c]{5}{*}{\shortstack{E\\n\\c\\o\\d\\e\\r}}", ENCODER_ORDER),
    ]:
        ref_rows = [df[df["Reference"].str.lower() == ref_type.lower()] for df in dfs]
        if all(r.empty for r in ref_rows):
            continue
        if ref_type == "Prior":
            lines.append(r"        \hhline{========}")
        ref_rows = [r.set_index("Encoder").reindex(row_order).reset_index() for r in ref_rows]
        total = len(row_order)
        for idx in range(total):
            if ref_type == "Adaptive" and idx == 5:
                lines.append(r"        \hhline{~|=======}")
            row_list = []
            row_list.append(multirow if idx == 0 else "")
            encoder = str(row_order[idx])
            row_list.append(encoder)
            for col in ["All", "Holdout"]:
                # Get 100k value for diff
                val_100k = None
                if idx < len(ref_rows[0]):
                    row_100k = ref_rows[0].iloc[idx]
                    val_100k = row_100k.get(col, None)
                    if isinstance(val_100k, int | float) and percentage:
                        val_100k = val_100k * 100
                for dfi in range(3):
                    val = None
                    if idx < len(ref_rows[dfi]):
                        row = ref_rows[dfi].iloc[idx]
                        val = row.get(col, None)
                    cell = ""
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        cell = ""
                    elif isinstance(val, int | float):
                        v = val * 100 if percentage else val
                        cell = f"{v:.2f}"
                        # Add difference for 10k/1k
                        if dfi > 0 and isinstance(val_100k, int | float) and val_100k is not None:
                            diff = v - val_100k
                            if abs(diff) > 1e-6:
                                cell += r" {\scriptsize \textit" + ("+" if diff >= 0 else "") + f"{diff:.2f}" + "}"
                    else:
                        cell = str(val)
                    row_list.append(cell)
            lines.append("        " + " & ".join(row_list) + r" \\")
            is_last = idx == total - 1
            is_before_hhline = ref_type == "Adaptive" and idx == 4
            if not is_last and not is_before_hhline:
                lines.append(r"        \cline{2-8}")
    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    # Output to file (optional: could return as string)
    out_path = outdir / "auc_scaling.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")


#: Canonical display order for datasets across the generalization figures;
#: each list below is filtered down from this fixed order.
generalization_dataset_order = ["ImageNet", "CheXpert", "MIMIC-CXR", "ChestX-ray8", "PadChest", "All", "Holdout"]

#: Pretty labels for datasets whose display name differs from the internal key.
generalization_dataset_labels = {"ChestX-ray8": "ChestX-ray14"}

#: Short x-tick abbreviations for the D_E encoder training datasets, keyed to
#: the internal dataset name (not the display label).
generalization_dataset_abbrev = {
    "ImageNet": "IN",
    "CheXpert": "CX",
    "MIMIC-CXR": "MM",
    "ChestX-ray8": "C14",
    "PadChest": "PC",
    "All": "All",
}


def _display_label(name: str) -> str:
    return generalization_dataset_labels.get(name, name)


#: Scoring (client) datasets D_k plotted as series -- always all six, so a
#: dataset's line color is stable across every generalization figure.
generalization_scoring_datasets = [
    name for name in generalization_dataset_order if name in [*base_datasets, "All", "Holdout"]
]

#: D_E encoder training datasets, in canonical order. ImageNet only has a
#: Self reference (an untrained encoder has no "own" data to be a Prior for),
#: so the Prior domain below drops it.
generalization_de_domain = [
    name for name in generalization_dataset_order if name in [*base_datasets, "ImageNet", "All"]
]

#: (display label, Reference value, D_E x-axis domain) per reference scheme.
generalization_reference_specs = [
    ("Prior", "Prior", [name for name in generalization_de_domain if name != "ImageNet"]),
    ("Adaptive", "Adaptive", generalization_de_domain),
]


def export_generalization_plots(
    auc_csv_path: Path,
    fpr_csv_path: Path,
    plot_dir: Path,
    subset: str,
) -> None:
    r"""Plot generalization AUC / FPR@95TPR against the encoder training data.

    Four figures are produced -- (Prior, Adaptive) reference x (AUC, FPR@95TPR)
    metric. The x-axis is the encoder training data :math:`\mathcal{D}_E`; one
    line per scoring (client) data :math:`\mathcal{D}_k`
    (:data:`generalization_scoring_datasets`), colored consistently with the
    rest of the paper's figures (:data:`DeMICAF.utils.colors.COLOR_DICT`). AUC
    lines are solid, FPR@95TPR lines are dashed.

    The Prior/Adaptive pair for each metric shares one y-axis scale (AUC on
    the left, FPR@95TPR on the right); only the outer panel of each pair
    (Prior AUC, Adaptive FPR@95TPR) draws its tick labels, so the four images
    compose into a single labeled row of subfigures.
    """
    metric_specs = [
        ("auc", auc_csv_path, "AUC (%)", "-"),
        ("fpr95", fpr_csv_path, "FPR@95TPR (%)", "--"),
    ]

    apply_paper_style()

    dataframes: dict[str, pd.DataFrame] = {}
    for metric_key, csv_path, _, _ in metric_specs:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"Skipping generalization plots: {csv_path} not found (run with --auc first).")
            return
        df = pd.read_csv(csv_path)
        required_columns = ["Reference", "Encoder", *generalization_scoring_datasets]
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for generalization plot in {csv_path}: {missing}")
        dataframes[metric_key] = df

    def series_values(df: pd.DataFrame, reference_mode: str, d_e_values: list[str], d_k: str) -> list[float]:
        values = []
        for d_e in d_e_values:
            row = df[(df["Reference"] == reference_mode) & (df["Encoder"] == d_e)]
            value = row.iloc[-1][d_k] if not row.empty else None
            values.append(float(value) * 100 if pd.notna(value) else np.nan)
        return values

    def metric_ylim(metric_key: str) -> tuple[float, float]:
        df = dataframes[metric_key]
        all_values = [
            value
            for _, reference_mode, d_e_values in generalization_reference_specs
            for d_k in generalization_scoring_datasets
            for value in series_values(df, reference_mode, d_e_values, d_k)
            if pd.notna(value)
        ]
        low, high = min(all_values), max(all_values)
        pad = (high - low) * 0.08 or 1.0
        return low - pad, high + pad

    ylims = {metric_key: metric_ylim(metric_key) for metric_key, _, _, _ in metric_specs}

    for metric_key, _, ylabel, linestyle in metric_specs:
        df = dataframes[metric_key]
        is_fpr = metric_key == "fpr95"

        for ref_label, reference_mode, d_e_values in generalization_reference_specs:
            fig, ax = plt.subplots(figsize=(2, 2))
            x_positions = list(range(len(d_e_values)))

            for d_k in generalization_scoring_datasets:
                y_values = series_values(df, reference_mode, d_e_values, d_k)
                ax.plot(
                    x_positions,
                    y_values,
                    color=COLOR_DICT.get(d_k, (0.5, 0.5, 0.5)),
                    linestyle=linestyle,
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    label=d_k,
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels([generalization_dataset_abbrev[d_e] for d_e in d_e_values])
            # ax.set_xlabel(r"$\mathcal{D}_E$")
            # plt.setp(ax.get_xticklabels(), rotation=00, ha="right")
            ax.set_ylim(*ylims[metric_key])
            ax.grid(alpha=0.15)

            # Only the outer panel of each metric's Prior/Adaptive pair labels its axis.
            show_labels = (ref_label == "Adaptive") if is_fpr else (ref_label == "Prior")
            if is_fpr:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.tick_params(labelright=show_labels)
            else:
                ax.tick_params(labelleft=show_labels)
            if show_labels:
                ax.set_ylabel(ylabel)

            plot_path = Path(plot_dir) / f"generalization_{ref_label.lower()}_{metric_key}_{subset}.pdf"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.canvas.draw()
            tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
            expanded_bbox = Bbox.from_extents(tight_bbox.x0, tight_bbox.y0, tight_bbox.x1, tight_bbox.y1)
            fig.savefig(plot_path, dpi=300, bbox_inches=expanded_bbox)
            plt.close(fig)
            print(f"File saved to {plot_path}.")


def export_generalization_csv(
    auc_csv_path: Path,
    fpr_csv_path: Path,
    csv_path: Path,
) -> None:
    """Export the combined AUC + FPR@95TPR table underlying :func:`export_generalization_plots`.

    Merges the ``auc_csv_path``/``fpr_csv_path`` tables (each ``Reference, Encoder``
    plus one column per scoring dataset) on ``(Reference, Encoder)`` into one wide
    table with a ``<dataset>_auc`` / ``<dataset>_fpr95`` column pair per scoring dataset.
    """
    auc_csv_path = Path(auc_csv_path)
    fpr_csv_path = Path(fpr_csv_path)
    for path in (auc_csv_path, fpr_csv_path):
        if not path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {path}")

    auc_df = pd.read_csv(auc_csv_path)
    fpr_df = pd.read_csv(fpr_csv_path)
    value_columns = [column for column in auc_df.columns if column not in ("Reference", "Encoder")]

    merged = auc_df.merge(fpr_df, on=["Reference", "Encoder"], suffixes=("_auc", "_fpr95"))
    ordered_columns = ["Reference", "Encoder"] + [
        f"{dataset}_{metric}" for dataset in value_columns for metric in ("auc", "fpr95")
    ]
    merged = merged[ordered_columns]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    print(f"File saved to {csv_path}.")


def export_generalization_legend(plot_dir: Path) -> None:
    r"""Export the standalone legend shared by the generalization figures.

    Meant to be placed once above the row of the four Prior/Adaptive x
    AUC/FPR@95TPR generalization figures produced by
    :func:`export_generalization_plots`: two rows of separate, equally-styled
    and titled legend groups. The top row holds, side by side and centered as
    a pair, one color entry per scoring dataset :math:`\mathcal{D}_k`
    (:data:`generalization_scoring_datasets`) under "$\\mathcal{D}_k$" and the
    two linestyle entries (AUC solid, FPR@95TPR dashed) under "Metric". The
    bottom row holds the D_E abbreviation key
    (:data:`generalization_de_domain`) under "$\\mathcal{D}_E$", centered on
    its own.
    """
    apply_paper_style()

    de_handles = [plt.Line2D([], [], linestyle="none") for _ in generalization_de_domain]
    de_labels = [f"{generalization_dataset_abbrev[d_e]} = {_display_label(d_e)}" for d_e in generalization_de_domain]

    dk_handles = [
        plt.Line2D(
            [0],
            [0],
            color=COLOR_DICT.get(d_k, (0.5, 0.5, 0.5)),
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=1.8,
        )
        for d_k in generalization_scoring_datasets
    ]
    dk_labels = [_display_label(d_k) for d_k in generalization_scoring_datasets]

    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.8),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.8),
    ]
    style_labels = ["AUC", "FPR@95TPR"]

    fig = plt.figure(figsize=(13, 1.0))
    # Placed at a shared, arbitrary anchor first -- real (title- and
    # entry-count-dependent) sizes are only known after a draw, so a fixed
    # guessed layout would drift out of sync whenever entries change.
    dk_legend = fig.legend(
        dk_handles, dk_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(dk_handles), frameon=True,
        title=r"$\mathcal{D}^{(1)}$",
    )
    metric_legend = fig.legend(
        style_handles, style_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(style_handles),
        frameon=True, title="Metric",
    )
    de_legend = fig.legend(
        de_handles, de_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(de_handles), frameon=True,
        title=r"$\mathcal{D}_E$", handlelength=0, handletextpad=0,
    )

    fig.canvas.draw()
    gap = 0.015
    vgap = 0.05

    # Row 1: D_k and Metric, side by side, centered as a pair.
    w_dk = dk_legend.get_window_extent().width / fig.bbox.width
    w_metric = metric_legend.get_window_extent().width / fig.bbox.width
    row1_height = max(dk_legend.get_window_extent().height, metric_legend.get_window_extent().height) / fig.bbox.height
    x = 0.5 - (w_dk + gap + w_metric) / 2
    dk_legend.set_bbox_to_anchor((x + w_dk / 2, 1.0), transform=fig.transFigure)
    x += w_dk + gap
    metric_legend.set_bbox_to_anchor((x + w_metric / 2, 1.0), transform=fig.transFigure)

    # Row 2: D_E, centered on its own, below row 1.
    de_legend.set_bbox_to_anchor((0.5, 1.0 - row1_height - vgap), transform=fig.transFigure)
    fig.canvas.draw()

    plot_path = Path(plot_dir) / "generalization_legend.pdf"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    tight_bbox = Bbox.union(
        [dk_legend.get_window_extent(), metric_legend.get_window_extent(), de_legend.get_window_extent()]
    ).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(plot_path, dpi=300, bbox_inches=tight_bbox)
    plt.close(fig)
    print(f"File saved to {plot_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoders, extract features, and compute scores.")
    parser.add_argument(
        "--subset",
        choices=["1k", "10k", "22k", "100k"],
        default=None,
        help="Run only the given subset (1k, 10k, or 100k). Runs all subsets if omitted.",
    )
    parser.add_argument(
        "--score", choices=["mahala", "chi"], default="mahala", help="Run only the given score type (mahala or chi)."
    )
    parser.add_argument("--train", action="store_true", help="Run only the training section.")
    parser.add_argument("--features", action="store_true", help="Run only the feature extraction section.")
    parser.add_argument("--scoring", action="store_true", help="Run only the score section.")
    parser.add_argument("--auc", action="store_true", help="Run only the AUC section.")
    parser.add_argument("--report", action="store_true", help="Run only the report section.")
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else ["1k", "10k", "100k"]
    score_name = "mahalanobis" if args.score == "mahala" else "chi"

    selected_any = args.train or args.features or args.scoring or args.auc or args.report
    run_train = args.train or not selected_any
    run_features = args.features or not selected_any
    run_score = args.scoring or not selected_any
    run_auc = args.auc or not selected_any
    run_report = args.report or not selected_any

    set_seed(0)
    mp.set_start_method("spawn", force=True)

    assets_root = get_results_root()
    results_root = get_results_root() / "generalization"
    data_root = get_cxr_root()

    encoder_names = [*base_datasets, "ImageNet", "Aggregated", "All"]

    for subset in subsets:
        subset_samples: int | None = None

        if subset == "1k":
            subset_name = "annotations.csv"
        elif subset == "22k":
            subset_name = "subset.csv"
        else:
            subset_name = f"subset{subset}.csv"

        dataframes: dict[str, pd.DataFrame] = {}
        datasets: dict[str, DatasetBase] = {}
        models_by_combo: dict[str, str] = {}

        scores_filename = assets_root / "scores" / "generalization" / f"{subset}_{score_name}.csv"

        # GET DATASETS
        if run_train or run_features or run_score:
            print(f"Retrieving {subset} dataframes...")
            # Build all metadata tables once.
            for dataset_name in base_datasets:
                subset_csv = data_root / dataset_name / subset_name
                dataframe = pd.read_csv(subset_csv)
                dataframes[dataset_name] = dataframe

            # Enforce identical dataset lengths for training parity.
            dataset_lengths = {name: len(df) for name, df in dataframes.items()}
            assert len(set(dataset_lengths.values())) == 1, f"Datasets must have the same length: {dataset_lengths}"
            subset_samples = len(dataframes["CheXpert"])

            # make All dataset
            mixed_df_parts = []
            for dataset_name in base_datasets:
                current_df = dataframes[dataset_name]
                n_samples = subset_samples // 4
                mixed_df_parts.append(current_df.iloc[:n_samples])
            dataframes["All"] = pd.concat(mixed_df_parts, ignore_index=True)

            if not run_score:
                print("Building datasets...")
                for dataset_name, dataframe in dataframes.items():
                    datasets[dataset_name] = ChestXrayDataset(
                        root_dir=str(data_root),
                        dataframe=dataframe,
                    )
                    print(f"{len(datasets[dataset_name]) / 1000:.0f}k samples in {dataset_name}")

        """
        TRAIN ENCODERS
        --------------------------------------------------------
        Train ResNet-18 encoders with the following data combinations:
        - N CheXpert
        - N ChestX-ray8
        - N MIMIC-CXR
        - N PadChest
        - N/4 from each dataset

        Then compute the average of the model weights, leading to a new "aggregated" encoder.
        """
        if run_train:
            print(f"Training {subset} encoders, using device: {device}.")
            for dataset_name, dataset in datasets.items():
                print(f"\n--- {dataset_name.replace('/', ' ').replace('_', ' ')} ---------------")

                date = datetime.datetime.now().strftime("%b_%d_%H_%M")
                modelname = date
                saveto_path = assets_root / "encoders" / subset / dataset_name / modelname
                saveto_path.mkdir(parents=True, exist_ok=True)

                encoder = dataset.get_network(bn_head=True, normalize_f=True).to(device)
                optimizer = torch.optim.Adam(encoder.parameters(), lr=l_r)

                view_dataset = AugmentedDataset(dataset, transforms=copy.deepcopy(ChestXray))

                dataloader = DataLoader(
                    view_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    collate_fn=multi_view_collate,
                    drop_last=False,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers > 0 else None,
                )

                train(
                    dataloader,
                    encoder,
                    optimizer,
                    saveto_path,
                    epochs=epochs,
                    framework=framework,
                )

                del dataloader
                torch.cuda.empty_cache()

                models_by_combo[dataset_name] = modelname

            # Aggregated encoder
            timestamp = datetime.datetime.now().strftime("%b_%d_%H_%M")

            state_dicts = []
            for dataset_name in datasets:
                modelname = models_by_combo.get(dataset_name, "SupCon")
                model_path = assets_root / "encoders" / subset / dataset_name / modelname / "model.pt"
                if not model_path.exists():
                    raise FileNotFoundError(f"Expected model checkpoint not found at: {model_path}")
                state_dicts.append(torch.load(model_path, map_location="cpu"))

            ref_keys = set(state_dicts[0].keys())
            for idx, state_dict in enumerate(state_dicts[1:], start=1):
                if set(state_dict.keys()) != ref_keys:
                    raise ValueError(f"Model at index {idx} has incompatible state_dict keys.")

            aggregated_state_dict = {}
            for key in state_dicts[0]:
                tensors = [state_dict[key] for state_dict in state_dicts]
                if torch.is_floating_point(tensors[0]):
                    stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
                    aggregated_state_dict[key] = stacked.mean(dim=0).to(dtype=tensors[0].dtype)
                else:
                    aggregated_state_dict[key] = tensors[0]

            aggregate_model_path = assets_root / "encoders" / subset / "Aggregated" / timestamp / "model.pt"
            aggregate_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(aggregated_state_dict, aggregate_model_path)
            models_by_combo["Aggregated"] = timestamp

            print("\n--- Aggregating trained checkpoints ---------------")
            print(f"Saved aggregated model to: {aggregate_model_path}")

        """
        GET FEATURES
        --------------------------------------------------------
        Get the features for the previous encoders, and also for a ResNet-18 ImageNet,
        representing a not trained encoder.
        The features are computed for the N images of each dataset which are fixed
        and for the 1000 annotated of each dataset.
        """
        if run_features:
            # Load encoders trained on this subset
            encoders = {}
            encoder_list = [encoder_name for encoder_name in encoder_names if encoder_name != "ImageNet"]
            for encoder in encoder_list:
                modelname = models_by_combo.get(encoder, "SupCon")
                model_path = assets_root / "encoders" / subset / encoder / modelname / "model.pt"
                encoders[encoder] = datasets["CheXpert"].get_network(bn_head=True, normalize_f=True).to(device)
                encoders[encoder].load_state_dict(torch.load(model_path, map_location=device))
            encoders["ImageNet"] = ImageNetResNet18Features().to(device)

            for dataset_name, dataset in list(datasets.items()):
                # skip All dataset for feature extraction, as it corresponds to the same samples in the other datasets
                if dataset_name == "All":
                    continue

                print(f"\n--- {dataset_name.replace('/', ' ')} ---------------")
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    collate_fn=None,
                    drop_last=False,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers > 0 else None,
                )

                for encoder_name, encoder in encoders.items():
                    features_path = assets_root / "features" / subset / f"{dataset_name}_using_{encoder_name}.npz"
                    encode(dataloader, encoder, features_path, desc=encoder_name)
                    print(f"Saved to {features_path}")

                del dataloader
                torch.cuda.empty_cache()

        """
        COMPUTE SCORES
        --------------------------------------------------------
        """
        if run_score:
            if subset_samples is None:
                raise RuntimeError("subset_samples must be set before scoring.")

            for encoder_name in encoder_names:
                # GET FEATURES
                refer_feats: dict[str, np.ndarray] = {}
                refer_names: dict[str, np.ndarray] = {}

                score_feats: dict[str, np.ndarray] = {}
                score_names: dict[str, np.ndarray] = {}

                refer_names_parts, refer_feats_parts = [], []
                score_names_parts, score_feats_parts = [], []
                for dataset_name in base_datasets:
                    refer_names[dataset_name], refer_feats[dataset_name] = load_features(
                        assets_root / "features" / subset / f"{dataset_name}_using_{encoder_name}.npz"
                    )
                    refer_names_parts.append(refer_names[dataset_name][: subset_samples // 4])
                    refer_feats_parts.append(refer_feats[dataset_name][: subset_samples // 4])

                    score_names[dataset_name], score_feats[dataset_name] = (
                        refer_names[dataset_name],
                        refer_feats[dataset_name],
                    )
                    if subset == "1k":
                        score_names_parts.append(refer_names[dataset_name])
                        score_feats_parts.append(refer_feats[dataset_name])

                refer_names["All"] = np.concatenate(refer_names_parts, axis=0)
                refer_feats["All"] = np.concatenate(refer_feats_parts, axis=0)
                if subset == "1k":
                    score_names["All"] = np.concatenate(score_names_parts, axis=0)
                    score_feats["All"] = np.concatenate(score_feats_parts, axis=0)
                else:
                    score_names["All"] = refer_names["All"]
                    score_feats["All"] = refer_feats["All"]

                # Build encoder-specific holdout sets from all other datasets.
                if encoder_name in base_datasets:
                    holdout_names_parts: list[np.ndarray] = []
                    holdout_feats_parts: list[np.ndarray] = []
                    score_names_parts, score_feats_parts = [], []

                    for dataset_name in base_datasets:
                        if dataset_name == encoder_name:
                            continue
                        holdout_names_parts.append(refer_names[dataset_name][: subset_samples // 3])
                        holdout_feats_parts.append(refer_feats[dataset_name][: subset_samples // 3])
                        if subset == "1k":
                            score_names_parts.append(refer_names[dataset_name])
                            score_feats_parts.append(refer_feats[dataset_name])

                    refer_names["Holdout"] = np.concatenate(holdout_names_parts, axis=0)
                    refer_feats["Holdout"] = np.concatenate(holdout_feats_parts, axis=0)
                    if subset == "1k":
                        score_names["Holdout"] = np.concatenate(score_names_parts, axis=0)
                        score_feats["Holdout"] = np.concatenate(score_feats_parts, axis=0)
                    else:
                        score_names["Holdout"] = refer_names["Holdout"]
                        score_feats["Holdout"] = refer_feats["Holdout"]

                for reference_mode in ["prior", "adaptive"]:
                    if reference_mode == "prior" and encoder_name not in [*base_datasets, "All"]:
                        continue
                    scoring_method = reference_mode.capitalize() + "_" + encoder_name
                    print(f"{encoder_name} {subset} [{reference_mode.capitalize()} Reference]")

                    mean, cov = None, None
                    if reference_mode == "prior":
                        mean, cov = reference_vector(refer_feats[encoder_name])

                    # Iterate over evaluation sets and their features
                    eval_set_iterator = tqdm(
                        score_feats.keys(),
                        desc="scoring",
                        unit="set",
                        leave=False,
                    )
                    for eval_set in eval_set_iterator:
                        image_names = score_names[eval_set]
                        if reference_mode == "adaptive":
                            ref_mean, ref_cov = reference_vector(refer_feats[eval_set])
                        else:
                            if mean is None or cov is None:
                                raise RuntimeError("mean and cov must be set for encoder reference mode.")
                            ref_mean, ref_cov = mean, cov

                        scores = mahalanobis_distance(score_feats[eval_set], [ref_mean], [ref_cov])
                        dataset_name = eval_set if eval_set != "Holdout" else f"{encoder_name}_Holdout"
                        save_scores_csv(
                            csv_path=scores_filename,
                            tag=scoring_method,
                            image_names=image_names,
                            dataset_name=dataset_name,
                            scores=np.asarray(scores),
                        )

        """
        COMPUTE AUC FROM SCORES CSV
        --------------------------------------------------------
        """
        if run_auc:
            scores_df = pd.read_csv(scores_filename)

            annotation_dfs = []
            for dataset_name in base_datasets:
                ann_path = data_root / dataset_name / "annotations.csv"
                df = pd.read_csv(ann_path)
                annotation_dfs.append(df)
            annotations_df = pd.concat(annotation_dfs, ignore_index=True)
            annotations_df["dataset"] = annotations_df["image_path"].str.split("/").str[0]

            auc_rows = []
            fpr_rows = []
            for encoder_name in encoder_names:
                for reference_mode in ["Prior", "Adaptive"]:
                    if reference_mode == "Prior" and encoder_name not in [*base_datasets, "All"]:
                        continue
                    scoring_method = reference_mode.capitalize() + "_" + encoder_name

                    focused_scores = scores_df.rename(columns={scoring_method: "score"})

                    auc_row = {"Reference": reference_mode, "Encoder": encoder_name}
                    fpr_row = {"Reference": reference_mode, "Encoder": encoder_name}
                    for test_data in [*base_datasets, "All"]:
                        filtered_scores = focused_scores.loc[focused_scores["dataset"] == test_data]

                        if test_data == "All":
                            filtered_annotations = annotations_df
                        else:
                            filtered_annotations = annotations_df.loc[annotations_df["dataset"] == test_data]

                        # invert_labels=False keeps the historical CA2 convention (positive = non-compliant).
                        auc_row[test_data] = compute_auc(
                            filtered_annotations, filtered_scores, percentage=False, invert_labels=False
                        )
                        fpr_row[test_data] = compute_fpr_at_tpr(
                            filtered_annotations, filtered_scores, percentage=False, invert_labels=False
                        )

                    if encoder_name in base_datasets:
                        filtered_scores = focused_scores.loc[focused_scores["dataset"] == f"{encoder_name}_Holdout"]
                        filtered_annotations = annotations_df.loc[annotations_df["dataset"] != encoder_name]
                        auc_row["Holdout"] = compute_auc(
                            filtered_annotations, filtered_scores, percentage=False, invert_labels=False
                        )
                        fpr_row["Holdout"] = compute_fpr_at_tpr(
                            filtered_annotations, filtered_scores, percentage=False, invert_labels=False
                        )
                    else:
                        auc_row["Holdout"] = None
                        fpr_row["Holdout"] = None
                    auc_rows.append(auc_row)
                    fpr_rows.append(fpr_row)
            auc = pd.DataFrame(auc_rows, columns=["Reference", "Encoder", *base_datasets, "All", "Holdout"])
            auc.to_csv(results_root / f"auc_{score_name}_{subset}.csv", index=False)
            fpr95 = pd.DataFrame(fpr_rows, columns=["Reference", "Encoder", *base_datasets, "All", "Holdout"])
            fpr95.to_csv(results_root / f"fpr95_{score_name}_{subset}.csv", index=False)

        if run_report:
            caption = "AUC percentage scores of encoders trained and evaluated across different datasets"
            if subset == "100k":
                caption += "."
            else:
                caption += f", using {subset} samples."
            export_methods_table(
                results_root / f"auc_{score_name}_{subset}.csv", caption=caption, label=f"tab:datagen_{subset}"
            )
            export_generalization_plots(
                auc_csv_path=results_root / f"auc_{score_name}_{subset}.csv",
                fpr_csv_path=results_root / f"fpr95_{score_name}_{subset}.csv",
                plot_dir=results_root,
                subset=subset,
            )
            export_generalization_csv(
                auc_csv_path=results_root / f"auc_{score_name}_{subset}.csv",
                fpr_csv_path=results_root / f"fpr95_{score_name}_{subset}.csv",
                csv_path=results_root / f"auc_fpr95_{score_name}_{subset}.csv",
            )

    # After all processing, check if auc_mahala_1k, 10k, 100k exist and plot comparison if so
    if run_report:
        export_generalization_legend(plot_dir=results_root)

        auc_1k, auc_10k, auc_100k = (
            results_root / "auc_mahalanobis_1k.csv",
            results_root / "auc_mahalanobis_10k.csv",
            results_root / "auc_mahalanobis_100k.csv",
        )

        if auc_1k.exists() and auc_10k.exists() and auc_100k.exists():
            # Load tables
            t1k = pd.read_csv(auc_1k)
            t10k = pd.read_csv(auc_10k)
            t100k = pd.read_csv(auc_100k)
            subset_mahala_tables = {"1k": t1k, "10k": t10k, "100k": t100k}
            export_scaling_table(
                subset_mahala_tables,
                caption=(
                    "AUC percentage scores of encoders trained and evaluated across different datasets, "
                    "on varying number of samples"
                ),
                label="tab:data_scaling",
                outdir=results_root,
            )


if __name__ == "__main__":
    main()
