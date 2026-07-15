"""Federated scoring with a static (frozen) encoder (thesis phase FA1).

Each client scores its images against reference statistics that are either
independent (encoder's own data), federated averages of client statistics,
or built from a few picked compliant samples. AUCs against the manual
compliance annotations are exported as LaTeX tables.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from DeMICAF.caf.scorer import mahalanobis_distance, reference_vector
from DeMICAF.utils.io import upsert_scores_csv as _upsert_scores_csv
from DeMICAF.utils.plotting import apply_thesis_style

datasets = ["CheXpert", "MIMIC-CXR", "ChestX-ray8", "PadChest"]
picked_samples = [1, 2, 5, 10, 20, 50, 100]
n_iter = 10

workspace_root = Path(__file__).resolve().parents[3]
results_root = workspace_root / "Results" / "DeMICAF"
data_root = Path("/eos/user/h/hpestana/CXR")
repo_root = Path(__file__).resolve().parents[2]
niqe_baseline_csv_path = repo_root / "Results" / "Baseline" / "perceptual_iqa_metrics.csv"


def upsert_scores_csv(
    csv_path: Path,
    image_names: np.ndarray,
    scores: np.ndarray,
    tag: str,
) -> None:
    """Upsert a score column, inferring the (single) client dataset from the image paths."""
    image_names = np.asarray(image_names, dtype=str)
    scores = np.asarray(scores, dtype=np.float64)

    dataset_name = None
    for part in Path(image_names[0]).parts:
        if part in datasets:
            dataset_name = part
            break
    if dataset_name is None:
        raise ValueError(f"Could not infer dataset name from image path: {image_names[0]}")

    inferred_datasets = [next((part for part in Path(name).parts if part in datasets), None) for name in image_names]
    if any(name_dataset != dataset_name for name_dataset in inferred_datasets):
        raise ValueError("Image paths passed to upsert_scores_csv must belong to a single dataset.")

    _upsert_scores_csv(csv_path, tag, image_names, scores, dataset_name)


def _align_scores_labels(
    scores: np.ndarray,
    feat_names: np.ndarray,
    annotation_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Align scores to annotation order and return ``(aligned_scores, labels)``.

    Labels follow the numeric convention used throughout this script when the
    annotation column is already numeric (compliant=1, non-compliant=0); string
    annotations are mapped with compliant=0, non-compliant=1.
    """
    score_df = pd.DataFrame(
        {
            "image_path": np.asarray(feat_names, dtype=str),
            "score": np.asarray(scores, dtype=np.float64),
        }
    )

    label_map = {"compliant": 0, "non-compliant": 1, "non compliant": 1}
    ann_df = annotation_df[["image_path", "annotation"]].copy()
    ann_df["image_path"] = ann_df["image_path"].astype(str)
    if pd.api.types.is_numeric_dtype(ann_df["annotation"]):
        ann_df["annotation"] = ann_df["annotation"].astype(np.int64)
    else:
        mapped = ann_df["annotation"].astype(str).str.strip().str.lower().map(label_map)
        if mapped.isna().any():
            raise ValueError("Unexpected annotation label found while computing metric.")
        ann_df["annotation"] = mapped.astype(np.int64)

    annotated_scores = ann_df.merge(
        score_df,
        on="image_path",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if annotated_scores["score"].isna().any():
        missing_paths = annotated_scores.loc[annotated_scores["score"].isna(), "image_path"].tolist()
        raise ValueError(
            f"Missing {len(missing_paths)} scores for annotated samples. Example paths: {missing_paths[0:5]}"
        )

    aligned_scores = annotated_scores["score"].to_numpy(dtype=np.float64)
    labels = annotated_scores["annotation"].to_numpy(dtype=np.int64)
    return aligned_scores, labels


def compute_auc(
    scores: np.ndarray,
    feat_names: np.ndarray,
    annotation_df: pd.DataFrame,
    percentage: bool = False,
    invert_labels: bool = False,
) -> float:
    """Compute an AUC from aligned score and annotation arrays."""
    aligned_scores, labels = _align_scores_labels(scores, feat_names, annotation_df)
    if invert_labels:
        aligned_scores = -aligned_scores

    auc = float(roc_auc_score(labels, aligned_scores))
    if percentage:
        auc *= 100

    return auc


def compute_ap(
    scores: np.ndarray,
    feat_names: np.ndarray,
    annotation_df: pd.DataFrame,
    percentage: bool = False,
) -> float:
    """Compute Average Precision for detecting non-compliant samples.

    The positive class is *non-compliant* (the numeric convention used here is
    compliant=1, non-compliant=0), ranked directly by the Mahalanobis distance
    (higher distance = more anomalous), so no score inversion is applied.
    """
    aligned_scores, labels = _align_scores_labels(scores, feat_names, annotation_df)
    positive = (labels == 0).astype(np.int64)  # non-compliant

    ap = float(average_precision_score(positive, aligned_scores))
    if percentage:
        ap *= 100

    return ap


def compute_fpr_at_tpr(
    scores: np.ndarray,
    feat_names: np.ndarray,
    annotation_df: pd.DataFrame,
    tpr_target: float = 0.95,
    percentage: bool = False,
) -> float:
    """Compute the FPR at a fixed TPR (default 95%) for detecting non-compliant samples.

    Like :func:`compute_ap`, the positive class is *non-compliant* (numeric
    convention compliant=1, non-compliant=0), ranked directly by the Mahalanobis
    distance, and the FPR is linearly interpolated on the ROC curve at the
    target TPR.
    """
    aligned_scores, labels = _align_scores_labels(scores, feat_names, annotation_df)
    positive = (labels == 0).astype(np.int64)  # non-compliant

    fpr, tpr, _ = roc_curve(positive, aligned_scores)
    fpr_at_tpr = float(np.interp(tpr_target, tpr, fpr))
    if percentage:
        fpr_at_tpr *= 100

    return fpr_at_tpr


def non_compliant_prevalence(
    feat_names: np.ndarray,
    scores: np.ndarray,
    annotation_df: pd.DataFrame,
    percentage: bool = False,
) -> float:
    """Fraction of annotated samples labelled non-compliant (AP chance level)."""
    _, labels = _align_scores_labels(scores, feat_names, annotation_df)
    prevalence = float((labels == 0).mean())
    return prevalence * 100 if percentage else prevalence


def compute_and_store_scores(
    ref_vectors: list[np.ndarray],
    ref_covmats: list[np.ndarray],
    client_feats: dict[str, np.ndarray],
    tag: str,
    chi: bool,
    scores_output_csv_path: Path,
) -> dict[str, pd.DataFrame]:
    """Compute scores for each client dataset and persist them to a scores CSV."""
    scores_by_client: dict[str, pd.DataFrame] = {}

    for client, feats in client_feats.items():
        feat_names = feats[:, 0].astype(str)
        feats_values = feats[:, 1:].astype(np.float32)
        scores = mahalanobis_distance(feats_values, ref_vectors, ref_covmats, chi=chi)

        upsert_scores_csv(
            csv_path=scores_output_csv_path,
            image_names=feat_names,
            scores=np.asarray(scores),
            tag=tag,
        )

        scores_by_client[client] = pd.DataFrame(
            {
                "image_path": feat_names,
                "score": np.asarray(scores, dtype=np.float64),
            }
        )

    return scores_by_client


def compute_client_metrics(
    scores_by_client: dict[str, pd.DataFrame],
    client_labels: dict[str, pd.DataFrame],
    encoder: str,
    percentage: bool = True,
    invert_labels: bool = True,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    """Compute AUC, AP, and FPR@95TPR per dataset/holdout plus the holdout prevalence.

    Returns
    -------
    tuple
        ``(aucs, aps, fprs, holdout_baseline)`` where ``aucs``/``aps``/``fprs``
        map each dataset (and ``"Holdout"``) to its metric, and
        ``holdout_baseline`` is the AP chance level (non-compliant fraction over
        the held-out clients).
    """
    aucs: dict[str, float] = dict.fromkeys(datasets, np.nan)
    aps: dict[str, float] = dict.fromkeys(datasets, np.nan)
    fprs: dict[str, float] = dict.fromkeys(datasets, np.nan)

    all_feat_names: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_annotations: list[pd.DataFrame] = []

    for client, score_df in scores_by_client.items():
        ann_df = client_labels[client][["image_path", "annotation"]].copy()
        ann_df["image_path"] = ann_df["image_path"].astype(str)

        client_scores = score_df["score"].to_numpy(dtype=np.float64)
        client_feat_names = score_df["image_path"].to_numpy(dtype=str)

        aucs[client] = compute_auc(
            scores=client_scores,
            feat_names=client_feat_names,
            annotation_df=ann_df,
            percentage=percentage,
            invert_labels=invert_labels,
        )
        aps[client] = compute_ap(
            scores=client_scores,
            feat_names=client_feat_names,
            annotation_df=ann_df,
            percentage=percentage,
        )
        fprs[client] = compute_fpr_at_tpr(
            scores=client_scores,
            feat_names=client_feat_names,
            annotation_df=ann_df,
            percentage=percentage,
        )

        all_feat_names.append(client_feat_names)
        all_scores.append(client_scores)
        all_annotations.append(ann_df)

    holdout_feat_names = np.concatenate(all_feat_names)
    holdout_scores = np.concatenate(all_scores)
    holdout_annotations = pd.concat(all_annotations, ignore_index=True)

    aucs["Holdout"] = compute_auc(
        scores=holdout_scores,
        feat_names=holdout_feat_names,
        annotation_df=holdout_annotations,
        percentage=percentage,
        invert_labels=invert_labels,
    )
    aps["Holdout"] = compute_ap(
        scores=holdout_scores,
        feat_names=holdout_feat_names,
        annotation_df=holdout_annotations,
        percentage=percentage,
    )
    fprs["Holdout"] = compute_fpr_at_tpr(
        scores=holdout_scores,
        feat_names=holdout_feat_names,
        annotation_df=holdout_annotations,
        percentage=percentage,
    )
    holdout_baseline = non_compliant_prevalence(
        feat_names=holdout_feat_names,
        scores=holdout_scores,
        annotation_df=holdout_annotations,
        percentage=percentage,
    )

    # Encoder dataset is not part of holdout clients for that row.
    aucs[encoder] = np.nan
    aps[encoder] = np.nan
    fprs[encoder] = np.nan
    return aucs, aps, fprs, holdout_baseline


def save_metrics(
    csv_path: Path,
    method_name: str,
    encoder: str,
    values: dict[str, float],
    baseline: float = 50.0,
) -> None:
    """Upsert one ``(method, encoder)`` metric row (per-dataset, Holdout, Baseline)."""
    expected_columns = [
        "method",
        "encoder",
        "CheXpert",
        "ChestX-ray8",
        "MIMIC-CXR",
        "PadChest",
        "Holdout",
        "Baseline",
    ]

    row = {
        "method": method_name,
        "encoder": encoder,
        "CheXpert": float(values.get("CheXpert", np.nan)),
        "ChestX-ray8": float(values.get("ChestX-ray8", np.nan)),
        "MIMIC-CXR": float(values.get("MIMIC-CXR", np.nan)),
        "PadChest": float(values.get("PadChest", np.nan)),
        "Holdout": float(values.get("Holdout", np.nan)),
        "Baseline": float(baseline),
    }

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Keep only the expected schema and create any missing columns.
        for column in expected_columns:
            if column not in df.columns:
                df[column] = np.nan
        df = df[expected_columns]

        if "method" in df.columns and "encoder" in df.columns:
            same_row_mask = (df["method"] == method_name) & (df["encoder"] == encoder)
        else:
            same_row_mask = pd.Series(False, index=df.index)
        if bool(same_row_mask.any()):
            df.loc[same_row_mask, list(row.keys())] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    method_order = [
        "Independent",
        "Average",
        "Picked_1_Samples",
        "Picked_3_Samples",
        "Picked_5_Samples",
        "Picked_10_Samples",
    ]
    encoder_order = datasets
    if "method" in df.columns and "encoder" in df.columns:
        df["_method_order"] = df["method"].apply(
            lambda m: method_order.index(m) if m in method_order else len(method_order)
        )
        df["_encoder_order"] = df["encoder"].apply(
            lambda e: encoder_order.index(e) if e in encoder_order else len(encoder_order)
        )
        df = df.sort_values(by=["_method_order", "_encoder_order", "method", "encoder"]).drop(
            columns=["_method_order", "_encoder_order"]
        )

    df = df[expected_columns]
    df.to_csv(csv_path, index=False)


def export_independent_auc_table(csv_path: Path, tex_path: Path) -> None:
    """Export Independent-only table with per-encoder row and dataset/holdout columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"AUC CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["method", "encoder", "CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest", "Holdout"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for independent table: {missing}")

    independent_df = df[df["method"] == "Independent"].copy()

    def value_for(encoder: str, column: str) -> str:
        row = independent_df[independent_df["encoder"] == encoder]
        if row.empty:
            return ""
        value = row.iloc[-1][column]
        if pd.isna(value):
            return "—"
        return f"{float(value):.2f}"

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \caption{AUC percentage scores for independent static encoder \ac{CAF} implementation"
        r" in \ac{FL} settings, varying encoder and reference dataset.}"
        r" \label{tab:static_encoder_independent_auc}",
        r"    \begin{tabular}{c|c|c|c|c|c}",
        r"        \hline",
        r"        \multirow{2}{*}{Reference} & \multicolumn{5}{c}{Evaluation} \\",
        r"        \cline{2-6}",
        r"        & CheXpert & ChestX-ray8 & MIMIC-CXR & PadChest & Holdout \\",
        r"        \hline",
        r"        \hline",
    ]

    for encoder in datasets:
        lines.append(
            "        "
            f"{encoder} & "
            f"{value_for(encoder, 'CheXpert')} & "
            f"{value_for(encoder, 'ChestX-ray8')} & "
            f"{value_for(encoder, 'MIMIC-CXR')} & "
            f"{value_for(encoder, 'PadChest')} & "
            f"{value_for(encoder, 'Holdout')} \\\\"
        )
        lines.append(r"        \hline")

    lines.extend(
        [
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def export_methods_holdout_table(csv_path: Path, tex_path: Path) -> None:
    """Export a single Holdout table with independent, average, and picked rows."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"AUC CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["method", "encoder", "Holdout"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for methods table: {missing}")

    def holdout_for(method: str, encoder: str) -> str:
        row = df[(df["method"] == method) & (df["encoder"] == encoder)]
        if row.empty:
            return ""
        value = row.iloc[-1]["Holdout"]
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}"

    def encoder_values(method: str) -> list[str]:
        return [holdout_for(method, encoder) for encoder in datasets]

    def row(first_cell: str, second_cell: str, values: list[str]) -> str:
        return f"        {first_cell} & {second_cell} & {values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\"

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \caption{Holdout AUC scores for static encoder \ac{CAF} implementation in \ac{FL} settings,"
        r" varying encoder dataset.} \label{tab:static_encoder_methods_auc}",
        r"    \begin{tabular}{c|c|c|c|c|c}",
        r"        \hline",
        r"        \multicolumn{2}{c|}{Method} & \multicolumn{4}{c}{Encoder Training Dataset} \\",
        r"        \hline",
        r"        Client Relation & Parameters & CheXpert & ChestX-ray8 & MIMIC-CXR & PadChest \\",
        r"        \hline",
        r"        \hline",
        "        "
        + "\\multicolumn{2}{c|}{Independent} & "
        + " & ".join(encoder_values("Independent"))
        + " "
        + chr(92) * 2,
        r"        \hline",
        "        " + "\\multicolumn{2}{c|}{Average} & " + " & ".join(encoder_values("Average")) + " " + chr(92) * 2,
    ]

    lines.extend(
        [
            r"        \hline",
            row("\\multirow{4}{*}{Picked}", "1 Sample", encoder_values("Picked_1_Samples")),
            r"        \cline{2-6}",
            row("", "3 Samples", encoder_values("Picked_3_Samples")),
            r"        \cline{2-6}",
            row("", "5 Samples", encoder_values("Picked_5_Samples")),
            r"        \cline{2-6}",
            row("", "10 Samples", encoder_values("Picked_10_Samples")),
            r"        \hline",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"File saved to {tex_path}.")


# Pretty row labels for the scheme table, matching the thesis tables.
scheme_train_labels = {
    "CheXpert": "CheXpert",
    "MIMIC-CXR": "MIMIC-CXR",
    "ChestX-ray8": "ChestX-ray14",
    "PadChest": "PadChest",
}


def export_scheme_table(
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
) -> None:
    """Export the per-train-dataset scheme table (Base./Independent/Picked/Average).

    One row per encoder (training dataset) using its Holdout metric. The ``Base.``
    column shows the per-encoder ``Baseline`` value: when it is constant across
    encoders (e.g. AUC chance level 50.0) it is rendered as a single multirow
    cell, otherwise (e.g. AP non-compliant prevalence) one value per row.
    In each row the best scheme value is bold and the worst is italic.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["method", "encoder", "Holdout"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scheme table: {missing}")

    def holdout_for(method: str, encoder: str) -> float | None:
        row = df[(df["method"] == method) & (df["encoder"] == encoder)]
        if row.empty or pd.isna(row.iloc[-1]["Holdout"]):
            return None
        return float(row.iloc[-1]["Holdout"])

    def baseline_for(encoder: str) -> float | None:
        if "Baseline" not in df.columns:
            return None
        row = df[df["encoder"] == encoder]
        if row.empty or pd.isna(row.iloc[-1]["Baseline"]):
            return None
        return float(row.iloc[-1]["Baseline"])

    # (method key, header label) for the scheme columns, in display order.
    scheme_methods = [
        ("Independent", "Independent"),
        ("Picked_1_Samples", "1"),
        ("Picked_2_Samples", "2"),
        ("Picked_5_Samples", "5"),
        ("Picked_10_Samples", "10"),
        ("Picked_20_Samples", "20"),
        ("Picked_50_Samples", "50"),
        ("Picked_100_Samples", "100"),
        ("Average", "Average"),
    ]

    def format_cells(values: list[float | None]) -> list[str]:
        present = [v for v in values if v is not None]
        max_value = max(present) if present else None
        min_value = min(present) if present else None
        cells = []
        for value in values:
            if value is None:
                cells.append("—")
            elif max_value is not None and value == max_value:
                cells.append(r"\textbf{" + f"{value:.1f}" + "}")
            elif min_value is not None and value == min_value:
                cells.append(r"\textit{" + f"{value:.1f}" + "}")
            else:
                cells.append(f"{value:.1f}")
        return cells

    baselines = {encoder: baseline_for(encoder) for encoder in datasets}
    present_baselines = [b for b in baselines.values() if b is not None]
    constant_baseline = bool(present_baselines) and (max(present_baselines) - min(present_baselines) < 0.05)

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \begin{tabular}{c|c|c|c|c|c|c|c|c|c}",
        r"    \toprule",
        r"    \multirow{2}{*}{\diagbox{Train}{Scheme}}"
        r" & \multirow{2}{*}{Independent} & \multicolumn{7}{c|}{Picked (Samples)}"
        r" & \multirow{2}{*}{Average} \\",
        r"    \cline{3-9}",
        r"    & & 1 & 2 & 5 & 10 & 20 & 50 & 100 & \\",
        r"    \midrule",
    ]

    for index, encoder in enumerate(datasets):
        values = [holdout_for(method, encoder) for method, _ in scheme_methods]
        cells = format_cells(values)

        # baseline = baselines[encoder]
        # if constant_baseline:
        #     if index == 0:
        #         base_cell = r"\multirow{" + str(len(datasets)) + r"}{*}{" + f"{present_baselines[0]:.1f}" + "}"
        #     else:
        #         base_cell = ""
        # else:
        #     base_cell = f"{baseline:.1f}" if baseline is not None else "—"

        train_label = scheme_train_labels.get(encoder, encoder)
        # lines.append(f"    {{{train_label}}} & {base_cell} & " + " & ".join(cells) + r" \\")
        lines.append(f"    {{{train_label}}} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"File saved to {tex_path}.")


# Scheme-plot line colors; linestyles keep the schemes distinguishable if the palette is unavailable.
scheme_plot_colors = {
    "Encoder": "#888888",
    "Full": "#64C7EA",
    "Compliant": "#7BDA19",
    "Prevalence": "#000000",
    "NIQE": "#A6A0F5",
}

# Legend entries grouped under the "Baseline" title instead of "Reference".
baseline_legend_labels = ("NIQE",)


def export_scheme_plot(
    csv_path: Path,
    plot_path: Path,
    ylabel: str,
    baseline_csv_path: Path | None = None,
    baseline_column: str = "niqe_auc",
    show_prevalence: bool = False,
) -> None:
    """Export the per-prior-dataset scheme figure (reference lines + picked-sample curve).

    One subplot per prior (encoder training) dataset, titled with its name. The
    Independent (``Encoder Reference``) and average schemes are horizontal
    reference lines; the picked scheme is a curve of the Holdout metric over
    the number of picked compliant samples, with a band of ±1 standard
    deviation across the picking iterations.

    When ``baseline_csv_path`` is given, the NIQE holdout value of each encoder
    dataset (its ``"<dataset> Holdout"`` row, ``baseline_column``) is drawn as a
    dashed horizontal baseline, and the legend is split into titled
    ``Reference`` and ``Baseline`` groups. ``show_prevalence`` additionally
    draws the per-encoder ``Baseline`` column of ``csv_path`` (the label
    prevalence, i.e. the AP chance level) as a solid black baseline.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    niqe_baselines: dict[str, float] = {}
    if baseline_csv_path is not None:
        baseline_csv_path = Path(baseline_csv_path)
        if not baseline_csv_path.exists():
            raise FileNotFoundError(f"Baseline metrics CSV not found: {baseline_csv_path}")
        baseline_df = pd.read_csv(baseline_csv_path)
        for encoder in datasets:
            row = baseline_df[baseline_df["set"] == f"{encoder} Holdout"]
            if not row.empty and not pd.isna(row.iloc[-1][baseline_column]):
                niqe_baselines[encoder] = float(row.iloc[-1][baseline_column])

    df = pd.read_csv(csv_path)
    required_columns = ["method", "encoder", "Holdout"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scheme plot: {missing}")

    def holdout_for(method: str, encoder: str) -> float | None:
        row = df[(df["method"] == method) & (df["encoder"] == encoder)]
        if row.empty or pd.isna(row.iloc[-1]["Holdout"]):
            return None
        return float(row.iloc[-1]["Holdout"])

    def prevalence_for(encoder: str) -> float | None:
        if "Baseline" not in df.columns:
            return None
        row = df[df["encoder"] == encoder]
        if row.empty or pd.isna(row.iloc[-1]["Baseline"]):
            return None
        return float(row.iloc[-1]["Baseline"])

    # (legend label, method key, linestyle) for the horizontal reference lines.
    reference_lines = [
        ("Encoder", "Independent", "-"),
        ("Full", "Average", "-"),
    ]

    apply_thesis_style()
    fig, axes = plt.subplots(1, len(datasets), figsize=(9, 2.5), sharey=True)

    for ax, encoder in zip(axes, datasets, strict=True):
        for label, method, linestyle in reference_lines:
            value = holdout_for(method, encoder)
            if value is not None:
                ax.axhline(value, color=scheme_plot_colors[label], linestyle=linestyle, linewidth=1.8, label=label)

        picked_points = [
            (
                n_samples,
                holdout_for(f"Picked_{n_samples}_Samples", encoder),
                holdout_for(f"Picked_{n_samples}_Samples_std", encoder),
            )
            for n_samples in picked_samples
        ]
        picked_points = [(n_samples, value, std) for n_samples, value, std in picked_points if value is not None]
        if picked_points:
            x_values, y_values, std_values = zip(*picked_points, strict=True)
            ax.plot(
                x_values,
                y_values,
                color=scheme_plot_colors["Compliant"],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label="Compliant",
            )
            if all(std is not None for std in std_values):
                means = np.asarray(y_values, dtype=np.float64)
                stds = np.asarray(std_values, dtype=np.float64)
                ax.fill_between(
                    x_values,
                    means - stds,
                    means + stds,
                    color=scheme_plot_colors["Compliant"],
                    alpha=0.2,
                    linewidth=0,
                )

        baseline_value = niqe_baselines.get(encoder)
        if baseline_value is not None:
            ax.axhline(
                baseline_value,
                color=scheme_plot_colors["NIQE"],
                linestyle="--",
                linewidth=1.8,
                label="NIQE",
            )

        if show_prevalence:
            prevalence_value = prevalence_for(encoder)
            if prevalence_value is not None:
                ax.axhline(
                    prevalence_value,
                    color=scheme_plot_colors["Prevalence"],
                    linestyle="-",
                    linewidth=1.8,
                    label="Prevalence",
                )

        ax.set_title(scheme_train_labels.get(encoder, encoder),fontsize=plt.rcParams["axes.labelsize"])
        ax.set_xscale("log")
        ax.set_xticks(picked_samples)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=8))
        ax.minorticks_off()
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(ylabel)

    # Legends, deduplicated across subplots in plot order.
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    grouped = {label: (handle, label) for handle, label in zip(handles, labels, strict=True)}
    prevalence_entries = [grouped.pop("Prevalence")] if "Prevalence" in grouped else []
    baseline_entries = [grouped.pop(label) for label in baseline_legend_labels if label in grouped]
    reference_entries = list(grouped.values())

    # Legend groups below the figure, left to right: the untitled, frameless
    # prevalence chance level (not a baseline method), then the titled
    # "Reference" and "Baseline" groups.
    legend_groups: list[tuple[list, str | None, bool]] = []
    if reference_entries:
        legend_groups.append((reference_entries, "Reference", True))
    if baseline_entries:
        legend_groups.append((baseline_entries, "Baseline", True))
    if prevalence_entries:
        legend_groups.append((prevalence_entries, " ", False))

    group_anchors = {1: (0.5,), 2: (0.865, 0.505), 3: (0.865, 0.505, 0.365)}[len(legend_groups)]
    for (entries, title, frameon), anchor in zip(legend_groups, group_anchors, strict=True):
        fig.legend(
            [handle for handle, _ in entries],
            [label for _, label in entries],
            loc="upper left",
            bbox_to_anchor=(0.865, anchor),
            ncol=1,
            frameon=frameon,
            title=title,
        )
    fig.tight_layout(rect=(-0.01, 0, 0.88, 0.97))
    fig.suptitle("Prior Dataset", y=0.995)#, fontsize=plt.rcParams["axes.labelsize"])
    fig.supxlabel("Number of Compliant Samples", y=0, fontsize=plt.rcParams["axes.labelsize"])


    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"File saved to {plot_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Assess CAF implementation in FL using a static encoder, using with previously computed features."
    )
    parser.add_argument("--compute", action="store_true", help="Compute AUC and Scores.")
    parser.add_argument("--report", action="store_true", help="Export results.")
    parser.add_argument(
        "--subset",
        choices=["1k", "10k", "100k"],
        default="100k",
        help="Run only the given subset (1k, 10k, or 100k). Runs all subsets if omitted.",
    )
    args = parser.parse_args()

    selected_any = args.compute or args.report
    run_compute = args.compute or not selected_any
    run_report = args.report or not selected_any

    subsets = ["1k", "10k", "100k"] if args.subset is None else [args.subset]

    for subset in subsets:
        csv_path = results_root / f"{subset}_AUCs.csv"
        ap_csv_path = results_root / f"{subset}_APs.csv"
        fpr_csv_path = results_root / f"{subset}_FPR95s.csv"
        mahala_scores_csv_path = results_root / f"{subset}_MahalaScores.csv"
        chi_scores_csv_path = results_root / f"{subset}_ChiScores.csv"
        independent_tex_output_path = results_root / f"{subset}_auc_independent.tex"
        methods_tex_output_path = results_root / f"{subset}_auc_methods.tex"
        scheme_auc_tex_output_path = results_root / f"{subset}_scheme_auc.tex"
        scheme_ap_tex_output_path = results_root / f"{subset}_scheme_ap.tex"
        scheme_auc_plot_output_path = results_root / "demicaf_auc.png"
        scheme_ap_plot_output_path = results_root / "demicaf_ap.png"
        scheme_fpr_plot_output_path = results_root / "demicaf_fpr.png"

        if run_compute:
            for encoder in datasets:
                print(f"REFERENCE: {encoder}")
                clients = [d for d in datasets if d != encoder]

                client_feats = {}
                client_labels = {}
                for client in clients:
                    client_labels[client] = pd.read_csv(data_root / client / "annotations.csv")
                    client_labels[client]["annotation"] = (
                        client_labels[client]["annotation"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map({"compliant": 1, "non-compliant": 0, "non compliant": 0})
                        .astype(np.int64)
                    )
                    if client_labels[client]["annotation"].isna().any():
                        raise ValueError(f"Unexpected annotation label found in {client}/annotations.csv")
                    client_labels[client]["image_path"] = client_labels[client]["image_path"].astype(str)

                    # get features
                    feat_file = workspace_root / "Assets" / "Features" / subset / f"{client}_using_{encoder}.npz"
                    feats = np.load(feat_file, allow_pickle=True)["arr_0"]
                    feats[:, 1:] = feats[:, 1:].astype(np.float32)

                    # Keep only annotated samples and preserve annotation order.
                    ann_paths = client_labels[client]["image_path"].to_numpy(dtype=str)
                    feat_paths = feats[:, 0].astype(str)
                    path_to_idx = {path: idx for idx, path in enumerate(feat_paths)}

                    missing_paths = [path for path in ann_paths if path not in path_to_idx]
                    if missing_paths:
                        raise ValueError(
                            f"Missing {len(missing_paths)} annotated samples in features for {client} using {encoder}. "
                            f"Example paths: {missing_paths[:5]}"
                        )

                    aligned_indices = np.asarray([path_to_idx[path] for path in ann_paths], dtype=np.int64)
                    client_feats[client] = feats[aligned_indices]

                """
                INDEPENDENT
                -------------------------------------------------
                Reference vector is independent of clients data.
                """
                for method in ["Independent"]:
                    method_name = method
                    print(f"Method: {method_name.replace('_', ' ')}")

                    reference_feat_file = (
                        workspace_root / "Assets" / "Features" / subset / f"{encoder}_using_{encoder}.npz"
                    )
                    reference_feats = np.load(reference_feat_file, allow_pickle=True)["arr_0"]
                    reference_feats[:, 1:] = reference_feats[:, 1:].astype(np.float32)

                    ref_vector, ref_covmat = reference_vector(reference_feats[:, 1:].astype(np.float32))
                    ref_vectors, ref_covmats = [ref_vector], [ref_covmat]

                    independent_scores = compute_and_store_scores(
                        ref_vectors,
                        ref_covmats,
                        client_feats,
                        tag=f"{encoder}_{method_name}_mahala",
                        chi=False,
                        scores_output_csv_path=mahala_scores_csv_path,
                    )
                    compute_and_store_scores(
                        ref_vectors,
                        ref_covmats,
                        client_feats,
                        tag=f"{encoder}_{method_name}_chi",
                        chi=True,
                        scores_output_csv_path=chi_scores_csv_path,
                    )
                    independent_aucs, independent_aps, independent_fprs, independent_baseline = compute_client_metrics(
                        scores_by_client=independent_scores,
                        client_labels=client_labels,
                        encoder=encoder,
                    )
                    save_metrics(csv_path=csv_path, method_name=method_name, encoder=encoder, values=independent_aucs)
                    save_metrics(
                        csv_path=ap_csv_path,
                        method_name=method_name,
                        encoder=encoder,
                        values=independent_aps,
                        baseline=independent_baseline,
                    )
                    save_metrics(
                        csv_path=fpr_csv_path,
                        method_name=method_name,
                        encoder=encoder,
                        values=independent_fprs,
                        baseline=95.0,
                    )

                """
                AVERAGE
                -------------------------------------------------
                Reference vector is the average of clients data.
                """
                client_stats = []
                for client, feats in client_feats.items():
                    feats_values = feats[:, 1:].astype(np.float32)
                    mu, cov = reference_vector(feats_values)
                    client_stats.append((client, mu, cov, feats_values.shape[0]))

                size_values = np.array([n_i for _, _, _, n_i in client_stats], dtype=np.float64)
                weights = size_values / size_values.sum()

                method_name = "Average"
                print(f"Method: {method_name}")

                means = np.array([mu for _, mu, _, _ in client_stats])
                covs = np.array([cov for _, _, cov, _ in client_stats])

                global_mean = np.sum(weights[:, None] * means, axis=0)
                d = means.shape[1]
                global_cov = np.zeros((d, d))

                for w_i, mu_i, cov_i in zip(weights, means, covs, strict=False):
                    diff = (mu_i - global_mean).reshape(-1, 1)
                    global_cov += w_i * (cov_i + diff @ diff.T)

                ref_vectors, ref_covmats = [global_mean], [global_cov]

                method_scores = compute_and_store_scores(
                    ref_vectors,
                    ref_covmats,
                    client_feats,
                    tag=f"{encoder}_{method_name}_mahala",
                    chi=False,
                    scores_output_csv_path=mahala_scores_csv_path,
                )
                compute_and_store_scores(
                    ref_vectors,
                    ref_covmats,
                    client_feats,
                    tag=f"{encoder}_{method_name}_chi",
                    chi=True,
                    scores_output_csv_path=chi_scores_csv_path,
                )
                method_aucs, method_aps, method_fprs, method_baseline = compute_client_metrics(
                    scores_by_client=method_scores,
                    client_labels=client_labels,
                    encoder=encoder,
                )
                save_metrics(csv_path=csv_path, method_name=method_name, encoder=encoder, values=method_aucs)
                save_metrics(
                    csv_path=ap_csv_path,
                    method_name=method_name,
                    encoder=encoder,
                    values=method_aps,
                    baseline=method_baseline,
                )
                save_metrics(
                    csv_path=fpr_csv_path,
                    method_name=method_name,
                    encoder=encoder,
                    values=method_fprs,
                    baseline=95.0,
                )

                """
                PICKED
                -------------------------------------------------
                Reference vector relies on the client picking compliant samples.
                """
                method = "Picked"
                for n_samples in picked_samples:
                    method_name = f"{method}_{n_samples}_Samples"
                    print(f"Method: {method_name.replace('_', ' ')}")
                    client_aucs_iter: dict[str, list[float]] = {}
                    client_aps_iter: dict[str, list[float]] = {}
                    client_fprs_iter: dict[str, list[float]] = {}
                    baselines_iter: list[float] = []

                    for _i in range(n_iter):
                        picked_feats = []
                        for client, feats in client_feats.items():
                            labels = client_labels[client]["annotation"]
                            compliant_indices = np.where(labels == 1)[0]
                            sampled_indices = np.random.choice(compliant_indices, size=n_samples, replace=False)
                            compliant_feats = feats[sampled_indices, 1:].astype(np.float32)

                            picked_feats.append(compliant_feats)

                        picked_feats = np.vstack(picked_feats)
                        ref_vector, ref_covmat = reference_vector(picked_feats)
                        ref_vectors, ref_covmats = [ref_vector], [ref_covmat]

                        iter_scores = compute_and_store_scores(
                            ref_vectors,
                            ref_covmats,
                            client_feats,
                            tag=f"{encoder}_{method_name}_mahala",
                            chi=False,
                            scores_output_csv_path=mahala_scores_csv_path,
                        )
                        compute_and_store_scores(
                            ref_vectors,
                            ref_covmats,
                            client_feats,
                            tag=f"{encoder}_{method_name}_chi",
                            chi=True,
                            scores_output_csv_path=chi_scores_csv_path,
                        )
                        client_aucs, client_aps, client_fprs, iter_baseline = compute_client_metrics(
                            scores_by_client=iter_scores,
                            client_labels=client_labels,
                            encoder=encoder,
                        )

                        for client, auc in client_aucs.items():
                            client_aucs_iter.setdefault(client, []).append(auc)
                        for client, ap in client_aps.items():
                            client_aps_iter.setdefault(client, []).append(ap)
                        for client, fpr in client_fprs.items():
                            client_fprs_iter.setdefault(client, []).append(fpr)
                        baselines_iter.append(iter_baseline)

                    client_aucs = {client: float(np.mean(aucs)) for client, aucs in client_aucs_iter.items()}
                    client_aps = {client: float(np.mean(aps)) for client, aps in client_aps_iter.items()}
                    client_fprs = {client: float(np.mean(fprs)) for client, fprs in client_fprs_iter.items()}
                    client_auc_stds = {client: float(np.std(aucs)) for client, aucs in client_aucs_iter.items()}
                    client_ap_stds = {client: float(np.std(aps)) for client, aps in client_aps_iter.items()}
                    client_fpr_stds = {client: float(np.std(fprs)) for client, fprs in client_fprs_iter.items()}
                    save_metrics(csv_path=csv_path, method_name=method_name, encoder=encoder, values=client_aucs)
                    save_metrics(
                        csv_path=csv_path,
                        method_name=f"{method_name}_std",
                        encoder=encoder,
                        values=client_auc_stds,
                    )
                    save_metrics(
                        csv_path=ap_csv_path,
                        method_name=method_name,
                        encoder=encoder,
                        values=client_aps,
                        baseline=float(np.mean(baselines_iter)),
                    )
                    save_metrics(
                        csv_path=ap_csv_path,
                        method_name=f"{method_name}_std",
                        encoder=encoder,
                        values=client_ap_stds,
                        baseline=float(np.mean(baselines_iter)),
                    )
                    save_metrics(
                        csv_path=fpr_csv_path,
                        method_name=method_name,
                        encoder=encoder,
                        values=client_fprs,
                        baseline=95.0,
                    )
                    save_metrics(
                        csv_path=fpr_csv_path,
                        method_name=f"{method_name}_std",
                        encoder=encoder,
                        values=client_fpr_stds,
                        baseline=95.0,
                    )

        if run_report:
            export_independent_auc_table(csv_path=csv_path, tex_path=independent_tex_output_path)
            export_methods_holdout_table(csv_path=csv_path, tex_path=methods_tex_output_path)
            export_scheme_table(
                csv_path=csv_path,
                tex_path=scheme_auc_tex_output_path,
                caption=(
                    r"Holdout \ac{AUC} (\%) for the static-encoder \ac{CAF} implementation in \ac{FL} settings,"
                    r" per training dataset and reference scheme. \textit{Base.} is the random-chance level."
                ),
                label="tab:static_encoder_scheme_auc",
            )
            export_scheme_table(
                csv_path=ap_csv_path,
                tex_path=scheme_ap_tex_output_path,
                caption=(
                    r"Holdout Average Precision (\%) for the static-encoder \ac{CAF} implementation in \ac{FL}"
                    r" settings, per training dataset and reference scheme. \textit{Base.} is the non-compliant"
                    r" prevalence (\ac{AP} chance level)."
                ),
                label="tab:static_encoder_scheme_ap",
            )
            export_scheme_plot(
                csv_path=csv_path,
                plot_path=scheme_auc_plot_output_path,
                ylabel="AUC (%)",
                baseline_csv_path=niqe_baseline_csv_path,
            )
            export_scheme_plot(
                csv_path=ap_csv_path,
                plot_path=scheme_ap_plot_output_path,
                ylabel="Average Precision (%)",
                baseline_csv_path=niqe_baseline_csv_path,
                baseline_column="niqe_ap",
                show_prevalence=True,
            )
            if fpr_csv_path.exists():
                export_scheme_plot(
                    csv_path=fpr_csv_path,
                    plot_path=scheme_fpr_plot_output_path,
                    ylabel="FPR@95TPR (%)",
                    baseline_csv_path=niqe_baseline_csv_path,
                    baseline_column="niqe_fpr95",
                )
            else:
                print(f"Skipping FPR plot: {fpr_csv_path} not found (run with --compute first).")


if __name__ == "__main__":
    main()
