"""Federated scoring with a static (frozen) encoder.

Each client scores its images against reference statistics that are either
independent (encoder's own data), federated averages of client statistics
(with optional quality-penalty weighting), or built from a few picked
compliant samples. AUCs against the manual compliance annotations are
exported as LaTeX tables.

The chest X-ray images are **not** shipped (registration-gated source datasets);
point the ``CXR_ROOT`` environment variable at a directory laid out as
``<Dataset>/...`` before running.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from demicaf.caf.scorer import mahalanobis_distance, reference_vector
from demicaf.utils.io import upsert_scores_csv as _upsert_scores_csv
from sklearn.metrics import average_precision_score, roc_auc_score

datasets = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
picked_samples = [1, 3, 5, 10]
n_iter = 10
quality_penalty_betas = [0.0, 1, 2, 5, 10, 20]

workspace_root = Path(__file__).resolve().parents[2]
data_root = Path(os.environ.get("CXR_ROOT", "data/chest_xray"))


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


def quality_fraction_weights(mean_scores: dict[str, float], beta: float) -> dict[str, float]:
    """Convert client mean scores into normalized quality weights."""
    score_values = np.array(list(mean_scores.values()), dtype=np.float64)
    score_values /= sum(score_values)
    shifted_scores = -beta * score_values
    shifted_scores -= shifted_scores.max()
    weights = np.exp(shifted_scores)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Unable to normalize client scores for beta={beta:g}.")
    weights /= total
    return {client: float(weight) for client, weight in zip(mean_scores.keys(), weights, strict=False)}


def normalized_mixture_weights(size_values: np.ndarray, quality_values: np.ndarray) -> np.ndarray:
    """Combine normalized size and quality fractions and renormalize the product."""
    weights = np.asarray(size_values, dtype=np.float64) * np.asarray(quality_values, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Unable to normalize combined client weights.")
    return weights / total


def compute_client_metrics(
    scores_by_client: dict[str, pd.DataFrame],
    client_labels: dict[str, pd.DataFrame],
    encoder: str,
    percentage: bool = True,
    invert_labels: bool = True,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Compute AUC and AP per dataset/holdout plus the holdout non-compliant prevalence.

    Returns
    -------
    tuple
        ``(aucs, aps, holdout_baseline)`` where ``aucs``/``aps`` map each dataset
        (and ``"Holdout"``) to its metric, and ``holdout_baseline`` is the AP
        chance level (non-compliant fraction over the held-out clients).
    """
    aucs: dict[str, float] = dict.fromkeys(datasets, np.nan)
    aps: dict[str, float] = dict.fromkeys(datasets, np.nan)

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
    holdout_baseline = non_compliant_prevalence(
        feat_names=holdout_feat_names,
        scores=holdout_scores,
        annotation_df=holdout_annotations,
        percentage=percentage,
    )

    # Encoder dataset is not part of holdout clients for that row.
    aucs[encoder] = np.nan
    aps[encoder] = np.nan
    return aucs, aps, holdout_baseline


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
        "Average_beta_0",
        "Average_beta_1",
        "Average_beta_2",
        "Average_beta_5",
        "Average_beta_10",
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
    """Export a single Holdout table with independent, beta-sweep, and picked rows."""
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

    beta_rows = [(beta, f"Beta {beta}") if beta != 0 else (beta, "Unweighted") for beta in quality_penalty_betas]

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \caption{Holdout AUC scores for static encoder \ac{CAF} implementation in \ac{FL} settings,"
        r" varying encoder dataset and quality penalty beta.} \label{tab:static_encoder_methods_auc}",
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
        row("\\multirow{5}{*}{Average}", beta_rows[0][1], encoder_values("Average_beta_0")),
        r"        \cline{2-6}",
    ]

    for beta_value, beta_label in beta_rows[1:]:
        lines.append(row("", beta_label, encoder_values(f"Average_beta_{beta_value:g}")))
        if beta_value != beta_rows[-1][0]:
            lines.append(r"        \cline{2-6}")

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
    "ChestX-ray8": "ChestX-ray",
    "MIMIC-CXR": "MIMIC",
    "PadChest": "PadChest",
}


def export_scheme_table(
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
    weighted_beta: float = 5,
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

    # (method key, header label) for the seven scheme columns, in display order.
    scheme_methods = [
        ("Independent", "Independent"),
        ("Picked_1_Samples", "1"),
        ("Picked_3_Samples", "3"),
        ("Picked_5_Samples", "5"),
        ("Picked_10_Samples", "10"),
        ("Average_beta_0", "Unweighted"),
        (f"Average_beta_{weighted_beta:g}", "Weighted"),
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
        r"    \begin{tabular}{c|c|c|c|c|c|c|c|c}",
        r"    \toprule",
        r"    \multirow{2}{*}{\diagbox{Train}{Scheme}} & \multirow{2}{*}{\textit{Base.}}"
        r" & \multirow{2}{*}{Independent} & \multicolumn{4}{c|}{Picked (Samples)}"
        r" & \multicolumn{2}{c}{Average} \\",
        r"    \cline{4-9}",
        r"    & & & 1 & 3 & 5 & 10 & Unweighted & Weighted \\",
        r"    \midrule",
    ]

    for index, encoder in enumerate(datasets):
        values = [holdout_for(method, encoder) for method, _ in scheme_methods]
        cells = format_cells(values)

        baseline = baselines[encoder]
        if constant_baseline:
            if index == 0:
                base_cell = r"\multirow{" + str(len(datasets)) + r"}{*}{" + f"{present_baselines[0]:.1f}" + "}"
            else:
                base_cell = ""
        else:
            base_cell = f"{baseline:.1f}" if baseline is not None else "—"

        train_label = scheme_train_labels.get(encoder, encoder)
        lines.append(f"    {{{train_label}}} & {base_cell} & " + " & ".join(cells) + r" \\")

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
        csv_path = workspace_root / f"Results/StaticEncoder/{subset}_AUCs.csv"
        ap_csv_path = workspace_root / f"Results/StaticEncoder/{subset}_APs.csv"
        mahala_scores_csv_path = workspace_root / f"Results/StaticEncoder/{subset}_MahalaScores.csv"
        chi_scores_csv_path = workspace_root / f"Results/StaticEncoder/{subset}_ChiScores.csv"
        independent_tex_output_path = workspace_root / f"Results/StaticEncoder/{subset}_auc_independent.tex"
        methods_tex_output_path = workspace_root / f"Results/StaticEncoder/{subset}_auc_methods.tex"
        scheme_auc_tex_output_path = workspace_root / f"Results/StaticEncoder/{subset}_scheme_auc.tex"
        scheme_ap_tex_output_path = workspace_root / f"Results/StaticEncoder/{subset}_scheme_ap.tex"

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
                    independent_aucs, independent_aps, independent_baseline = compute_client_metrics(
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

                """
                AVERAGE
                -------------------------------------------------
                Reference vector is the average of clients data.
                """
                mean_scores = {
                    client: float(score_df["score"].mean()) for client, score_df in independent_scores.items()
                }
                client_stats = []
                for client, feats in client_feats.items():
                    feats_values = feats[:, 1:].astype(np.float32)
                    print(feats_values.shape)
                    mu, cov = reference_vector(feats_values)
                    client_stats.append((client, mu, cov, feats_values.shape[0]))

                size_values = np.array([n_i for _, _, _, n_i in client_stats], dtype=np.float64)
                size_fractions = size_values / size_values.sum()
                beta_quality_scores = {
                    beta: quality_fraction_weights(mean_scores, beta) for beta in quality_penalty_betas
                }

                for quality_penalty_beta in quality_penalty_betas:
                    method_name = f"Average_beta_{quality_penalty_beta:g}"
                    print(f"Method: {method_name.replace('_', ' ')}")

                    quality_values = np.array(
                        [beta_quality_scores[quality_penalty_beta][client] for client, _, _, _ in client_stats],
                        dtype=np.float64,
                    )
                    weights = normalized_mixture_weights(size_fractions, quality_values)
                    # print(weights)

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
                    method_aucs, method_aps, method_baseline = compute_client_metrics(
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
                        client_aucs, client_aps, iter_baseline = compute_client_metrics(
                            scores_by_client=iter_scores,
                            client_labels=client_labels,
                            encoder=encoder,
                        )

                        for client, auc in client_aucs.items():
                            client_aucs_iter.setdefault(client, []).append(auc)
                        for client, ap in client_aps.items():
                            client_aps_iter.setdefault(client, []).append(ap)
                        baselines_iter.append(iter_baseline)

                    client_aucs = {client: float(np.mean(aucs)) for client, aucs in client_aucs_iter.items()}
                    client_aps = {client: float(np.mean(aps)) for client, aps in client_aps_iter.items()}
                    save_metrics(csv_path=csv_path, method_name=method_name, encoder=encoder, values=client_aucs)
                    save_metrics(
                        csv_path=ap_csv_path,
                        method_name=method_name,
                        encoder=encoder,
                        values=client_aps,
                        baseline=float(np.mean(baselines_iter)),
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


if __name__ == "__main__":
    main()
