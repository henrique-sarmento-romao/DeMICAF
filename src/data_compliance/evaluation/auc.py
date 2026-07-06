"""Compliance-score evaluation: AUC against manual compliance annotations."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def filter_cause(
    cause: str,
    df_cause: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only compliant samples plus non-compliant samples with the given cause."""
    causes = [column for column in df_cause.columns if column not in ["image_path", "compliance"]]
    cause = cause.lower().replace(" ", "_").replace("-", "_")

    cause_df = df_cause.loc[df_cause[cause] == 1]
    compliant_df = df_cause.loc[(df_cause[causes] != 1).all(axis=1)]
    return pd.concat([cause_df, compliant_df], ignore_index=True)


def _align_labels_and_scores(
    annotation_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    score: str,
    *,
    invert_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Align scores to annotations and drop rows with missing scores.

    Parameters
    ----------
    annotation_df : pd.DataFrame
        Must contain ``image_path`` and ``annotation`` (compliant / non-compliant).
    scores_df : pd.DataFrame
        Must contain ``image_path`` and the ``score`` column.
    score : str
        Name of the score column in ``scores_df``.
    invert_labels : bool
        When True the positive class is *compliant*; when False it is *non-compliant*.

    Returns
    -------
    tuple of np.ndarray
        ``(labels, scores)`` as aligned, NaN-free arrays.
    """
    annotation_map = {"compliant": 0, "non-compliant": 1, "non compliant": 1}
    if invert_labels:
        annotation_map = {k: 1 - v for k, v in annotation_map.items()}

    ann_df = annotation_df[["image_path", "annotation"]].copy()
    ann_df["image_path"] = ann_df["image_path"].astype(str)
    ann_df["annotation"] = ann_df["annotation"].astype(str).str.strip().str.lower().map(annotation_map).astype(np.int64)

    annotated_scores = ann_df.merge(
        scores_df,
        on="image_path",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    aligned_scores = annotated_scores[score].to_numpy(dtype=np.float64)
    labels = annotated_scores["annotation"].to_numpy(dtype=np.int64)

    valid_mask = ~np.isnan(aligned_scores)
    return labels[valid_mask], aligned_scores[valid_mask]


def compute_auc(
    annotation_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    score: str = "score",
    *,
    percentage: bool = False,
    invert_labels: bool = True,
) -> float | None:
    """Compute an AUC for annotated samples after aligning scores to annotation order.

    Parameters
    ----------
    annotation_df : pd.DataFrame
        Must contain ``image_path`` and ``annotation`` (compliant / non-compliant).
    scores_df : pd.DataFrame
        Must contain ``image_path`` and the ``score`` column; rows with missing
        scores are dropped before computing the AUC.
    score : str
        Name of the score column in ``scores_df``.
    percentage : bool
        Return the AUC as a percentage instead of a fraction.
    invert_labels : bool
        When True (default) the positive class is *compliant*; when False it is
        *non-compliant*.

    Returns
    -------
    float or None
        The AUC, or None when no valid labels remain or only one class is present.
    """
    labels, aligned_scores = _align_labels_and_scores(annotation_df, scores_df, score, invert_labels=invert_labels)

    if labels.size == 0:
        warnings.warn("No valid labels available to compute AUC.", stacklevel=2)
        return None

    if np.all(labels == labels[0]):
        warnings.warn("All labels are the same; AUC cannot be computed.", stacklevel=2)
        return None

    auc = float(roc_auc_score(labels, aligned_scores))
    if percentage:
        auc *= 100

    return auc


def compute_average_precision(
    annotation_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    score: str = "score",
    *,
    percentage: bool = False,
    invert_labels: bool = True,
) -> float | None:
    """Compute the Average Precision (area under the precision-recall curve).

    Parameters
    ----------
    annotation_df : pd.DataFrame
        Must contain ``image_path`` and ``annotation`` (compliant / non-compliant).
    scores_df : pd.DataFrame
        Must contain ``image_path`` and the ``score`` column; rows with missing
        scores are dropped before computing the metric.
    score : str
        Name of the score column in ``scores_df``.
    percentage : bool
        Return the Average Precision as a percentage instead of a fraction.
    invert_labels : bool
        When True (default) the positive class is *compliant*; when False it is
        *non-compliant*.

    Returns
    -------
    float or None
        The Average Precision, or None when no valid labels remain or only one
        class is present.
    """
    labels, aligned_scores = _align_labels_and_scores(annotation_df, scores_df, score, invert_labels=invert_labels)

    if labels.size == 0:
        warnings.warn("No valid labels available to compute Average Precision.", stacklevel=2)
        return None

    if np.all(labels == labels[0]):
        warnings.warn("All labels are the same; Average Precision cannot be computed.", stacklevel=2)
        return None

    ap = float(average_precision_score(labels, aligned_scores))
    if percentage:
        ap *= 100

    return ap


def compute_fpr_at_tpr(
    annotation_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    score: str = "score",
    *,
    tpr_target: float = 0.95,
    percentage: bool = False,
    invert_labels: bool = True,
) -> float | None:
    """Compute the False Positive Rate at a fixed True Positive Rate (default 95%).

    The FPR is read off the ROC curve at the first operating point whose TPR
    reaches ``tpr_target`` (linearly interpolated between curve vertices).

    Parameters
    ----------
    annotation_df : pd.DataFrame
        Must contain ``image_path`` and ``annotation`` (compliant / non-compliant).
    scores_df : pd.DataFrame
        Must contain ``image_path`` and the ``score`` column; rows with missing
        scores are dropped before computing the metric.
    score : str
        Name of the score column in ``scores_df``.
    tpr_target : float
        Target True Positive Rate (fraction in ``[0, 1]``) at which to read the FPR.
    percentage : bool
        Return the FPR as a percentage instead of a fraction.
    invert_labels : bool
        When True (default) the positive class is *compliant*; when False it is
        *non-compliant*.

    Returns
    -------
    float or None
        The FPR at the target TPR, or None when no valid labels remain or only one
        class is present.
    """
    labels, aligned_scores = _align_labels_and_scores(annotation_df, scores_df, score, invert_labels=invert_labels)

    if labels.size == 0:
        warnings.warn("No valid labels available to compute FPR at TPR.", stacklevel=2)
        return None

    if np.all(labels == labels[0]):
        warnings.warn("All labels are the same; FPR at TPR cannot be computed.", stacklevel=2)
        return None

    fpr, tpr, _ = roc_curve(labels, aligned_scores)
    fpr_at_tpr = float(np.interp(tpr_target, tpr, fpr))
    if percentage:
        fpr_at_tpr *= 100

    return fpr_at_tpr


def save_auc(
    auc: float,
    dataset: str,
    framework: str,
    rank_method: str,
    k: int,
    filename: str | Path = "AUCs.csv",
) -> None:
    """Insert or update one ``(dataset, framework, rank_method, k)`` AUC row in a CSV."""
    path = Path(filename)
    if path.exists():
        df = pd.read_csv(path)
        mask = (
            (df["Dataset"] == dataset)
            & (df["Framework"] == framework)
            & (df["Ranking Method"] == rank_method)
            & (df["Clusters"] == k)
        )
        if mask.any():
            df.loc[mask, "AUC"] = auc
        else:
            df = pd.concat(
                [df, pd.DataFrame([[dataset, framework, rank_method, k, auc]], columns=df.columns)], ignore_index=True
            )
    else:
        df = pd.DataFrame(
            [[dataset, framework, rank_method, k, auc]],
            columns=["Dataset", "Framework", "Ranking Method", "Clusters", "AUC"],
        )

    df.to_csv(path, index=False)
