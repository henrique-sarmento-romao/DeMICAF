"""Point metrics and stratified bootstrap CIs for compliance-score evaluation.

Scores are assumed oriented so that higher = more non-compliant (``y=1``); callers
apply any per-method sign flip before these functions see the scores (see the
per-cause benchmark's orientation table). Metrics are never auto-flipped here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

LOW_N_THRESHOLD = 50


def _has_both_classes(y_true: np.ndarray) -> bool:
    """Return False when ``y_true`` is single-class — sklearn's ranking metrics are undefined then.

    Callers use this to short-circuit before calling into sklearn: a degenerate
    (single-class) sample stays degenerate under every resampling too (see
    :func:`stratified_bootstrap`), so skipping here also avoids spamming
    ``UndefinedMetricWarning`` a bootstrap's worth of times over.
    """
    return bool((y_true == 1).any() and (y_true == 0).any())


def fpr_at_tpr(y_true: ArrayLike, scores: ArrayLike, tpr_target: float = 0.95) -> float:
    """Compute the False Positive Rate at the first operating point reaching ``tpr_target``."""
    y_true = np.asarray(y_true)
    if not _has_both_classes(y_true):
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= tpr_target
    return float(fpr[mask].min()) if mask.any() else np.nan


class PointMetrics(NamedTuple):
    """AUROC, AP, and FPR@95TPR for one scored set, plus its class balance."""

    auroc: float
    ap: float
    fpr95: float
    n_pos: int
    n_neg: int
    prevalence: float
    low_n: bool


def point_metrics(y_true: ArrayLike, scores: ArrayLike, tpr_target: float = 0.95) -> PointMetrics:
    """Compute AUROC, AP, and FPR@``tpr_target`` for one ``(y_true, scores)`` pair."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    prevalence = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else np.nan
    return PointMetrics(
        auroc=_auroc(y_true, scores),
        ap=_ap(y_true, scores),
        fpr95=fpr_at_tpr(y_true, scores, tpr_target),
        n_pos=n_pos,
        n_neg=n_neg,
        prevalence=prevalence,
        low_n=n_pos < LOW_N_THRESHOLD,
    )


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if not _has_both_classes(y_true):
        return np.nan
    return float(roc_auc_score(y_true, scores))


def _ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if not _has_both_classes(y_true):
        return np.nan
    return float(average_precision_score(y_true, scores))


def _fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    return fpr_at_tpr(y_true, scores, tpr_target=0.95)


#: Single-metric functions matching :class:`PointMetrics`' fields, for reuse with
#: :func:`stratified_bootstrap` (one bootstrap pass per metric).
METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": _auroc,
    "ap": _ap,
    "fpr95": _fpr95,
}


def stratified_bootstrap(
    y_true: ArrayLike,
    scores: ArrayLike,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap replicates of ``fn(y_true, scores)``, resampling each class separately.

    Returns the raw (NaN-free) array of ``n_boot`` replicate values so callers can pool
    replicates across multiple split seeds before taking percentiles.
    """
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, pos.size, replace=True),
                rng.choice(neg, neg.size, replace=True),
            ]
        )
        out.append(fn(y_true[idx], scores[idx]))
    out_arr = np.asarray(out, dtype=np.float64)
    return out_arr[~np.isnan(out_arr)]
