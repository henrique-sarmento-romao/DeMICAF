r"""Per-cause compliance benchmark: DeMICAF vs. baselines on ``annotated_causes.csv``.

For each non-compliance cause and each scoring method, scores every pooled
client-dataset test image and reports AUROC, AP, and FPR@95TPR (with stratified
bootstrap 95% CIs) for separating cause-`c` images from fully-compliant images, plus
an aggregate (all non-compliant vs. compliant) row per method. See
``per_cause_benchmark_spec.md`` for the full protocol.

P0 methods (``demicaf_aggfull``, ``deepclean_central``, ``niqe``, ``brisque``,
``qalign``, ``topiq``) plus P1's ``knn``, ``maha_pp`` (classifier penultimate-layer
features — see ``scripts/extract_classifier_features.py``) and ``selfclean``
(pretrained-ViT off-topic-sample score, no self-supervised training). ``demicaf_prior``
and ``gen`` (P2) are not implemented — not needed for the target table.

Reference/test split: compliant images of the three client datasets (CheXpert is the
excluded prior) are split 50/50 per client dataset into a reference split (used only
to fit the two Mahalanobis-distance methods' reference Gaussians) and a test split;
all non-compliant images go to the test split. Repeated over multiple seeds.

Typical usage (run as a module so ``scripts.baseline`` is importable)::

    uv run python -m scripts.benchmarking
    uv run python -m scripts.benchmarking --methods niqe brisque --n-seeds 1 --n-boot 100
    uv run python -m scripts.benchmarking --resume
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyiqa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from DeMICAF.caf.scorer import mahalanobis_distance, reference_vector
from DeMICAF.evaluation.bootstrap import LOW_N_THRESHOLD, METRIC_FNS, point_metrics, stratified_bootstrap
from DeMICAF.utils.io import load_features
from DeMICAF.utils.paths import get_repo_root, resolve_path
from DeMICAF.utils.plotting import apply_thesis_style
from scripts.baseline import _safe_metric, image_to_tensor, load_image

# ── Constants ─────────────────────────────────────────────────────────────────

PRIOR_DATASET = "CheXpert"
CLIENTS = ["ChestX-ray8", "MIMIC-CXR", "PadChest"]

#: Spec's short cause key -> actual `annotated_causes.csv` column name.
CAUSE_COLUMN_MAP = {
    "corrupted": "corrupted_images",
    "not_thorax": "not_a_thorax",
    "incomplete_thorax": "incomplete_thorax",
    "image_quality": "image_quality_problems",
    "area_not_valid": "area_not_valid",
    "overlaying_objects": "overlaying_objects",
    "non_canonical": "non_canonical_positions",
}

#: Score orientation per method: "asis" (higher = more non-compliant already) or
#: "negate" (native metric is higher = better quality, so flip the sign). Set
#: explicitly per spec §6 — never auto-flipped.
ORIENTATION = {
    "demicaf_aggfull": "asis",
    "deepclean_central": "asis",
    "niqe": "asis",
    "brisque": "asis",
    "qalign": "negate",
    "topiq": "negate",
    "knn": "asis",
    "maha_pp": "asis",
    # Empirically verified against a full run (per spec §6, "check convention" for
    # selfclean): raw off_topic_samples score correlates with *compliant*, not
    # non-compliant, in this label-free setup — negate to match the "higher = more
    # non-compliant" convention. Caught by the aggregate-AUROC assertion below.
    "selfclean": "negate",
    # Empirically verified (per spec §6): our multi-label adaptation of GEN correlates
    # with *compliant*, not non-compliant — negate to match the "higher = more
    # non-compliant" convention. Caught by the aggregate-AUROC assertion below.
    "gen": "negate",
}

PYIQA_METRIC_NAME = {"qalign": "qalign", "topiq": "topiq_nr"}

#: Neighbours for the KNN-OOD scorer (Sun et al. 2022). Not specified by the
#: spec; 10 is a reasonable default for a ~1300-image reference bank (this
#: isn't ImageNet-scale, so no literature value to match).
KNN_K = 10

TPR_TARGET = 0.95
N_BOOT_DEFAULT = 1000
N_SEEDS_DEFAULT = 3
SEED_BASE = 0

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = WORKSPACE_ROOT / "Assets" / "Features" / "100k"

# ── Data loading ──────────────────────────────────────────────────────────────


def load_annotations(annotations_path: Path) -> pd.DataFrame:
    """Read the annotation CSV, tag each row with its dataset, and restrict to the clients."""
    df = pd.read_csv(annotations_path)
    df["dataset"] = df["image_path"].astype(str).str.split("/", n=1).str[0]
    df = df[df["dataset"].isin(CLIENTS)].reset_index(drop=True)
    df["compliant"] = df["compliance"].astype(str).str.strip().str.lower() == "compliant"
    return df


@dataclass
class Split:
    """One seed's reference/test partition of the client-pooled annotations."""

    seed: int
    reference_ids: dict[str, np.ndarray]  # client -> compliant image_paths in the reference split
    test_df: pd.DataFrame  # test-compliant + all non-compliant, all clients pooled


def build_split(df: pd.DataFrame, seed: int) -> Split:
    """Split compliant client images 50/50 per client dataset; non-compliant images all go to test."""
    compliant = df[df["compliant"]]
    non_compliant = df[~df["compliant"]]

    reference_ids: dict[str, np.ndarray] = {}
    test_compliant_frames = []
    for client, group in compliant.groupby("dataset", sort=False):
        ref_group, test_group = train_test_split(group, test_size=0.5, random_state=seed)
        reference_ids[client] = ref_group["image_path"].to_numpy(dtype=str)
        test_compliant_frames.append(test_group)

    test_df = pd.concat([*test_compliant_frames, non_compliant], ignore_index=True)
    return Split(seed=seed, reference_ids=reference_ids, test_df=test_df)


# ── DeMICAF / DeepClean (SupCon Mahalanobis) scorers ─────────────────────────


class FeatureBank(NamedTuple):
    """Precomputed SupCon features for one client dataset under one encoder."""

    names: np.ndarray
    feats: np.ndarray
    index: dict[str, int]


def load_demicaf_features(encoder: str) -> dict[str, FeatureBank]:
    """Load precomputed ``{client}_using_{encoder}.npz`` feature banks for every client."""
    banks: dict[str, FeatureBank] = {}
    for client in CLIENTS:
        names, feats = load_features(FEATURES_ROOT / f"{client}_using_{encoder}.npz")
        banks[client] = FeatureBank(names=names, feats=feats, index=dict(zip(names, range(len(names)), strict=True)))
    return banks


def load_classifier_features(features_dir: Path) -> dict[str, FeatureBank]:
    """Load ``classifier_features_{client}.npz`` banks written by ``extract_classifier_features.py``."""
    banks: dict[str, FeatureBank] = {}
    for client in CLIENTS:
        names, feats = load_features(features_dir / f"classifier_features_{client}.npz")
        banks[client] = FeatureBank(names=names, feats=feats, index=dict(zip(names, range(len(names)), strict=True)))
    return banks


def _select_features(bank: FeatureBank, ids: np.ndarray) -> np.ndarray:
    """Return the feature rows for the given image paths, in ``ids`` order."""
    positions = np.asarray([bank.index[image_id] for image_id in ids], dtype=np.int64)
    return bank.feats[positions]


def _l2_normalize_bank(bank: FeatureBank) -> FeatureBank:
    """Return a copy of ``bank`` with every feature row scaled to unit L2 norm."""
    norms = np.linalg.norm(bank.feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return FeatureBank(names=bank.names, feats=bank.feats / norms, index=bank.index)


def _pooled_reference_features(split: Split, client_feats: dict[str, FeatureBank]) -> np.ndarray:
    """Concatenate all 3 clients' reference-split features into one pooled array."""
    return np.concatenate(
        [_select_features(client_feats[client], split.reference_ids[client]) for client in CLIENTS],
        axis=0,
    )


def fit_aggfull_reference(
    split: Split,
    client_feats: dict[str, FeatureBank],
) -> tuple[np.ndarray, np.ndarray]:
    """Agg-Full reference: per-client Gaussians fused by size-weighted mean/covariance.

    Matches the federated aggregation formula used by ``scripts/scoring/score.py``'s
    ``"Average"`` method, but fit on this seed's reference split rather than the full
    client data.
    """
    means, covs, sizes = [], [], []
    for client in CLIENTS:
        ref_feats = _select_features(client_feats[client], split.reference_ids[client])
        mean, cov = reference_vector(ref_feats)
        means.append(mean)
        covs.append(cov)
        sizes.append(ref_feats.shape[0])

    weights = np.asarray(sizes, dtype=np.float64)
    weights /= weights.sum()
    means_arr = np.asarray(means)
    covs_arr = np.asarray(covs)

    global_mean = np.sum(weights[:, None] * means_arr, axis=0)
    global_cov = np.zeros_like(covs_arr[0])
    for weight, mean, cov in zip(weights, means_arr, covs_arr, strict=True):
        diff = (mean - global_mean).reshape(-1, 1)
        global_cov += weight * (cov + diff @ diff.T)

    return global_mean, global_cov


def fit_central_reference(
    split: Split,
    client_feats: dict[str, FeatureBank],
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (centralised) reference: a single Gaussian over all clients' reference-split features.

    Also used for ``maha_pp`` (L2-normalised classifier features are passed in as
    ``client_feats``) — the pooling/fitting math is identical, only the feature
    source differs.
    """
    return reference_vector(_pooled_reference_features(split, client_feats))


def score_md(
    test_df: pd.DataFrame,
    client_feats: dict[str, FeatureBank],
    mean: np.ndarray,
    cov: np.ndarray,
) -> dict[str, float]:
    """Mahalanobis distance of every test image to ``(mean, cov)``; higher = more OOD."""
    scores: dict[str, float] = {}
    for client, group in test_df.groupby("dataset", sort=False):
        ids = group["image_path"].to_numpy(dtype=str)
        points = _select_features(client_feats[client], ids)
        md = mahalanobis_distance(points, [mean], [cov])
        scores.update(dict(zip(ids, md, strict=True)))
    return scores


def score_knn(
    split: Split,
    client_feats: dict[str, FeatureBank],
    k: int = KNN_K,
) -> dict[str, float]:
    """Distance to the k-th nearest neighbour in the pooled, L2-normalised reference bank.

    ``client_feats`` must already be L2-normalised (see :func:`_l2_normalize_bank`) —
    both the reference bank and the test points need to be on the unit hypersphere
    for this to match the Sun et al. 2022 KNN-OOD formulation.
    """
    pooled_ref = _pooled_reference_features(split, client_feats)
    nn_model = NearestNeighbors(n_neighbors=k).fit(pooled_ref)

    scores: dict[str, float] = {}
    for client, group in split.test_df.groupby("dataset", sort=False):
        ids = group["image_path"].to_numpy(dtype=str)
        points = _select_features(client_feats[client], ids)
        distances, _ = nn_model.kneighbors(points)
        kth_distance = distances[:, -1]
        scores.update(dict(zip(ids, kth_distance, strict=True)))
    return scores


# ── No-reference IQA scorers ──────────────────────────────────────────────────


def load_cached_iqa(metric: str, cache_path: Path) -> dict[str, float]:
    """Load a metric column already computed by ``scripts/baseline.py``."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found; run `uv run python scripts/baseline.py` first to populate niqe/brisque."
        )
    df = pd.read_csv(cache_path, usecols=["image_path", metric]).dropna(subset=[metric])
    return dict(zip(df["image_path"].astype(str), df[metric].astype(float), strict=True))


def _write_iqa_rows(path: Path, rows: list[dict[str, Any]], *, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "score"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def compute_iqa(
    metric_key: str,
    df: pd.DataFrame,
    data_root: Path,
    cache_path: Path,
    *,
    resume: bool,
) -> dict[str, float]:
    """Compute a pyiqa no-reference metric over ``df``'s images, incrementally and resumably."""
    existing: dict[str, float] = {}
    if resume and cache_path.exists():
        cached = pd.read_csv(cache_path)
        existing = dict(zip(cached["image_path"].astype(str), cached["score"].astype(float), strict=True))

    todo = df[~df["image_path"].isin(existing)] if existing else df
    write_header = not (resume and cache_path.exists())

    if not todo.empty:
        metric_fn = pyiqa.create_metric(PYIQA_METRIC_NAME[metric_key])
        buffer: list[dict[str, Any]] = []
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc=metric_key, unit="img"):
            image_path = str(row["image_path"])
            full_path = Path(image_path)
            if not full_path.is_absolute():
                full_path = data_root / full_path

            image = load_image(full_path)
            score = _safe_metric(metric_fn, image_to_tensor(image)) if image is not None else None
            buffer.append({"image_path": image_path, "score": score})

            if len(buffer) >= 50:
                _write_iqa_rows(cache_path, buffer, write_header=write_header)
                write_header = False
                buffer.clear()

        if buffer:
            _write_iqa_rows(cache_path, buffer, write_header=write_header)

    cached = pd.read_csv(cache_path).dropna(subset=["score"])
    return dict(zip(cached["image_path"].astype(str), cached["score"].astype(float), strict=True))


# ── SelfClean scorer ──────────────────────────────────────────────────────────


class _SelfCleanImageDataset(TorchDataset):
    """Minimal Dataset exposing ``samples`` so SelfClean auto-detects image paths.

    SelfClean overwrites ``self.transform`` in place (see its
    ``set_dataset_transformation``), so this class starts with ``transform=None``
    and applies whatever gets assigned there — the same convention as
    ``torchvision.datasets.ImageFolder``.
    """

    def __init__(self, paths: list[str], data_root: Path) -> None:
        self.samples = [(path, 0) for path in paths]
        self.data_root = data_root
        self.transform = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self.samples[index]
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = self.data_root / full_path
        image = load_image(full_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def compute_selfclean(df: pd.DataFrame, data_root: Path, cache_path: Path, *, resume: bool) -> dict[str, float]:
    """Off-topic-sample score for every image in ``df``, cached (no reference/seed dependence).

    Uses a pretrained ImageNet ViT-tiny embedder (``PretrainingType.IMAGENET_VIT``)
    rather than SelfClean's default DINO self-supervised pretraining — training a
    DINO encoder from scratch is a multi-hour-to-days undertaking on CPU with no
    precomputed-embedding shortcut in the library; the pretrained-ViT path is a
    supported, much faster mode of the same library (confirmed against the
    installed source, not the docs, which don't fully cover this). Orientation is
    "asis" (higher = more off-topic/anomalous) — verified against the installed
    source's ``get_off_topic_ranking``, and cross-checked by the harness's own
    aggregate-AUROC assertion.
    """
    if resume and cache_path.exists():
        cached = pd.read_csv(cache_path)
        return dict(zip(cached["image_path"].astype(str), cached["score"].astype(float), strict=True))

    from selfclean.cleaner.issue_manager import IssueTypes
    from selfclean.cleaner.selfclean import PretrainingType, SelfClean

    paths = df["image_path"].tolist()
    dataset = _SelfCleanImageDataset(paths, data_root)

    cleaner = SelfClean(plot_distribution=False)
    issues = cleaner.run_on_dataset(
        dataset=dataset,
        pretraining_type=PretrainingType.IMAGENET_VIT,
        issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES],
        num_workers=4,
    )
    result = issues.get_issues("off_topic_samples")
    scores = {paths[int(idx)]: float(score) for idx, score in zip(result["indices"], result["scores"], strict=True)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": list(scores.keys()), "score": list(scores.values())}).to_csv(cache_path, index=False)
    return scores


# ── GEN (generalized entropy) scorer ──────────────────────────────────────────

#: Liu et al. 2023's default gamma for the generalized-entropy OOD score.
GEN_GAMMA = 0.1


def _generalized_entropy(probs: np.ndarray, gamma: float = GEN_GAMMA) -> np.ndarray:
    """Mean gamma-weighted binary entropy across labels, per image.

    Adapts Liu et al. 2023's GEN score — originally ``-sum_i p_i^gamma * log(p_i)``
    over a single-label softmax — to this multi-label sigmoid classifier: each
    label's ``[p, 1-p]`` is its own 2-outcome distribution, and the
    generalized-entropy term is averaged over the 6 labels. Same masked-averaging
    spirit as ``predictive_entropy`` elsewhere in this project's audit code, minus
    the mask (there's no uncertain/ignored label at pure-inference time — every
    probability here is a live model output, not a training target).
    """
    eps = 1e-12
    p = np.clip(probs, eps, 1 - eps)
    term = -((p**gamma) * np.log(p) + ((1 - p) ** gamma) * np.log(1 - p))
    return term.mean(axis=1)


def compute_gen(probs_dir: Path, cache_path: Path, *, resume: bool) -> dict[str, float]:
    """Generalized-entropy score for every classifier-scored image, cached (no reference/seed dependence).

    Reads ``classifier_probs_{client}.npz`` written by
    ``scripts/extract_classifier_features.py`` (sigmoid per-label probabilities from
    the trained classifier's final layer — a different output than the penultimate
    features ``knn``/``maha_pp`` use, extracted in the same forward pass). Orientation
    is the paper's own convention (higher = more OOD) as a default; verify against
    the harness's aggregate-AUROC assertion, same as every other new scorer here.
    """
    if resume and cache_path.exists():
        cached = pd.read_csv(cache_path)
        return dict(zip(cached["image_path"].astype(str), cached["score"].astype(float), strict=True))

    scores: dict[str, float] = {}
    for client in CLIENTS:
        names, probs = load_features(probs_dir / f"classifier_probs_{client}.npz")
        gen_scores = _generalized_entropy(probs)
        scores.update(dict(zip(names, gen_scores.tolist(), strict=True)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": list(scores.keys()), "score": list(scores.values())}).to_csv(cache_path, index=False)
    return scores


# ── Scorer registry ───────────────────────────────────────────────────────────


def build_scorers(
    split: Split,
    methods: list[str],
    demicaf_feats: dict[str, FeatureBank] | None,
    deepclean_feats: dict[str, FeatureBank] | None,
    classifier_feats: dict[str, FeatureBank] | None,
    iqa_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute this seed's ``{image_path: score}`` map for every requested method."""
    scores: dict[str, dict[str, float]] = {}

    if "demicaf_aggfull" in methods:
        mean, cov = fit_aggfull_reference(split, demicaf_feats)
        scores["demicaf_aggfull"] = score_md(split.test_df, demicaf_feats, mean, cov)
    if "deepclean_central" in methods:
        mean, cov = fit_central_reference(split, deepclean_feats)
        scores["deepclean_central"] = score_md(split.test_df, deepclean_feats, mean, cov)

    if "maha_pp" in methods or "knn" in methods:
        normalized_classifier_feats = {client: _l2_normalize_bank(bank) for client, bank in classifier_feats.items()}
        if "maha_pp" in methods:
            mean, cov = fit_central_reference(split, normalized_classifier_feats)
            scores["maha_pp"] = score_md(split.test_df, normalized_classifier_feats, mean, cov)
        if "knn" in methods:
            scores["knn"] = score_knn(split, normalized_classifier_feats)

    # "iqa_scores" also holds selfclean's and gen's precomputed per-image scores here —
    # all are {image_path: float} lookups with no per-seed reference dependence, so the
    # same subsetting loop covers all of them.
    test_ids = set(split.test_df["image_path"])
    for method in ("niqe", "brisque", "qalign", "topiq", "selfclean", "gen"):
        if method in methods:
            scores[method] = {k: v for k, v in iqa_scores[method].items() if k in test_ids}

    return scores


# ── Evaluation loop ────────────────────────────────────────────────────────────


def _cause_subset(test_df: pd.DataFrame, cause_column: str | None) -> pd.DataFrame:
    """Rows for one cause's evaluation: cause-present ∪ fully compliant (None = aggregate)."""
    if cause_column is None:
        return test_df
    is_cause = test_df[cause_column] == 1
    return test_df[is_cause | test_df["compliant"]]


def evaluate_all(
    all_scores: dict[int, dict[str, dict[str, float]]],
    splits: dict[int, Split],
    methods: list[str],
    n_boot: int,
) -> pd.DataFrame:
    """Per (method, cause, metric): mean point estimate over seeds + a seed-pooled bootstrap CI."""
    rows: list[dict[str, Any]] = []
    cause_levels: list[tuple[str, str | None]] = [*CAUSE_COLUMN_MAP.items(), ("aggregate", None)]

    for method in methods:
        sign = 1.0 if ORIENTATION[method] == "asis" else -1.0

        for cause_key, cause_column in cause_levels:
            point_values: dict[str, list[float]] = {name: [] for name in METRIC_FNS}
            replicate_pool: dict[str, list[np.ndarray]] = {name: [] for name in METRIC_FNS}
            n_pos_list: list[int] = []
            n_neg_list: list[int] = []

            for seed, split in splits.items():
                subset = _cause_subset(split.test_df, cause_column)
                y_all = (~subset["compliant"]).to_numpy(dtype=np.int64)
                score_map = all_scores[seed][method]
                s_all = sign * subset["image_path"].map(score_map).to_numpy(dtype=np.float64)

                valid = ~np.isnan(s_all)
                y, s = y_all[valid], s_all[valid]

                pm = point_metrics(y, s, TPR_TARGET)
                n_pos_list.append(pm.n_pos)
                n_neg_list.append(pm.n_neg)
                # A degenerate (single-class) sample bootstraps to the same degenerate
                # result n_boot times over — skip it rather than spamming sklearn's
                # UndefinedMetricWarning; ci_low/ci_high stay NaN for this seed.
                degenerate = pm.n_pos == 0 or pm.n_neg == 0
                for metric_name, fn in METRIC_FNS.items():
                    point_values[metric_name].append(getattr(pm, metric_name))
                    if not degenerate:
                        replicate_pool[metric_name].append(stratified_bootstrap(y, s, fn, n_boot=n_boot, seed=seed))

            n_pos = round(np.mean(n_pos_list))
            n_neg = round(np.mean(n_neg_list))
            prevalence = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else np.nan
            low_n = n_pos < LOW_N_THRESHOLD

            for metric_name in METRIC_FNS:
                pooled = np.concatenate(replicate_pool[metric_name]) if replicate_pool[metric_name] else np.array([])
                rows.append(
                    {
                        "method": method,
                        "cause": cause_key,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        "prevalence": prevalence,
                        "metric": metric_name,
                        "value": float(np.mean(point_values[metric_name])),
                        "ci_low": float(np.percentile(pooled, 2.5)) if pooled.size else np.nan,
                        "ci_high": float(np.percentile(pooled, 97.5)) if pooled.size else np.nan,
                        "low_n": low_n,
                    }
                )

            if cause_key == "aggregate":
                agg_auroc = float(np.mean(point_values["auroc"]))
                if agg_auroc < 0.5:
                    raise AssertionError(
                        f"Aggregate AUROC < 0.5 for method {method!r} ({agg_auroc:.3f}) — "
                        "check its entry in ORIENTATION; scores are never auto-flipped."
                    )

    columns = ["method", "cause", "n_pos", "n_neg", "prevalence", "metric", "value", "ci_low", "ci_high", "low_n"]
    return pd.DataFrame(rows, columns=columns)


def save_results_csv(results: pd.DataFrame, path: Path) -> None:
    """Write the long-format per-cause results table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    print(f"Results saved to {path}")


# ── LaTeX AP table ─────────────────────────────────────────────────────────────

#: Short row labels for the LaTeX table (rows are sorted by prevalence at export time,
#: not in this dict's order).
CAUSE_ABBREVIATIONS = {
    "corrupted": "CI",
    "not_thorax": "NT",
    "incomplete_thorax": "INC",
    "image_quality": "IQ",
    "area_not_valid": "ANV",
    "overlaying_objects": "OVL",
    "non_canonical": "NCP",
}

#: (column header, method key) for the non-"Ours" columns, left to right.
LATEX_BASELINE_COLUMNS = [
    ("NIQE", "niqe"),
    ("BRIS", "brisque"),
    ("QAl", "qalign"),
    ("TOPQ", "topiq"),
    ("KNN", "knn"),
    ("MPP", "maha_pp"),
    ("GEN", "gen"),
    ("SC", "selfclean"),
    ("DC", "deepclean_central"),
]
LATEX_OURS_METHOD = "demicaf_aggfull"


def _fmt_pct(value: float | None) -> str:
    r"""Format as a bare percentage number (no literal ``\%``) — the caption states the unit."""
    return "--" if value is None or np.isnan(value) else f"{value * 100:04.1f}"


def _format_ranked_cell(value: float | None, best: float | None, second_best: float | None) -> str:
    """Bold the row's best value, underline its second-best (a distinct, strictly lower value)."""
    cell = _fmt_pct(value)
    if value is None:
        return cell
    if best is not None and value == best:
        return r"\textbf{" + cell + "}"
    if second_best is not None and value == second_best:
        return r"\underline{" + cell + "}"
    return cell


def export_ap_latex_table(results: pd.DataFrame, tex_path: Path) -> None:
    """Export the per-cause AP (%) LaTeX table, best method bolded and second-best underlined per row.

    Missing columns (P1 methods not yet run) render as "--" rather than blocking the
    export — fill them in once those scorers are registered, without touching this
    function.
    """
    ap = results[results["metric"] == "ap"].set_index(["cause", "method"])["value"]
    prevalence = (
        results[(results["metric"] == "ap") & (results["method"] == LATEX_OURS_METHOD)]
        .set_index("cause")["prevalence"]
    )
    all_methods = [method for _, method in LATEX_BASELINE_COLUMNS] + [LATEX_OURS_METHOD]

    def row_values(cause_key: str) -> dict[str, float | None]:
        return {m: (ap.loc[(cause_key, m)] if (cause_key, m) in ap.index else None) for m in all_methods}

    def row_line(label: str, cause_key: str) -> str:
        values = row_values(cause_key)
        present = sorted({v for v in values.values() if v is not None}, reverse=True)
        best = present[0] if present else None
        second_best = present[1] if len(present) > 1 else None
        prev_cell = _fmt_pct(prevalence.get(cause_key))
        cells = [_format_ranked_cell(values[m], best, second_best) for _, m in LATEX_BASELINE_COLUMNS]
        ours_cell = _format_ranked_cell(values[LATEX_OURS_METHOD], best, second_best)
        return f"    {label} & {prev_cell} & " + " & ".join(cells) + f" & {ours_cell} \\\\"

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{\ac{AP} (\%) of image compliance assessment methods per-cause"
        r" (Fig \ref{fig:causes}), using CheXpert as prior dataset, with \textbf{best method} in bold"
        r" and \underline{second-best} underlined.}",
        r"    \label{tab:percause_t}",
        r"    \setlength{\tabcolsep}{3pt}\small",
        r"    \begin{tabular}{c c|c c c c|c c c|c c|c}",
        r"    \hline",
        r"     \multicolumn{2}{c|}{Cause} & \multicolumn{4}{c|}{IQA} & \multicolumn{3}{c|}{OOD}"
        r" & \multicolumn{2}{c|}{Audit} & \multirow{2}{*}{Ours} \\",
        r"    \cline{1-5}\cline{6-9}\cline{10-11}",
        r"     & Prev. & NIQE & BRIS & QAl & TOPQ & KNN & MPP & GEN & SC & DC & \\",
        r"    \hline",
    ]
    cause_order = sorted(CAUSE_ABBREVIATIONS, key=lambda cause_key: prevalence.get(cause_key, np.inf))
    lines += [row_line(CAUSE_ABBREVIATIONS[cause_key], cause_key) for cause_key in cause_order]
    lines += [r"    \hline", row_line("All", "aggregate"), r"    \hline", r"    \end{tabular}", r"\end{table}", ""]

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX table saved to {tex_path}")


# ── Plot ──────────────────────────────────────────────────────────────────────

CAUSE_LABELS = {
    "corrupted": "Corrupted",
    "not_thorax": "Not a Thorax",
    "incomplete_thorax": "Incomplete Thorax",
    "image_quality": "Image Quality",
    "area_not_valid": "Area Not Valid",
    "overlaying_objects": "Overlaying Objects",
    "non_canonical": "Non-Canonical",
    "aggregate": "Aggregate",
}
CAUSE_PANEL_ORDER = list(CAUSE_LABELS)

METHOD_LABELS = {
    "demicaf_aggfull": "DeMICAF",
    "deepclean_central": "DeepClean",
    "niqe": "NIQE",
    "brisque": "BRISQUE",
    "qalign": "Q-Align",
    "topiq": "TOPIQ",
    "knn": "KNN",
    "maha_pp": "Maha++",
    "gen": "GEN",
    "selfclean": "SelfClean",
}
#: Reuses the repo's own design tokens (``data/colors.json``); validated colorblind-safe
#: (deuteranopia/tritanopia ΔE well above the 8-point floor) via the dataviz skill's
#: palette checker.
METHOD_COLORS = {
    "demicaf_aggfull": "#64C7EA",  # Steel Blue
    "deepclean_central": "#FF810C",  # Orange
    "niqe": "#A6A0F5",  # Lilac
    "brisque": "#FFD730",  # Yellow
    "qalign": "#DA291C",  # Red
    "topiq": "#888888",  # Grey
    "knn": "#0033A0",  # Blue
    "maha_pp": "#77FAD9",  # Turquoise
    "gen": "#1120C7",  # Sky Blue
    "selfclean": "#008BBE",  # Deep Blue
}


def plot_per_cause_auroc(results: pd.DataFrame, plot_path: Path, methods: list[str]) -> None:
    """Hardcoded 2x4 grid: one panel per cause, AUROC bars with 95% CI whiskers, methods on x."""
    apply_thesis_style()
    auroc = results[results["metric"] == "auroc"]

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharey=True)
    x = np.arange(len(methods))

    for ax, cause_key in zip(axes.flat, CAUSE_PANEL_ORDER, strict=True):
        cause_rows = auroc[auroc["cause"] == cause_key].set_index("method")
        values = [cause_rows.loc[m, "value"] * 100 if m in cause_rows.index else np.nan for m in methods]
        ci_low = [cause_rows.loc[m, "ci_low"] * 100 if m in cause_rows.index else np.nan for m in methods]
        ci_high = [cause_rows.loc[m, "ci_high"] * 100 if m in cause_rows.index else np.nan for m in methods]
        errors = np.abs(np.vstack([np.subtract(values, ci_low), np.subtract(ci_high, values)]))

        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(x, values, yerr=errors, color=colors, capsize=2, width=0.7, error_kw={"linewidth": 1})
        ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(CAUSE_LABELS[cause_key], fontsize=plt.rcParams["axes.labelsize"])
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=90)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.15)

    axes[0, 0].set_ylabel("AUROC (%)")
    axes[1, 0].set_ylabel("AUROC (%)")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Per-cause compliance benchmark: DeMICAF vs. baselines.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repo_root / "data" / "annotations" / "annotated_causes.csv",
    )
    parser.add_argument("--data-root", type=str, default="/eos/user/h/hpestana/CXR")
    parser.add_argument(
        "--niqe-brisque-cache",
        type=Path,
        default=repo_root / "Results" / "Baseline" / "perceptual_iqa.csv",
        help="Pre-computed niqe/brisque CSV from scripts/baseline.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=repo_root / "Results" / "PerCause")
    parser.add_argument(
        "--classifier-features-dir",
        type=Path,
        default=repo_root / "Results" / "PerCause",
        help="Directory with classifier_features_{client}.npz from extract_classifier_features.py.",
    )
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=list(ORIENTATION),
        help="Subset of method keys to run (default: all P0 methods).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume the qalign/topiq incremental cache.")
    return parser.parse_args()


def main() -> None:
    """Build the per-seed splits, score every method, evaluate per-cause, and save the outputs."""
    args = parse_args()
    data_root = resolve_path(args.data_root)
    methods = args.methods or list(ORIENTATION)

    print("Loading annotations ...")
    annotations = load_annotations(args.annotations)

    print(f"Building {args.n_seeds} seeded reference/test splits ...")
    seeds = list(range(SEED_BASE, SEED_BASE + args.n_seeds))
    splits = {seed: build_split(annotations, seed) for seed in seeds}

    demicaf_feats = load_demicaf_features(PRIOR_DATASET) if "demicaf_aggfull" in methods else None
    deepclean_feats = load_demicaf_features("All") if "deepclean_central" in methods else None
    classifier_feats = (
        load_classifier_features(args.classifier_features_dir) if {"knn", "maha_pp"} & set(methods) else None
    )

    print("Loading/computing NR-IQA scores ...")
    iqa_scores: dict[str, dict[str, float]] = {}
    for method in ("niqe", "brisque"):
        if method in methods:
            iqa_scores[method] = load_cached_iqa(method, args.niqe_brisque_cache)
    for method in ("qalign", "topiq"):
        if method in methods:
            cache_path = args.out_dir / f"{method}_cache.csv"
            iqa_scores[method] = compute_iqa(method, annotations, data_root, cache_path, resume=args.resume)
    if "selfclean" in methods:
        cache_path = args.out_dir / "selfclean_cache.csv"
        iqa_scores["selfclean"] = compute_selfclean(annotations, data_root, cache_path, resume=args.resume)
    if "gen" in methods:
        cache_path = args.out_dir / "gen_cache.csv"
        iqa_scores["gen"] = compute_gen(args.classifier_features_dir, cache_path, resume=args.resume)

    print("Fitting references and scoring the test set per seed ...")
    all_scores: dict[int, dict[str, dict[str, float]]] = {
        seed: build_scorers(split, methods, demicaf_feats, deepclean_feats, classifier_feats, iqa_scores)
        for seed, split in splits.items()
    }

    print("Computing per-cause metrics with stratified bootstrap CIs ...")
    results = evaluate_all(all_scores, splits, methods, n_boot=args.n_boot)

    save_results_csv(results, args.out_dir / "per_cause_results.csv")
    plot_per_cause_auroc(results, args.out_dir / "per_cause_auroc.png", methods)

    if LATEX_OURS_METHOD in methods:
        export_ap_latex_table(results, args.out_dir / "per_cause_ap.tex")
    else:
        print(f"Skipping LaTeX table: {LATEX_OURS_METHOD!r} not in --methods.")


if __name__ == "__main__":
    main()
