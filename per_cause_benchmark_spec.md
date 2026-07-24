# Per-Cause Compliance Benchmark — Implementation Spec

**Context:** Final experiment for DeMICAF (DeCaF 2026). We already have per-image
compliance scores for our method; this task adds a **per-cause** comparison of
DeMICAF against baselines on the manual annotations.

**Deliverable:** a tidy results table (long-format CSV) of
`AUROC`, `AP`, `FPR@95TPR` (with bootstrap CIs) for every `(method, cause)` pair,
plus a compact per-cause plot. British English throughout.

---

## 1. Goal

For each non-compliance **cause** `c`, and each **method** `m`, score every test
image and evaluate how well `m` separates **cause-`c` images** from
**fully-compliant images**. Report AUROC, AP, FPR@95TPR + 95% CIs.

Also report an **aggregate** row per method: all non-compliant vs compliant
(this is the stable setting where FPR@95TPR is trustworthy).

---

## 2. Data & label semantics

- Annotations: ~1,000 labelled images per dataset across four datasets
  (CheXpert, MIMIC-CXR, ChestX-ray14, PadChest); evaluation pooled over the
  three **client** datasets (prior dataset excluded, matching the Reference
  Comparison setup). One prior dataset trains/anchors as in the paper.
- Seven causes: `corrupted`, `not_thorax`, `incomplete_thorax`,
  `image_quality`, `area_not_valid`, `overlaying_objects`, `non_canonical`.
  Multi-label: an image may carry several causes; compliant = no cause.
- **Positive set for cause `c` (primary):** every image where cause `c` is
  present (regardless of co-occurring causes).
- **Negative set (shared across all causes):** fully-compliant images.
- Read the annotation CSV with `pandas`; iterate with
  `for _, r in df.iterrows(): row = r.to_dict()` (NOT `itertuples` — it mangles
  hyphenated dataset/column names).

**DECISION-1 (positives):** primary = *cause-present*. Optional robustness pass:
*sole-cause* positives (image whose only cause is `c`) — run only for causes
with sufficient n; report in appendix if used.

---

## 3. Evaluation protocol

Fixed test set, identical across all methods.

1. **Split compliant images** into `reference-compliant` and `test-compliant`
   (e.g. 50/50, stratified by client dataset). Non-compliant images all go to
   test (they are positives). This prevents leakage of test negatives into any
   reference/feature bank.
2. **Reference fitting** (methods that need it) uses the *reference split only*.
   To isolate the encoder/scorer contribution, fit the distance-based baselines
   (KNN, Maha++) on the **same label-free reference construction as DeMICAF**
   (all reference-split images of the clients), not a hand-picked clean set.
   NR-IQA methods need no reference.
3. **Scoring:** every method scores the full test set once → per-image scores.
4. **Per-cause metrics:** for cause `c`, restrict to
   `test-compliant ∪ {test images with c}`; compute metrics with
   `y=1` for cause-present, `y=0` for compliant.
5. **Seeds & CIs:** repeat the split over ≥3 seeds; within each, compute
   metrics and a **stratified bootstrap** (resample positives and negatives
   separately, ≥1000 resamples) to get 95% CIs. Report mean over seeds and the
   pooled CI. Bootstrap over realizations, as elsewhere in the project.

**DECISION-2 (reference for OOD baselines):** default = label-free (as above).
Alternative = compliant-only reference; if you want it, run as a second variant
rather than replacing the default.

---

## 4. Metrics

Higher score ⇒ more non-compliant (see §6 for per-method orientation).

- **AUROC** — primary; comparable across causes (prevalence-independent).
- **AP** (`average_precision_score`) — curation-relevant; compare
  *within* a cause only, since its floor is the prevalence. Always emit
  `prevalence` and `n_pos` per cause so AP is interpretable.
- **FPR@95TPR** — secondary; unstable when `n_pos` is small. Emit CIs and set a
  flag `low_n = n_pos < 50`. Optionally also emit `TPR@FPR=5%` as a steadier
  operating point.

Reference helper (`metrics.py`):

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def fpr_at_tpr(y_true, scores, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= tpr_target
    return float(fpr[mask].min()) if mask.any() else np.nan

def point_metrics(y_true, scores):
    return {
        "auroc": roc_auc_score(y_true, scores),
        "ap":    average_precision_score(y_true, scores),
        "fpr95": fpr_at_tpr(y_true, scores, 0.95),
    }

def stratified_bootstrap(y_true, scores, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    pos = np.flatnonzero(y_true == 1); neg = np.flatnonzero(y_true == 0)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        out.append(fn(y_true[idx], scores[idx]))
    out = np.asarray(out); out = out[~np.isnan(out)]
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
```

Pass raw scores to `roc_auc_score` (no thresholding). Do **not** auto-flip signs
to force AUROC > 0.5 — set orientation explicitly per §6 and add an assertion
that flags any method whose aggregate AUROC < 0.5 (that signals a wrong sign to
fix at source, not to paper over).

---

## 5. Methods to evaluate

Each scorer is a function `ids -> np.ndarray[float]`, higher = more
non-compliant. Register them so the eval loop is method-agnostic:

```python
SCORERS = {name: fn, ...}   # fn(test_ids) -> scores aligned to test_ids
```

| Method key | Family | Score source | Reference / fit | Package / code | Priority |
|---|---|---|---|---|---|
| `demicaf_aggfull` | Ours | MD to aggregated ref (SupCon encoder) | Agg-Full over reference split; Ledoit–Wolf + λ=1e-5 | existing repo | P0 |
| `demicaf_prior` | Ours | MD to prior-dataset ref | Prior scheme | existing repo | P1 |
| `deepclean_central` | Same-objective | MD to centralised/self ref (SupCon) | pooled reference split | existing repo | P0 |
| `selfclean` | Same-objective | self-supervised quality-issue score | per method | SelfClean repo | P1 |
| `niqe` | Classical NR-IQA | perceptual quality | none | `pyiqa` `niqe` | P0 |
| `brisque` | Classical NR-IQA | perceptual quality | none | `pyiqa` `brisque` | P0 |
| `qalign` | SOTA NR-IQA | LMM quality score | none | `pyiqa` `qalign` | P0 |
| `topiq` | SOTA NR-IQA | semantic-guided quality | none | `pyiqa` `topiq_nr` | P0 |
| `knn` | OOD on classifier | k-th NN distance in feature bank | label-free ref bank (L2-norm feats) | Sun 2022 / OpenOOD | P1 |
| `maha_pp` | OOD on classifier | L2-normalised Mahalanobis | single Gaussian on ref feats | Mueller & Hein 2025 | P1 |
| `gen` | OOD on classifier | generalized entropy of softmax | trained pathology classifier | Liu 2023 / OpenOOD | P2 |

Notes:
- **Supervised classifier** for `knn`/`maha_pp`/`gen`: a pathology multi-label
  classifier (e.g. DenseNet-121/CheXNet-style) trained on pooled CXR labels.
  Features = penultimate layer (for `knn`, `maha_pp`); logits/softmax over
  pathology classes (for `gen`). If this classifier is **not already trained**,
  treat `gen` as deferrable and `knn`/`maha_pp` as blocked until it exists.
- `maha_pp` uses a **single** (non-class-conditional) Gaussian on the label-free
  reference, i.e. L2-normalised MD — directly comparable to DeMICAF's MD.
- `pyiqa` methods are grayscale CXRs on a natural-image model: expect failure on
  non-perceptual causes — that is the intended finding, report as external
  baselines.

**DECISION-3 (Maha++ placement):** default = OOD-on-classifier baseline (table
above). Optional variant `maha_pp_supcon` = Maha++ scorer on the SupCon encoder,
as a scorer ablation of DeMICAF. Include only if time permits.

---

## 6. Score orientation

Set explicitly; assert, do not auto-flip.

| Method | Native meaning | Use |
|---|---|---|
| `demicaf_*`, `deepclean_central`, `knn`, `maha_pp` | distance, higher = more OOD | as-is |
| `niqe`, `brisque` | higher = worse quality | as-is |
| `qalign`, `topiq` | higher = better quality | negate |
| `gen` | generalized entropy (check repo convention) | **verify**, then set |
| `selfclean` | quality-issue score/rank (check convention) | **verify**, then set |

---

## 7. Outputs

Long-format CSV `per_cause_results.csv`:

```
method, cause, n_pos, n_neg, prevalence, metric, value, ci_low, ci_high, low_n
```

- `cause` includes an `aggregate` level (all non-compliant vs compliant).
- One row per `(method, cause, metric)`.
- Plots (PNG, 150 DPI): a per-cause grouped bar/point chart of AUROC with CIs,
  methods on x within each cause panel; optionally a second panel for AP.
  Prefer a hardcoded, reliable layout over a fragile dynamic one.

---

## 8. Open decisions (confirm before running)

1. Positive definition: cause-present (default) vs sole-cause. [§2]
2. OOD-baseline reference: label-free (default) vs compliant-only. [§3]
3. Maha++ placement: classifier baseline (default) vs SupCon scorer variant. [§5]
4. Is the pathology classifier for `knn`/`gen`/`maha_pp` already available? If
   not, ship P0 first (ours + DeepClean + 4× NR-IQA) and add the OOD family after.

---

## 9. Priority / scope (July 24 deadline)

- **P0 (must-run):** `demicaf_aggfull`, `deepclean_central`, `niqe`, `brisque`,
  `qalign`, `topiq`. Full per-cause + aggregate table from these alone.
- **P1:** `demicaf_prior`, `selfclean`, `knn`, `maha_pp`.
- **P2:** `gen` (needs classifier + logit path).

Build the registry + eval loop against P0 first, verify the table/plot, then
drop the remaining scorers in without touching the harness.
