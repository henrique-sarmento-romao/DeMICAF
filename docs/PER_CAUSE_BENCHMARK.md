# Per-Cause Compliance Benchmark — Protocol

Implemented by `scripts/benchmarking.py`. Compares DeMICAF against NR-IQA,
OOD-detection, and centralised-compliance baselines on the manual annotations,
broken down per non-compliance cause (Table 1 of the paper).

## 1. Goal

For each non-compliance **cause** `c`, and each **method** `m`, score every test
image and evaluate how well `m` separates **cause-`c` images** from
**fully-compliant images**. Report AUROC, AP, FPR@95TPR + 95% CIs.

Also report an **aggregate** row per method: all non-compliant vs compliant
(the stable setting where FPR@95TPR is trustworthy).

## 2. Data & label semantics

- Annotations: ~1,000 labelled images per dataset across four datasets
  (CheXpert, MIMIC-CXR, ChestX-ray14, PadChest); evaluation pooled over the
  three **client** datasets (CheXpert is the prior dataset, excluded, matching
  the Reference Scheme Comparison setup).
- Seven causes: `corrupted`, `not_thorax`, `incomplete_thorax`,
  `image_quality`, `area_not_valid`, `overlaying_objects`, `non_canonical`.
  Multi-label: an image may carry several causes; compliant = no cause.
- **Positive set for cause `c`:** every image where cause `c` is present
  (regardless of co-occurring causes).
- **Negative set (shared across all causes):** fully-compliant images.

## 3. Evaluation protocol

Fixed test set, identical across all methods.

1. **Split compliant images** into `reference-compliant` and `test-compliant`
   (50/50, stratified by client dataset). Non-compliant images all go to test
   (they are positives). This prevents leakage of test negatives into any
   reference/feature bank.
2. **Reference fitting** (methods that need it) uses the *reference split only*.
   The distance-based baselines (KNN, Maha++) are fit on the same label-free
   reference construction as DeMICAF (all reference-split images of the
   clients), not a hand-picked clean set. NR-IQA methods need no reference.
3. **Scoring:** every method scores the full test set once → per-image scores.
4. **Per-cause metrics:** for cause `c`, restrict to
   `test-compliant ∪ {test images with c}`; compute metrics with
   `y=1` for cause-present, `y=0` for compliant.
5. **Seeds & CIs:** repeated over ≥3 seeds; within each, compute metrics and a
   **stratified bootstrap** (resample positives and negatives separately,
   ≥1000 resamples) to get 95% CIs. Report mean over seeds and the pooled CI.

## 4. Metrics

Higher score ⇒ more non-compliant (see §6 for per-method orientation).

- **AUROC** — primary; comparable across causes (prevalence-independent).
- **AP** (`average_precision_score`) — curation-relevant; compare *within* a
  cause only, since its floor is the prevalence. `prevalence`/`n_pos` are
  reported per cause so AP is interpretable.
- **FPR@95TPR** — secondary; unstable when `n_pos` is small. CIs are reported
  with a `low_n = n_pos < 50` flag.

Reference implementation: `DeMICAF.evaluation.bootstrap` (`point_metrics`,
`stratified_bootstrap`, `METRIC_FNS`).

Pass raw scores to `roc_auc_score` (no thresholding). Signs are set explicitly
per §6 and never auto-flipped; an assertion flags any method whose aggregate
AUROC < 0.5 (a wrong sign to fix at source, not to paper over).

## 5. Methods evaluated

Each scorer is a function `ids -> np.ndarray[float]`, higher = more
non-compliant, registered in a method-agnostic eval loop.

| Method key | Family | Score source | Reference / fit |
|---|---|---|---|
| `demicaf_aggfull` | Ours | MD to aggregated ref (SupCon encoder) | Aggregated over reference split; Ledoit–Wolf + λ=1e-5 |
| `deepclean_central` | Same-objective | MD to centralised/pooled ref (SupCon) | pooled reference split |
| `niqe`, `brisque` | Classical NR-IQA | perceptual quality | none (`pyiqa`) |
| `qalign`, `topiq` | SOTA NR-IQA | LMM / semantic-guided quality | none (`pyiqa`) |
| `knn` | OOD on classifier | k-th NN distance in feature bank | label-free ref bank (L2-norm feats) |
| `maha_pp` | OOD on classifier | L2-normalised Mahalanobis (single Gaussian) | label-free reference |
| `gen` | OOD on classifier | generalized entropy of softmax | trained pathology classifier |
| `selfclean` | Same-objective | self-supervised quality-issue score | pretrained-ViT embedding |

The pathology classifier for `knn`/`maha_pp`/`gen` is the FedAvg baseline
trained by `scripts/federated/train.py`; its features are extracted by
`scripts/extract_classifier_features.py`.

## 6. Score orientation

Set explicitly; assert, do not auto-flip.

| Method | Native meaning | Use |
|---|---|---|
| `demicaf_*`, `deepclean_central`, `knn`, `maha_pp` | distance, higher = more OOD | as-is |
| `niqe`, `brisque` | higher = worse quality | as-is |
| `qalign`, `topiq` | higher = better quality | negate |
| `gen` | our multi-label adaptation correlates with *compliant* | negate |
| `selfclean` | raw off-topic-sample score correlates with *compliant* | negate |

## 7. Outputs

Long-format CSV `per_cause_results.csv` under `results/per_cause/`:

```
method, cause, n_pos, n_neg, prevalence, metric, value, ci_low, ci_high, low_n
```

- `cause` includes an `aggregate` level (all non-compliant vs compliant).
- One row per `(method, cause, metric)`.
- `per_cause_auroc.png` — per-cause grouped point chart of AUROC with CIs.
- `per_cause_ap.tex` — the paper's Table 1 (AP per cause, best bold / second-best underlined).
