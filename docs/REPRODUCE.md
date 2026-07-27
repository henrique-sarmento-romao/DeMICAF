# Reproduction Guide

All scripts run as modules from the repository root with `src/` on the path:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
export CXR_ROOT=/path/to/CXR   # root holding one subfolder per source dataset
```

`CXR_ROOT` is only needed for steps that read the actual radiographs. Steps that use
only the bundled annotations work out of the box.

## 0. Environment sanity check

```bash
python -c "import DeMICAF.caf.scorer, DeMICAF.data.datasets, DeMICAF.evaluation.auc; print('ok')"
```

## 1. Statistics and figures (no images needed)

```bash
# Regenerate data/stats/causes_stats.md and data/stats/Causes_Chart.png
python -m scripts.stats.causes

# Optional: paper overlay figures + progress video (need CAF score CSVs)
python -m scripts.stats.causes --scores path/to/scores.csv --encoder-scores path/to/encoder_scores.csv
```

Dataset demographics (age/sex/position distributions) from the raw metadata CSVs:

```bash
python -m scripts.stats.eda --out out/eda
```

## 2. Annotation database (provenance)

The released `annotations/annotated_causes.csv` is the canonical artifact. To
rebuild it from raw per-annotator exports:

```bash
# Build the SQLite DB from per-dataset CSVs (columns: image_path, compliance)
python -m scripts.preprocessing.make_database --raw-dir path/to/raw_csvs --db-out annotations.db

# Flatten the DB back into the wide annotated_causes.csv (+ optional per-dataset splits)
python -m scripts.preprocessing.export_annotations --db annotations.db \
    --out annotations/annotated_causes.csv --per-dataset
```

## 3. Image subsets (needs `$CXR_ROOT`)

```bash
# Draw the seeded 10k and 100k subsets per dataset (always includes the annotated images)
python -m scripts.preprocessing.draw_subsets --data-root "$CXR_ROOT" --dataset CheXpert
```

Outputs `subset10k.csv` / `subset100k.csv` under each `$CXR_ROOT/<Dataset>/`.

## 4. Scoring with the Compliance Assessment Framework (needs `$CXR_ROOT`)

```bash
# Minimal end-to-end: load images, extract features, Mahalanobis score, AUC vs. annotations
python -m examples.quickstart --checkpoint path/to/encoder.pt

# Generalisation Analysis (Fig. 4): cross-dataset encoder generalization + size scaling,
# under the Prior/Adaptive reference schemes. Only run the 100k scale for the paper's
# reported setting:
python -m scripts.scoring.generalization --subset 100k

# Reference Scheme Comparison (Fig. 5): federated static-encoder scoring under the
# Prior / Aggregated / Compliant reference schemes
python -m scripts.scoring.score

# Loss/scorer ablation backing Sec. 5's choice of SupCon + unclustered Mahalanobis
python -m scripts.scoring.configuration

# Normalize / aggregate raw Mahalanobis scores into compliance scores
python -m scripts.scoring.compliance_scores
```

## 5. Per-cause benchmark (Table 1, needs `$CXR_ROOT`)

Full protocol: [`docs/PER_CAUSE_BENCHMARK.md`](PER_CAUSE_BENCHMARK.md).

```bash
# NR-IQA baselines (NIQE, BRISQUE) vs. annotations, feeding scripts.benchmarking's cache
python -m scripts.baseline

# FedAvg pathology classifier (feeds the knn/maha_pp/gen baselines)
python -m scripts.federated.train

# Penultimate features + per-label probabilities from the trained classifier
python -m scripts.extract_classifier_features --checkpoint results/per_cause/classifier/fedavg_resnet18/seed_0/final_model.pt

# DeMICAF vs. every baseline, per non-compliance cause
python -m scripts.benchmarking
```

The scoring scripts expect pre-computed features / encoders and write their outputs to
`results/<experiment>/`, overridable via CLI flags. Evaluate any score column against
the annotations with `DeMICAF.evaluation.auc.compute_auc` (positive class = *compliant*
by default); use `filter_cause` to compute a per-cause AUC.

## Notes

- Reproducibility relies on the global RNG seeded via `DeMICAF.utils.seeding`
  and the fixed `RNG_SEED = 42` in the sampling scripts.
- Override the repository root with `DEMICAF_ROOT` if running from a relocated checkout.
