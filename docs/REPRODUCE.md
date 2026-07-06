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
python -c "import data_compliance.caf.scorer, data_compliance.data.datasets, data_compliance.evaluation.auc; print('ok')"
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

The released `data/annotations/annotated_causes.csv` is the canonical artifact. To
rebuild it from raw per-annotator exports:

```bash
# Build the SQLite DB from per-dataset CSVs (columns: image_path, compliance)
python -m scripts.preprocessing.make_database --raw-dir path/to/raw_csvs --db-out annotations.db

# Flatten the DB back into the wide annotated_causes.csv (+ optional per-dataset splits)
python -m scripts.preprocessing.export_annotations --db annotations.db \
    --out data/annotations/annotated_causes.csv --per-dataset
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

# Contrastive-loss / ranking-function comparison
python -m scripts.scoring.configuration

# Cross-dataset encoder generalization + size scaling
python -m scripts.scoring.generalization

# Federated static-encoder scoring under different reference schemes
python -m scripts.scoring.score

# Normalize / aggregate raw Mahalanobis scores into compliance scores
python -m scripts.scoring.compliance_scores --scores-csv raw_scores.csv --out-csv compliance_normalized.csv
```

The scoring scripts expect pre-computed features / encoders and write their outputs to
paths overridable via CLI flags. Evaluate any score column against the annotations with
`data_compliance.evaluation.auc.compute_auc` (positive class = *compliant* by default);
use `filter_cause` to compute a per-cause AUC.

## Notes

- Reproducibility relies on the global RNG seeded via `data_compliance.utils.seeding`
  and the fixed `RNG_SEED = 42` in the sampling scripts.
- Override the repository root with `DECAF_ROOT` if running from a relocated checkout.
