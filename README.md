# DeCaF — A Compliance-Annotation Dataset for Chest Radiography

This repository accompanies the paper *MICAFFe: Medical Imaging Compliance Assessment
Framework for Federated Learning*. It releases:

1. **The dataset** — manual **compliance annotations** for 4,000 chest X-rays
   (1,000 each from four public datasets), labelled *compliant* / *non-compliant* with
   a **7-cause non-compliance taxonomy**.
2. **Preprocessing code** — reproducible image subset sampling, the annotation-database
   builder, the dataset loader, and dataset statistics/figures.
3. **The Compliance Assessment Framework (CAF)** — the contrastive encoder plus the
   Mahalanobis / cosine scorer and federated score normalization used to produce
   per-image compliance scores.

> **Images are not redistributed.** The four source datasets are registration-gated;
> DeCaF ships the annotations keyed by image path plus the code. See
> [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) for how to obtain the images.

## What is "compliance"?

A chest radiograph is **compliant** when it is a technically valid, standard-protocol
frontal thorax image suitable for downstream analysis. **Non-compliant** images are
flagged with one or more of seven causes (corrupted, not a thorax, incomplete thorax,
image-quality problems, area not valid, overlaying objects, non-canonical position).
Full definitions: [`docs/TAXONOMY.md`](docs/TAXONOMY.md).

## The dataset at a glance

| Source dataset | Annotated | Compliant | Non-compliant |
| -------------- | --------: | --------: | ------------: |
| CheXpert       |     1,000 |       888 |           112 |
| ChestX-ray8    |     1,000 |       920 |            80 |
| MIMIC-CXR      |     1,000 |       778 |           222 |
| PadChest       |     1,000 |       863 |           137 |
| **Total**      | **4,000** | **3,449** |       **551** |

The annotations live in [`data/annotations/annotated_causes.csv`](data/annotations/annotated_causes.csv):

| column | meaning |
| ------ | ------- |
| `image_path` | dataset-prefixed relative path, e.g. `CheXpert/CheXpert-v1.0/train/patient.../view1_frontal.jpg` |
| `compliance` | `compliant` or `non-compliant` |
| `corrupted_images`, `not_a_thorax`, `incomplete_thorax`, `image_quality_problems`, `area_not_valid`, `overlaying_objects`, `non_canonical_positions` | `1` if that cause applies, empty otherwise |

## Repository layout

```
data/          annotations (CSV + taxonomy.json), figure palette, cause statistics
docs/          dataset card, taxonomy definitions, reproduction guide
src/           importable library: CAF framework (caf/), loader (data/), evaluation, utils
scripts/       preprocessing/ · scoring/ · stats/  (run as modules)
examples/      quickstart.py — load + score + evaluate end-to-end
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Regenerate the cause statistics + bar chart from the bundled annotations
PYTHONPATH=src python -m scripts.stats.causes

# 2) Load images, score them with CAF, and evaluate against the annotations
#    (requires the source images under $CXR_ROOT — see docs/DATASET_CARD.md)
CXR_ROOT=/path/to/CXR PYTHONPATH=src python -m examples.quickstart --checkpoint encoder.pt
```

The library resolves the repo root automatically; override it with `DECAF_ROOT` when
running from a relocated copy. Source-image roots are configured via the `CXR_ROOT`
environment variable (or `--data-root` / `--cxr-root` flags).

Reproduction details for every script are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Citing

If you use these annotations or code, please cite the paper (see [`CITATION.cff`](CITATION.cff)).

## Licensing

- **Code** — MIT (see [`LICENSE`](LICENSE)).
- **Annotations** (`data/annotations/`) — CC BY 4.0.
- **Source images** — governed by each source dataset's own license / data-use
  agreement; not included here.
