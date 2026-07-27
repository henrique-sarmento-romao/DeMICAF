# Dataset Card — DeMICAF Compliance Annotations

## Summary

DeMICAF provides manual **compliance annotations** for 4,000 frontal chest radiographs,
1,000 drawn from each of four large public datasets. Each image is labelled *compliant*
or *non-compliant*, and non-compliant images are tagged with one or more of seven
non-compliance causes (see [`TAXONOMY.md`](TAXONOMY.md)). The purpose is to enable
**per-image data-quality / compliance assessment**, including in federated settings
where images cannot be centrally inspected.

Only the **annotations** and code are distributed here. The radiographs themselves come
from registration-gated datasets and must be obtained from their original providers.

## Source datasets and how to obtain them

Each source dataset has its own registration and/or data-use agreement. Download the
images from the official source and lay them out under a common root (`$CXR_ROOT`),
one subfolder per dataset.

| Dataset | Reference | Where to obtain |
| ------- | --------- | --------------- |
| **CheXpert** | Irvin et al., 2019 | https://stanfordmlgroup.github.io/competitions/chexpert/ |
| **ChestX-ray8** (ChestX-ray14) | Wang et al., 2017 | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| **MIMIC-CXR** | Johnson et al., 2019 | https://physionet.org/content/mimic-cxr/ (credentialed access) |
| **PadChest** | Bustos et al., 2019 | https://bimcv.cipf.es/bimcv-projects/padchest/ |

### Image-path convention

`image_path` in the annotations is dataset-prefixed and resolves as follows (this is
also encoded in `scripts/preprocessing/draw_subsets.py`):

| Dataset | `image_path` pattern |
| ------- | -------------------- |
| CheXpert | `CheXpert/CheXpert-v1.0/{Path}` |
| ChestX-ray8 | `ChestX-ray8/images/{Image Index}` |
| MIMIC-CXR | `MIMIC-CXR/files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.png` |
| PadChest | `PadChest/{ImageDir}/{ImageID}` |

To join annotations to files on disk, strip the leading `<Dataset>/` component and join
with `$CXR_ROOT/<Dataset>/...` (the loader and quickstart do this for you).

## Composition

- **Size:** 4,000 images (1,000 per dataset).
- **Class balance:** 3,449 compliant / 551 non-compliant. Per-dataset counts and
  per-cause frequencies are in [`../data/stats/causes_stats.md`](../data/stats/causes_stats.md).
- **Modality:** frontal chest radiographs (X-ray).
- **Labels:** binary compliance + 7 binary non-compliance causes (multi-label; an image
  can carry several causes).

## Sampling

The 1,000 annotated images per dataset were selected for annotation; larger 10k/100k
image pools used for encoder training and reference computation are drawn (seeded,
`RNG_SEED = 42`) by `scripts/preprocessing/draw_subsets.py`, always including the
annotated images. A seeded 250-image compliant validation split per dataset is used for
the picked-reference experiments.

## Annotation protocol

- Labelling followed the taxonomy in [`TAXONOMY.md`](TAXONOMY.md).
- A primary annotator labelled each image; **ambiguous cases were adjudicated jointly
  with a senior supervisor**, and the most diagnostically challenging cases were further
  reviewed by a **qualified radiologist** to ensure label reliability.
- Annotators are anonymized in this release (`annotator_1`, …); the annotation database
  schema is in `scripts/preprocessing/data_definition.sql` and the builder in
  `scripts/preprocessing/make_database.py`.

## Uses

- **Intended:** benchmarking per-image compliance / quality assessment; evaluating
  out-of-distribution and image-quality scoring; federated data-quality studies.
- **Out of scope:** the annotations describe *technical compliance*, not diagnosis;
  they are not disease/pathology labels and should not be used as such.

## Limitations and biases

- Single-annotator primary labelling (with adjudication) — some subjectivity remains,
  particularly at the *Area Not Valid* / *Image Quality Problems* boundary.
- 1,000 images per dataset is a small, class-imbalanced sample of each source dataset.
- Non-compliance-cause frequencies vary substantially across datasets, reflecting
  differing acquisition protocols and populations.

## Licensing

- **Annotations** (`annotations/`): CC BY 4.0.
- **Code**: MIT (see [`../LICENSE`](../LICENSE)).
- **Source images**: governed by each provider's license / data-use agreement; not
  redistributed here.

## Citation

See [`../CITATION.cff`](../CITATION.cff).
