# Non-Compliance Taxonomy

A chest radiograph is **compliant** when it is a technically valid, standard-protocol
frontal thorax image fit for downstream analysis. Otherwise it is **non-compliant**,
annotated with one or more of the seven causes below. The machine-readable version of
this taxonomy is [`../data/annotations/taxonomy.json`](../data/annotations/taxonomy.json);
the CSV column id for each cause is given in parentheses.

| # | Cause (`column_id`) | Definition |
| - | ------------------- | ---------- |
| 1 | **Corrupted Images** (`corrupted_images`) | Images where most of the thoracic information is either absent or severely damaged, rendering the image incoherent. |
| 2 | **Not a Thorax** (`not_a_thorax`) | Images that do not depict a thoracic region, such as X-rays of other body parts. |
| 3 | **Incomplete Thorax** (`incomplete_thorax`) | More than ~5% of the lung parenchyma is missing; significant portions of the thorax are not visible, preventing a full assessment of the lungs and heart. |
| 4 | **Image Quality Problems** (`image_quality_problems`) | Insufficient contrast or low resolution, often from acquisition issues such as motion artifacts or exposure errors. |
| 5 | **Area Not Valid** (`area_not_valid`) | Less than ~80% of the frame is valid thoracic content, typically due to collimation problems, cropping, or acquisition artifacts. |
| 6 | **Overlaying Objects** (`overlaying_objects`) | Removable objects (defibrillation pads, stretcher bars, electrodes) obscuring significant portions of the thorax. Non-removable support devices (endotracheal tubes, pacemakers) are **not** considered non-compliant. |
| 7 | **Non-Canonical Positions** (`non_canonical_positions`) | Patient position deviates from standard protocols (e.g. head overlaying the thorax), making correct interpretation difficult and risking model bias. |

An image may carry more than one cause. In the released CSV each cause is a binary
column (`1` = present).

## Notes

- **Area Not Valid** was introduced after the initial protocol (previously subsumed
  under *Image Quality Problems*) as a dedicated category, given its high recurrence.
  > **TODO (author):** confirm the exact clinical wording of the *Area Not Valid*
  > definition before public release; the text above is a working formulation.
- During labelling a third *doubt* state was available for ambiguous cases; these were
  adjudicated to *compliant* / *non-compliant* before release, so the distributed
  annotations contain only the two final states.

## Example images

Figure 1 of the paper shows one representative example per cause. A montage can be
regenerated from locally-available images with `scripts/stats/causes.py` (see the
overlay options in [`REPRODUCE.md`](REPRODUCE.md)).
