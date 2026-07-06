"""Build the SQLite annotation database from the raw per-annotator CSV exports.

This is the (anonymized) script used to assemble the compliance-annotation database
from which ``data/annotations/annotated_causes.csv`` is exported. It is included for
provenance and reproducibility.

The raw per-annotator CSVs (one per dataset, each row = ``image_path, compliance``)
and the resulting ``.db`` file are **not** shipped: the images they reference come
from registration-gated public datasets. The distributed artifact is the flattened
``annotated_causes.csv`` (see ``export_annotations.py``).

Schema is defined in the sibling ``data_definition.sql``. Usage::

    python -m scripts.preprocessing.make_database \
        --raw-dir path/to/per_annotator_csvs \
        --db-out annotations.db
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Anonymized annotators. The compliance labelling was performed by a single primary
#: annotator; ambiguous cases were adjudicated with a senior supervisor and, for the
#: most challenging cases, a qualified radiologist.
ANNOTATORS = ["annotator_1", "annotator_2", "annotator_3"]
PRIMARY_ANNOTATOR = ANNOTATORS[0]

DATASETS = ["CheXpert", "PadChest", "ChestX-ray8", "MIMIC-CXR"]

COMPLIANCY_STATES = ["Compliant", "Non-Compliant", "Doubt"]

#: The non-compliance taxonomy (also mirrored in ``data/annotations/taxonomy.json``).
CAUSES: list[tuple[str, str]] = [
    (
        "Corrupted Images",
        "Images where most of the thoracic information is either absent or severely "
        "damaged, rendering the image incoherent.",
    ),
    (
        "Not a Thorax",
        "Images that do not depict a thoracic region, such as X-rays of other body parts.",
    ),
    (
        "Incomplete Thorax",
        "Images where more than approximately 5% of the lung parenchyma is missing. This "
        "includes cases where significant portions of the thorax are not visible, making "
        "it impossible to perform a full assessment of the lungs and heart.",
    ),
    (
        "Image Quality Problems",
        "Includes issues such as insufficient contrast or low resolution, often due to "
        "acquisition issues such as motion artifacts or exposure errors.",
    ),
    (
        # NOTE: Split out of "Image Quality Problems" as a dedicated category given its
        # high recurrence. TODO(author): confirm the exact clinical wording of this
        # definition before public release.
        "Area Not Valid",
        "Images where less than approximately 80% of the frame is valid thoracic content, "
        "with the invalid area typically caused by collimation problems, cropping, or "
        "acquisition artifacts.",
    ),
    (
        "Overlaying Objects",
        "Removable objects (e.g., defibrillation pads, stretcher bars, electrodes) that "
        "obscure significant portions of the thoracic region, potentially affecting the "
        "diagnosis and causing artifacts that interfere with image interpretation. Objects "
        "that cannot be removed, such as endotracheal tubes or pacemakers, fall under the "
        "category of support devices and are not considered non-compliant.",
    ),
    (
        "Non-Canonical Positions",
        "Images where the patient's position deviates from standard protocols, such as the "
        "head overlaying the thorax. These positional errors can make it difficult to "
        "interpret the image correctly and can introduce significant biases into the model "
        "if not addressed.",
    ),
]

COMPLIANCY_MAP = {"compliant": "Compliant", "non-compliant": "Non-Compliant", "doubt": "Doubt"}


def _next_image_id(cursor: sqlite3.Cursor) -> int:
    cursor.execute("SELECT MAX(id) FROM Image")
    max_id = cursor.fetchone()[0]
    return max_id + 1 if max_id is not None else 0


def build_database(raw_dir: Path, db_out: Path, schema_sql: Path) -> None:
    """Create the annotation database and populate it from the per-dataset CSVs."""
    db_out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_out))
    cursor = conn.cursor()

    cursor.executescript(schema_sql.read_text(encoding="utf-8"))

    cursor.executemany("INSERT INTO User(username) VALUES(?)", [(u,) for u in ANNOTATORS])
    cursor.executemany("INSERT INTO Dataset(name) VALUES(?)", [(d,) for d in DATASETS])
    cursor.executemany("INSERT INTO Compliancy(compliancy) VALUES(?)", [(c,) for c in COMPLIANCY_STATES])
    cursor.executemany(
        "INSERT INTO Cause(id, name, description) VALUES(?, ?, ?)",
        [(i, name, desc) for i, (name, desc) in enumerate(CAUSES)],
    )

    for dataset in DATASETS:
        csv_path = raw_dir / f"{dataset}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)  # header
            for row in reader:
                image_id = _next_image_id(cursor)
                cursor.execute(
                    "INSERT INTO Image(id, path, dataset) VALUES(?, ?, ?)",
                    (image_id, row[0], dataset),
                )
                compliancy = COMPLIANCY_MAP[row[1].strip().lower()]
                cursor.execute(
                    "INSERT INTO Annotation(image, annotator, compliancy) VALUES(?, ?, ?)",
                    (image_id, PRIMARY_ANNOTATOR, compliancy),
                )

    conn.commit()
    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SQLite annotation database.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory with one '<Dataset>.csv' per dataset (columns: image_path, compliance).",
    )
    parser.add_argument("--db-out", type=Path, default=Path("annotations.db"), help="Output SQLite file.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=HERE / "data_definition.sql",
        help="Schema SQL file (default: sibling data_definition.sql).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_database(args.raw_dir, args.db_out, args.schema)
    print(f"Wrote annotation database to: {args.db_out}")


if __name__ == "__main__":
    main()
