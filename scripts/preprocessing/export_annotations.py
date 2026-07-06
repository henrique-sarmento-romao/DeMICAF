"""Export the compliance annotations from the SQLite database to flat CSVs.

Produces the distributed ``annotated_causes.csv`` (one row per annotated image, with a
``compliance`` column and one binary column per non-compliance cause) and, optionally,
per-dataset splits.

The canonical release CSV is already shipped at
``data/annotations/annotated_causes.csv``; this script documents how it is derived from
the annotation database built by ``make_database.py`` and lets you regenerate it.

Usage::

    python -m scripts.preprocessing.export_annotations --db annotations.db \
        --out data/annotations/annotated_causes.csv --per-dataset
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TAXONOMY_PATH = REPO_ROOT / "data" / "annotations" / "taxonomy.json"


#: Prefix prepended to each dataset's stored path to form the canonical ``image_path``.
DATASET_PREFIXES = {
    "CheXpert": "CheXpert/CheXpert-v1.0",
    "ChestX-ray8": "ChestX-ray8",
    "MIMIC-CXR": "MIMIC-CXR",
    "PadChest": "PadChest",
}


def _cause_columns() -> list[tuple[str, str]]:
    """Return ``(column_id, display_name)`` pairs for the causes, in taxonomy order."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    return [(cause["id"], cause["name"]) for cause in taxonomy["causes"]]


def _canonical_image_path(dataset: str, stored_path: str) -> str:
    """Prepend the dataset prefix so paths match the released ``annotated_causes.csv``."""
    prefix = DATASET_PREFIXES.get(dataset, dataset)
    return f"{prefix}/{str(stored_path).lstrip('/')}"


def export_annotations(db_path: Path, out_csv: Path, *, per_dataset: bool = False) -> pd.DataFrame:
    """Flatten the annotation database into a wide ``image_path × cause`` table."""
    cause_cols = _cause_columns()
    name_to_id = {name: col_id for col_id, name in cause_cols}

    conn = sqlite3.connect(str(db_path))
    try:
        images = pd.read_sql_query("SELECT id, path, dataset FROM Image", conn)
        annotations = pd.read_sql_query("SELECT image, annotator, compliancy FROM Annotation", conn)
        annotation_causes = pd.read_sql_query("SELECT image, cause FROM AnnotationCause", conn)
        cause_id_to_name = dict(pd.read_sql_query("SELECT id, name FROM Cause", conn).itertuples(index=False))
    finally:
        conn.close()

    # ``AnnotationCause.cause`` may hold either the Cause id or the Cause name; normalise to name.
    def _to_cause_name(value: object) -> str:
        try:
            return cause_id_to_name[int(value)]
        except (ValueError, TypeError, KeyError):
            return str(value)

    annotation_causes["cause"] = annotation_causes["cause"].map(_to_cause_name)

    # One compliance label per image (single primary annotator per the protocol).
    compliance = annotations.groupby("image")["compliancy"].first()

    records: list[dict[str, object]] = []
    causes_by_image = annotation_causes.groupby("image")["cause"].apply(set).to_dict()
    for _, image in images.iterrows():
        image_id = image["id"]
        state = str(compliance.get(image_id, "")).strip().lower()
        if state not in {"compliant", "non-compliant"}:
            continue
        present = causes_by_image.get(image_id, set())
        image_path = _canonical_image_path(str(image["dataset"]), str(image["path"]))
        record: dict[str, object] = {"image_path": image_path, "compliance": state}
        for col_id, display_name in cause_cols:
            record[col_id] = 1 if display_name in present else ""
        records.append(record)

    columns = ["image_path", "compliance", *[col_id for col_id, _ in cause_cols]]
    df = pd.DataFrame.from_records(records, columns=columns)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} annotations to {out_csv}")

    if per_dataset:
        df_dataset = df.assign(_dataset=df["image_path"].str.split("/", n=1).str[0])
        for dataset, group in df_dataset.groupby("_dataset"):
            dataset_csv = out_csv.parent / f"{dataset}.csv"
            group.drop(columns="_dataset").to_csv(dataset_csv, index=False)
            print(f"  {dataset}: {len(group)} rows -> {dataset_csv}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compliance annotations to CSV.")
    parser.add_argument("--db", type=Path, required=True, help="SQLite annotation database (see make_database.py).")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "annotations" / "annotated_causes.csv",
        help="Output CSV path (default: data/annotations/annotated_causes.csv).",
    )
    parser.add_argument("--per-dataset", action="store_true", help="Also write one CSV per source dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_annotations(args.db, args.out, per_dataset=args.per_dataset)


if __name__ == "__main__":
    main()
