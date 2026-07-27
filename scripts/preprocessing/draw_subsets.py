"""Draw the fixed 10k and 100k image subsets per dataset (seeded, reproducible).

Each subset always includes the 1,000 quality-annotated images for the dataset (read
from ``<data-root>/<Dataset>/annotations.csv``), then tops up to the target size with a
seeded random sample of the remaining images from the raw metadata CSV.

The raw metadata CSVs and images are **not** shipped (registration-gated source
datasets). Point ``--data-root`` (or the ``CXR_ROOT`` environment variable) at a
directory laid out as ``<Dataset>/<metadata>.csv`` plus ``<Dataset>/annotations.csv``.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from DeMICAF.utils.paths import get_cxr_root

TARGET_10K = 10_000
TARGET_100K = 100_000
RNG_SEED = 42


def build_chexpert_path(row: dict[str, str]) -> str:
    return f"CheXpert/CheXpert-v1.0/{row['Path']}"


def build_chestxray8_path(row: dict[str, str]) -> str:
    return f"ChestX-ray8/images/{row['Image Index']}"


def build_mimic_path(row: dict[str, str]) -> str:
    subject_id = str(row["subject_id"]).strip()
    study_id = str(row["study_id"]).strip()
    dicom_id = str(row["dicom_id"]).strip()
    return f"MIMIC-CXR/files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.png"


def build_padchest_path(row: dict[str, str]) -> str:
    image_dir = str(row["ImageDir"]).strip()
    image_id = str(row["ImageID"]).strip()
    return f"PadChest/{image_dir}/{image_id}"


#: Per-dataset raw metadata filename and the row → relative-image-path builder.
DATASETS = {
    "CheXpert": {"metadata": "train.csv", "path_builder": build_chexpert_path},
    "ChestX-ray8": {"metadata": "Data_Entry_2017_v2020.csv", "path_builder": build_chestxray8_path},
    "MIMIC-CXR": {"metadata": "mimic-cxr-2.0.0-metadata.csv", "path_builder": build_mimic_path},
    "PadChest": {
        "metadata": "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        "path_builder": build_padchest_path,
    },
}


def read_annotation_paths(path: Path) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = str(row["image_path"]).strip()
            if image_path and image_path not in seen:
                seen.add(image_path)
                paths.append(image_path)

    return paths


def collect_metadata_paths(metadata_path: Path, path_builder) -> tuple[list[str], set[str]]:
    all_paths: list[str] = []
    seen: set[str] = set()

    with metadata_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = path_builder(row)
            if image_path and image_path not in seen:
                seen.add(image_path)
                all_paths.append(image_path)

    return all_paths, seen


def write_subset(path: Path, image_paths: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path"])
        for image_path in image_paths:
            writer.writerow([image_path])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw 10k and 100k subsets for chest X-ray datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS.keys()),
        help="Dataset(s) to process. Repeat the flag to process multiple datasets.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=get_cxr_root(),
        help="Root holding <Dataset>/<metadata>.csv and <Dataset>/annotations.csv "
        "(default: $CXR_ROOT or <repo_root>/data/chest_xray).",
    )
    return parser.parse_args()


def existing_paths_only(paths: list[str], existence_cache: dict[str, bool], data_root: Path) -> list[str]:
    existing: list[str] = []
    for image_path in paths:
        in_cache = existence_cache.get(image_path)
        if in_cache is None:
            in_cache = (data_root / image_path).is_file()
            existence_cache[image_path] = in_cache
        if in_cache:
            existing.append(image_path)
    return existing


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    rng = random.Random(RNG_SEED)
    selected_datasets = set(args.dataset or DATASETS.keys())

    for dataset_name, config in DATASETS.items():
        if dataset_name not in selected_datasets:
            continue

        metadata_path: Path = data_root / dataset_name / config["metadata"]
        annotations_path: Path = data_root / dataset_name / "annotations.csv"
        path_builder = config["path_builder"]
        existence_cache: dict[str, bool] = {}

        annotation_paths = read_annotation_paths(annotations_path)

        if dataset_name == "PadChest":
            annotation_paths_existing = existing_paths_only(annotation_paths, existence_cache, data_root)
            missing_annotation_files = len(annotation_paths) - len(annotation_paths_existing)
            annotation_paths = annotation_paths_existing
            if missing_annotation_files:
                print(
                    f"{dataset_name}: warning - {missing_annotation_files} annotation image files are missing; "
                    "they are replaced by other existing images in subsets."
                )

        annotation_set = set(annotation_paths)

        metadata_paths, metadata_set = collect_metadata_paths(metadata_path, path_builder)
        candidate_paths = [p for p in metadata_paths if p not in annotation_set]

        if dataset_name == "PadChest":
            candidate_paths = existing_paths_only(candidate_paths, existence_cache, data_root)

        missing_annotations = len(annotation_set - metadata_set)
        print(
            f"{dataset_name}: total={len(metadata_paths)} | annotations={len(annotation_paths)} | "
            f"candidates={len(candidate_paths)}"
        )
        if missing_annotations:
            print(
                f"{dataset_name}: warning - {missing_annotations} annotation paths are not in metadata; "
                "they are still kept in subsets."
            )

        extra_needed_10k = max(0, TARGET_10K - len(annotation_paths))
        extra_needed_100k = max(0, TARGET_100K - len(annotation_paths))

        sampled_for_100k = rng.sample(candidate_paths, k=min(extra_needed_100k, len(candidate_paths)))

        sampled_for_10k = sampled_for_100k[: min(extra_needed_10k, len(sampled_for_100k))]

        subset10k_paths = annotation_paths + sampled_for_10k
        subset100k_paths = annotation_paths + sampled_for_100k

        if len(subset10k_paths) < TARGET_10K:
            print(
                f"{dataset_name}: warning - only {len(subset10k_paths)} rows available for 10k subset "
                "after applying constraints."
            )
        if len(subset100k_paths) < TARGET_100K:
            print(
                f"{dataset_name}: warning - only {len(subset100k_paths)} rows available for 100k subset "
                "after applying constraints."
            )

        output_dir = metadata_path.parent
        subset10k_file = output_dir / "subset10k.csv"
        subset100k_file = output_dir / "subset100k.csv"

        write_subset(subset10k_file, subset10k_paths)
        write_subset(subset100k_file, subset100k_paths)

        print(
            f"{dataset_name}: wrote {subset10k_file.name} ({len(subset10k_paths)} rows) and "
            f"{subset100k_file.name} ({len(subset100k_paths)} rows)"
        )


if __name__ == "__main__":
    main()
