"""Dataset implementations and dataset-specific transform bundles."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nn_functional
from PIL import Image, ImageFile

from DeMICAF.caf.transforms.transforms import (
    Transforms,
    add_random_objects,
    adjust_exposure,
    blur,
    color_jitter,
    extreme_resized_crop,
    maybe_zoom_out,
    partial_illumination,
    random_flip,
    random_rotation,
    reduce_gray_levels,
    slice_shift,
    zoom_in,
)
from DeMICAF.data.base import Dataset2D

if TYPE_CHECKING:
    from collections.abc import Callable

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = [
    "KNOWN_DATASETS",
    "ChestXray",
    "ChestXrayDataset",
    "DatasetCXR",
    "build_multi_cxr_dataset",
]

KNOWN_DATASETS: list[str] = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]


ChestXray = Transforms(
    transforms_id=[
        partial(maybe_zoom_out, scale_range=(1.05, 1.4), size=(224, 224), prob=0.0),
        partial(zoom_in, scale_range=(1.05, 1.4), size=(224, 224)),
        partial(random_flip, prob=0.1),
        partial(random_rotation, degrees=(-20, 20)),
        partial(color_jitter, brightness=(0.5, 1.5), contrast=(0.7, 1.3)),
        partial(blur, sigma=(1.0, 5.0)),
    ],
    transforms_ood=[
        partial(extreme_resized_crop, size=(224, 224)),
        partial(slice_shift, num_slices=(2, 4), dislocation_factor=0.3),
        adjust_exposure,
        partial(reduce_gray_levels, levels=4),
        add_random_objects,
        partial_illumination,
    ],
)


class DatasetCXR(Dataset2D):
    """
    Dataset for 2D chest X-ray images.

    Metadata is expected to contain a path column such as `image_path` (as in
    `subset.csv`). If absent, the first dataframe column is used as a fallback.
    Optional label information is read from a `label` column when available.
    """

    __slots__ = (
        "_label_map",
        "eval_resize",
        "image_column",
        "label_cols",
        "label_column",
        "output_mode",
        "path_builder",
        "train_resize",
        "uncertain_policy",
    )

    def __init__(
        self,
        root_dir: str | Path,
        dataframe: Any,
        metadata: Any | None = None,
        metadata_join_on: str | tuple[str, str] | list[str] | None = None,
        transform: Any | None = None,
        image_column: str = "image_path",
        label_column: str = "label",
        label_cols: list[str] | None = None,
        path_builder: Any | None = None,
        output_mode: str = "label",
        uncertain_policy: str = "ignore",
        train_resize: int = 256,
        eval_resize: int = 224,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        root_dir
            Dataset root directory.
        dataframe
            Filter table defining which images to include.
        metadata
            Optional metadata table/CSV to merge with `dataframe`.
        metadata_join_on
            Join key(s) to merge `dataframe` and `metadata`.
            Use a single column name for same-name joins, a list of names for
            multi-column joins, or `(left_key, right_key)` for different names.
        transform
            Optional transform applied to a single sample.
        image_column
            Name of the column containing image paths.
        label_column
            Name of the column containing labels for single-label datasets.
        label_cols
            Label columns for multi-label classification.
        path_builder
            Optional callable that builds image relative paths from each row.
        output_mode
            Either `label` for single-label output or `labels_mask` for masked multi-label output.
        uncertain_policy
            How to handle uncertain multi-label values: `ignore`, `zeros`, or `ones`.
        train_resize
            Resize used before applying augmentations.
        eval_resize
            Final spatial size for model input tensors.
        """
        self.image_column = image_column
        self.label_column = label_column
        self.label_cols = label_cols
        self.path_builder = path_builder
        self.output_mode = output_mode
        self.uncertain_policy = uncertain_policy
        self.train_resize = train_resize
        self.eval_resize = eval_resize

        self._label_map = {
            "frontal": 0,
            "lateral": 1,
        }

        if self.output_mode not in {"label", "labels_mask"}:
            msg = f"Unsupported output_mode: {self.output_mode}"
            raise ValueError(msg)

        base_frame = self._ensure_dataframe(dataframe)
        if metadata is not None:
            meta_frame = self._ensure_dataframe(metadata)
            base_frame = self._merge_filter_with_metadata(base_frame, meta_frame, metadata_join_on)

        super().__init__(root_dir=root_dir, dataframe=base_frame, transform=transform)

    @staticmethod
    def _ensure_dataframe(source: Any) -> pd.DataFrame:
        """Load a dataframe from an in-memory table or a CSV path."""
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return pd.read_csv(source)

    @staticmethod
    def _merge_filter_with_metadata(
        filter_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        metadata_join_on: str | tuple[str, str] | list[str] | None,
    ) -> pd.DataFrame:
        """Merge a filter dataframe with dataset-specific metadata."""
        if metadata_join_on is None:
            common = [col for col in filter_df.columns if col in metadata_df.columns]
            if not common:
                msg = "Could not infer join keys between filter and metadata. Pass metadata_join_on explicitly."
                raise ValueError(msg)
            return filter_df.merge(metadata_df, on=common, how="left", suffixes=("", "_meta"))

        if isinstance(metadata_join_on, tuple):
            left_key, right_key = metadata_join_on
            return filter_df.merge(
                metadata_df,
                left_on=left_key,
                right_on=right_key,
                how="left",
                suffixes=("", "_meta"),
            )

        return filter_df.merge(metadata_df, on=metadata_join_on, how="left", suffixes=("", "_meta"))

    def _build_entries(self) -> None:
        """Build valid dataset entries from the metadata table."""
        if self.dataframe is None:
            self.entries = []
            return

        path_column = self.image_column if self.image_column in self.dataframe.columns else self.dataframe.columns[0]

        entries: list[dict[str, Any]] = []
        for idx, row in self.dataframe.iterrows():
            rel_path = str(self.path_builder(row)) if self.path_builder is not None else str(row[path_column])
            candidate = Path(rel_path)
            full_path = candidate if candidate.is_absolute() else self.root_dir / rel_path

            if not full_path.exists():
                continue

            entry: dict[str, Any] = {
                "fname": rel_path,
                "path": full_path,
                "row_index": idx,
            }

            if self.output_mode == "label" and self.label_column in self.dataframe.columns:
                raw_label = str(row[self.label_column]).strip().lower()
                entry["label"] = self._label_map.get(raw_label, -1)

            entries.append(entry)

        self.entries = entries

    def _load_data(self, entry: dict[str, Any]) -> Image.Image:
        """Load and normalize a chest X-ray as a grayscale PIL image."""
        image: Image.Image = Image.open(entry["path"])

        if "I" in image.mode:
            arr = np.array(image)
            arr = arr // 256
            image = Image.fromarray(arr.astype(np.uint8), mode="L")
        else:
            image = image.convert("L")

        return image

    def _resolve_multilabels(self, row: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert multi-label values into labels and a training mask."""
        if self.label_cols is None:
            msg = "label_cols must be provided when output_mode='labels_mask'"
            raise ValueError(msg)

        values = np.asarray(row[self.label_cols], dtype=float)
        labels = values.copy()
        mask = torch.ones(len(self.label_cols), dtype=torch.float32)

        for idx, value in enumerate(values):
            if np.isnan(value) or value == -1:
                if self.uncertain_policy == "ignore":
                    mask[idx] = 0.0
                    labels[idx] = 0.0
                elif self.uncertain_policy == "zeros":
                    labels[idx] = 0.0
                elif self.uncertain_policy == "ones":
                    labels[idx] = 1.0
                else:
                    msg = f"Unsupported uncertain_policy: {self.uncertain_policy}"
                    raise ValueError(msg)

        return torch.tensor(labels, dtype=torch.float32), mask

    @staticmethod
    def _resize_pil(image: Image.Image, size: int) -> Image.Image:
        """Resize a PIL image to a square while preserving grayscale mode."""
        return image.resize((size, size), resample=Image.Resampling.BILINEAR)

    def _to_model_tensor(self, image_like: Any, resize: int, normalize: bool | None = None) -> torch.Tensor:
        """Convert image-like input to a normalized model tensor."""
        tensor = self._ensure_tensor(image_like)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 3:
            msg = f"Expected image tensor with 3 dims (C,H,W), got shape={tuple(tensor.shape)}"
            raise ValueError(msg)

        if tensor.shape[0] > 1:
            tensor = tensor[:1]

        tensor = tensor.float()

        if tensor.max() > 1:
            tensor = tensor / 255.0

        tensor = nn_functional.interpolate(
            tensor.unsqueeze(0),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        do_normalize = True if normalize is None else normalize
        if do_normalize:
            tensor = (tensor - 0.5) / 0.5

        return tensor

    def get_view_input(self, index: int) -> tuple[str, Any, Any]:
        """Return the raw input used by augmentation wrappers."""
        entry = self.entries[index]
        fname = entry["fname"]
        image = self._load_data(entry)
        base = self._resize_pil(image, self.train_resize)

        if self.transform is not None:
            base = self.transform(base)

        if self.output_mode == "labels_mask":
            if self.dataframe is None:
                msg = "Dataset dataframe is not available"
                raise RuntimeError(msg)
            row = self.dataframe.iloc[entry["row_index"]]
            return fname, base, self._resolve_multilabels(row)

        return fname, base, entry.get("label", -1)

    def postprocess_view(self, view: Any) -> torch.Tensor:
        """Convert a view into the normalized tensor expected by the encoder."""
        return self._to_model_tensor(view, self.eval_resize, normalize=True)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        """Return a single sample for supervised training or inference."""
        entry = self.entries[index]
        fname = entry["fname"]
        image = self._load_data(entry)

        if self.transform is not None:
            image = self.transform(image)

        tensor = self._to_model_tensor(image, self.eval_resize, normalize=True)
        if self.output_mode == "labels_mask":
            if self.dataframe is None:
                msg = "Dataset dataframe is not available"
                raise RuntimeError(msg)
            row = self.dataframe.iloc[entry["row_index"]]
            return fname, tensor, self._resolve_multilabels(row)

        return fname, tensor, entry.get("label", -1)


ChestXrayDataset = DatasetCXR


def _make_multi_dataset_path_builder(
    dataset_image_roots: dict[str, Path],
) -> Callable[[Any], str]:
    """Return a path builder that resolves absolute image paths for multi-dataset rows.

    Each CSV stores paths prefixed with the dataset name (e.g. ``ChestX-ray8/images/foo.png``).
    The builder strips that prefix so it can be joined with the dataset-specific image root.
    The ``_dataset`` column added by :func:`build_multi_cxr_dataset` identifies the source.
    """

    def _build(row: Any) -> str:
        source = str(row["_dataset"])
        raw = str(row["image_path"])
        root = dataset_image_roots[source]
        prefix = source + "/"
        stripped = raw.removeprefix(prefix)
        return str(root / stripped)

    return _build


def build_multi_cxr_dataset(
    datasets_root: str | Path,
    *,
    dataset_names: list[str] | None = None,
    subset_csv: str = "annotations.csv",
    dataset_image_roots: dict[str, str | Path] | None = None,
    transform: Any | None = None,
    label_cols: list[str] | None = None,
    output_mode: str = "labels_mask",
    uncertain_policy: str = "ignore",
    train_resize: int = 256,
    eval_resize: int = 224,
    **dataset_kwargs: Any,
) -> DatasetCXR:
    """Build a :class:`DatasetCXR` combining subsets from multiple CXR datasets.

    Parameters
    ----------
    datasets_root
        Common parent directory containing one subfolder per dataset.
        Used to locate per-dataset CSVs at ``datasets_root/{name}/{subset_csv}``
        and as the default image root when ``dataset_image_roots`` is not given.
    dataset_names
        Datasets to include. Defaults to all four in :data:`KNOWN_DATASETS`.
    subset_csv
        CSV filename inside each dataset folder.
        ``'annotations.csv'`` → 1k quality-labelled samples per dataset (4k total).
        ``'subset100k.csv'`` → large unlabelled pool per dataset.
        ``'subset.csv'`` → standard scored subset.
    dataset_image_roots
        Optional per-dataset image root override, keyed by dataset name.
        Defaults to ``datasets_root/{name}`` for each dataset.
    transform
        Transform applied to each sample (e.g. ``train_tfm`` or ``val_tfm``).
    label_cols
        Multi-label columns for ``output_mode='labels_mask'``.
    output_mode
        ``'label'`` for single-label or ``'labels_mask'`` for multi-label.
    uncertain_policy
        How to treat uncertain/missing labels: ``'ignore'``, ``'zeros'``, ``'ones'``.
    train_resize
        Resize used before augmentations.
    eval_resize
        Final spatial size for model input tensors.
    **dataset_kwargs
        Additional keyword arguments forwarded to :class:`DatasetCXR`.
    """
    root = Path(datasets_root)
    names = dataset_names if dataset_names is not None else KNOWN_DATASETS

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for name in names:
        csv_path = root / name / subset_csv
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        df = pd.read_csv(csv_path)
        df["_dataset"] = name
        frames.append(df)

    if not frames:
        msg = f"No '{subset_csv}' found for any of {names} under {root}. Missing:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)

    if missing:
        import warnings

        warnings.warn("Missing subset CSVs (skipped):\n" + "\n".join(missing), stacklevel=2)

    combined = pd.concat(frames, ignore_index=True)

    image_roots: dict[str, Path] = {
        name: (
            Path(dataset_image_roots[name]) if dataset_image_roots and name in dataset_image_roots else root / name
        ).resolve()
        for name in names
    }
    path_builder = _make_multi_dataset_path_builder(image_roots)

    return DatasetCXR(
        root_dir=root,
        dataframe=combined,
        transform=transform,
        image_column="image_path",
        path_builder=path_builder,
        label_cols=label_cols,
        output_mode=output_mode,
        uncertain_policy=uncertain_policy,
        train_resize=train_resize,
        eval_resize=eval_resize,
        **dataset_kwargs,
    )
