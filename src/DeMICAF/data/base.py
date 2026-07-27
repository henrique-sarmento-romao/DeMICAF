"""Core dataset abstractions and augmentation wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset

from DeMICAF.caf.networks import ResNet2D

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

__all__ = [
    "AugmentedDataset",
    "Dataset2D",
    "DatasetBase",
    "multi_view_collate",
]


def _unpack_sample(sample: Any) -> tuple[Any, Any, Any | None]:
    """Normalize dataset samples into (name, data, label) tuples."""
    msg = "Expected dataset sample to be a tuple or list"
    if not isinstance(sample, list | tuple):
        raise TypeError(msg)
    if len(sample) == 2:
        fname, data = sample
        return fname, data, None
    if len(sample) == 3:
        fname, data, label = sample
        return fname, data, label
    msg = "Expected dataset sample to have 2 or 3 items"
    raise ValueError(msg)


def _normalize_labels(label: Any, n_views: int) -> tuple[Any, ...] | None:
    """Repeat or validate labels across views."""
    if label is None:
        return None
    if isinstance(label, list | tuple):
        if len(label) != n_views:
            msg = "If label is a sequence, it must match number of views"
            raise ValueError(msg)
        return tuple(label)
    return tuple(label for _ in range(n_views))


def _make_view_names(fname: Any, n_views: int) -> str | tuple[str, ...]:
    """Generate view names for contrastive training outputs."""
    if isinstance(fname, list | tuple):
        msg = "AugmentedDataset expects a single sample name, got a sequence"
        raise TypeError(msg)

    base = str(fname)
    if n_views == 1:
        return base
    if n_views == 4:
        shifted = f"{base}_shifted"
        return (base, base, shifted, shifted)
    return tuple(base for _ in range(n_views))


def _apply_transforms(transforms: Any, input_obj: Any) -> tuple[Any, ...]:
    """Apply ID and OOD transforms to generate multiple views."""
    if not transforms:
        return (input_obj,)

    transform_id = transforms.ID
    transform_ood = transforms.OOD

    view_id_a = transform_id(input_obj)
    view_id_b = transform_id(input_obj)

    if transform_ood:
        view_ood = transform_ood(input_obj)
        view_ood_a = transform_id(view_ood)
        view_ood_b = transform_id(view_ood)
        del view_ood
        return (view_id_a, view_id_b, view_ood_a, view_ood_b)

    return (view_id_a, view_id_b)


def multi_view_collate(batch: list[Any]) -> tuple[list[Any], torch.Tensor] | tuple[list[Any], torch.Tensor, list[Any]]:
    """
    Collate multi-view samples into a flat batch.

    Parameters
    ----------
    batch
            Batch produced by a dataset or wrapper that returns views.

    Returns
    -------
    tuple
            Filenames and stacked images, optionally followed by labels.
    """
    if not batch:
        return [], torch.empty(0)

    fnames: list[Any] = []
    images: list[torch.Tensor] = []
    labels: list[Any] = []
    has_labels = False
    n_views: int | None = None

    for sample in batch:
        fname, views, label = _unpack_sample(sample)

        if not isinstance(views, list | tuple):
            views = (views,)

        if n_views is None:
            n_views = len(views)
        elif len(views) != n_views:
            msg = "Inconsistent number of views in batch"
            raise ValueError(msg)

        if isinstance(fname, list | tuple):
            if len(fname) != len(views):
                msg = "If fname is a sequence, it must match number of views"
                raise ValueError(msg)
            fname_per_view = list(fname)
        else:
            fname_per_view = [fname] * len(views)

        if label is None:
            label_per_view = None
        else:
            has_labels = True
            label_per_view = _normalize_labels(label, len(views))

        for idx, view in enumerate(views):
            if not torch.is_tensor(view):
                msg = "Expected tensor views in batch"
                raise TypeError(msg)
            fnames.append(fname_per_view[idx])
            images.append(view)
            if label_per_view is not None:
                labels.append(label_per_view[idx])

    images_tensor = torch.stack(images, dim=0)
    if has_labels:
        return fnames, images_tensor, labels
    return fnames, images_tensor


class DatasetBase(Dataset[Any], ABC):
    """
    Abstract base dataset for medical imaging data.

    Subclasses must implement `_build_entries`, `_load_data`, and `__getitem__`,
    and define the `NETWORK` class attribute.
    """

    __slots__ = ("dataframe", "entries", "root_dir", "transform")

    NETWORK: ClassVar[type[Any] | None] = None

    def __init__(self, root_dir: str | Path, dataframe: pd.DataFrame | None, transform: Any | None = None) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        root_dir
                Root directory containing the dataset assets.
        dataframe
                Metadata table backing the dataset.
        transform
                Optional transform applied to a single sample.
        """
        self.root_dir = Path(root_dir)
        self.dataframe = dataframe
        self.transform = transform
        self.entries: list[dict[str, Any]] = []

        self._build_entries()

    @abstractmethod
    def _build_entries(self) -> None:
        """Populate `self.entries` from the metadata table."""

    @abstractmethod
    def _load_data(self, entry: dict[str, Any]) -> Any:
        """Load raw data for a single entry."""

    def __len__(self) -> int:
        """Return the number of entries in the dataset."""
        return len(self.entries)

    def get_view_input(self, index: int) -> tuple[Any, Any, Any | None]:
        """
        Provide inputs consumed by augmentation wrappers.

        Parameters
        ----------
        index
                Index of the requested sample.

        Returns
        -------
        tuple
                Sample name, raw input, and optional label.
        """
        sample = self[index]
        return _unpack_sample(sample)

    def postprocess_view(self, view: Any) -> Any:
        """Post-process a single augmented view before returning it."""
        return view

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return a processed sample by index."""

    def get_network(self, **network_kwargs: Any) -> Any:
        """
        Instantiate the dataset's encoder network.

        Parameters
        ----------
        **network_kwargs
                Keyword arguments forwarded to the network constructor.
        """
        network_cls = self.NETWORK
        if network_cls is None or not callable(network_cls):
            msg = f"{self.__class__.__name__} must define a callable NETWORK class attribute"
            raise NotImplementedError(msg)
        return network_cls(**network_kwargs)


class Dataset2D(DatasetBase):
    """Base class for 2D medical imaging datasets."""

    __slots__ = ()

    NETWORK = ResNet2D

    @staticmethod
    def _to_grayscale_uint8(image: Any) -> np.ndarray:
        """
        Convert input images to 8-bit grayscale arrays.

        Parameters
        ----------
        image
                Input image-like object.
        """
        mode = getattr(image, "mode", "")
        if "I" in mode:
            arr = np.asarray(image)
            arr = arr // 256
            return arr.astype(np.uint8)

        if hasattr(image, "convert"):
            image = image.convert("L")
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _ensure_tensor(image_like: Any) -> torch.Tensor:
        """
        Ensure an image-like object is returned as a torch tensor.

        Parameters
        ----------
        image_like
                Image-like input from the dataset.
        """
        if torch.is_tensor(image_like):
            return image_like

        arr = Dataset2D._to_grayscale_uint8(image_like)
        return torch.from_numpy(arr).unsqueeze(0).float() / 255.0
class AugmentedDataset(Dataset[Any]):
    """Dataset wrapper that generates augmented multi-view samples."""

    __slots__ = ("dataset", "transforms", "view_input_fn", "view_postprocess")

    def __init__(
        self,
        dataset: Dataset[Any],
        transforms: Any,
        view_input_fn: Callable[[int], Any] | None = None,
        view_postprocess: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize the segmentation dataset.

        Parameters
        ----------
        dataset
                Base dataset providing raw inputs.
        transforms
                Transform bundle with ID and optional OOD transforms.
        view_input_fn
                Optional custom hook for fetching view inputs.
        view_postprocess
                Optional hook for post-processing a single view.
        """
        self.dataset = dataset
        self.transforms = transforms
        self.view_input_fn = view_input_fn
        if view_postprocess is None:
            view_postprocess = getattr(dataset, "postprocess_view", None)
        self.view_postprocess = view_postprocess

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped dataset."""
        return getattr(self.dataset, name)

    def _get_view_input(self, index: int) -> tuple[Any, Any, Any | None]:
        if self.view_input_fn is not None:
            return _unpack_sample(self.view_input_fn(index))

        get_view_input = getattr(type(self.dataset), "get_view_input", None)
        if callable(get_view_input):
            return _unpack_sample(get_view_input(self.dataset, index))

        return _unpack_sample(self.dataset[index])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return augmented multi-view samples for contrastive learning."""
        fname, input_obj, _ = self._get_view_input(index)
        if isinstance(input_obj, list | tuple):
            msg = "AugmentedDataset expects a single input tensor or image, got a sequence"
            raise TypeError(msg)

        views = _apply_transforms(self.transforms, input_obj)

        if self.view_postprocess is not None:
            views = tuple(self.view_postprocess(view) for view in views)

        view_names = _make_view_names(fname, len(views))

        if len(views) == 1:
            return view_names, views[0]
        return view_names, views
