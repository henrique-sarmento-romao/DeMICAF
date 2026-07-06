"""Core dataset abstractions, augmentation wrappers, and patch extraction utilities."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import Dataset, IterableDataset

from data_compliance.caf.networks import ResNet2D, ResNet3D

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pandas as pd

__all__ = [
    "AugmentedDataset",
    "ContinuousPatchDataset",
    "Dataset2D",
    "Dataset3D",
    "Dataset3DSegmentation",
    "DatasetBase",
    "PatchExtractorMixin",
    "RandomPatchDataset",
    "contrastive_collate",
    "multi_view_collate",
    "random_patch_collate",
    "sliding_patch_collate",
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


def _split_views(data: Any) -> tuple[tuple[Any, ...], bool]:
    """Ensure data is represented as a tuple of views."""
    if isinstance(data, list | tuple):
        return tuple(data), True
    return (data,), False


def _pack_views(views: tuple[Any, ...], was_multi: bool) -> Any:
    """Return views as a tuple or a single item depending on original format."""
    if was_multi:
        return views
    return views[0]


def _append_patch_suffix(fname: Any, suffix: Any, n_views: int) -> Any:
    """Append patch suffixes to view names while preserving view structure."""
    if isinstance(fname, list | tuple):
        if len(fname) != n_views:
            msg = "If fname is a sequence, it must match number of views"
            raise ValueError(msg)
        return tuple(f"{name}_{suffix}" for name in fname)
    return f"{fname}_{suffix}"


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


class Dataset3D(DatasetBase):
    """Base class for 3D medical imaging datasets."""

    __slots__ = ("crop",)

    NETWORK = ResNet3D

    def __init__(
        self,
        root_dir: str | Path,
        dataframe: pd.DataFrame | None,
        transform: Any | None = None,
        crop: bool | None = None,
    ) -> None:
        """Initialize the 2D dataset.

        Parameters
        ----------
        root_dir
                Root directory containing the dataset assets.
        dataframe
                Metadata table backing the dataset.
        transform
                Optional transform applied to a single sample.
        crop
                Whether to crop volumes to the region of interest.
        """
        self.crop = True if crop is None else crop
        super().__init__(root_dir=root_dir, dataframe=dataframe, transform=transform)
        self._validate_patch_attributes()

    def _validate_patch_attributes(self) -> None:
        """Ensure subclasses define patch extraction attributes."""
        required_attrs = ("patch_size", "patches_per_volume", "patch_stride")
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            msg = "Dataset3D subclasses must define patch attributes: " + ", ".join(missing)
            raise NotImplementedError(msg)

    def _crop_volume(
        self, volume: torch.Tensor, bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ) -> torch.Tensor:
        """
        Crop a 3D volume using provided bounds with margin.

        Parameters
        ----------
        volume
                Volume tensor of shape (C, D, H, W).
        bounds
                Spatial bounds as ((minx, miny, minz), (maxx, maxy, maxz)).
        """
        if bounds is None:
            return volume

        mini = list(bounds[0])
        maxi = list(bounds[1])

        dist = 0.0
        for i in range(3):
            dist += (maxi[i] - mini[i]) ** 2
        margin = int(0.05 * float(np.sqrt(dist)))

        for i in range(3):
            mini[i] = max(0, mini[i] - margin)
            maxi[i] = min(maxi[i] + margin, volume.shape[i + 1])

        min_x, min_y, min_z = mini
        max_x, max_y, max_z = maxi

        return volume[:, min_x:max_x, min_y:max_y, min_z:max_z]

    @abstractmethod
    def get_spacing(self, study_id: str) -> tuple[float, float, float]:
        """Return voxel spacing for a specific study."""


class Dataset3DSegmentation(Dataset3D):
    """Base class for 3D segmentation datasets."""

    __slots__ = ("input_mode", "segmentations")

    def __init__(
        self,
        root_dir: str | Path,
        dataframe: pd.DataFrame | None,
        segmentations: list[str],
        input_mode: str = "Segmentation",
        transform: Any | None = None,
        crop: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the 3D dataset.

        Parameters
        ----------
        root_dir
                Root directory containing the dataset assets.
        dataframe
                Metadata table backing the dataset.
        segmentations
                Segmentation names to load for each entry.
        input_mode
                Input type ("Segmentation", "Image", or "Border").
        transform
                Optional transform applied to a single sample.
        crop
                Whether to crop volumes to the region of interest.
        **kwargs
                Backward-compatible aliases such as `input` for `input_mode`.
        """
        if "input" in kwargs:
            input_mode = str(kwargs.pop("input"))
        if kwargs:
            msg = f"Unexpected keyword arguments: {', '.join(sorted(kwargs))}"
            raise TypeError(msg)
        self.segmentations = segmentations
        self.input_mode = input_mode
        super().__init__(root_dir=root_dir, dataframe=dataframe, transform=transform, crop=crop)

    def get_input_channels(self) -> int:
        """Return the number of input channels for the configured input mode."""
        if self.input_mode == "Segmentation":
            return 1
        if self.input_mode == "Image":
            return 2
        if self.input_mode == "Border":
            return 3
        msg = f"Unknown input type: {self.input_mode}"
        raise ValueError(msg)

    def get_network(self, **network_kwargs: Any) -> Any:
        """Return an initialized network with the correct input channels."""
        in_channels = self.get_input_channels()
        return super().get_network(in_channels=in_channels, **network_kwargs)

    @abstractmethod
    def _load_image(self, path: Path) -> torch.Tensor:
        """Load a medical image volume for an entry."""

    @abstractmethod
    def _load_segmentation(self, seg_paths: list[Path]) -> torch.Tensor:
        """Load and merge segmentation masks for an entry."""

    def _compute_border(self, seg_tensor: np.ndarray, img_tensor: torch.Tensor) -> torch.Tensor:
        """Compute a border tensor from the segmentation using morphology."""
        dil = binary_dilation(seg_tensor, iterations=3)
        ero = binary_erosion(seg_tensor, iterations=3)

        dil_t = torch.from_numpy(dil).to(img_tensor.device)
        ero_t = torch.from_numpy(ero).to(img_tensor.device)
        return (dil_t ^ ero_t) * img_tensor

    def process_entry(self, seg: torch.Tensor, img: torch.Tensor | None) -> torch.Tensor:
        """
        Build the final input tensor for a segmentation entry.

        Parameters
        ----------
        seg
                Segmentation tensor of shape (C, D, H, W).
        img
                Optional image tensor of shape (C, D, H, W).
        """
        border: torch.Tensor | None = None
        if self.input_mode == "Border" and img is not None:
            border = self._compute_border(seg.numpy(), img)

        if self.input_mode == "Segmentation":
            return seg
        if self.input_mode == "Image" and img is not None:
            img_cast = img.to(seg.dtype) if img.dtype != seg.dtype else img
            return torch.cat((seg, img_cast), dim=0)
        if self.input_mode == "Border" and img is not None and border is not None:
            img_cast = img.to(seg.dtype) if img.dtype != seg.dtype else img
            border_cast = border.to(seg.dtype) if border.dtype != seg.dtype else border
            return torch.cat((seg, img_cast, border_cast), dim=0)
        return seg


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


DatasetAugmented = AugmentedDataset


class PatchExtractorMixin:
    """Mixin providing common patch extraction utilities.

    Concrete users must define ``patch_size`` (and the patcher-specific stride /
    per-volume attributes).
    """

    __slots__ = ()

    patch_size: tuple[int, int, int]

    def _pad_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Pad a patch to reach the required patch size."""
        p_d, p_h, p_w = self.patch_size
        pad_z = max(0, p_d - patch.shape[1])
        pad_y = max(0, p_h - patch.shape[2])
        pad_x = max(0, p_w - patch.shape[3])

        if pad_z or pad_y or pad_x:
            patch = torch.nn.functional.pad(patch, (0, pad_x, 0, pad_y, 0, pad_z))
        return patch

    @staticmethod
    def get_patch_bounds(
        volume: torch.Tensor, patch_size: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Compute bounds for extracting a random patch from a volume."""
        _, depth, height, width = volume.shape
        p_d, p_h, p_w = patch_size

        min_z, min_y, min_x = 0, 0, 0
        max_z = max(0, depth - p_d)
        max_y = max(0, height - p_h)
        max_x = max(0, width - p_w)

        return (min_z, min_y, min_x), (max_z, max_y, max_x)

    def get_patch(
        self,
        volume: torch.Tensor,
        bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    ) -> torch.Tensor:
        """Extract a patch from a volume at a random location."""
        _, depth, height, width = volume.shape
        p_d, p_h, p_w = self.patch_size

        if bounds is None:
            min_z, min_y, min_x = 0, 0, 0
            max_z, max_y, max_x = depth - p_d, height - p_h, width - p_w
        else:
            (min_z, min_y, min_x), (max_z, max_y, max_x) = bounds
            min_z = max(0, min(min_z, depth - p_d))
            min_y = max(0, min(min_y, height - p_h))
            min_x = max(0, min(min_x, width - p_w))
            max_z = max(0, min(max_z, depth - p_d))
            max_y = max(0, min(max_y, height - p_h))
            max_x = max(0, min(max_x, width - p_w))

        z0 = min_z if max_z <= min_z else random.randint(min_z, max_z)
        y0 = min_y if max_y <= min_y else random.randint(min_y, max_y)
        x0 = min_x if max_x <= min_x else random.randint(min_x, max_x)

        patch = volume[:, z0 : z0 + p_d, y0 : y0 + p_h, x0 : x0 + p_w]
        return self._pad_patch(patch)


class ContinuousPatchDataset(IterableDataset[Any], PatchExtractorMixin):
    """Iterable dataset that emits sliding-window patches from volumes."""

    __slots__ = ("ds", "patch_size", "stride")

    def __init__(self, dataset: Dataset[Any]) -> None:
        """Initialize the sliding-window patch dataset.

        Parameters
        ----------
        dataset
                Base dataset to extract patches from.
        """
        self.ds = dataset
        self.patch_size = tuple(dataset.patch_size)  # type: ignore[attr-defined]
        self.stride = tuple(dataset.patch_stride)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped dataset."""
        return getattr(self.ds, name)

    def __iter__(self) -> Iterator[tuple[Any, Any] | tuple[Any, Any, Any]]:
        """Yield sliding-window patches across the dataset."""
        for idx in range(len(self.ds)):  # type: ignore[arg-type]
            fname, data, label = _unpack_sample(self.ds[idx])
            views, was_multi = _split_views(data)
            seg = views[0]
            _, depth, height, width = seg.shape
            p_d, p_h, p_w = self.patch_size
            s_d, s_h, s_w = self.stride

            z_starts = list(range(0, max(1, depth - p_d + 1), s_d))
            y_starts = list(range(0, max(1, height - p_h + 1), s_h))
            x_starts = list(range(0, max(1, width - p_w + 1), s_w))

            for z_idx, z in enumerate(z_starts):
                for y_idx, y in enumerate(y_starts):
                    for x_idx, x in enumerate(x_starts):
                        patched_views = tuple(
                            self._pad_patch(view[:, z : z + p_d, y : y + p_h, x : x + p_w]) for view in views
                        )
                        suffix = f"{x_idx}{y_idx}{z_idx}"
                        patch_name = _append_patch_suffix(fname, suffix, len(views))
                        patch_data = _pack_views(patched_views, was_multi)
                        if label is None:
                            yield patch_name, patch_data
                        else:
                            labels_out = _normalize_labels(label, len(views)) if was_multi else label
                            yield patch_name, patch_data, labels_out


class RandomPatchDataset(Dataset[Any], PatchExtractorMixin):
    """Map-style dataset that extracts random patches from volumes."""

    __slots__ = ("_cache", "cache_size", "ds", "patch_size", "patches_per_volume")

    def __init__(self, dataset: Dataset[Any], cache_size: int = 1) -> None:
        """Initialize the patch extractor.

        Parameters
        ----------
        dataset
                Base dataset to extract patches from.
        cache_size
                Number of volumes to keep in-memory per worker.
        """
        self.ds = dataset
        self.patch_size = tuple(dataset.patch_size)  # type: ignore[attr-defined]
        self.patches_per_volume = dataset.patches_per_volume  # type: ignore[attr-defined]
        self.cache_size = max(1, cache_size)
        self._cache: dict[int, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped dataset."""
        return getattr(self.ds, name)

    def __len__(self) -> int:
        """Return the total number of patches across all volumes."""
        return len(self.ds) * self.patches_per_volume  # type: ignore[arg-type]

    @property
    def transforms(self) -> Any | None:
        """Forward transforms from the wrapped dataset."""
        return getattr(self.ds, "transforms", None)

    def __getitem__(self, index: int) -> tuple[Any, Any] | tuple[Any, Any, Any]:
        """Extract a random patch from the wrapped dataset."""
        volume_idx = index // self.patches_per_volume
        patch_num = index % self.patches_per_volume

        fname, data, label = _unpack_sample(self.ds[volume_idx])
        views, was_multi = _split_views(data)

        bounds = self.get_patch_bounds(views[0], self.patch_size)
        patched_views = tuple(self.get_patch(view, bounds=bounds) for view in views)

        patch_name = _append_patch_suffix(fname, patch_num, len(views))
        patch_data = _pack_views(patched_views, was_multi)

        if label is None:
            return patch_name, patch_data

        labels_out = _normalize_labels(label, len(views)) if was_multi else label
        return patch_name, patch_data, labels_out


contrastive_collate = multi_view_collate


def random_patch_collate(batch: list[Any], patching: dict[str, Any], framework: str) -> tuple[list[Any], torch.Tensor]:
    """
    Legacy collate that extracts random patches from transformed volumes.

    Parameters
    ----------
    batch
            List of samples yielded by the dataset.
    patching
            Patching configuration with `size` and `number`.
    framework
            Framework name ("SimCLR" or similar).
    """
    fnames: list[Any] = []
    images: list[torch.Tensor] = []
    patch_size = patching["size"]
    patch_num = patching["number"]

    class _TempPatcher(PatchExtractorMixin):
        __slots__ = ("patch_size",)

        def __init__(self, size: tuple[int, int, int]) -> None:
            self.patch_size = size

    temp_patcher = _TempPatcher(patch_size)

    for subj in batch:
        if framework == "SimCLR":
            fname, (vol_id_a, vol_id_b) = subj
        else:
            fname, (vol_id_a, vol_id_b, vol_ood_a, vol_ood_b) = subj

        for i in range(patch_num):
            bounds = PatchExtractorMixin.get_patch_bounds(vol_id_a, patch_size)

            patch_id_a = temp_patcher.get_patch(vol_id_a, bounds=bounds)
            fnames.append(f"{fname}_{i}")
            images.append(patch_id_a)

            patch_id_b = temp_patcher.get_patch(vol_id_b, bounds=bounds)
            fnames.append(f"{fname}_{i}")
            images.append(patch_id_b)

            if framework != "SimCLR":
                patch_ood_a = temp_patcher.get_patch(vol_ood_a, bounds=bounds)
                fnames.append(f"{fname}_{i}_shifted")
                images.append(patch_ood_a)

                patch_ood_b = temp_patcher.get_patch(vol_ood_b, bounds=bounds)
                fnames.append(f"{fname}_{i}_shifted")
                images.append(patch_ood_b)

    return fnames, torch.stack(images, dim=0)


def sliding_patch_collate(batch: list[Any], patching: dict[str, Any]) -> tuple[list[Any], torch.Tensor]:
    """
    Legacy collate that extracts sliding-window patches from volumes.

    Parameters
    ----------
    batch
            List of samples yielded by the dataset.
    patching
            Patching configuration with `size` and optional `stride`.
    """
    fnames: list[Any] = []
    images: list[torch.Tensor] = []
    patch_size = patching["size"]
    stride = patching.get("stride", patch_size)

    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)

    p_d, p_h, p_w = patch_size
    s_d, s_h, s_w = stride

    class _TempPatcher(PatchExtractorMixin):
        __slots__ = ("patch_size",)

        def __init__(self, size: tuple[int, int, int]) -> None:
            self.patch_size = size

    temp_patcher = _TempPatcher(patch_size)

    for subj in batch:
        fname, segmentation = subj
        _, depth, height, width = segmentation.shape

        starts_z = list(range(0, max(1, depth - p_d + 1), s_d))
        starts_y = list(range(0, max(1, height - p_h + 1), s_h))
        starts_x = list(range(0, max(1, width - p_w + 1), s_w))

        patches = []
        for z in starts_z:
            for y in starts_y:
                for x in starts_x:
                    patch = temp_patcher._pad_patch(  # noqa: SLF001 — helper patcher of the same class family
                        segmentation[:, z : z + p_d, y : y + p_h, x : x + p_w]
                    )
                    patches.append(patch)

        for idx, patch in enumerate(patches):
            fnames.append(f"{fname}_{idx}")
            images.append(patch)

    return fnames, torch.stack(images, dim=0)
