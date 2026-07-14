"""
Pure PIL/numpy implementations of 2D CXR transforms.

Input can be tensor, PIL image, or numpy array. Transforms operate on grayscale
2D arrays and return 2D numpy arrays to integrate with framework `Transforms`.
"""

import random
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageOps

ImageLike = torch.Tensor | np.ndarray | Image.Image | Any


class Transforms:
    """Container pairing in-distribution (ID) and out-of-distribution (OOD) augmentations."""

    class ID:
        """Apply the full ID pipeline, randomly dropping one transform 25% of the time."""

        def __init__(self, transforms: list[Any]) -> None:
            self.transforms = transforms

        def __call__(self, image: ImageLike) -> torch.Tensor:
            """Apply the ID augmentation pipeline to one image."""
            # Keep dtype only when input is a tensor; numpy/PIL dtypes are not valid for Tensor.to(dtype).
            input_dtype = image.dtype if isinstance(image, torch.Tensor) else None

            arr = to_numpy(image)

            # remove one transform randomly
            transforms = self.transforms.copy()
            if random.random() < 0.25:
                transforms.pop(random.randrange(len(transforms)))

            # apply transforms
            out = arr
            for T in transforms:
                out = T(out)

            result = to_tensor(out)
            # Ensure output has same dtype as input
            if input_dtype is not None and result.dtype != input_dtype:
                result = result.to(input_dtype)
            return result

    class OOD:
        """Apply one randomly chosen OOD (non-compliance-simulating) transform."""

        def __init__(self, transforms: list[Any]) -> None:
            self.transforms = transforms

        def __call__(self, image: ImageLike) -> torch.Tensor:
            """Apply one random OOD transform to one image."""
            # Keep dtype only when input is a tensor; numpy/PIL dtypes are not valid for Tensor.to(dtype).
            input_dtype = image.dtype if isinstance(image, torch.Tensor) else None

            arr = to_numpy(image)

            # apply random transform
            T = random.choice(self.transforms)
            out = T(arr)

            result = to_tensor(out)
            # Ensure output has same dtype as input
            if input_dtype is not None and result.dtype != input_dtype:
                result = result.to(input_dtype)
            return result

    def __init__(self, transforms_id: list[Any], transforms_ood: list[Any]) -> None:
        # Instance attributes intentionally shadow the nested classes of the same name.
        self.ID = Transforms.ID(transforms_id)  # type: ignore[misc,assignment]
        self.OOD = Transforms.OOD(transforms_ood)  # type: ignore[misc,assignment]


def to_numpy(x: ImageLike) -> np.ndarray:
    """Convert tensor/PIL/array input to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def to_tensor(x: ImageLike) -> torch.Tensor:
    """Convert numpy array to torch tensor, preserving dtype.

    Converts bool to uint8. Does not modify dimensions.
    """
    arr = np.asarray(x)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr)


def ensure_2d(image: ImageLike) -> np.ndarray:
    """Ensure image is 2D grayscale, squeezing channel dimension if present."""
    arr = to_numpy(image)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Accept (1, H, W), (H, W, 1), or multi-channel images.
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.shape[-1] in (3, 4):
            return arr[..., 0]
        if arr.shape[0] in (3, 4):
            return arr[0]

    raise ValueError(f"Expected 2D image, got shape {arr.shape}")


def as_uint8(image: ImageLike) -> np.ndarray:
    """Convert image to uint8 grayscale range [0, 255]."""
    arr = ensure_2d(image)
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def to_pil(image: ImageLike) -> Image.Image:
    """Convert image-like input to PIL grayscale image."""
    return Image.fromarray(as_uint8(image), mode="L")


def from_pil(image: Image.Image) -> np.ndarray:
    """Convert PIL grayscale image to numpy uint8."""
    return np.asarray(image.convert("L"), dtype=np.uint8)


def resize_with_random_padding(
    image: Image.Image, new_size: tuple[int, int], target_size: tuple[int, int]
) -> Image.Image:
    """Resize image and pad randomly to target size, then center-crop if needed."""
    img = image.resize(new_size, Image.Resampling.BILINEAR)

    pad_h = max(target_size[1] - new_size[1], 0)
    pad_w = max(target_size[0] - new_size[0], 0)

    left = random.randint(0, pad_w)
    top = random.randint(0, pad_h)
    right = pad_w - left
    bottom = pad_h - top

    img = ImageOps.expand(img, (left, top, right, bottom), fill=0)
    return img.crop((0, 0, target_size[0], target_size[1]))


def _resize_256(image: Image.Image) -> Image.Image:
    """Match legacy preprocessing: resize shortest side target to 256 with bilinear interpolation."""
    return transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)(image)


# ID TRANSFORMS ========================================================
def maybe_zoom_out(
    image: ImageLike, scale_range: tuple[float, float], size: tuple[int, int], prob: float
) -> np.ndarray:
    """Randomly zoom out using resize+padding, aligned with torchvision preprocessing style."""
    img = _resize_256(to_pil(image))
    if random.random() >= prob:
        out = transforms.CenterCrop(size)(img)
        return from_pil(out)

    scale_factor = random.uniform(*scale_range)
    new_w = max(1, int(size[0] / scale_factor))
    new_h = max(1, int(size[1] / scale_factor))
    small = transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)(img)

    pad_w = max(size[0] - new_w, 0)
    pad_h = max(size[1] - new_h, 0)
    left = random.randint(0, pad_w)
    top = random.randint(0, pad_h)
    right = pad_w - left
    bottom = pad_h - top
    padded = transforms.Pad((left, top, right, bottom), fill=0)(small)
    out = transforms.CenterCrop(size)(padded)
    return from_pil(out)


def zoom_in(image: ImageLike, scale_range: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    """Randomly zoom in via torchvision RandomResizedCrop semantics."""
    img = _resize_256(to_pil(image))
    # Convert zoom-in factors (e.g., 1.05-1.4) to crop area scales.
    crop_scale = (1.0 / max(scale_range), 1.0 / min(scale_range))
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        img,
        scale=list(crop_scale),
        ratio=[0.95, 1.05],
    )
    out = TF.resized_crop(
        img,
        top=i,
        left=j,
        height=h,
        width=w,
        size=size,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    return from_pil(out)


def random_flip(image: ImageLike, prob: float) -> np.ndarray:
    """Apply random horizontal flip."""
    img = to_pil(image)
    out = transforms.RandomHorizontalFlip(p=prob)(img)
    return from_pil(out)


def random_rotation(image: ImageLike, degrees: tuple[float, float] | float) -> np.ndarray:
    """Apply random in-plane rotation."""
    img = to_pil(image)
    out = transforms.RandomRotation(
        degrees=degrees,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=0,
    )(img)
    return from_pil(out)


def color_jitter(
    image: ImageLike, brightness: tuple[float, float] | float, contrast: tuple[float, float] | float
) -> np.ndarray:
    """Apply random brightness and contrast change."""
    img = to_pil(image)
    out = transforms.ColorJitter(brightness=brightness, contrast=contrast)(img)
    return from_pil(out)


def blur(image: ImageLike, sigma: tuple[float, float] | float) -> np.ndarray:
    """Apply Gaussian blur with random sigma."""
    img = to_pil(image)
    out = transforms.GaussianBlur(kernel_size=3, sigma=sigma)(img)
    return from_pil(out)


# OOD TRANSFORMS =======================================================
def extreme_resized_crop(image: ImageLike, size: tuple[int, int]) -> np.ndarray:
    """Apply aggressive zoom-in crop or extreme zoom-out padding."""
    img = _resize_256(to_pil(image))
    option = random.choice(["zoom_in", "zoom_out"])

    if option == "zoom_in":
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img,
            scale=[0.1, 0.6],
            ratio=[0.5, 2.0],
        )
        out = TF.resized_crop(
            img,
            top=i,
            left=j,
            height=h,
            width=w,
            size=size,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        return from_pil(out)

    scale_factor = random.uniform(2.5, 3.0)
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
    out = resize_with_random_padding(img, new_size, size)
    return from_pil(out)


def slice_shift(image: ImageLike, num_slices: tuple[int, int] = (2, 4), dislocation_factor: float = 0.3) -> np.ndarray:
    """Split image into stripes and shift each stripe independently."""
    img = to_pil(image)
    direction = random.choice(["horizontal", "vertical"])
    n_slices = random.randint(*num_slices)

    slices = []
    span = (img.width if direction == "vertical" else img.height) // n_slices
    for i in range(n_slices):
        start = i * span
        end = start + span if i < n_slices - 1 else (img.width if direction == "vertical" else img.height)
        if direction == "horizontal":
            slices.append(img.crop((0, start, img.width, end)))
        else:
            slices.append(img.crop((start, 0, end, img.height)))

    shifted = []
    for piece in slices:
        if direction == "horizontal":
            shift = random.randint(-int(dislocation_factor * piece.size[0]), int(dislocation_factor * piece.size[0]))
            params = (1, 0, shift, 0, 1, 0)
        else:
            shift = random.randint(-int(dislocation_factor * piece.size[1]), int(dislocation_factor * piece.size[1]))
            params = (1, 0, 0, 0, 1, shift)
        shifted.append(piece.transform(piece.size, Image.Transform.AFFINE, params))

    out = Image.new("L", img.size)
    for idx, piece in enumerate(shifted):
        if direction == "horizontal":
            out.paste(piece, (0, idx * piece.height))
        else:
            out.paste(piece, (idx * piece.width, 0))

    return from_pil(out)


def adjust_exposure(image: ImageLike) -> np.ndarray:
    """Apply overexposure or underexposure artifact."""
    img = to_pil(image)
    choice = random.choice(["overexposure", "underexposure"])

    if choice == "overexposure":
        out = transforms.ColorJitter(
            brightness=(2.0, 5.0),
            contrast=(0.1, 0.5),
        )(img)
    else:
        out = transforms.ColorJitter(
            brightness=(0.02, 0.05),
            contrast=(1.5, 2.0),
        )(img)

    return from_pil(out)


def reduce_gray_levels(image: ImageLike, levels: int = 4) -> np.ndarray:
    """Quantize grayscale values to a reduced set of levels."""
    arr = as_uint8(image)
    step = max(1, 256 // levels)
    return ((arr // step) * step).astype(np.uint8)


def add_random_objects(image: ImageLike) -> np.ndarray:
    """Overlay random bright/dark geometric objects."""
    img = to_pil(image)
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    total_area = width * height

    for _ in range(random.randint(1, 3)):
        target_area = random.uniform(0.05, 0.3) * total_area
        aspect_ratio = random.uniform(0.5, 2.0)
        obj_w = int((target_area * aspect_ratio) ** 0.5)
        obj_h = int(target_area / max(obj_w, 1))
        obj_w = min(max(1, obj_w), width)
        obj_h = min(max(1, obj_h), height)
        x1 = random.randint(0, width - obj_w)
        y1 = random.randint(0, height - obj_h)
        x2, y2 = x1 + obj_w, y1 + obj_h
        color = random.randint(0, 255)
        if random.choice(["rectangle", "ellipse"]) == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)

    return from_pil(canvas)


def partial_illumination(image: ImageLike) -> np.ndarray:
    """Replace one random region with noise to emulate localized illumination shifts."""
    img = to_pil(image)
    width, height = img.size

    min_area = int(0.3 * width * height)
    max_area = int(0.8 * width * height)
    target_area = random.randint(min_area, max_area)

    aspect_ratio = 1.0 if random.choice([True, False]) else random.uniform(0.5, 2.0)
    target_w = int((target_area * aspect_ratio) ** 0.5)
    target_h = int(target_area / max(target_w, 1))
    target_w = min(max(1, target_w), width)
    target_h = min(max(1, target_h), height)

    x1 = random.randint(0, width - target_w)
    y1 = random.randint(0, height - target_h)
    x2, y2 = x1 + target_w, y1 + target_h

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    noise = Image.effect_noise((width, height), 64)

    out = Image.composite(img, noise, mask)
    return from_pil(out)
