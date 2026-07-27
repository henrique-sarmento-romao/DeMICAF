"""Pathology classifier shared by the federated training pipeline.

A grayscale-adapted ResNet-18/152 multi-label classifier used as the FedAvg
baseline pathology model (``scripts/federated/train.py``), whose penultimate
features and per-label logits feed the per-cause benchmark's ``knn``/
``maha_pp``/``gen`` baselines. ``federated.server`` implements its own
centralized training loop, so only the FL-path pieces live here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet152_Weights

if TYPE_CHECKING:
    import pandas as pd
    from torch.utils.data import DataLoader

train_tfm = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

val_tfm = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def build_model(
    *,
    pretrained: bool = True,
    num_labels: int = 6,
    architecture: str = "resnet18",
    dropout_rate: float = 0.1,
) -> nn.Module:
    """Build a ResNet model adapted for single-channel CXR inputs.

    Parameters
    ----------
    pretrained
        Whether to initialize from ImageNet weights.
    num_labels
        Number of output labels.
    architecture
        Backbone architecture: "resnet18" or "resnet152".
    dropout_rate
        Dropout probability in the classifier head (0.1-0.25 recommended).
    """
    arch = architecture.lower()
    if arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet152(weights=weights)
    else:
        msg = f"Unsupported architecture: {architecture!r}. Choose 'resnet18' or 'resnet152'."
        raise ValueError(msg)

    conv1_rgb = model.conv1
    conv1_gray = nn.Conv2d(1, conv1_rgb.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        conv1_gray.weight.copy_(conv1_rgb.weight.mean(dim=1, keepdim=True))
    model.conv1 = conv1_gray

    in_features = model.fc.in_features
    head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Dropout(dropout_rate),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.Dropout(dropout_rate),
        nn.ReLU(),
        nn.Linear(128, num_labels),
    )
    model.fc = head
    return model


def compute_pos_weights(df: pd.DataFrame, label_cols: list[str]) -> torch.Tensor:
    """Compute per-label positive weights for class imbalance handling."""
    pos = (df[label_cols] == 1).sum()
    neg = (df[label_cols] == 0).sum()
    weights = (neg / pos.clip(lower=1)).clip(upper=50)
    return torch.tensor(weights.values, dtype=torch.float32)


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute masked BCE loss to handle uncertain labels."""
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device), reduction="none")
    losses = loss_fn(logits, targets)
    return (losses * masks).sum() / masks.sum().clamp(min=1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label_cols: list[str],
) -> dict[str, float]:
    """Compute mean AUC across labels."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    with torch.no_grad():
        for _, imgs, targets in loader:
            labels, masks = targets
            logits = model(imgs.to(device)).cpu()
            all_logits.append(torch.sigmoid(logits).numpy())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())

    probs = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_masks)
    aucs: dict[str, float] = {}
    for idx, col in enumerate(label_cols):
        valid = masks[:, idx] == 1
        if valid.sum() < 2:
            continue
        y_true = labels[valid, idx]
        y_pred = probs[valid, idx]
        if (y_true == 1).sum() > 0 and (y_true != 1).sum() > 0:
            aucs[col] = float(roc_auc_score(y_true, y_pred))
    aucs["mean"] = float(np.mean(list(aucs.values()))) if aucs else 0.0
    return aucs


__all__ = ["build_model", "compute_pos_weights", "evaluate", "masked_bce_loss", "train_tfm", "val_tfm"]
