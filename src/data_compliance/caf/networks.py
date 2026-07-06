"""Encoder backbones (2D and 3D ResNet-18) with optional projection heads."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18
from torchvision.models.video import r3d_18


class ResNet2D(nn.Module):
    """Grayscale ResNet-18 encoder with an optional batch-normalized projection head."""

    def __init__(self, bn_head: bool, normalize_f: bool, out_dim: int = 128) -> None:
        super().__init__()

        self.normalize_f = normalize_f
        self.bn_head = bn_head

        resnet = resnet18(pretrained=False)
        self.feats = nn.Sequential(*list(resnet.children())[:-1])

        # Modify the first convolutional layer to accept 1-channel input
        self.feats[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if self.bn_head:
            self.head = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, out_dim))

    def forward(self, x: torch.Tensor, head: bool = True) -> torch.Tensor:
        """Return projected embeddings when ``head`` is True, else the backbone features."""
        features = self.feats(x).flatten(1)
        if head:
            if self.normalize_f:
                return self.head(F.normalize(features, dim=1, p=2))
            return self.head(features)

        return features


class ResNet3D(nn.Module):
    """Volumetric ResNet-18 (r3d_18) encoder with an optional projection head."""

    def __init__(
        self, in_channels: int = 1, bn_head: bool = True, normalize_f: bool = True, out_dim: int = 128
    ) -> None:
        super().__init__()

        self.normalize_f = normalize_f
        self.bn_head = bn_head

        resnet3d = r3d_18(pretrained=False)

        # configurable input channels
        resnet3d.stem[0] = nn.Conv3d(
            in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )

        self.feats = nn.Sequential(*list(resnet3d.children())[:-1])

        if self.bn_head:
            self.head = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, out_dim))

    def forward(self, x: torch.Tensor, head: bool = True) -> torch.Tensor:
        """Return projected embeddings when ``head`` is True, else the backbone features."""
        features = self.feats(x).flatten(1)
        if head:
            feats = F.normalize(features, dim=1, p=2) if self.normalize_f else features
            return self.head(feats)
        return features
