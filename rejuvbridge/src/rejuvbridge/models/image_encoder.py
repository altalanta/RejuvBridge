"""ResNet18-based image encoder for RejuvBridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


@dataclass
class ImageEncoderConfig:
    embed_dim: int = 64


class ImageEncoder(nn.Module):
    """Wraps a torchvision ResNet-18 and projects to L2-normalized embeddings."""

    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features  # type: ignore[assignment]
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(in_features, config.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(images)
        embeds = self.projection(features)
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds
