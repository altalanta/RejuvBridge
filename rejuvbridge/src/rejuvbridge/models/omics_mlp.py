"""Multi-layer perceptron for omics embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class OmicsEncoderConfig:
    input_dim: int
    hidden_dim: int = 256
    embed_dim: int = 64
    dropout: float = 0.1


class OmicsEncoder(nn.Module):
    """Simple MLP mapping gene vectors into the shared embedding space."""

    def __init__(self, config: OmicsEncoderConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.network(features)
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds
