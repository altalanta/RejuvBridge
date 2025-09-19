"""Contrastive utilities for CLIP-style training."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F
import torch


def info_nce_loss(image_embeds: Tensor, omics_embeds: Tensor, temperature: float) -> Tensor:
    """Compute symmetric InfoNCE loss between image and omics embeddings."""

    if image_embeds.shape != omics_embeds.shape:
        raise ValueError("Embedding tensors must share shape")
    logits = (image_embeds @ omics_embeds.T) / temperature
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_j) / 2


def cosine_similarity_matrix(image_embeds: Tensor, omics_embeds: Tensor) -> Tensor:
    """Return cosine similarity matrix (assumes inputs are normalized)."""

    return image_embeds @ omics_embeds.T
