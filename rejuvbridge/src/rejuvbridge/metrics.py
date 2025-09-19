"""Metric utilities for retrieval evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def ranks_from_similarity(sim_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ranks (1-indexed) for image->omics and omics->image retrieval."""

    if sim_matrix.size(0) != sim_matrix.size(1):
        raise ValueError("Similarity matrix must be square")
    device = sim_matrix.device
    n = sim_matrix.size(0)
    targets = torch.arange(n, device=device)

    img_order = torch.argsort(sim_matrix, dim=1, descending=True)
    omics_order = torch.argsort(sim_matrix, dim=0, descending=True)

    img_positions = (img_order == targets.unsqueeze(1)).nonzero(as_tuple=False)
    omics_positions = (omics_order == targets.unsqueeze(0)).nonzero(as_tuple=False)

    img_rank = torch.zeros(n, device=device, dtype=torch.long)
    omics_rank = torch.zeros(n, device=device, dtype=torch.long)
    img_rank[img_positions[:, 0]] = img_positions[:, 1] + 1
    omics_rank[omics_positions[:, 1]] = omics_positions[:, 0] + 1
    return img_rank, omics_rank


def retrieval_at_k(ranks: torch.Tensor, k: int) -> float:
    """Compute Recall@k given 1-indexed ranks."""

    return float((ranks <= k).float().mean().item())


def retrieval_metrics(sim_matrix: torch.Tensor, ks: Iterable[int] = (1, 5)) -> Dict[str, float]:
    """Compute retrieval metrics for both directions."""

    img_rank, omics_rank = ranks_from_similarity(sim_matrix)
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = retrieval_at_k(img_rank, k)
        metrics[f"R@{k}_omics2img"] = retrieval_at_k(omics_rank, k)
    metrics["mean_rank_img2omics"] = float(img_rank.float().mean().item())
    metrics["mean_rank_omics2img"] = float(omics_rank.float().mean().item())
    return metrics


def bootstrap_retrieval(
    img_ranks: torch.Tensor,
    omics_ranks: torch.Tensor,
    n_bootstrap: int = 1000,
    seed: int = 123,
    ks: Iterable[int] = (1, 5),
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap retrieval metrics and return confidence intervals."""

    rng = np.random.default_rng(seed)
    n = img_ranks.numel()
    metrics = {k: [] for k in ks}
    metrics_omics = {k: [] for k in ks}

    img_np = img_ranks.cpu().numpy()
    omics_np = omics_ranks.cpu().numpy()

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_img = img_np[idx]
        sample_omics = omics_np[idx]
        for k in ks:
            metrics[k].append(float(np.mean(sample_img <= k)))
            metrics_omics[k].append(float(np.mean(sample_omics <= k)))

    intervals: Dict[str, Tuple[float, float]] = {}
    for k in ks:
        lower = float(np.percentile(metrics[k], 2.5))
        upper = float(np.percentile(metrics[k], 97.5))
        intervals[f"R@{k}"] = (lower, upper)
        lower_o = float(np.percentile(metrics_omics[k], 2.5))
        upper_o = float(np.percentile(metrics_omics[k], 97.5))
        intervals[f"R@{k}_omics2img"] = (lower_o, upper_o)
    return intervals
