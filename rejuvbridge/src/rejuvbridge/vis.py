"""Visualization helpers for RejuvBridge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def plot_umap(
    image_embeds: torch.Tensor,
    omics_embeds: torch.Tensor,
    labels: np.ndarray,
    out_path: Path,
    seed: int = 42,
) -> None:
    """Create a UMAP embedding of the joint space."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = torch.cat([image_embeds, omics_embeds], dim=0).cpu().numpy()
    try:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
        import umap

        reducer = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(combined)
        method = "UMAP"
    except Exception:  # pragma: no cover - fallback when numba unavailable
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=seed)
        embedding = reducer.fit_transform(combined)
        method = "PCA"
    modes = np.array(["image"] * len(image_embeds) + ["omics"] * len(omics_embeds))
    plt.figure(figsize=(6, 5))
    for idx, mode in enumerate(["image", "omics"]):
        mask = modes == mode
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            c=labels[mask],
            cmap="viridis",
            alpha=0.7,
            label=mode,
        )
    plt.legend()
    plt.title(f"{method} of joint embedding space")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_gradcam_overlays(
    image_encoder,
    dataset,
    omics_embeds: torch.Tensor,
    indices: Iterable[int],
    out_path: Path,
) -> None:
    """Generate Grad-CAM overlays for selected indices."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(image_encoder.parameters()).device

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    handle_f = image_encoder.backbone.layer4.register_forward_hook(forward_hook)
    handle_b = image_encoder.backbone.layer4.register_full_backward_hook(backward_hook)

    overlays = []
    for idx in indices:
        image, _ = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        target_vec = omics_embeds[idx].unsqueeze(0).to(device)
        image_encoder.zero_grad()
        embed = image_encoder(input_tensor)
        score = torch.sum(embed * target_vec)
        score.backward()
        if activations is None or gradients is None:
            continue
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        img_np = image.permute(1, 2, 0).cpu().numpy()

        overlays.append((idx, img_np, cam))

    if overlays:
        cols = len(overlays)
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
        if cols == 1:
            axes = [axes]
        for axis, (idx, img_np, cam) in zip(axes, overlays):
            axis.imshow(img_np, alpha=0.7)
            axis.imshow(cam, cmap="hot", alpha=0.5)
            axis.set_title(f"Tile {idx}")
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
    handle_f.remove()
    handle_b.remove()
