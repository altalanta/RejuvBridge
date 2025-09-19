"""Evaluation script for RejuvBridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .metrics import bootstrap_retrieval, ranks_from_similarity, retrieval_metrics
from .models.contrastive import cosine_similarity_matrix
from .models.image_encoder import ImageEncoder, ImageEncoderConfig
from .models.omics_mlp import OmicsEncoder, OmicsEncoderConfig
from .train import TileOmicsDataset, load_metadata, set_seed
from .vis import plot_umap, save_gradcam_overlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained RejuvBridge model.")
    parser.add_argument("--tiles", type=Path, required=True, help="Directory containing synthetic tiles")
    parser.add_argument("--omics", type=Path, required=True, help="Parquet file with omics data")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--seed", type=int, default=123, help="Bootstrap seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    embed_dim = cfg.get("train", {}).get("embed_dim", 64)
    hidden_dim = cfg.get("train", {}).get("hidden_dim", 256)
    dropout = cfg.get("train", {}).get("dropout", 0.1)
    img_size = cfg.get("data", {}).get("img_size", 64)
    seed = cfg.get("data", {}).get("seed", 42)
    set_seed(seed)

    metadata = pd.read_csv(Path(args.tiles) / "metadata.csv")
    omics = pd.read_parquet(args.omics)
    dataset = TileOmicsDataset(metadata, omics, img_size=img_size)

    image_encoder = ImageEncoder(ImageEncoderConfig(embed_dim=embed_dim))
    image_encoder.load_state_dict(ckpt["image_state_dict"])
    omics_encoder = OmicsEncoder(
        OmicsEncoderConfig(
            input_dim=len(dataset.genes),
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
    )
    omics_encoder.load_state_dict(ckpt["omics_state_dict"])
    image_encoder.eval()
    omics_encoder.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    image_embeds = []
    omics_embeds = []
    with torch.no_grad():
        for images, genes in loader:
            image_embeds.append(image_encoder(images))
            omics_embeds.append(omics_encoder(genes))
    image_embeds_t = torch.cat(image_embeds, dim=0)
    omics_embeds_t = torch.cat(omics_embeds, dim=0)

    sim_matrix = cosine_similarity_matrix(image_embeds_t, omics_embeds_t)
    metrics = retrieval_metrics(sim_matrix)
    img_ranks, omics_ranks = ranks_from_similarity(sim_matrix)
    ci = bootstrap_retrieval(img_ranks, omics_ranks, seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        **metrics,
        "R@1_ci": ci["R@1"],
        "R@5_ci": ci["R@5"],
        "R@1_omics2img_ci": ci["R@1_omics2img"],
        "R@5_omics2img_ci": ci["R@5_omics2img"],
    }
    (out_dir / "retrieval.json").write_text(json.dumps(metrics_payload, indent=2))

    labels = metadata["class_id"].to_numpy()
    stacked_labels = np.concatenate([labels, labels])
    plot_umap(image_embeds_t, omics_embeds_t, stacked_labels, out_dir / "umap.png")

    top_indices = torch.topk(sim_matrix.diagonal(), k=min(2, sim_matrix.shape[0])).indices.tolist()
    save_gradcam_overlays(image_encoder, dataset, omics_embeds_t, top_indices, out_dir / "gradcam.png")


if __name__ == "__main__":
    main()
