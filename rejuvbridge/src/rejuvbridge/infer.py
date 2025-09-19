"""Embedding inference utility for RejuvBridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .models.image_encoder import ImageEncoder, ImageEncoderConfig
from .models.omics_mlp import OmicsEncoder, OmicsEncoderConfig
from .train import TileOmicsDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode tiles and omics using a trained checkpoint.")
    parser.add_argument("--tiles", type=Path, required=True, help="Directory with metadata.csv and images")
    parser.add_argument("--omics", type=Path, required=True, help="Omics parquet file")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint file from training")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    embed_dim = cfg.get("train", {}).get("embed_dim", 64)
    hidden_dim = cfg.get("train", {}).get("hidden_dim", 256)
    dropout = cfg.get("train", {}).get("dropout", 0.1)
    img_size = cfg.get("data", {}).get("img_size", 64)

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(image_embeds, dim=0), out_dir / "image_embeddings.pt")
    torch.save(torch.cat(omics_embeds, dim=0), out_dir / "omics_embeddings.pt")
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
