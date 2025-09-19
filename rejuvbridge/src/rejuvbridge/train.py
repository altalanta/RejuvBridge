"""Training entrypoint for RejuvBridge."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .metrics import ranks_from_similarity, retrieval_metrics
from .models.contrastive import cosine_similarity_matrix, info_nce_loss
from .models.image_encoder import ImageEncoder, ImageEncoderConfig
from .models.omics_mlp import OmicsEncoder, OmicsEncoderConfig


class TileOmicsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset returning paired image tensors and omics vectors."""

    def __init__(self, metadata: pd.DataFrame, omics: pd.DataFrame, img_size: int) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.omics = omics.set_index("tile_path").loc[self.metadata["tile_path"]].reset_index(drop=True)
        self.genes = [c for c in self.omics.columns if c.startswith("gene_")]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        img = Image.open(row["tile_path"]).convert("RGB")
        image_tensor = self.transform(img)
        gene_values = self.omics.loc[idx, self.genes].astype("float32").to_numpy()
        genes = torch.from_numpy(gene_values)
        return image_tensor, genes


@dataclass
class TrainArtifacts:
    metrics: dict
    checkpoint_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:  # pragma: no cover - backend not compiled
        pass


def load_metadata(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tiles_dir = Path(to_absolute_path(cfg.paths.tiles_dir))
    metadata = pd.read_csv(tiles_dir / "metadata.csv")
    omics = pd.read_parquet(to_absolute_path(cfg.paths.omics_path))
    return metadata, omics


def train_model(cfg: DictConfig) -> TrainArtifacts:
    set_seed(cfg.data.seed)
    metadata, omics = load_metadata(cfg)

    dataset = TileOmicsDataset(metadata, omics, img_size=cfg.data.img_size)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cpu")
    image_encoder = ImageEncoder(ImageEncoderConfig(embed_dim=cfg.train.embed_dim)).to(device)
    omics_encoder = OmicsEncoder(
        OmicsEncoderConfig(
            input_dim=len(dataset.genes),
            hidden_dim=cfg.train.hidden_dim,
            embed_dim=cfg.train.embed_dim,
            dropout=cfg.train.dropout,
        )
    ).to(device)

    optimizer = Adam(list(image_encoder.parameters()) + list(omics_encoder.parameters()), lr=cfg.train.lr)

    for epoch in range(cfg.train.epochs):
        image_encoder.train()
        omics_encoder.train()
        running_loss = 0.0
        for images, genes in loader:
            images = images.to(device)
            genes = genes.to(device)
            optimizer.zero_grad()
            image_embeds = image_encoder(images)
            omics_embeds = omics_encoder(genes)
            loss = info_nce_loss(image_embeds, omics_embeds, cfg.train.temperature)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item() * images.size(0))
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{cfg.train.epochs} - loss: {epoch_loss:.4f}")

    image_encoder.eval()
    omics_encoder.eval()
    with torch.no_grad():
        all_images = []
        all_omics = []
        for images, genes in DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False):
            images = images.to(device)
            genes = genes.to(device)
            all_images.append(image_encoder(images))
            all_omics.append(omics_encoder(genes))
        image_embeds = torch.cat(all_images, dim=0)
        omics_embeds = torch.cat(all_omics, dim=0)
    sim_matrix = cosine_similarity_matrix(image_embeds, omics_embeds)
    metrics = retrieval_metrics(sim_matrix)

    ckpt_dir = Path(to_absolute_path(cfg.paths.ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "clip.ckpt"
    torch.save(
        {
            "image_state_dict": image_encoder.state_dict(),
            "omics_state_dict": omics_encoder.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        ckpt_path,
    )
    return TrainArtifacts(metrics=metrics, checkpoint_path=ckpt_path)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    artifacts = train_model(cfg)
    metrics_path = Path(to_absolute_path(cfg.paths.ckpt_dir)) / "train_metrics.json"
    metrics_path.write_text(pd.Series(artifacts.metrics).to_json())
    print(f"Saved checkpoint to {artifacts.checkpoint_path}")


if __name__ == "__main__":
    main()
