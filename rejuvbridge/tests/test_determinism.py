from pathlib import Path

from omegaconf import OmegaConf

from rejuvbridge.data.synth_omics import generate_omics
from rejuvbridge.data.synth_tiles import generate_tiles
from rejuvbridge.train import train_model


def test_training_determinism(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    omics_path = tmp_path / "omics.parquet"
    generate_tiles(tiles_dir, n_tiles=64, img_size=32, seed=7, n_classes=4)
    generate_omics(tiles_dir, omics_path, n_genes=32, seed=7)

    cfg = OmegaConf.create(
        {
            "data": {"seed": 123, "img_size": 32},
            "paths": {
                "tiles_dir": str(tiles_dir),
                "omics_path": str(omics_path),
                "ckpt_dir": str(tmp_path / "ckpt"),
            },
            "train": {
                "batch_size": 16,
                "epochs": 2,
                "lr": 1e-3,
                "embed_dim": 16,
                "hidden_dim": 32,
                "dropout": 0.1,
                "temperature": 0.1,
            },
        }
    )

    metrics_a = train_model(cfg).metrics
    metrics_b = train_model(cfg).metrics

    for key in ("R@1", "R@5"):
        assert metrics_a[key] == metrics_b[key]
