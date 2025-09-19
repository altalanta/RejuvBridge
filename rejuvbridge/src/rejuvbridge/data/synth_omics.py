"""Synthetic omics generator aligned to histology tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_omics(
    tiles_dir: Path,
    out_path: Path,
    n_genes: int,
    seed: int,
    class_signal_scale: float = 2.5,
    noise_scale: float = 0.5,
) -> pd.DataFrame:
    """Generate gene expression vectors aligned with tile metadata."""

    tiles_dir = Path(tiles_dir)
    metadata_path = tiles_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "metadata.csv not found. Run synth_tiles.py before synth_omics.py or pass the correct directory."
        )

    metadata = pd.read_csv(metadata_path)
    n_classes = metadata["class_id"].nunique()

    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]

    class_means = rng.normal(0.0, class_signal_scale, size=(n_classes, n_genes))
    records = []
    for _, row in metadata.iterrows():
        class_id = int(row["class_id"])
        base = class_means[class_id]
        expression = base + rng.normal(0.0, noise_scale, size=n_genes)
        records.append(
            {
                "tile_id": row["tile_id"],
                "class_id": class_id,
                "tile_path": row["tile_path"],
                **{gene: expr for gene, expr in zip(gene_names, expression)},
            }
        )

    df = pd.DataFrame(records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic omics aligned to tiles.")
    parser.add_argument("--tiles", type=Path, required=True, help="Directory containing metadata.csv from synth_tiles")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet file path")
    parser.add_argument("--n_genes", type=int, default=200, help="Number of genes per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_omics(args.tiles, args.out, args.n_genes, args.seed)


if __name__ == "__main__":
    main()
