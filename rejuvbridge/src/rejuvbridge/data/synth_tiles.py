"""Synthetic histology tile generator for RejuvBridge."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

Palette = Tuple[int, int, int]


def _class_palette(n_classes: int) -> list[Palette]:
    rng = np.random.default_rng(123)
    angles = np.linspace(0, 2 * math.pi, n_classes, endpoint=False)
    colors: list[Palette] = []
    for angle in angles:
        r = int(127 + 120 * math.sin(angle))
        g = int(127 + 120 * math.sin(angle + 2 * math.pi / 3))
        b = int(127 + 120 * math.sin(angle + 4 * math.pi / 3))
        colors.append((r, g, b))
    return colors


def _draw_blob(image: Image.Image, color: Palette, rng: np.random.Generator) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    num_blobs = rng.integers(3, 6)
    for _ in range(num_blobs):
        radius = int(rng.uniform(width * 0.1, width * 0.3))
        x = int(rng.uniform(radius, width - radius))
        y = int(rng.uniform(radius, height - radius))
        alpha = int(rng.uniform(120, 200))
        jitter = tuple(int(min(255, max(0, c + rng.normal(0, 25)))) for c in color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=jitter + (alpha,))


def generate_tiles(
    out_dir: Path,
    n_tiles: int,
    img_size: int,
    seed: int,
    n_classes: int = 8,
) -> pd.DataFrame:
    """Generate synthetic RGB tiles and return metadata."""

    rng = np.random.default_rng(seed)
    colors = _class_palette(n_classes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx in range(n_tiles):
        class_id = idx % n_classes
        class_dir = out_dir / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)
        tile_id = f"tile_{idx:05d}"
        tile_path = class_dir / f"{tile_id}.png"

        base = np.full((img_size, img_size, 3), colors[class_id], dtype=np.uint8)
        noise = rng.normal(0, 12, size=base.shape)
        tile_array = np.clip(base + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(tile_array)
        _draw_blob(image, colors[class_id], rng)
        image.save(tile_path)

        records.append(
            {
                "tile_id": tile_id,
                "class_id": int(class_id),
                "tile_path": str(tile_path.resolve()),
            }
        )

    metadata = pd.DataFrame(records)
    metadata_path = out_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic histology tiles.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for tiles")
    parser.add_argument("--n", type=int, default=512, help="Number of tiles to generate")
    parser.add_argument("--img_size", type=int, default=64, help="Square tile size in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--classes", type=int, default=8, help="Number of morphological classes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_tiles(args.out, args.n, args.img_size, args.seed, args.classes)


if __name__ == "__main__":
    main()
