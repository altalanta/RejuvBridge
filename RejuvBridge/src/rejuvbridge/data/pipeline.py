from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def ingest_data(input_dir: str, out_dir: str) -> None:
    """Stub: copy/scan input into standard layout and write a minimal parquet manifest.

    In real usage, add modality-aware loaders (EM/LM/WSI), basic QC and checksum capture.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame([
        {"sample_id": "SAMPLE_DEMO", "path": str(Path(input_dir)), "modality": "histology"}
    ])
    table = pa.Table.from_pandas(manifest)
    pq.write_table(table, out / "manifest.parquet")
    print(f"Wrote manifest to {out/'manifest.parquet'}")


def prepare_shards(raw_dir: str, out_dir: str, tile: int = 512, stride: Optional[int] = None) -> None:
    """Stub: tile images and shard to WebDataset/Zarr; here we emit a dummy parquet for demo."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stride = stride or tile
    tiles = pd.DataFrame([
        {"tile_id": 0, "x": 0, "y": 0, "tile": tile, "stride": stride, "path": "tile_000000.jpg"}
    ])
    pq.write_table(pa.Table.from_pandas(tiles), out / "tiles.parquet")
    print(f"Wrote tiles metadata to {out/'tiles.parquet'} (stub)")

