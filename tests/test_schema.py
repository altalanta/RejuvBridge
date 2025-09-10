from pathlib import Path
import pytest


def test_manifests_and_tiles(tmp_path: Path):
    # Import here so base installs without extras can still import tests
    from rejuvbridge.data.pipeline import ingest_data, prepare_shards
    try:
        import pyarrow.parquet as pq
    except Exception:
        pytest.skip("pyarrow not installed; install with '.[data]' to run this test")

    samples = tmp_path / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    raw = tmp_path / "data" / "raw"
    shards = tmp_path / "data" / "shards"
    ingest_data(str(samples), str(raw))
    prepare_shards(str(raw), str(shards), tile=256, stride=256)

    manifest = pq.read_table(raw / "manifest.parquet").to_pandas()
    tiles = pq.read_table(shards / "tiles.parquet").to_pandas()

    assert set(["sample_id", "path", "modality"]).issubset(manifest.columns)
    assert set(["tile_id", "x", "y", "tile", "stride", "path"]).issubset(tiles.columns)

