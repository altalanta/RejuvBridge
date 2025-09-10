from pathlib import Path

from typer.testing import CliRunner

from rejuvbridge.cli import app


def test_demo(tmp_path: Path):
    runner = CliRunner()
    res = runner.invoke(app, ["demo", "--workdir", str(tmp_path)])
    assert res.exit_code == 0
    assert (tmp_path / "samples" / "demo_tile.png").exists()
    assert (tmp_path / "data" / "raw" / "manifest.parquet").exists()
    assert (tmp_path / "data" / "shards" / "tiles.parquet").exists()

