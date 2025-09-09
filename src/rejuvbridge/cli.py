import typer
from typing import Optional
from .utils.env import check_env
from .data.pipeline import ingest_data, prepare_shards
from .training.runner import train_from_config
from .deploy.app import run_api
from .ui.app import launch_ui
from .utils.demo import write_embedded_tile

app = typer.Typer(help="RejuvBridge CLI")


@app.command()
def env() -> None:
    """Validate environment and print versions."""
    check_env()


@app.command()
def ingest(input: str, out: str) -> None:
    """Ingest raw data into standardized layout."""
    ingest_data(input, out)


@app.command()
def prepare(raw: str, out: str, tile: int = 512, stride: Optional[int] = None) -> None:
    """Tile images and write WebDataset/Zarr shards with Parquet metadata."""
    prepare_shards(raw, out, tile, stride)


@app.command()
def train(config: str) -> None:
    """Train models using a Hydra-style YAML config."""
    train_from_config(config)


@app.command()
def api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI inference service."""
    run_api(host, port)


@app.command()
def ui(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Launch the Gradio UI."""
    launch_ui(host, port)


@app.command()
def demo(workdir: str = ".") -> None:
    """Create a tiny sample tile, ingest and prepare shards in WORKDIR."""
    from pathlib import Path

    wd = Path(workdir)
    samples = wd / "samples"
    raw = wd / "data" / "raw"
    shards = wd / "data" / "shards"
    samples.mkdir(parents=True, exist_ok=True)
    tile_path = samples / "demo_tile.png"
    write_embedded_tile(tile_path)
    ingest_data(str(samples), str(raw))
    prepare_shards(str(raw), str(shards), tile=256, stride=256)
    print(f"Demo ready: {shards}")
