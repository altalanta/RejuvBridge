import base64
from pathlib import Path

# 1x1 transparent PNG (valid base64). Sufficient for demo pipeline.
_EMBEDDED_TILE_BASE64 = (
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


def write_embedded_tile(path: Path) -> None:
    """Write a tiny PNG tile to path (embedded base64)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(_EMBEDDED_TILE_BASE64)
    path.write_bytes(data)
