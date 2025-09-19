"""RejuvBridge-lite synthetic cross-modal alignment package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rejuvbridge-lite")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
