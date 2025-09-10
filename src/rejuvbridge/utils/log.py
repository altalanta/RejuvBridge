import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    Respects REJUV_LOG_LEVEL env var (default INFO). Avoids adding multiple
    handlers if called repeatedly.
    """
    level_str = os.getenv("REJUV_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger = logging.getLogger(name if name else "rejuvbridge")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

