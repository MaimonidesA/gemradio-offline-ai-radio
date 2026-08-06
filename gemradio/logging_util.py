"""Logging shared by every GemRadio component."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_CONFIGURED = False


def setup() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    from . import config

    config.ensure_dirs()
    level = logging.DEBUG if os.environ.get("GEMRADIO_DEBUG") else logging.INFO
    root = logging.getLogger("gemradio")
    root.setLevel(level)
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
                            datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    try:
        fileh = RotatingFileHandler(config.LOG_PATH, maxBytes=2_000_000, backupCount=2)
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
    except OSError:
        pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup()
    short = name.split(".")[-1]
    return logging.getLogger(f"gemradio.{short}")
