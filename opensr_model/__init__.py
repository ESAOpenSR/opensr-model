# __init__.py
# type: ignore[attr-defined]
"""Latent diffusion model trained in RGBN optical remote sensing imagery"""

import sys

# from opensr_model import *
from opensr_model.srmodel import *

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    try:
        # If your distribution name on PyPI/setup.py is different (e.g. "opensr-model"),
        # pass that exact string instead of __name__.
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version  # <-- this makes `opensr_model.__version__` work
__all__ = ["__version__"]