"""Additive PyTorch Lightning training support for OpenSR."""

from .modules.autoencoder import AutoencoderTrainingModule
from .data import OpenSRDataModule
from .modules.diffusion import DiffusionTrainingModule

__all__ = [
    "AutoencoderTrainingModule",
    "DiffusionTrainingModule",
    "OpenSRDataModule",
]


def _install_legacy_imports():
    """Keep existing scripts and pickled class references working after moves."""
    from importlib import import_module
    import sys

    for old, new in {
        "autoencoder_module": "modules.autoencoder",
        "diffusion_module": "modules.diffusion",
        "losses": "modules.losses",
        "metrics": "modules.metrics",
        "optim": "modules.optim",
        "callbacks": "runtime.callbacks",
        "checkpoints": "runtime.checkpoints",
        "config": "runtime.config",
        "provenance": "runtime.provenance",
    }.items():
        module = import_module(f".{new}", __name__)
        sys.modules[f"{__name__}.{old}"] = module
        globals()[old] = module


_install_legacy_imports()
del _install_legacy_imports
