"""Public API and provenance guarantees across the training-package moves."""

from importlib import import_module
import json

from omegaconf import OmegaConf


def test_previous_imports_resolve_to_canonical_modules():
    from opensr_model.training import (
        AutoencoderTrainingModule,
        DiffusionTrainingModule,
        OpenSRDataModule,
    )

    paths = {
        "autoencoder_module": "modules.autoencoder",
        "diffusion_module": "modules.diffusion",
        "losses": "modules.losses",
        "metrics": "modules.metrics",
        "optim": "modules.optim",
        "callbacks": "runtime.callbacks",
        "checkpoints": "runtime.checkpoints",
        "config": "runtime.config",
        "provenance": "runtime.provenance",
    }
    for old, new in paths.items():
        assert import_module(f"opensr_model.training.{old}") is import_module(
            f"opensr_model.training.{new}"
        )
    assert (
        AutoencoderTrainingModule
        is import_module(
            "opensr_model.training.modules.autoencoder"
        ).AutoencoderTrainingModule
    )
    assert (
        DiffusionTrainingModule
        is import_module(
            "opensr_model.training.modules.diffusion"
        ).DiffusionTrainingModule
    )
    from opensr_model.training.data import _aligned_crop
    from opensr_model.training.data import taco

    assert OpenSRDataModule is taco.OpenSRDataModule
    assert _aligned_crop is taco._aligned_crop


def test_relocated_provenance_still_snapshots_whole_package(tmp_path):
    from opensr_model.training.runtime.provenance import record_provenance

    config = OmegaConf.create({"model": {"pretrained_checkpoint": None}, "seed": 42})
    record_provenance(config, tmp_path)
    manifest = json.loads((tmp_path / "provenance.json").read_text())
    for name in (
        "autoencoder/utils.py",
        "configs/config_10m.yaml",
        "training/modules/diffusion.py",
    ):
        assert name in manifest["sources"]
        assert (tmp_path / "source" / name).is_file()
    record_provenance(config, tmp_path)
