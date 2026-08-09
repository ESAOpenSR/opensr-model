from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from opensr_model.training.callbacks import NativeCheckpointCallback
from opensr_model.training.cli import (
    _build_callbacks,
    _build_module,
    _validate_resume_checkpoint,
)
from opensr_model.training.config import load_model_config, load_training_config


def test_diffusion_module_resume_does_not_reload_initializer(
    tiny_model_config,
) -> None:
    config = OmegaConf.create(
        {
            "seed": 1,
            "model": {
                "pretrained_checkpoint": "/path/that/no/longer/exists.ckpt",
                "autoencoder_checkpoint": None,
                "allow_random_first_stage": False,
            },
            "training": {
                "optimizer": {"learning_rate": 1e-4},
                "scheduler": {"name": "constant"},
                "validation": {"sampling_steps": 5, "num_samples": 0},
            },
        }
    )
    module = _build_module(
        "diffusion", OmegaConf.create(tiny_model_config), config, resuming=True
    )
    assert module.hparams.pretrained_checkpoint is None
    assert module.hparams.autoencoder_checkpoint is None


def test_diffusion_cli_fallbacks_use_quality_sampling_defaults(
    tiny_model_config,
) -> None:
    config = OmegaConf.create(
        {
            "seed": 17,
            "model": {
                "pretrained_checkpoint": None,
                "autoencoder_checkpoint": None,
                "allow_random_first_stage": False,
            },
            "training": {
                "optimizer": {"learning_rate": 1e-4},
                "scheduler": {"name": "constant"},
                "validation": {},
            },
        }
    )
    module = _build_module(
        "diffusion", OmegaConf.create(tiny_model_config), config, resuming=True
    )
    assert module.validation_sampling_steps == 100
    assert module.validation_sampling_eta == pytest.approx(0.95)
    assert module.validation_sampling_temperature == pytest.approx(1.0)
    assert module.validation_use_ema is True


def test_diffusion_native_export_defaults_to_ema_and_allows_raw(
    tmp_path,
) -> None:
    config = load_training_config(
        "opensr_model/configs/train_diffusion.yaml",
        overrides=[
            "trainer.enable_checkpointing=false",
            "logging.images.enabled=false",
        ],
        expected_stage="diffusion",
    )
    callbacks = _build_callbacks(
        "diffusion", config, tmp_path.resolve(), has_logger=False
    )
    native = next(
        callback
        for callback in callbacks
        if isinstance(callback, NativeCheckpointCallback)
    )
    assert native.weight_source == "ema"

    config.checkpoint.native.weight_source = "raw"
    callbacks = _build_callbacks(
        "diffusion", config, tmp_path.resolve(), has_logger=False
    )
    native = next(
        callback
        for callback in callbacks
        if isinstance(callback, NativeCheckpointCallback)
    )
    assert native.weight_source == "raw"


def test_resume_validation_rejects_malformed_optimizer_state(tmp_path) -> None:
    path = tmp_path / "malformed.ckpt"
    torch.save(
        {
            "state_dict": {"weight": torch.ones(1)},
            "optimizer_states": "not-a-list",
            "loops": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="not a complete Lightning resume"):
        _validate_resume_checkpoint(path)


def test_disabling_lightning_checkpointing_omits_model_checkpoint(tmp_path) -> None:
    config = load_training_config(
        "opensr_model/configs/train_autoencoder.yaml",
        overrides=[
            "trainer.enable_checkpointing=false",
            "checkpoint.native.enabled=false",
            "logging.images.enabled=false",
        ],
        expected_stage="autoencoder",
    )
    callbacks = _build_callbacks(
        "autoencoder", config, tmp_path.resolve(), has_logger=False
    )
    assert not any(isinstance(callback, ModelCheckpoint) for callback in callbacks)


def test_missing_nested_architecture_path_does_not_fall_back_by_basename() -> None:
    with pytest.raises(FileNotFoundError):
        load_model_config("missing/config_10m.yaml")
