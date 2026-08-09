from __future__ import annotations

from copy import deepcopy

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from opensr_model.training.autoencoder_module import AutoencoderTrainingModule
from opensr_model.training.callbacks import (
    NativeCheckpointCallback,
    ReconstructionImageLogger,
)
from opensr_model.training.checkpoints import (
    load_autoencoder_weights,
    load_checkpoint_payload,
    load_diffusion_weights,
)
from opensr_model.training.data import OpenSRDataModule
from opensr_model.training.diffusion_module import DiffusionTrainingModule


def _data(fake_taco_factory) -> OpenSRDataModule:
    bundle = fake_taco_factory(hr_size=16, include_nodata=False)
    return OpenSRDataModule(
        taco_path=bundle.root,
        train_patch_size=16,
        batch_size=1,
        val_batch_size=1,
        num_workers=0,
        pin_memory=False,
    )


def _trainer(tmp_path, callbacks) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        logger=False,
        callbacks=callbacks,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        log_every_n_steps=1,
    )


def test_autoencoder_fit_logs_images_and_both_checkpoint_formats(
    tmp_path,
    tiny_model_config,
    tiny_autoencoder_training_config,
    fake_taco_factory,
) -> None:
    module = AutoencoderTrainingModule(
        tiny_model_config, tiny_autoencoder_training_config
    )
    resume = ModelCheckpoint(dirpath=tmp_path / "resume", save_last=True)
    native = NativeCheckpointCallback(dirpath="native")
    images = ReconstructionImageLogger(max_images=1, output_dir="images")
    trainer = _trainer(tmp_path, [resume, native, images])
    trainer.fit(module, datamodule=_data(fake_taco_factory))

    resume_path = tmp_path / "resume" / "last.ckpt"
    native_path = tmp_path / "native" / "last-autoencoder.ckpt"
    assert resume_path.exists()
    assert native_path.exists()
    assert (tmp_path / "images" / "epoch_0000" / "reconstructions_rgb.png").exists()
    assert "optimizer_states" in torch.load(
        resume_path, map_location="cpu", weights_only=True
    )
    restored = AutoencoderTrainingModule(
        tiny_model_config, tiny_autoencoder_training_config
    )
    assert load_autoencoder_weights(restored.autoencoder, native_path).is_exact
    assert load_autoencoder_weights(restored.autoencoder, resume_path).is_exact


def test_diffusion_fit_exports_strict_native_checkpoint(
    tmp_path,
    tiny_model_config,
    fake_taco_factory,
) -> None:
    module = DiffusionTrainingModule(
        tiny_model_config,
        scheduler_patience=None,
        validation_num_samples=1,
        validation_image_batches=1,
        validation_sampling_steps=5,
        validation_seed=7,
    )
    native = NativeCheckpointCallback(dirpath="native")
    images = ReconstructionImageLogger(max_images=1, output_dir="images")
    trainer = _trainer(tmp_path, [native, images])
    trainer.fit(module, datamodule=_data(fake_taco_factory))

    resume_path = tmp_path / "diffusion-resume.ckpt"
    trainer.save_checkpoint(resume_path)
    resume_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    assert resume_payload["opensr_inference_config"] == (
        module.opensr_inference_config()
    )

    native_path = tmp_path / "native" / "last-inference.ckpt"
    assert native_path.exists()
    assert (tmp_path / "images" / "epoch_0000" / "reconstructions_rgb.png").exists()
    payload = load_checkpoint_payload(native_path)
    assert payload["metadata"]["inference_unet_weight_source"] == "ema"
    assert payload["metadata"]["opensr_inference_config"] == (
        module.opensr_inference_config()
    )
    raw_key = next(
        key for key in payload["state_dict"] if key.startswith("model.diffusion_model.")
    )
    shadow_key = "model_ema." + raw_key.removeprefix("model.").replace(".", "")
    assert torch.equal(
        payload["state_dict"][raw_key], payload["state_dict"][shadow_key]
    )
    restored = DiffusionTrainingModule(
        tiny_model_config,
        scheduler_patience=None,
        validation_num_samples=0,
        validation_sampling_steps=5,
    )
    assert load_diffusion_weights(restored.diffusion, native_path).is_exact
    assert all(
        not parameter.requires_grad
        for parameter in restored.first_stage_model.parameters()
    )


def test_manual_gan_accumulation_runs_and_steps_on_partial_cycle(
    tmp_path,
    tiny_model_config,
    tiny_autoencoder_training_config,
    fake_taco_factory,
) -> None:
    training = deepcopy(tiny_autoencoder_training_config)
    training["accumulate_grad_batches"] = 2
    training["scheduler"] = {
        "name": "cosine",
        "warmup_steps": 1,
        "min_lr_ratio": 0.1,
    }
    training["loss"].update(
        {
            "discriminator_factor": 1.0,
            "discriminator_start": 0,
            "discriminator_channels": 16,
            "discriminator_layers": 1,
        }
    )
    module = AutoencoderTrainingModule(tiny_model_config, training)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        max_epochs=1,
        limit_train_batches=3,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
    )
    trainer.fit(module, datamodule=_data(fake_taco_factory))

    assert int(module.generator_updates) == 2
    assert trainer.global_step == 4  # one generator and discriminator step each


def test_gan_uses_independent_generator_and_discriminator_reconstructions(
    tmp_path,
    monkeypatch,
    tiny_model_config,
    tiny_autoencoder_training_config,
    fake_taco_factory,
) -> None:
    training = deepcopy(tiny_autoencoder_training_config)
    training["loss"].update(
        {
            "discriminator_factor": 1.0,
            "discriminator_start": 0,
            "discriminator_channels": 16,
            "discriminator_layers": 1,
        }
    )
    module = AutoencoderTrainingModule(tiny_model_config, training)
    reconstruct_calls = []
    subset_calls = []
    reconstruct = module._reconstruct
    select_subset = module._random_three_band_pair

    def tracked_reconstruct(image, *, sample):
        reconstruct_calls.append(sample)
        return reconstruct(image, sample=sample)

    def tracked_subset(image, reconstruction):
        subset_calls.append(True)
        return select_subset(image, reconstruction)

    monkeypatch.setattr(module, "_reconstruct", tracked_reconstruct)
    monkeypatch.setattr(module, "_random_three_band_pair", tracked_subset)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
    )
    trainer.fit(module, datamodule=_data(fake_taco_factory))

    assert reconstruct_calls == [True, True]
    assert len(subset_calls) == 2
