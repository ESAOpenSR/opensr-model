from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import opensr_model.training.diffusion_module as diffusion_module_helpers
from opensr_model.diffusion.utils import LitEma
from opensr_model.training.autoencoder_module import AutoencoderTrainingModule
from opensr_model.training.cli import _prepare_trainer_kwargs
from opensr_model.training.diffusion_module import DiffusionTrainingModule
from opensr_model.training.losses import (
    PatchDiscriminator,
    adaptive_generator_weight,
)
from opensr_model.training.metrics import (
    batch_clipped_fraction,
    compute_rgb_nir_reconstruction_metrics,
    spectral_angle,
)


def test_diffusion_validation_randomness_is_locally_reproducible(
    tiny_model_config,
) -> None:
    module = DiffusionTrainingModule(
        tiny_model_config,
        scheduler_patience=None,
        validation_num_samples=0,
        validation_sampling_steps=5,
        validation_seed=19,
    )
    hr = torch.rand(2, 4, 16, 16)
    lr = torch.rand(2, 4, 4, 4)

    first_generator = module._make_generator(hr.device, 123)
    first_z, first_condition = module._prepare_latents(
        hr, lr, generator=first_generator
    )
    first_loss, first_stats = module._diffusion_objective(
        first_z, first_condition, generator=first_generator
    )

    second_generator = module._make_generator(hr.device, 123)
    second_z, second_condition = module._prepare_latents(
        hr, lr, generator=second_generator
    )
    second_loss, second_stats = module._diffusion_objective(
        second_z, second_condition, generator=second_generator
    )

    assert torch.equal(first_z, second_z)
    assert torch.equal(first_condition, second_condition)
    assert torch.equal(first_loss, second_loss)
    assert torch.equal(first_stats["x0_mse"], second_stats["x0_mse"])


def test_validation_images_compare_ema_and_raw_with_aligned_detail_crop(
    monkeypatch, tiny_model_config
) -> None:
    module = DiffusionTrainingModule(
        tiny_model_config,
        scheduler_patience=None,
        validation_num_samples=1,
        validation_sampling_steps=5,
        validation_seed=19,
        validation_detail_crop_size=16,
    )
    lr = torch.rand(1, 4, 8, 8)
    hr = F.interpolate(lr, scale_factor=4, mode="nearest")
    calls: list[bool] = []

    def sample(input_lr, *, use_ema, **kwargs):
        calls.append(use_ema)
        upsampled = F.interpolate(input_lr, scale_factor=4, mode="nearest")
        return upsampled if use_ema else (upsampled + 0.01).clamp(0.0, 1.0)

    monkeypatch.setattr(module, "sample_super_resolution", sample)
    monkeypatch.setattr(module, "_encode_hr", lambda value, **_: value)
    monkeypatch.setattr(module, "_decode_first_stage_float32", lambda value: value)
    display_values = iter((0.25, 0.75))
    monkeypatch.setattr(
        module,
        "_histogram_align_for_display",
        lambda sr, _: torch.full_like(sr, next(display_values)),
    )

    images, details, sample_ids, metrics = module._validation_images(
        {"sample_id": ["tile"]}, hr, lr, batch_idx=2
    )

    assert calls == [True, False]
    assert tuple(images) == (
        "low_resolution",
        "target",
        "autoencoder_reconstruction",
        "super_resolution_standard",
        "super_resolution_ema",
        "absolute_error_standard",
        "absolute_error_ema",
    )
    assert sample_ids == ["tile"]
    assert details["target"].shape[-2:] == (16, 16)
    assert details["low_resolution"].shape[-2:] == (4, 4)
    assert torch.equal(
        details["target"],
        F.interpolate(details["low_resolution"], scale_factor=4, mode="nearest"),
    )
    assert "super_resolution_standard" in details
    assert "super_resolution_ema" in details
    assert torch.all(images["super_resolution_ema"] == 0.25)
    assert torch.all(images["super_resolution_standard"] == 0.75)
    assert "ema_psnr" in metrics
    assert "raw_psnr" in metrics
    assert metrics["psnr"] == metrics["ema_psnr"]
    assert metrics["ema_mae"].item() == pytest.approx(0.0)


def test_display_histogram_alignment_uses_bilinearly_interpolated_lr(
    monkeypatch,
) -> None:
    lr = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 16
    sr = torch.flip(
        F.interpolate(lr, scale_factor=4, mode="nearest"),
        dims=(-1,),
    )
    references = []

    def capture_reference(source, reference, *, channel_axis):
        assert channel_axis == 0
        references.append(torch.from_numpy(reference.copy()))
        return source

    monkeypatch.setattr(diffusion_module_helpers, "match_histograms", capture_reference)
    aligned = DiffusionTrainingModule._histogram_align_for_display(sr, lr)

    expected = F.interpolate(
        lr,
        size=sr.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    assert len(references) == 4
    for channel, reference in enumerate(references):
        assert torch.equal(reference, expected[:, channel])
    assert torch.equal(aligned, sr)


def test_inference_incompatible_parameterization_is_rejected(
    tiny_model_config,
) -> None:
    config = deepcopy(tiny_model_config)
    config["denoiser_settings"]["parameterization"] = "x0"
    with pytest.raises(ValueError, match="requires parameterization='eps'"):
        DiffusionTrainingModule(config, validation_sampling_steps=5)


def test_ema_restore_releases_collected_parameter_copies() -> None:
    model = torch.nn.Linear(3, 2)
    ema = LitEma(model)
    original = [parameter.detach().clone() for parameter in model.parameters()]
    ema.store(model.parameters())
    for parameter in model.parameters():
        parameter.data.zero_()
    ema.restore(model.parameters())

    assert not ema.collected_params
    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(model.parameters(), original)
    )


def test_spectral_angle_treats_matching_zero_vectors_as_zero() -> None:
    image = torch.zeros(1, 4, 3, 3)
    mask = torch.ones(1, 3, 3)
    assert spectral_angle(image, image, mask).item() == pytest.approx(0.0)


def test_rgb_nir_metrics_keep_single_band_sam_out_of_telemetry() -> None:
    target = torch.zeros(1, 4, 12, 12)
    prediction = torch.empty_like(target)
    prediction[:, :3].fill_(0.25)
    prediction[:, 3:].fill_(0.5)
    # A per-channel mask exercises safe mask slicing as well as metric slicing.
    mask = torch.ones_like(target)

    metrics = compute_rgb_nir_reconstruction_metrics(prediction, target, mask)

    assert metrics["rgb_mae"].item() == pytest.approx(0.25)
    assert metrics["nir_mae"].item() == pytest.approx(0.5)
    assert {
        name.removeprefix("nir_") for name in metrics if name.startswith("nir_")
    } == {"mae", "mse", "rmse", "psnr", "out_of_range"}
    assert "rgb_sam" in metrics
    assert (
        compute_rgb_nir_reconstruction_metrics(prediction[:, :2], target[:, :2]) == {}
    )


def test_optional_clipped_fraction_is_reduced_and_validated() -> None:
    reference = torch.zeros(2, 4, 3, 3)
    assert batch_clipped_fraction({}, reference) is None
    value = batch_clipped_fraction(
        {"clipped_fraction": torch.tensor([0.25, 0.75])}, reference
    )
    assert value is not None
    assert value.item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        batch_clipped_fraction({"clipped_fraction": [0.0, 1.01]}, reference)


def test_both_training_modules_weight_clipping_telemetry_by_batch(
    monkeypatch,
    tiny_model_config,
    tiny_autoencoder_training_config,
) -> None:
    batch = {"clipped_fraction": torch.tensor([0.25, 0.75])}
    hr = torch.zeros(2, 4, 16, 16)
    lr = torch.zeros(2, 4, 4, 4)

    autoencoder = AutoencoderTrainingModule(
        tiny_model_config, tiny_autoencoder_training_config
    )
    autoencoder_logs = []
    monkeypatch.setattr(
        autoencoder,
        "log",
        lambda name, value, **kwargs: autoencoder_logs.append((name, value, kwargs)),
    )
    autoencoder._log_input_clipped_fraction(batch, "val", hr)
    assert autoencoder_logs[0][0] == "val/input_clipped_fraction"
    assert autoencoder_logs[0][1].item() == pytest.approx(0.5)
    assert autoencoder_logs[0][2]["batch_size"] == 2

    diffusion = DiffusionTrainingModule(
        tiny_model_config,
        scheduler_patience=None,
        validation_num_samples=0,
        validation_sampling_steps=5,
    )
    diffusion_logs = []
    monkeypatch.setattr(
        diffusion,
        "log",
        lambda name, value, **kwargs: diffusion_logs.append((name, value, kwargs)),
    )
    monkeypatch.setattr(diffusion, "_paired_images", lambda _: (hr, lr))
    monkeypatch.setattr(diffusion, "_valid_mask", lambda *_: None)
    latent = torch.zeros(2, 2, 8, 8)
    monkeypatch.setattr(
        diffusion, "_prepare_latents", lambda *_, **__: (latent, latent)
    )
    stat_names = (
        "prediction_rms",
        "target_rms",
        "latent_std",
        "conditioning_std",
        "timestep_mean",
        "x0_mse",
        "valid_fraction",
    )
    zero = torch.zeros(())
    monkeypatch.setattr(
        diffusion,
        "_diffusion_objective",
        lambda *_, **__: (zero, {name: zero for name in stat_names}),
    )
    diffusion._shared_step(batch, stage="val", batch_idx=0)
    clipping_log = next(
        entry for entry in diffusion_logs if entry[0] == "val/input_clipped_fraction"
    )
    assert clipping_log[1].item() == pytest.approx(0.5)
    assert clipping_log[2]["batch_size"] == 2


def test_cli_reconciles_manual_gan_ddp_and_accumulation(
    tiny_model_config,
    tiny_autoencoder_training_config,
) -> None:
    training = deepcopy(tiny_autoencoder_training_config)
    training["accumulate_grad_batches"] = 3
    training["loss"].update(
        {
            "discriminator_factor": 1.0,
            "discriminator_layers": 1,
            "discriminator_channels": 16,
        }
    )
    module = AutoencoderTrainingModule(tiny_model_config, training)
    kwargs = {
        "accelerator": "cpu",
        "devices": 2,
        "strategy": "auto",
        "accumulate_grad_batches": 1,
    }
    _prepare_trainer_kwargs("autoencoder", module, kwargs)
    assert kwargs["strategy"] == "ddp_find_unused_parameters_true"

    kwargs["accumulate_grad_batches"] = 2
    with pytest.raises(ValueError, match="training.accumulate_grad_batches"):
        _prepare_trainer_kwargs("autoencoder", module, kwargs)


def test_legacy_autoencoder_loss_sums_l1_lpips_and_kl_per_batch(
    tiny_model_config,
    tiny_autoencoder_training_config,
) -> None:
    training = deepcopy(tiny_autoencoder_training_config)
    training["loss"].update({"kl_weight": 0.1, "perceptual_weight": 1.0})
    # Avoid constructing the external LPIPS dependency; this deterministic stand-in
    # has its exact Bx1x1x1 output shape and therefore tests broadcast semantics.
    training["loss"]["perceptual_weight"] = 0.0
    module = AutoencoderTrainingModule(tiny_model_config, training)

    class DummyPerceptual(torch.nn.Module):
        def forward(self, image, reconstruction):
            assert image.shape == reconstruction.shape == (2, 3, 2, 2)
            return image.new_tensor([0.5, 1.0]).view(2, 1, 1, 1)

    class DummyPosterior:
        @staticmethod
        def kl():
            return torch.tensor([2.0, 4.0])

    module.perceptual_loss = DummyPerceptual()
    module.perceptual_weight = 1.0
    image = torch.zeros(2, 3, 2, 2)
    reconstruction = torch.ones_like(image)
    components = module._generator_components(
        reconstruction, image, DummyPosterior(), training=False
    )

    # Per sample: 12*(1 + LPIPS); then sum both samples and divide by B.
    assert components["nll_loss"].item() == pytest.approx((18.0 + 24.0) / 2.0)
    assert components["kl_loss"].item() == pytest.approx(3.0)
    assert components["loss"].item() == pytest.approx(21.3)


def test_legacy_autoencoder_random_subset_is_shared_by_all_image_losses(
    monkeypatch,
) -> None:
    calls = []

    def sample(population, count):
        calls.append((list(population), count))
        return [3, 1, 0]

    monkeypatch.setattr(
        "opensr_model.training.autoencoder_module.random.sample",
        sample,
    )
    image = torch.stack(
        [torch.full((2, 2), float(channel)) for channel in range(4)]
    ).unsqueeze(0)
    reconstruction = image + 10.0
    selected_image, selected_reconstruction, selected = (
        AutoencoderTrainingModule._random_three_band_pair(image, reconstruction)
    )

    assert selected.tolist() == [3, 1, 0]
    assert calls == [([0, 1, 2, 3], 3)]
    assert selected_image[:, :, 0, 0].tolist() == [[3.0, 1.0, 0.0]]
    assert torch.equal(selected_reconstruction, selected_image + 10.0)


def test_legacy_patchgan_and_adaptive_weight_match_reference() -> None:
    discriminator = PatchDiscriminator(in_channels=3, base_channels=16, num_layers=2)
    convolutions = [
        layer for layer in discriminator.network if isinstance(layer, torch.nn.Conv2d)
    ]
    assert convolutions[0].in_channels == 3
    assert any(
        isinstance(layer, torch.nn.BatchNorm2d) for layer in discriminator.network
    )

    last_layer = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    reconstruction_loss = (last_layer.square()).sum()
    adversarial_loss = (3.0 * last_layer).sum()
    expected = torch.norm(2.0 * last_layer.detach()) / (
        torch.norm(torch.full_like(last_layer, 3.0)) + 1e-4
    )
    actual = adaptive_generator_weight(
        reconstruction_loss,
        adversarial_loss,
        last_layer,
        discriminator_weight=0.5,
    )
    assert actual.item() == pytest.approx((expected * 0.5).item())
    assert not actual.requires_grad


def test_legacy_autoencoder_optimizer_excludes_fixed_logvar_and_uses_plateau(
    tiny_model_config,
    tiny_autoencoder_training_config,
) -> None:
    training = deepcopy(tiny_autoencoder_training_config)
    training["loss"].update(
        {
            "discriminator_factor": 1.0,
            "discriminator_channels": 16,
            "discriminator_layers": 1,
        }
    )
    training["scheduler"] = {
        "name": "plateau",
        "monitor": "train/total_loss",
        "factor": 0.5,
        "patience": 25,
    }
    module = AutoencoderTrainingModule(tiny_model_config, training)
    module._trainer = SimpleNamespace(estimated_stepping_batches=8)
    optimizers, schedulers = module.configure_optimizers()

    assert len(optimizers) == len(schedulers) == 2
    assert all(type(optimizer) is torch.optim.Adam for optimizer in optimizers)
    assert all(optimizer.defaults["betas"] == (0.5, 0.9) for optimizer in optimizers)
    generator_parameter_ids = {
        id(parameter)
        for group in optimizers[0].param_groups
        for parameter in group["params"]
    }
    assert id(module.logvar) not in generator_parameter_ids
    assert all(
        isinstance(entry["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
        for entry in schedulers
    )
    assert all(entry["monitor"] == "train/total_loss" for entry in schedulers)
