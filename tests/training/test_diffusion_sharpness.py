from __future__ import annotations

import pytest
import torch

from opensr_model.training.diffusion_module import DiffusionTrainingModule


def _module(tiny_model_config, **sharpness) -> DiffusionTrainingModule:
    return DiffusionTrainingModule(
        tiny_model_config,
        scheduler_patience=None,
        sharpness_config=sharpness,
        validation_num_samples=0,
        validation_sampling_steps=5,
    )


def test_p2_weights_epsilon_mse_without_hiding_the_base_metric(
    monkeypatch, tiny_model_config
) -> None:
    module = _module(
        tiny_model_config,
        p2_gamma=1.0,
        p2_k=1.0,
        latent_detail_weight=0.0,
    )
    z = torch.zeros(2, 2, 8, 8)
    conditioning = torch.zeros_like(z)
    noise = torch.ones_like(z)
    timesteps = torch.tensor([0, 9], dtype=torch.long)
    monkeypatch.setattr(
        module.diffusion,
        "apply_model",
        lambda noisy_z, timestep, condition: torch.zeros_like(noisy_z),
    )

    loss, stats = module._diffusion_objective(
        z,
        conditioning,
        timesteps=timesteps,
        noise=noise,
    )

    alpha_bar = module.diffusion.alphas_cumprod[timesteps]
    snr = alpha_bar / (1.0 - alpha_bar)
    expected = (1.0 + snr).reciprocal().mean()
    assert stats["eps_mse"].item() == pytest.approx(1.0)
    assert stats["p2_loss"].item() == pytest.approx(expected.item())
    assert loss.item() == pytest.approx(expected.item())


def test_latent_laplacian_is_low_timestep_gated_and_differentiable(
    tiny_model_config,
) -> None:
    module = _module(
        tiny_model_config,
        p2_gamma=0.0,
        latent_detail_weight=1.0,
        latent_detail_max_timestep_fraction=0.25,
    )
    predicted = torch.zeros(2, 2, 8, 8, requires_grad=True)
    target = torch.zeros_like(predicted)
    target[0, :, ::2, ::2] = 1.0
    target[1] = 100.0

    loss, eligible_fraction = module._latent_detail_loss(
        predicted,
        target,
        torch.tensor([1, 9]),
        latent_mask=None,
    )
    loss.backward()

    assert loss.item() > 0
    assert eligible_fraction.item() == pytest.approx(0.5)
    assert predicted.grad is not None
    assert predicted.grad[0].abs().sum().item() > 0
    assert predicted.grad[1].abs().sum().item() == pytest.approx(0.0)


def test_decoded_rgb_auxiliary_is_bounded_gated_and_backpropagates(
    monkeypatch, tiny_model_config
) -> None:
    module = _module(
        tiny_model_config,
        p2_gamma=0.0,
        latent_detail_weight=0.0,
        decoded_reconstruction_weight=0.1,
        decoded_detail_weight=0.1,
        decoded_max_timestep_fraction=0.20,
        decoded_every_n_batches=3,
        decoded_max_batch_size=1,
    )
    z = torch.randn(2, 2, 8, 8)
    predicted_noise = torch.zeros_like(z, requires_grad=True)
    monkeypatch.setattr(
        module.diffusion,
        "apply_model",
        lambda noisy_z, timestep, condition: predicted_noise,
    )
    target = torch.rand(2, 4, 16, 16)

    loss, stats = module._diffusion_objective(
        z,
        torch.zeros_like(z),
        timesteps=torch.tensor([1, 9]),
        noise=torch.zeros_like(z),
        valid_mask=torch.ones(2, 1, 16, 16),
        decoded_target=target,
        decode_auxiliary=True,
    )
    loss.backward()

    assert stats["eps_mse"].item() == pytest.approx(0.0)
    assert stats["decoded_reconstruction_loss"].item() > 0
    assert stats["decoded_detail_loss"].item() > 0
    assert stats["decoded_perceptual_loss"].item() == pytest.approx(0.0)
    assert stats["decoded_active_samples"].item() == pytest.approx(1.0)
    assert predicted_noise.grad is not None
    assert predicted_noise.grad[0].abs().sum().item() > 0
    assert predicted_noise.grad[1].abs().sum().item() == pytest.approx(0.0)
    assert module._should_compute_decoded_auxiliary(0)
    assert not module._should_compute_decoded_auxiliary(1)
    assert module._should_compute_decoded_auxiliary(3)


def test_sharpness_does_not_change_native_keys_and_metadata_is_weights_only_safe(
    tmp_path, tiny_model_config
) -> None:
    baseline = _module(
        tiny_model_config,
        p2_gamma=0.0,
        latent_detail_weight=0.0,
    )
    sharp = _module(
        tiny_model_config,
        p2_gamma=1.0,
        latent_detail_weight=0.25,
        decoded_reconstruction_weight=0.1,
        decoded_detail_weight=0.1,
    )
    assert baseline.native_state_dict().keys() == sharp.native_state_dict().keys()

    checkpoint = {}
    sharp.on_save_checkpoint(checkpoint)
    metadata = checkpoint["opensr_inference_config"]
    denoiser = metadata["denoiser_settings"]
    assert denoiser["timesteps"] == sharp.diffusion.num_timesteps
    assert denoiser["linear_start"] == pytest.approx(sharp.diffusion.betas[0].item())
    assert denoiser["linear_end"] == pytest.approx(sharp.diffusion.betas[-1].item())
    assert denoiser["parameterization"] == "eps"

    path = tmp_path / "metadata.ckpt"
    torch.save(checkpoint, path)
    assert torch.load(path, weights_only=True) == checkpoint


def test_sharpness_configuration_rejects_typos(tiny_model_config) -> None:
    with pytest.raises(ValueError, match="Unknown sharpness_config"):
        _module(tiny_model_config, latent_details_weight=0.1)
