"""Optional losses must respect masks, band selection, and frozen backbones."""

from copy import deepcopy
from types import SimpleNamespace

import lpips
import pytest
import torch
from omegaconf import OmegaConf

from opensr_model.training.modules.autoencoder import AutoencoderTrainingModule
from opensr_model.training.modules.diffusion import DiffusionTrainingModule


class PerceptualProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.inputs = None

    def forward(self, a, b):
        self.inputs = (a.detach().clone(), b.detach().clone())
        return self.weight * (a - b).square().mean((1, 2, 3), keepdim=True)


@pytest.mark.parametrize("mask_channels", [1, 4])
def test_decoded_rgb_and_perceptual_losses_ignore_masked_pixels_and_nir(
    tiny_model_config, monkeypatch, mask_channels
):
    probe = PerceptualProbe()
    monkeypatch.setattr(lpips, "LPIPS", lambda **kwargs: probe)
    module = DiffusionTrainingModule(
        tiny_model_config,
        validation_num_samples=0,
        sharpness_config={
            "decoded_rgb_only": True,
            "decoded_perceptual_weight": 0.1,
            "decoded_reconstruction_weight": 0.1,
            "decoded_detail_weight": 0.1,
        },
    )
    module.train()
    assert not probe.training and not probe.weight.requires_grad
    decoded = torch.full((1, 4, 16, 16), 0.3, requires_grad=True)
    target = torch.full_like(decoded, 0.2)
    target[:, 3] = 0.9
    mask = torch.ones(1, mask_channels, 16, 16)
    mask[..., 6:10, 6:10] = 0
    monkeypatch.setattr(module, "_decode_first_stage_float32", lambda _: decoded)
    reconstruction, detail, perceptual, active = module._decoded_auxiliary_losses(
        torch.zeros(1, 2, 8, 8), target, torch.tensor([0]), mask
    )
    assert reconstruction.item() == pytest.approx(0.1)
    assert perceptual.item() == pytest.approx(0.04 * 240 / 256)
    assert active.item() == 1
    a, b = probe.inputs
    assert a.shape[1] == 3
    torch.testing.assert_close(a[..., 6:10, 6:10], b[..., 6:10, 6:10])
    (reconstruction + detail + perceptual).backward()
    assert decoded.grad[:, :3].abs().sum() > 0
    assert decoded.grad[:, 3].count_nonzero() == 0
    assert decoded.grad[..., 6:10, 6:10].count_nonzero() == 0


def test_decoded_loss_skips_ineligible_timesteps_without_decoding(
    tiny_model_config, monkeypatch
):
    module = DiffusionTrainingModule(
        tiny_model_config,
        validation_num_samples=0,
        sharpness_config={"decoded_max_timestep_fraction": 0.1},
    )

    def unexpected(*args):
        pytest.fail("High-timestep latent should not be decoded")

    monkeypatch.setattr(module, "_decode_first_stage_float32", unexpected)
    z = torch.randn(1, 2, 8, 8, requires_grad=True)
    losses = module._decoded_auxiliary_losses(
        z, torch.zeros(1, 4, 16, 16), torch.tensor([9]), None
    )
    assert all(loss.item() == 0 for loss in losses)
    sum(losses[:3]).backward()
    assert z.grad.count_nonzero() == 0


def test_autoencoder_perceptual_backbone_is_frozen_and_receives_reflectance(
    tiny_model_config, tiny_autoencoder_training_config, monkeypatch
):
    probe = PerceptualProbe()
    monkeypatch.setattr(lpips, "LPIPS", lambda **kwargs: probe)
    config = deepcopy(tiny_autoencoder_training_config)
    config["loss"]["perceptual_weight"] = 1.0
    module = AutoencoderTrainingModule(
        OmegaConf.create(tiny_model_config), OmegaConf.create(config)
    )
    module.train()
    assert not probe.training and not probe.weight.requires_grad
    image = torch.full((1, 3, 16, 16), 0.2)
    prediction = torch.full_like(image, 0.3, requires_grad=True)
    loss = module._perceptual(prediction, image).mean()
    assert loss.item() == pytest.approx(0.01)
    torch.testing.assert_close(probe.inputs[0], image)
    torch.testing.assert_close(probe.inputs[1], prediction)
    loss.backward()
    assert prediction.grad.abs().sum() > 0
    assert probe.weight.grad is None


def test_autoencoder_mask_preserves_target_and_blocks_invalid_gradient():
    prediction = torch.ones(2, 4, 3, 3, requires_grad=True)
    target = torch.zeros_like(prediction, requires_grad=True)
    mask = torch.ones(2, 3, 3)
    mask[:, 0, 0] = float("nan")
    masked = AutoencoderTrainingModule._mask_prediction(prediction, target, mask)
    assert masked[:, :, 0, 0].count_nonzero() == 0
    masked.sum().backward()
    assert prediction.grad[:, :, 0, 0].count_nonzero() == 0
    assert target.grad is None
    assert (
        AutoencoderTrainingModule._mask_prediction(prediction, target, None)
        is prediction
    )


@pytest.mark.parametrize(
    "mask", [torch.ones(1, 3, 3), torch.ones(2, 2), torch.ones(2, 1, 2, 3)]
)
def test_autoencoder_mask_requires_matching_batch_and_grid(mask):
    image = torch.ones(2, 4, 3, 3)
    with pytest.raises(ValueError, match="valid_mask"):
        AutoencoderTrainingModule._mask_prediction(image, image, mask)


def test_three_band_objective_keeps_all_rgb_and_rejects_incompatible_shapes():
    image = torch.randn(2, 3, 8, 8)
    reference, prediction, selected = AutoencoderTrainingModule._random_three_band_pair(
        image, image + 1
    )
    torch.testing.assert_close(reference, image)
    torch.testing.assert_close(prediction, image + 1)
    assert selected.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="shapes"):
        AutoencoderTrainingModule._random_three_band_pair(image, image[:1])
    with pytest.raises(ValueError, match="at least three"):
        AutoencoderTrainingModule._random_three_band_pair(image[:, :2], image[:, :2])


def test_autoencoder_plateau_scheduler_waits_for_validation_metric(
    tiny_model_config, tiny_autoencoder_training_config, monkeypatch
):
    config = deepcopy(tiny_autoencoder_training_config)
    config["loss"].update(
        discriminator_factor=1.0, discriminator_channels=8, discriminator_layers=1
    )
    module = AutoencoderTrainingModule(tiny_model_config, config)
    optimizer = torch.optim.SGD(module.autoencoder.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=0, factor=0.5
    )
    module._manual_scheduler_metadata = [{"monitor": "val/loss"}]
    module._trainer = SimpleNamespace(sanity_checking=False, callback_metrics={})
    monkeypatch.setattr(module, "lr_schedulers", lambda: scheduler)
    with pytest.raises(RuntimeError, match="was not logged"):
        module.on_validation_epoch_end()
    module._trainer.callback_metrics["val/loss"] = torch.tensor(1.0)
    module.on_validation_epoch_end()
    module._trainer.callback_metrics["val/loss"] = torch.tensor(2.0)
    module.on_validation_epoch_end()
    assert optimizer.param_groups[0]["lr"] == 0.05


def test_validation_sampler_is_reproducible_without_advancing_global_rng(
    tiny_model_config,
):
    module = DiffusionTrainingModule(
        tiny_model_config, validation_num_samples=0, validation_sampling_steps=5
    )
    lr = torch.full((1, 4, 4, 4), 0.2)
    before = torch.random.get_rng_state().clone()
    first = module.sample_super_resolution(
        lr, seed=31, eta=0.5, temperature=0.7, use_ema=False
    )
    second = module.sample_super_resolution(
        lr, seed=31, eta=0.5, temperature=0.7, use_ema=False
    )
    assert first.shape == (1, 4, 16, 16)
    assert torch.isfinite(first).all() and first.min() >= 0 and first.max() <= 1
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_nondividing_validation_ddim_steps_are_rejected(tiny_model_config):
    module = DiffusionTrainingModule(tiny_model_config, validation_num_samples=0)
    with pytest.raises(ValueError, match="divides"):
        module.sample_super_resolution(torch.ones(1, 4, 4, 4), steps=4)


def test_training_rejects_parameterization_incompatible_with_public_inference(
    tiny_model_config,
):
    config = deepcopy(tiny_model_config)
    config["denoiser_settings"]["parameterization"] = "x0"
    with pytest.raises(ValueError, match="training requires parameterization='eps'"):
        DiffusionTrainingModule(config)
