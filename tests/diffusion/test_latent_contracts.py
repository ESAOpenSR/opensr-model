"""Latent encoding, conditioning, EMA restoration and schedule contracts."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from opensr_model.autoencoder.autoencoder import DiagonalGaussianDistribution
from opensr_model.diffusion.latentdiffusion import DDPM, LatentDiffusion
from opensr_model.srmodel import SRLatentDiffusion


@pytest.fixture
def model(tiny_model_config):
    return SRLatentDiffusion(OmegaConf.create(tiny_model_config)).model


def test_ema_scope_restores_training_weights_even_on_exception(model, capsys):
    parameter = next(model.model.parameters())
    with torch.no_grad():
        parameter.add_(2)
    original = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="inside"):
        with model.ema_scope("validation"):
            torch.testing.assert_close(parameter, original - 2)
            raise RuntimeError("inside")
    torch.testing.assert_close(parameter, original)
    assert "Restored training weights" in capsys.readouterr().out
    model.use_ema = False
    with model.ema_scope():
        torch.testing.assert_close(parameter, original)


def test_first_stage_encoding_scaling_and_invalid_type(model):
    model.scale_factor = 2
    x = torch.randn(1, 2, 4, 4)
    torch.testing.assert_close(model.get_first_stage_encoding(x), x * 2)
    posterior = DiagonalGaussianDistribution(
        torch.cat([x, torch.zeros_like(x)], 1), deterministic=True
    )
    torch.testing.assert_close(model.get_first_stage_encoding(posterior), x * 2)
    with pytest.raises(NotImplementedError, match="encoder_posterior"):
        model.get_first_stage_encoding({})
    model.split_input_params = {"patch_distributed_vq": False}
    torch.testing.assert_close(
        model.decode_first_stage(x), model.first_stage_model.decode(x / 2)
    )


@pytest.mark.parametrize("mode", ["identity", "encode", "posterior", "custom"])
def test_conditioning_supports_encoder_modes(model, mode):
    class Conditioner(nn.Module):
        def encode(self, x):
            return (
                DiagonalGaussianDistribution(torch.cat([x * 2, torch.zeros_like(x)], 1))
                if mode == "posterior"
                else x * 2
            )

        def custom(self, x):
            return x + 3

    x = torch.randn(1, 2, 4, 4)
    if mode != "identity":
        model.cond_stage_model = Conditioner()
    if mode == "custom":
        model.cond_stage_forward = "custom"
    expected = x if mode == "identity" else x + 3 if mode == "custom" else x * 2
    torch.testing.assert_close(model.get_learned_conditioning(x), expected)
    model.cond_stage_forward = "missing"
    with pytest.raises(AssertionError):
        model.get_learned_conditioning(x)


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_get_input_returns_aligned_truncated_latents_and_reconstructions(
    model, conditional, trainable
):
    model.model.conditioning_key = "concat" if conditional else None
    model.cond_stage_trainable = trainable
    images = torch.randn(2, 4, 16, 16)
    condition = torch.randn(2, 2, 8, 8)
    latent, encoded_cond, original, reconstruction, original_cond = model.get_input(
        {"image": images, "LR_image": condition},
        "image",
        bs=1,
        return_first_stage_outputs=True,
        return_original_cond=True,
    )
    assert latent.shape == (1, 2, 8, 8) and not latent.requires_grad
    torch.testing.assert_close(original, images[:1])
    torch.testing.assert_close(reconstruction, model.decode_first_stage(latent))
    if conditional:
        torch.testing.assert_close(encoded_cond, condition[:1])
        torch.testing.assert_close(original_cond, condition)
    else:
        assert encoded_cond is None and original_cond is None


def test_base_input_converts_to_float_and_expands_grayscale(model):
    x = torch.ones(2, 4, 4, dtype=torch.int16)
    result = DDPM.get_input(model, {"image": x}, "image")
    assert result.shape == (2, 4, 4, 1) and result.dtype == torch.float32


def test_short_condition_schedule_and_buffer_scale(tiny_model_config):
    model = LatentDiffusion(
        tiny_model_config["first_stage_config"],
        tiny_model_config["cond_stage_config"],
        unet_config=tiny_model_config["cond_stage_config"],
        timesteps=10,
        num_timesteps_cond=3,
        scale_by_std=True,
        scale_factor=2,
        use_ema=False,
    )
    assert "scale_factor" in dict(model.named_buffers())
    assert model.shorten_cond_schedule
    torch.testing.assert_close(
        model.cond_ids, torch.tensor([0, 4, 9, 9, 9, 9, 9, 9, 9, 9])
    )
    assert all(not p.requires_grad for p in model.first_stage_model.parameters())


@pytest.mark.parametrize("form", ["tensor", "list", "dict"])
def test_apply_model_conditioning_forms_are_equivalent(model, form):
    x = torch.randn(1, 2, 8, 8)
    condition = torch.randn_like(x)
    t = torch.zeros(1, dtype=torch.long)
    supplied = (
        condition
        if form == "tensor"
        else [condition] if form == "list" else {"c_concat": [condition]}
    )
    expected = model.model.diffusion_model(torch.cat([x, condition], 1), t)
    torch.testing.assert_close(model.apply_model(x, t, supplied), expected)
