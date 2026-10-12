from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest
import torch

from opensr_model.diffusion.dpm_solver import DPMSolverPlusPlusSampler


def noise_model():
    betas = torch.linspace(0.0015**0.5, 0.0155**0.5, 100).square()
    return SimpleNamespace(
        num_timesteps=100,
        parameterization="eps",
        betas=betas,
        apply_model=lambda x, t, c: torch.zeros_like(x),
    )


@pytest.mark.parametrize("steps", [1, 2, 15, 30, 99])
def test_zero_epsilon_matches_closed_form_and_exact_step_count(steps):
    pytest.importorskip("diffusers")
    model = noise_model()
    calls = []
    conditioning = torch.randn(2, 4, 8, 8)

    def predict(x, t, c):
        assert c is conditioning
        assert t.shape == (2,)
        assert t.dtype == torch.long
        assert t.device == x.device
        calls.append(t[0].item())
        return torch.zeros_like(x)

    model.apply_model = predict
    sampler = DPMSolverPlusPlusSampler(model, steps)
    initial = torch.randn_like(conditioning)
    latent = initial.clone()
    for t in sampler.timesteps:
        latent = sampler.step(latent, conditioning, t)

    alpha_bar = (1 - model.betas).cumprod(0)
    expected = initial / alpha_bar[sampler.timesteps[0]].sqrt()
    torch.testing.assert_close(latent, expected, rtol=2e-5, atol=2e-5)
    assert calls == sampler.timesteps
    assert len(calls) == len(set(calls)) == steps
    assert calls == sorted(calls, reverse=True)
    torch.testing.assert_close(sampler.scheduler.betas, model.betas)


def test_exact_clean_prediction_reaches_clean_sample_without_clipping():
    pytest.importorskip("diffusers")
    model = noise_model()
    # Latents are not constrained to pixel-space [0, 1] or [-1, 1].
    clean = torch.full((2, 4, 8, 8), 3.5)
    alpha_bar = (1 - model.betas).cumprod(0)

    def predict(x, t, c):
        alpha = alpha_bar[t].reshape(-1, 1, 1, 1)
        return (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()

    model.apply_model = predict
    sampler = DPMSolverPlusPlusSampler(model, 15)
    latent = torch.randn_like(clean)
    for t in sampler.timesteps:
        latent = sampler.step(latent, clean, t)
    torch.testing.assert_close(latent, clean)


@pytest.mark.parametrize("steps", [0, -1, 100, 101, True, 2.5, "20", None])
def test_invalid_step_counts_fail_before_importing_diffusers(steps):
    with pytest.raises(ValueError, match="sampling_steps"):
        DPMSolverPlusPlusSampler(noise_model(), steps)


def test_non_epsilon_model_is_rejected():
    model = noise_model()
    model.parameterization = "x0"
    with pytest.raises(ValueError, match="parameterization='eps'"):
        DPMSolverPlusPlusSampler(model, 20)


def test_missing_optional_dependency_has_installation_hint(monkeypatch):
    original = builtins.__import__

    def without_diffusers(name, *args, **kwargs):
        if name == "diffusers":
            raise ModuleNotFoundError("No module named 'diffusers'")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_diffusers)
    with pytest.raises(ImportError, match=r"opensr-model\[dpm\]"):
        DPMSolverPlusPlusSampler(noise_model(), 20)
