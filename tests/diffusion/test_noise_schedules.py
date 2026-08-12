from __future__ import annotations

import numpy as np
import pytest
import torch

from opensr_model.diffusion.latentdiffusion import DDPM, LatentDiffusion
from opensr_model.diffusion.utils import make_beta_schedule


def _schedule_model(
    schedule: str,
    *,
    timesteps: int = 100,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    v_posterior: float = 0.0,
    parameterization: str = "eps",
) -> DDPM:
    model = DDPM.__new__(DDPM)
    torch.nn.Module.__init__(model)
    model.v_posterior = v_posterior
    model.parameterization = parameterization
    DDPM.register_schedule(
        model,
        beta_schedule=schedule,
        timesteps=timesteps,
        linear_start=linear_start,
        linear_end=linear_end,
    )
    return model


@pytest.mark.parametrize("schedule", ["linear", "cosine", "sqrt_linear", "sqrt"])
def test_registered_noise_schedule_matches_every_defining_coefficient(schedule) -> None:
    model = _schedule_model(schedule, timesteps=100)
    beta = torch.from_numpy(
        make_beta_schedule(schedule, 100, linear_start=1e-4, linear_end=2e-2)
    ).to(torch.float64)
    alpha = 1.0 - beta
    alpha_bar = alpha.cumprod(0)
    alpha_bar_previous = torch.cat((torch.ones(1), alpha_bar[:-1]))

    assert torch.allclose(model.betas.double(), beta, rtol=1e-6, atol=1e-8)
    assert torch.allclose(model.alphas_cumprod.double(), alpha_bar, rtol=1e-6)
    assert torch.allclose(
        model.alphas_cumprod_prev.double(), alpha_bar_previous, rtol=1e-6
    )
    assert torch.allclose(
        model.sqrt_alphas_cumprod.double(), alpha_bar.sqrt(), rtol=1e-6
    )
    assert torch.allclose(
        model.sqrt_one_minus_alphas_cumprod.double(),
        (1.0 - alpha_bar).sqrt(),
        rtol=1e-6,
    )
    assert torch.allclose(
        model.sqrt_recip_alphas_cumprod.double(), alpha_bar.rsqrt(), rtol=1e-6
    )
    assert torch.allclose(
        model.sqrt_recipm1_alphas_cumprod.double(),
        (alpha_bar.reciprocal() - 1.0).sqrt(),
        rtol=1e-6,
    )
    noise = model.sqrt_one_minus_alphas_cumprod
    assert torch.all(noise[1:] > noise[:-1])


@pytest.mark.parametrize("schedule", ["linear", "cosine", "sqrt_linear", "sqrt"])
def test_q_sample_adds_exact_schedule_noise_for_fixed_draw(schedule) -> None:
    model = _schedule_model(schedule, timesteps=32)
    clean = torch.linspace(-1.0, 1.0, 32).reshape(32, 1, 1, 1)
    fixed_noise = torch.linspace(1.0, -1.0, 32).reshape(32, 1, 1, 1)
    timestep = torch.arange(32)

    noisy = LatentDiffusion.q_sample(model, clean, timestep, fixed_noise)
    signal_coefficient = model.sqrt_alphas_cumprod.reshape(32, 1, 1, 1)
    noise_coefficient = model.sqrt_one_minus_alphas_cumprod.reshape(32, 1, 1, 1)
    expected = signal_coefficient * clean + noise_coefficient * fixed_noise

    assert torch.equal(noisy, expected)
    recovered_noise = (noisy - signal_coefficient * clean) / noise_coefficient
    assert torch.allclose(recovered_noise, fixed_noise, atol=1e-6)


@pytest.mark.parametrize("schedule", ["linear", "cosine", "sqrt_linear", "sqrt"])
def test_empirical_forward_noise_mean_and_variance_match_schedule(schedule) -> None:
    model = _schedule_model(schedule, timesteps=100)
    selected = torch.tensor([0, 49, 99])
    clean = torch.ones(3, 100_000)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(1234))

    noisy = LatentDiffusion.q_sample(model, clean, selected, noise)
    expected_mean = model.sqrt_alphas_cumprod[selected]
    expected_variance = 1.0 - model.alphas_cumprod[selected]
    mean_standard_error = (expected_variance / clean.shape[1]).sqrt()

    assert torch.all(
        (noisy.mean(1) - expected_mean).abs() < 4 * mean_standard_error + 1e-5
    )
    assert torch.allclose(noisy.var(1), expected_variance, rtol=0.02, atol=2e-5)


def test_posterior_noise_variance_matches_q_posterior_formula() -> None:
    model = _schedule_model("linear", timesteps=50, v_posterior=0.35)
    beta = model.betas.double()
    alpha_bar = model.alphas_cumprod.double()
    alpha_bar_previous = model.alphas_cumprod_prev.double()
    beta_tilde = beta * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
    expected_variance = 0.65 * beta_tilde + 0.35 * beta

    assert torch.allclose(
        model.posterior_variance.double(), expected_variance, rtol=1e-5, atol=1e-8
    )
    assert torch.allclose(
        model.posterior_log_variance_clipped.double(),
        expected_variance.clamp_min(1e-20).log(),
        rtol=1e-5,
    )


def test_explicit_beta_schedule_controls_noise_without_reinterpretation() -> None:
    beta = np.array([0.01, 0.05, 0.2], dtype=np.float64)
    model = DDPM.__new__(DDPM)
    torch.nn.Module.__init__(model)
    model.v_posterior = 0.0
    model.parameterization = "x0"

    DDPM.register_schedule(model, given_betas=beta)

    expected_alpha_bar = torch.tensor([0.99, 0.99 * 0.95, 0.99 * 0.95 * 0.8])
    assert torch.allclose(model.betas, torch.from_numpy(beta).float())
    assert torch.allclose(model.alphas_cumprod, expected_alpha_bar)
    assert model.num_timesteps == 3
    assert torch.isfinite(model.lvlb_weights).all()
