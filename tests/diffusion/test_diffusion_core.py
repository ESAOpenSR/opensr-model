from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from opensr_model.diffusion.utils import (
    DDIMSampler,
    extract_into_tensor,
    make_beta_schedule,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


@pytest.mark.parametrize("schedule", ["linear", "cosine", "sqrt_linear", "sqrt"])
def test_beta_schedules_are_finite_bounded_and_have_requested_length(schedule) -> None:
    betas = make_beta_schedule(schedule, 20, linear_start=1e-4, linear_end=2e-2)

    assert betas.shape == (20,)
    assert np.isfinite(betas).all()
    assert ((betas > 0) & (betas < 1)).all()


def test_uniform_ddim_schedule_and_eta_zero_are_deterministic() -> None:
    timesteps = make_ddim_timesteps("uniform", 4, 20, verbose=False)
    alphas = np.linspace(0.99, 0.7, 20)
    sigmas, selected, previous = make_ddim_sampling_parameters(
        alphas, timesteps, eta=0.0, verbose=False
    )

    assert np.array_equal(timesteps, np.array([1, 6, 11, 16]))
    assert np.array_equal(selected, alphas[timesteps])
    assert previous[0] == alphas[0]
    assert np.count_nonzero(sigmas) == 0


def test_q_sample_uses_the_requested_timestep_coefficients() -> None:
    diffusion = SimpleNamespace(
        sqrt_alphas_cumprod=torch.tensor([1.0, 0.5]),
        sqrt_one_minus_alphas_cumprod=torch.tensor([0.0, 0.25]),
    )
    clean = torch.full((2, 1, 2, 2), 4.0)
    noise = torch.full_like(clean, 8.0)
    timesteps = torch.tensor([0, 1])

    result = LatentDiffusion.q_sample(diffusion, clean, timesteps, noise)

    assert torch.equal(result[0], clean[0])
    assert torch.equal(result[1], torch.full((1, 2, 2), 4.0))
    assert extract_into_tensor(
        torch.tensor([2.0, 3.0]), timesteps, clean.shape
    ).shape == (
        2,
        1,
        1,
        1,
    )


def test_ddim_reverse_step_with_zero_noise_matches_closed_form() -> None:
    class ZeroNoiseModel:
        num_timesteps = 4
        device = torch.device("cpu")
        alphas_cumprod = torch.tensor([0.9, 0.8, 0.7, 0.6])
        alphas_cumprod_prev = torch.tensor([0.9, 0.9, 0.8, 0.7])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        betas = torch.ones(4) * 0.1

        @staticmethod
        def apply_model(x, timestep, conditioning):
            return torch.zeros_like(x)

    sampler = DDIMSampler(ZeroNoiseModel())
    sampler.make_schedule(ddim_num_steps=2, ddim_eta=0.0, verbose=False)
    latent = torch.ones(2, 1, 2, 2)

    previous, predicted_clean = sampler.p_sample_ddim(
        latent, torch.zeros_like(latent), t=3, index=1, temperature=100.0
    )

    expected_clean = latent / torch.as_tensor(sampler.ddim_alphas[1]).sqrt()
    expected_previous = (
        torch.as_tensor(sampler.ddim_alphas_prev[1]).sqrt() * expected_clean
    )
    assert torch.allclose(predicted_clean, expected_clean)
    assert torch.allclose(previous, expected_previous)


def test_repeated_noise_reuses_one_draw_for_every_batch_item() -> None:
    repeated = noise_like((4, 2, 3, 3), "cpu", repeat=True)
    independent = noise_like((4, 2, 3, 3), "cpu", repeat=False)

    assert torch.equal(repeated[0], repeated[1])
    assert not torch.equal(independent[0], independent[1])
