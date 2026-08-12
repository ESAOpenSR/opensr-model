from __future__ import annotations

import torch

from opensr_model.autoencoder.autoencoder import (
    AutoencoderKL,
    DiagonalGaussianDistribution,
)


def test_diagonal_gaussian_matches_closed_form_statistics() -> None:
    mean = torch.tensor([[[[1.0]], [[-2.0]]]])
    logvar = torch.log(torch.tensor([[[[4.0]], [[0.25]]]]))
    distribution = DiagonalGaussianDistribution(torch.cat((mean, logvar), dim=1))

    expected_kl = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).sum()
    expected_nll = 0.5 * (torch.log(torch.tensor(2.0 * torch.pi)) + logvar).sum()

    assert torch.equal(distribution.mode(), mean)
    assert torch.allclose(distribution.kl(), expected_kl.unsqueeze(0))
    assert torch.allclose(distribution.nll(mean), expected_nll.unsqueeze(0))


def test_deterministic_diagonal_gaussian_never_adds_noise() -> None:
    parameters = torch.randn(2, 6, 3, 3)
    distribution = DiagonalGaussianDistribution(parameters, deterministic=True)

    assert torch.equal(distribution.sample(), distribution.mean)
    assert torch.equal(distribution.kl(), torch.tensor([0.0]))
    assert torch.equal(distribution.nll(distribution.mean), torch.tensor([0.0]))


def test_tiny_autoencoder_round_trip_preserves_contract() -> None:
    config = {
        "embed_dim": 2,
        "double_z": True,
        "z_channels": 2,
        "resolution": 16,
        "in_channels": 4,
        "out_ch": 4,
        "ch": 32,
        "ch_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    model = AutoencoderKL(config, config["embed_dim"]).eval()
    image = torch.randn(2, 4, 16, 16)

    with torch.no_grad():
        posterior = model.encode(image)
        reconstruction = model.decode(posterior.mode())
        forward_reconstruction, forward_posterior = model(image, sample_posterior=False)

    assert posterior.mean.shape == (2, 2, 8, 8)
    assert reconstruction.shape == image.shape
    assert torch.equal(forward_posterior.mean, posterior.mean)
    assert torch.allclose(forward_reconstruction, reconstruction)
    assert torch.isfinite(reconstruction).all()
