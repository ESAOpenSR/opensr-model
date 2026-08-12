from __future__ import annotations

import pytest
import torch

from opensr_model.training.losses import (
    discriminator_hinge_loss,
    discriminator_vanilla_loss,
    generator_adversarial_loss,
    reconstruction_l1,
    reconstruction_l2,
)
from opensr_model.training.optim import build_adamw, build_scheduler


def test_masked_reconstruction_losses_ignore_invalid_pixels() -> None:
    prediction = torch.tensor([[[[1.0, 100.0], [3.0, 100.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    assert reconstruction_l1(prediction, target, mask) == 2.0
    assert reconstruction_l2(prediction, target, mask) == 5.0


def test_adversarial_losses_match_their_reference_formulas() -> None:
    real = torch.tensor([2.0, 0.0])
    fake = torch.tensor([-2.0, 0.0])

    assert discriminator_hinge_loss(real, fake) == 0.5
    assert torch.allclose(
        discriminator_vanilla_loss(real, fake),
        0.5
        * (
            torch.nn.functional.softplus(-real).mean()
            + torch.nn.functional.softplus(fake).mean()
        ),
    )
    assert generator_adversarial_loss(fake, "hinge") == 1.0
    assert torch.allclose(
        generator_adversarial_loss(fake, "vanilla"),
        torch.nn.functional.softplus(-fake).mean(),
    )
    with pytest.raises(ValueError, match="Unknown adversarial loss"):
        generator_adversarial_loss(fake, "typo")


def test_optimizer_and_cosine_scheduler_honor_configuration() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = build_adamw(
        [parameter],
        {"learning_rate": 0.01, "betas": [0.8, 0.9], "weight_decay": 0.1},
    )
    scheduler_config = build_scheduler(
        optimizer,
        {"name": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.1},
        total_steps=6,
    )

    assert optimizer.defaults["betas"] == (0.8, 0.9)
    assert optimizer.defaults["weight_decay"] == 0.1
    assert scheduler_config is not None
    assert scheduler_config["interval"] == "step"
    scheduler = scheduler_config["scheduler"]
    rates = [scheduler.get_last_lr()[0]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0])
    assert rates[0] < rates[1]
    assert rates[-1] == pytest.approx(0.001)


def test_scheduler_rejects_unknown_name_and_bad_betas() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="exactly two"):
        build_adamw([parameter], {"betas": [0.9]})
    optimizer = build_adamw([parameter])
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(optimizer, {"name": "cyclic"}, total_steps=10)
