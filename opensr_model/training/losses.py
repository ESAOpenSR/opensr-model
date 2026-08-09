"""Training-only losses; none of these modules leak into native model exports."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import masked_mean


def reconstruction_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return masked_mean((prediction - target).abs(), mask)


def reconstruction_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return masked_mean((prediction - target).square(), mask)


class PatchDiscriminator(nn.Module):
    """The ``NLayerDiscriminator`` used by the released autoencoder recipe.

    This intentionally mirrors the PatchGAN imported by the checkpoint-era
    latent-diffusion code: 4x4 convolutions, BatchNorm, and normal ``N(0, 0.02)``
    convolution weights. Autoencoder training always feeds it three selected
    bands even though the native autoencoder itself remains four-channel.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_layers: int = 3,
        max_channels: int = 512,
        use_actnorm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if use_actnorm:
            raise ValueError(
                "The checkpoint-era autoencoder used BatchNorm, not ActNorm"
            )

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for level in range(1, num_layers):
            next_channels = min(base_channels * 2**level, max_channels)
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels
        next_channels = min(channels * 2, max_channels)
        layers.extend(
            [
                nn.Conv2d(channels, next_channels, 4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(next_channels, 1, 4, stride=1, padding=1),
            ]
        )
        self.network = nn.Sequential(*layers)
        self.apply(_legacy_patchgan_weights_init)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


def _legacy_patchgan_weights_init(module: nn.Module) -> None:
    """Initialization from ``taming.modules.discriminator.model.weights_init``."""

    class_name = module.__class__.__name__
    if "Conv" in class_name and hasattr(module, "weight"):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name and hasattr(module, "weight"):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def discriminator_hinge_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def discriminator_vanilla_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean())


def generator_adversarial_loss(
    fake_logits: torch.Tensor, kind: str = "hinge"
) -> torch.Tensor:
    if kind == "hinge":
        return -fake_logits.mean()
    if kind == "vanilla":
        return F.softplus(-fake_logits).mean()
    raise ValueError(
        f"Unknown adversarial loss {kind!r}; expected 'hinge' or 'vanilla'"
    )


def adaptive_generator_weight(
    reconstruction_loss: torch.Tensor,
    adversarial_loss: torch.Tensor,
    last_layer: torch.Tensor,
    *,
    discriminator_weight: float = 0.5,
) -> torch.Tensor:
    """Legacy gradient-norm balancing for the generator's GAN contribution."""

    reconstruction_gradient = torch.autograd.grad(
        reconstruction_loss, last_layer, retain_graph=True
    )[0]
    adversarial_gradient = torch.autograd.grad(
        adversarial_loss, last_layer, retain_graph=True
    )[0]
    weight = torch.norm(reconstruction_gradient) / (
        torch.norm(adversarial_gradient) + 1e-4
    )
    return weight.clamp(0.0, 1e4).detach() * float(discriminator_weight)


__all__ = [
    "PatchDiscriminator",
    "adaptive_generator_weight",
    "discriminator_hinge_loss",
    "discriminator_vanilla_loss",
    "generator_adversarial_loss",
    "reconstruction_l1",
    "reconstruction_l2",
]
