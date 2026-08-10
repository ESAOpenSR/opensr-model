"""PyTorch Lightning training module for the native OpenSR diffusion model.

The inference checkpoint contract in :mod:`opensr_model.srmodel` is the bare
``LatentDiffusion.state_dict()`` stored under a top-level ``"state_dict"``
key.  This module deliberately keeps Lightning's training state separate from
that contract; :meth:`native_state_dict` is the integration point used by the
native-checkpoint callback.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from opensr_model.diffusion.utils import DDIMSampler
from opensr_model.utils import linear_transform_4b, linear_transform_placeholder

from .metrics import (
    batch_clipped_fraction,
    compute_reconstruction_metrics,
    compute_rgb_nir_reconstruction_metrics,
    downsample_consistency,
    select_channel_mask,
)

_MISSING = object()

_SHARPNESS_DEFAULTS: dict[str, Any] = {
    # P2 weighting de-emphasizes the already easy, very-high-SNR denoising
    # states. gamma=0 restores the uniform epsilon-MSE objective exactly.
    "p2_gamma": 1.0,
    "p2_k": 1.0,
    # A cheap x0-space latent Laplacian loss gives low-noise predictions an
    # explicit incentive to preserve spatial detail.
    "latent_detail_weight": 0.05,
    "latent_detail_max_timestep_fraction": 0.25,
    # Decoding predicted x0 is substantially more expensive at 512 px. These
    # terms are opt-in and cadence/batch bounded.
    "decoded_reconstruction_weight": 0.0,
    "decoded_detail_weight": 0.0,
    "decoded_perceptual_weight": 0.0,
    "decoded_max_timestep_fraction": 0.20,
    "decoded_every_n_batches": 8,
    "decoded_max_batch_size": 1,
    "decoded_rgb_only": True,
    "perceptual_backbone": "vgg",
}


def _config_value(config: Any, *path: str, default: Any = _MISSING) -> Any:
    """Read a value from a nested mapping, dataclass-like object, or DictConfig."""

    value = config
    for part in path:
        if isinstance(value, Mapping):
            if part not in value:
                if default is not _MISSING:
                    return default
                raise KeyError(".".join(path))
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        elif default is not _MISSING:
            return default
        else:
            raise KeyError(".".join(path))
    return value


def _mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def _laplacian(tensor: torch.Tensor) -> torch.Tensor:
    """Return a channel-wise four-neighbour Laplacian without parameters."""

    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(tensor.shape)}")
    kernel = tensor.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).reshape(1, 1, 3, 3)
    return F.conv2d(
        tensor,
        kernel.expand(tensor.shape[1], 1, 3, 3),
        padding=1,
        groups=tensor.shape[1],
    )


def _masked_per_sample_mean(
    values: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Reduce BCHW values independently for every sample."""

    if mask is None:
        return _mean_flat(values)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != values.ndim or mask.shape[0] != values.shape[0]:
        raise ValueError("mask must be Bx1xHxW or BxCxHxW and match values")
    if mask.shape[-2:] != values.shape[-2:]:
        raise ValueError("mask and values must have the same spatial shape")
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand(-1, values.shape[1], -1, -1)
    if mask.shape[1] != values.shape[1]:
        raise ValueError("mask must have one channel or match the value channels")
    reduce_dims = tuple(range(1, values.ndim))
    return (values * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(
        1.0
    )


def _resolve_sharpness_config(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    values = dict(_SHARPNESS_DEFAULTS)
    if config is not None:
        unknown = set(config) - set(values)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"Unknown sharpness_config setting(s): {names}")
        values.update(dict(config))

    nonnegative = (
        "p2_gamma",
        "latent_detail_weight",
        "decoded_reconstruction_weight",
        "decoded_detail_weight",
        "decoded_perceptual_weight",
    )
    for name in nonnegative:
        value = float(values[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"sharpness_config.{name} must be finite and non-negative")
        values[name] = value

    p2_k = float(values["p2_k"])
    if not math.isfinite(p2_k) or p2_k <= 0:
        raise ValueError("sharpness_config.p2_k must be finite and positive")
    values["p2_k"] = p2_k

    for name in (
        "latent_detail_max_timestep_fraction",
        "decoded_max_timestep_fraction",
    ):
        value = float(values[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"sharpness_config.{name} must be in [0, 1]")
        values[name] = value

    for name in ("decoded_every_n_batches", "decoded_max_batch_size"):
        value = int(values[name])
        if value <= 0:
            raise ValueError(f"sharpness_config.{name} must be positive")
        values[name] = value
    values["decoded_rgb_only"] = bool(values["decoded_rgb_only"])
    values["perceptual_backbone"] = str(values["perceptual_backbone"])
    if not values["perceptual_backbone"]:
        raise ValueError("sharpness_config.perceptual_backbone must not be empty")
    return values


class DiffusionTrainingModule(pl.LightningModule):
    """Train the existing conditional ``LatentDiffusion`` without modifying it.

    The first-stage autoencoder is always frozen.  For each paired batch, the
    HR target is encoded to a 4-channel latent while the LR image is bilinearly
    enlarged by 4x and encoded by the same autoencoder.  The latter exactly
    matches ``SRLatentDiffusion._tensor_encode`` and the published checkpoint's
    concatenative conditioning path.

    Batches are mappings containing ``image`` (HR) and ``LR_image`` (LR) by
    default.  Both tensors must be float-compatible ``B x C x H x W`` tensors;
    for the bundled model the shapes are ``B x 4 x 512 x 512`` and
    ``B x 4 x 128 x 128`` respectively.
    """

    checkpoint_stage = "diffusion"

    def __init__(
        self,
        config: Any,
        *,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        scheduler_factor: float = 0.5,
        scheduler_patience: Optional[int] = 1000,
        scheduler_min_lr: float = 0.0,
        hr_key: Optional[str] = None,
        lr_key: Optional[str] = None,
        pretrained_checkpoint: Optional[str | Path] = None,
        autoencoder_checkpoint: Optional[str | Path] = None,
        sharpness_config: Optional[Mapping[str, Any]] = None,
        validation_num_samples: int = 4,
        validation_image_batches: int = 1,
        validation_sample_every_n_epochs: int = 1,
        validation_sampling_steps: int = 100,
        validation_sampling_eta: float = 0.95,
        validation_sampling_temperature: float = 1.0,
        validation_seed: int = 0,
        validation_use_ema: bool = True,
    ) -> None:
        super().__init__()

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if scheduler_patience is not None and scheduler_patience < 0:
            raise ValueError("scheduler_patience must be non-negative or None")
        if not 0.0 < scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between 0 and 1")
        if validation_num_samples < 0 or validation_image_batches < 0:
            raise ValueError("validation image counts must be non-negative")
        if validation_sample_every_n_epochs <= 0:
            raise ValueError("validation_sample_every_n_epochs must be positive")
        if validation_sampling_steps <= 0:
            raise ValueError("validation_sampling_steps must be positive")
        if validation_sampling_eta < 0:
            raise ValueError("validation_sampling_eta must be non-negative")
        if validation_sampling_temperature < 0:
            raise ValueError("validation_sampling_temperature must be non-negative")

        self.model_config = config
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.adam_betas = tuple(float(value) for value in adam_betas)
        self.scheduler_factor = float(scheduler_factor)
        self.scheduler_patience = scheduler_patience
        self.scheduler_min_lr = float(scheduler_min_lr)
        self.sharpness_config = _resolve_sharpness_config(sharpness_config)

        self.hr_key = hr_key or str(
            _config_value(config, "other", "first_stage_key", default="image")
        )
        self.lr_key = lr_key or str(
            _config_value(config, "other", "cond_stage_key", default="LR_image")
        )
        self.encode_conditioning = bool(
            _config_value(config, "encode_conditioning", default=True)
        )
        self.apply_normalization = bool(
            _config_value(config, "apply_normalization", default=False)
        )
        if self.apply_normalization:
            raise ValueError(
                "Training currently supports only apply_normalization=false, the "
                "released v1 inference contract"
            )
        self.linear_transform = (
            linear_transform_4b
            if self.apply_normalization
            else linear_transform_placeholder
        )

        self.validation_num_samples = int(validation_num_samples)
        self.validation_image_batches = int(validation_image_batches)
        self.validation_sample_every_n_epochs = int(validation_sample_every_n_epochs)
        self.validation_sampling_steps = int(validation_sampling_steps)
        self.validation_sampling_eta = float(validation_sampling_eta)
        self.validation_sampling_temperature = float(validation_sampling_temperature)
        self.validation_seed = int(validation_seed)
        self.validation_use_ema = bool(validation_use_ema)

        self.decoded_perceptual_loss: Optional[torch.nn.Module] = None
        if self.sharpness_config["decoded_perceptual_weight"] > 0:
            self.decoded_perceptual_loss = self._build_perceptual_loss(
                self.sharpness_config["perceptual_backbone"]
            )

        first_stage_config = _config_value(config, "first_stage_config")
        unet_config = _config_value(config, "cond_stage_config")
        parameterization = str(
            _config_value(
                config, "denoiser_settings", "parameterization", default="eps"
            )
        )
        if parameterization != "eps":
            raise ValueError(
                "Current SRLatentDiffusion inference always interprets denoiser "
                "outputs as epsilon; training requires parameterization='eps'"
            )

        self.diffusion = LatentDiffusion(
            first_stage_config=first_stage_config,
            cond_stage_config=unet_config,
            timesteps=int(
                _config_value(config, "denoiser_settings", "timesteps", default=1000)
            ),
            unet_config=unet_config,
            linear_start=float(
                _config_value(
                    config, "denoiser_settings", "linear_start", default=1.5e-3
                )
            ),
            linear_end=float(
                _config_value(
                    config, "denoiser_settings", "linear_end", default=1.55e-2
                )
            ),
            concat_mode=bool(
                _config_value(config, "other", "concat_mode", default=True)
            ),
            cond_stage_trainable=bool(
                _config_value(config, "other", "cond_stage_trainable", default=False)
            ),
            first_stage_key=self.hr_key,
            cond_stage_key=self.lr_key,
            parameterization=parameterization,
            use_ema=True,
        )
        # ``LatentDiffusion`` predates device-agnostic Lightning code and some
        # of its sampling utilities read this ordinary Python attribute.
        self.diffusion.device = torch.device("cpu")

        if self.diffusion.model.conditioning_key != "concat":
            raise ValueError(
                "OpenSR diffusion training requires concat conditioning; "
                f"got {self.diffusion.model.conditioning_key!r}"
            )
        if self.diffusion.cond_stage_trainable:
            raise ValueError("The OpenSR conditioning stage must remain frozen")

        self._freeze_first_stage()
        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint, strict=True)
        if autoencoder_checkpoint is not None:
            from .checkpoints import load_autoencoder_weights

            load_autoencoder_weights(
                self.first_stage_model, autoencoder_checkpoint, strict=True
            )
            self._freeze_first_stage()

        # Keep only small, reconstructable training settings in Lightning's
        # hyperparameter metadata.  The model config is supplied by the CLI.
        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "adam_betas": self.adam_betas,
                "scheduler_factor": self.scheduler_factor,
                "scheduler_patience": self.scheduler_patience,
                "scheduler_min_lr": self.scheduler_min_lr,
                "hr_key": self.hr_key,
                "lr_key": self.lr_key,
                "sharpness_config": dict(self.sharpness_config),
                "validation_num_samples": self.validation_num_samples,
                "validation_image_batches": self.validation_image_batches,
                "validation_sample_every_n_epochs": self.validation_sample_every_n_epochs,
                "validation_sampling_steps": self.validation_sampling_steps,
                "validation_sampling_eta": self.validation_sampling_eta,
                "validation_sampling_temperature": self.validation_sampling_temperature,
                "validation_seed": self.validation_seed,
                "validation_use_ema": self.validation_use_ema,
                "pretrained_checkpoint": (
                    None
                    if pretrained_checkpoint is None
                    else str(pretrained_checkpoint)
                ),
                "autoencoder_checkpoint": (
                    None
                    if autoencoder_checkpoint is None
                    else str(autoencoder_checkpoint)
                ),
            }
        )

    @property
    def first_stage_model(self) -> torch.nn.Module:
        return self.diffusion.first_stage_model

    def _freeze_first_stage(self) -> None:
        self.first_stage_model.eval()
        for parameter in self.first_stage_model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _build_perceptual_loss(backbone: str) -> torch.nn.Module:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "Decoded perceptual diffusion loss was enabled but lpips is not "
                "installed. Install opensr-model[train]."
            ) from exc
        module = lpips.LPIPS(net=backbone)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module

    def train(self, mode: bool = True) -> "DiffusionTrainingModule":
        super().train(mode)
        # The legacy model already replaces ``first_stage_model.train`` with
        # a no-op, but reasserting eval mode here makes the invariant explicit.
        self.first_stage_model.eval()
        if self.decoded_perceptual_loss is not None:
            self.decoded_perceptual_loss.eval()
        return self

    def opensr_inference_config(self) -> dict[str, Any]:
        """Return plain, weights-only-safe settings for inference exports.

        Schedule endpoints come from the live buffers rather than the input
        YAML. This matters when a pretrained native checkpoint replaces the
        constructor's schedule during initialization.
        """

        betas = self.diffusion.betas.detach().float().cpu()
        return {
            "apply_normalization": bool(self.apply_normalization),
            "encode_conditioning": bool(self.encode_conditioning),
            "denoiser_settings": {
                "timesteps": int(self.diffusion.num_timesteps),
                "linear_start": float(betas[0].item()),
                "linear_end": float(betas[-1].item()),
                "parameterization": str(self.diffusion.parameterization),
                "sampling_steps": int(self.validation_sampling_steps),
                "sampling_eta": float(self.validation_sampling_eta),
                "sampling_temperature": float(self.validation_sampling_temperature),
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["opensr_inference_config"] = self.opensr_inference_config()

    def load_pretrained(
        self, checkpoint_path: str | Path, *, strict: bool = True
    ) -> Any:
        """Load a native OpenSR checkpoint through the shared helper."""

        from .checkpoints import load_diffusion_weights

        result = load_diffusion_weights(self.diffusion, checkpoint_path, strict=strict)
        self._freeze_first_stage()
        return result

    def native_state_dict(
        self, *, keep_vars: bool = False
    ) -> OrderedDict[str, torch.Tensor]:
        """Return the unprefixed state expected by ``SRLatentDiffusion``."""

        return OrderedDict(self.diffusion.state_dict(keep_vars=keep_vars))

    def _batch_tensor(
        self, batch: Mapping[str, Any], key: str, *, name: str
    ) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"Batch is missing {name} tensor at key {key!r}")
        tensor = batch[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Batch value {key!r} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(
                f"Batch value {key!r} must have shape BxCxHxW; got {tuple(tensor.shape)}"
            )
        return torch.nan_to_num(
            tensor.to(memory_format=torch.contiguous_format).float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _paired_images(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hr = self._batch_tensor(batch, self.hr_key, name="HR")
        lr = self._batch_tensor(batch, self.lr_key, name="LR")
        if hr.shape[0] != lr.shape[0]:
            raise ValueError("HR and LR tensors must have the same batch size")
        if hr.shape[1] != lr.shape[1]:
            raise ValueError("HR and LR tensors must have the same channel count")
        expected_hr_size = (lr.shape[-2] * 4, lr.shape[-1] * 4)
        if tuple(hr.shape[-2:]) != expected_hr_size:
            raise ValueError(
                "OpenSR expects 4x paired crops: "
                f"LR spatial size {tuple(lr.shape[-2:])} requires HR {expected_hr_size}, "
                f"got {tuple(hr.shape[-2:])}"
            )
        return hr, lr

    @staticmethod
    def _valid_mask(
        batch: Mapping[str, Any], reference: torch.Tensor
    ) -> Optional[torch.Tensor]:
        mask = batch.get("valid_mask")
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Batch value 'valid_mask' must be a torch.Tensor")
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[0] != reference.shape[0]:
            raise ValueError(
                "valid_mask must have shape Bx1xHxW (or BxHxW) and match the batch"
            )
        if mask.shape[-2:] != reference.shape[-2:]:
            raise ValueError("valid_mask must have the HR image spatial shape")
        return torch.nan_to_num(mask.float(), nan=0.0).clamp_(0.0, 1.0)

    @staticmethod
    def _randn_like(
        tensor: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )

    def _posterior_latent(
        self,
        posterior: Any,
        *,
        sample: bool,
        generator: Optional[torch.Generator] = None,
        apply_scale: bool,
    ) -> torch.Tensor:
        if sample:
            # The legacy distribution sampler first allocates CPU float32 noise
            # and then transfers it.  Drawing directly beside the posterior is
            # mathematically identical, preserves mixed-precision dtype, and
            # also lets validation use an isolated deterministic generator.
            latent = posterior.mean + posterior.std * self._randn_like(
                posterior.mean, generator
            )
        else:
            latent = posterior.mode()
        if apply_scale:
            latent = self.diffusion.scale_factor * latent
        return latent.detach()

    @staticmethod
    def _full_precision_context(tensor: torch.Tensor) -> torch.autocast:
        """Disable AMP around the numerically sensitive 512px autoencoder."""

        return torch.autocast(device_type=tensor.device.type, enabled=False)

    def _decode_first_stage_float32(self, latent: torch.Tensor) -> torch.Tensor:
        # The canonical autoencoder applies full 128x128 spatial attention. Its
        # FP16 attention logits can overflow before softmax, so both frozen
        # encoding and differentiable auxiliary decoding must execute in FP32.
        with self._full_precision_context(latent):
            return self.diffusion.decode_first_stage(latent.float()).float()

    @torch.no_grad()
    def _encode_hr(
        self,
        hr: torch.Tensor,
        *,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        with self._full_precision_context(hr):
            posterior = self.diffusion.encode_first_stage(hr.float())
            return self._posterior_latent(
                posterior,
                sample=sample,
                generator=generator,
                apply_scale=True,
            )

    @torch.no_grad()
    def _encode_conditioning(
        self,
        lr: torch.Tensor,
        *,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if not self.encode_conditioning:
            # This is the legacy, non-encoded path.  The bundled checkpoint
            # uses encoded conditioning and therefore does not enter it.
            return self.linear_transform(lr.clone(), stage="norm").detach()

        with self._full_precision_context(lr):
            lr_up = F.interpolate(
                lr.float(),
                size=(lr.shape[-2] * 4, lr.shape[-1] * 4),
                mode="bilinear",
                align_corners=False,
            )
            posterior = self.diffusion.first_stage_model.encode(lr_up)
            # Inference calls ``posterior.sample()`` directly rather than
            # ``get_first_stage_encoding``.  Keep that behavior exactly.
            return self._posterior_latent(
                posterior,
                sample=sample,
                generator=generator,
                apply_scale=False,
            )

    def _prepare_latents(
        self,
        hr: torch.Tensor,
        lr: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = self._encode_hr(hr, sample=True, generator=generator)
            conditioning = self._encode_conditioning(
                lr, sample=True, generator=generator
            )
        if z.shape != conditioning.shape:
            raise ValueError(
                "Encoded HR and conditioning must have identical shapes; "
                f"got {tuple(z.shape)} and {tuple(conditioning.shape)}"
            )
        return z, conditioning

    @staticmethod
    def _erode_mask_for_laplacian(
        mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        # A detail pixel is valid only if its four-neighbour receptive field is
        # valid. max-pooling the inverse is a cheap binary erosion that also
        # behaves sensibly for fractional masks.
        return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)

    def _timestep_gate(
        self, timesteps: torch.Tensor, maximum_fraction: float
    ) -> torch.Tensor:
        maximum = int(math.floor((self.diffusion.num_timesteps - 1) * maximum_fraction))
        return timesteps <= maximum

    def _latent_detail_loss(
        self,
        predicted_x0: torch.Tensor,
        target_x0: torch.Tensor,
        timesteps: torch.Tensor,
        latent_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eligible = self._timestep_gate(
            timesteps,
            self.sharpness_config["latent_detail_max_timestep_fraction"],
        )
        if not bool(eligible.any()):
            return predicted_x0.sum() * 0.0, eligible.float().mean()

        detail_error = (_laplacian(predicted_x0) - _laplacian(target_x0)).abs()
        detail_mask = self._erode_mask_for_laplacian(latent_mask)
        per_sample = _masked_per_sample_mean(detail_error, detail_mask)
        return per_sample[eligible].mean(), eligible.float().mean()

    def _decoded_auxiliary_losses(
        self,
        predicted_x0: torch.Tensor,
        target_image: torch.Tensor,
        timesteps: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a bounded low-timestep subset and compare it in RGB space."""

        eligible = self._timestep_gate(
            timesteps,
            self.sharpness_config["decoded_max_timestep_fraction"],
        )
        indices = eligible.nonzero(as_tuple=False).flatten()
        indices = indices[: self.sharpness_config["decoded_max_batch_size"]]
        if indices.numel() == 0:
            zero = predicted_x0.sum() * 0.0
            return zero, zero, zero, zero.detach()

        decoded = self._decode_first_stage_float32(
            predicted_x0.index_select(0, indices)
        )
        target = target_image.index_select(0, indices).float()
        mask = (
            None
            if valid_mask is None
            else valid_mask.index_select(0, indices).to(decoded.device)
        )

        if self.sharpness_config["decoded_rgb_only"]:
            if decoded.shape[1] < 3 or target.shape[1] < 3:
                raise ValueError("decoded_rgb_only requires at least three channels")
            decoded_for_loss = decoded[:, :3]
            target_for_loss = target[:, :3]
            if mask is not None and mask.shape[1] > 1:
                mask_for_loss = mask[:, :3]
            else:
                mask_for_loss = mask
        else:
            decoded_for_loss = decoded
            target_for_loss = target
            mask_for_loss = mask

        reconstruction = _masked_per_sample_mean(
            (decoded_for_loss - target_for_loss).abs(), mask_for_loss
        ).mean()
        detail_mask = self._erode_mask_for_laplacian(mask_for_loss)
        detail = _masked_per_sample_mean(
            (_laplacian(decoded_for_loss) - _laplacian(target_for_loss)).abs(),
            detail_mask,
        ).mean()

        perceptual = decoded.new_zeros(())
        if self.decoded_perceptual_loss is not None:
            if decoded.shape[1] < 3 or target.shape[1] < 3:
                raise ValueError("Decoded LPIPS requires at least three channels")
            lpips_prediction = decoded[:, :3]
            lpips_target = target[:, :3]
            lpips_mask = mask
            if lpips_mask is not None:
                if lpips_mask.shape[1] > 1:
                    lpips_mask = lpips_mask[:, :3]
                if lpips_mask.shape[1] == 1:
                    lpips_mask = lpips_mask.expand(-1, 3, -1, -1)
                lpips_prediction = (
                    lpips_prediction * lpips_mask
                    + lpips_target.detach() * (1.0 - lpips_mask)
                )
            lpips_prediction = 2.0 * lpips_prediction.clamp(0.0, 1.0) - 1.0
            lpips_target = 2.0 * lpips_target.clamp(0.0, 1.0) - 1.0
            perceptual = self.decoded_perceptual_loss(
                lpips_prediction, lpips_target
            ).mean()

        active_samples = decoded.new_tensor(float(indices.numel()))
        return reconstruction, detail, perceptual, active_samples

    def _should_compute_decoded_auxiliary(self, batch_idx: int) -> bool:
        weights = (
            self.sharpness_config["decoded_reconstruction_weight"],
            self.sharpness_config["decoded_detail_weight"],
            self.sharpness_config["decoded_perceptual_weight"],
        )
        return any(weight > 0 for weight in weights) and (
            batch_idx % self.sharpness_config["decoded_every_n_batches"] == 0
        )

    def _diffusion_objective(
        self,
        z: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        valid_mask: Optional[torch.Tensor] = None,
        decoded_target: Optional[torch.Tensor] = None,
        decode_auxiliary: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = z.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (batch_size,),
                device=z.device,
                generator=generator,
                dtype=torch.long,
            )
        if noise is None:
            noise = self._randn_like(z, generator)

        noisy_z = self.diffusion.q_sample(z, timesteps, noise=noise)
        prediction = self.diffusion.apply_model(noisy_z, timesteps, conditioning)

        if self.diffusion.parameterization == "eps":
            target = noise
        elif self.diffusion.parameterization == "x0":
            target = z
        else:  # guarded by DDPM.__init__, retained for a clear training error
            raise ValueError(
                f"Unsupported parameterization {self.diffusion.parameterization!r}"
            )

        squared_error = (prediction.float() - target.float()).square()
        broadcast_shape = (batch_size,) + (1,) * (z.ndim - 1)
        sqrt_alpha = self.diffusion.sqrt_alphas_cumprod[timesteps].reshape(
            broadcast_shape
        )
        sqrt_one_minus_alpha = self.diffusion.sqrt_one_minus_alphas_cumprod[
            timesteps
        ].reshape(broadcast_shape)
        predicted_x0 = (
            noisy_z.float() - sqrt_one_minus_alpha.float() * prediction.float()
        ) / sqrt_alpha.float().clamp_min(1.0e-12)
        x0_squared_error = (predicted_x0 - z.float()).square()
        latent_mask = None
        if valid_mask is not None:
            latent_mask = F.interpolate(
                valid_mask.float(), size=z.shape[-2:], mode="nearest"
            ).to(device=z.device)
            if latent_mask.shape[1] == 1 and squared_error.shape[1] != 1:
                latent_mask = latent_mask.expand(-1, squared_error.shape[1], -1, -1)
        per_sample_eps_mse = _masked_per_sample_mean(squared_error, latent_mask)
        per_sample_x0_mse = _masked_per_sample_mean(x0_squared_error, latent_mask)
        eps_mse = per_sample_eps_mse.mean()
        x0_mse = per_sample_x0_mse.mean()
        valid_fraction = (
            squared_error.new_ones(()) if latent_mask is None else latent_mask.mean()
        )

        alpha_bar = self.diffusion.alphas_cumprod[timesteps].float()
        snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1.0e-12)
        p2_weight = (self.sharpness_config["p2_k"] + snr).pow(
            -self.sharpness_config["p2_gamma"]
        )
        p2_loss = (p2_weight * per_sample_eps_mse).mean()

        zero = prediction.float().sum() * 0.0
        latent_detail = zero
        latent_detail_eligible_fraction = timesteps.new_zeros((), dtype=torch.float32)
        if self.sharpness_config["latent_detail_weight"] > 0:
            latent_detail, latent_detail_eligible_fraction = self._latent_detail_loss(
                predicted_x0,
                z.float(),
                timesteps,
                latent_mask,
            )

        decoded_reconstruction = zero
        decoded_detail = zero
        decoded_perceptual = zero
        decoded_active_samples = zero.detach()
        if decode_auxiliary:
            if decoded_target is None:
                raise ValueError(
                    "decoded_target is required when decode_auxiliary is enabled"
                )
            (
                decoded_reconstruction,
                decoded_detail,
                decoded_perceptual,
                decoded_active_samples,
            ) = self._decoded_auxiliary_losses(
                predicted_x0,
                decoded_target,
                timesteps,
                valid_mask,
            )

        auxiliary_loss = (
            self.sharpness_config["latent_detail_weight"] * latent_detail
            + self.sharpness_config["decoded_reconstruction_weight"]
            * decoded_reconstruction
            + self.sharpness_config["decoded_detail_weight"] * decoded_detail
            + self.sharpness_config["decoded_perceptual_weight"] * decoded_perceptual
        )
        loss = p2_loss + auxiliary_loss
        stats = {
            "loss": loss.detach(),
            "eps_mse": eps_mse.detach(),
            "p2_loss": p2_loss.detach(),
            "p2_weight_mean": p2_weight.mean().detach(),
            "snr_mean": snr.mean().detach(),
            "auxiliary_loss": auxiliary_loss.detach(),
            "latent_detail_loss": latent_detail.detach(),
            "latent_detail_eligible_fraction": (
                latent_detail_eligible_fraction.detach()
            ),
            "decoded_reconstruction_loss": decoded_reconstruction.detach(),
            "decoded_detail_loss": decoded_detail.detach(),
            "decoded_perceptual_loss": decoded_perceptual.detach(),
            "decoded_active_samples": decoded_active_samples.detach(),
            "prediction_rms": prediction.float().square().mean().sqrt().detach(),
            "target_rms": target.float().square().mean().sqrt().detach(),
            "latent_std": z.float().std().detach(),
            "conditioning_std": conditioning.float().std().detach(),
            "timestep_mean": timesteps.float().mean().detach(),
            "x0_mse": x0_mse.detach(),
            "valid_fraction": valid_fraction.detach(),
        }
        return loss, stats

    @staticmethod
    def _require_finite_loss(
        loss: torch.Tensor,
        stats: Mapping[str, torch.Tensor],
        *,
        stage: str,
        batch_idx: int,
    ) -> None:
        """Abort before backward instead of allowing one bad batch to poison weights."""

        if bool(torch.isfinite(loss.detach()).all()):
            return
        nonfinite = sorted(
            name
            for name, value in stats.items()
            if not bool(torch.isfinite(value.detach()).all())
        )
        names = ", ".join(nonfinite) if nonfinite else "unknown component"
        raise FloatingPointError(
            f"Non-finite {stage} loss at batch {batch_idx}; "
            f"non-finite components: {names}"
        )

    def forward(
        self,
        hr: torch.Tensor,
        lr: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the scalar DDPM training objective for an explicit pair."""

        z, conditioning = self._prepare_latents(hr.float(), lr.float())
        loss, _ = self._diffusion_objective(
            z,
            conditioning,
            timesteps=timesteps,
            noise=noise,
            valid_mask=valid_mask,
        )
        return loss

    def _shared_step(
        self, batch: Mapping[str, Any], *, stage: str, batch_idx: int
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hr, lr = self._paired_images(batch)
        valid_mask = self._valid_mask(batch, hr)
        generator = None
        if stage == "val":
            # Keep the full validation objective stable across epochs without
            # perturbing the global training RNG or the seeded DDIM sampler.
            generator = self._make_generator(
                hr.device, self.validation_seed + 1_000_003 + batch_idx
            )
        z, conditioning = self._prepare_latents(hr, lr, generator=generator)
        decode_auxiliary = stage == "train" and self._should_compute_decoded_auxiliary(
            batch_idx
        )
        loss, stats = self._diffusion_objective(
            z,
            conditioning,
            generator=generator,
            valid_mask=valid_mask,
            decoded_target=hr if decode_auxiliary else None,
            decode_auxiliary=decode_auxiliary,
        )
        self._require_finite_loss(loss, stats, stage=stage, batch_idx=batch_idx)
        batch_size = hr.shape[0]
        clipped_fraction = batch_clipped_fraction(batch, hr)
        if clipped_fraction is not None:
            self.log(
                f"{stage}/input_clipped_fraction",
                clipped_fraction,
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, value in stats.items():
            if name == "loss":
                continue
            self.log(
                f"{stage}/{name}",
                value,
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss, stats, hr, lr, z, conditioning

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _, _, _, _ = self._shared_step(
            batch, stage="train", batch_idx=batch_idx
        )
        return loss

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        """Update EMA once, immediately after each actual optimizer step."""

        super().optimizer_step(*args, **kwargs)
        if self.diffusion.use_ema:
            self.diffusion.model_ema(self.diffusion.model)

    def _make_generator(self, device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def _ddim_sample_latents(
        self,
        conditioning: torch.Tensor,
        *,
        steps: int,
        eta: float,
        temperature: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # Reuse the repository's exact schedule construction, while handling
        # both eps- and x0-parameterized model outputs below.
        self.diffusion.device = conditioning.device
        sampler = DDIMSampler(self.diffusion)
        sampler.make_schedule(
            ddim_num_steps=steps,
            ddim_eta=eta,
            verbose=False,
        )
        if len(sampler.ddim_timesteps) != steps:
            raise ValueError(
                f"The native uniform DDIM schedule produced "
                f"{len(sampler.ddim_timesteps)} steps for a request of {steps}. "
                f"Choose a step count that divides {self.diffusion.num_timesteps}."
            )
        if int(np.max(sampler.ddim_timesteps)) >= self.diffusion.num_timesteps:
            raise ValueError("The native DDIM schedule produced an invalid timestep")

        latent = self._randn_like(conditioning, generator)
        time_range = np.flip(sampler.ddim_timesteps)
        total_steps = len(time_range)
        batch_size = conditioning.shape[0]

        for index_from_start, step in enumerate(time_range):
            index = total_steps - index_from_start - 1
            timesteps = torch.full(
                (batch_size,),
                int(step),
                device=latent.device,
                dtype=torch.long,
            )
            model_output = self.diffusion.apply_model(latent, timesteps, conditioning)

            alpha = torch.as_tensor(
                sampler.ddim_alphas[index],
                device=latent.device,
                dtype=latent.dtype,
            ).reshape(1, 1, 1, 1)
            alpha_prev = torch.as_tensor(
                sampler.ddim_alphas_prev[index],
                device=latent.device,
                dtype=latent.dtype,
            ).reshape(1, 1, 1, 1)
            sigma = torch.as_tensor(
                sampler.ddim_sigmas[index],
                device=latent.device,
                dtype=latent.dtype,
            ).reshape(1, 1, 1, 1)
            sqrt_one_minus_alpha = torch.as_tensor(
                sampler.ddim_sqrt_one_minus_alphas[index],
                device=latent.device,
                dtype=latent.dtype,
            ).reshape(1, 1, 1, 1)

            if self.diffusion.parameterization == "eps":
                predicted_noise = model_output
                predicted_x0 = (
                    latent - sqrt_one_minus_alpha * predicted_noise
                ) / alpha.sqrt()
            elif self.diffusion.parameterization == "x0":
                predicted_x0 = model_output
                predicted_noise = (
                    latent - alpha.sqrt() * predicted_x0
                ) / sqrt_one_minus_alpha.clamp_min(1.0e-12)
            else:
                raise ValueError(
                    f"Unsupported parameterization {self.diffusion.parameterization!r}"
                )

            direction = (1.0 - alpha_prev - sigma.square()).clamp_min(
                0.0
            ).sqrt() * predicted_noise
            if eta == 0.0 or temperature == 0.0:
                stochastic = torch.zeros_like(latent)
            else:
                stochastic = sigma * self._randn_like(latent, generator) * temperature
            latent = alpha_prev.sqrt() * predicted_x0 + direction + stochastic

        return latent

    @torch.no_grad()
    def sample_super_resolution(
        self,
        lr: torch.Tensor,
        *,
        steps: Optional[int] = None,
        eta: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        use_ema: Optional[bool] = None,
    ) -> torch.Tensor:
        """Produce a reproducible validation SR image without post-hoc matching."""

        if lr.ndim != 4:
            raise ValueError("lr must have shape BxCxHxW")
        lr = torch.nan_to_num(lr.float(), nan=0.0, posinf=0.0, neginf=0.0)
        sample_steps = self.validation_sampling_steps if steps is None else int(steps)
        sample_eta = self.validation_sampling_eta if eta is None else float(eta)
        sample_temperature = (
            self.validation_sampling_temperature
            if temperature is None
            else float(temperature)
        )
        sample_seed = self.validation_seed if seed is None else int(seed)
        sample_with_ema = self.validation_use_ema if use_ema is None else use_ema

        generator = self._make_generator(lr.device, sample_seed)
        conditioning = self._encode_conditioning(lr, sample=True, generator=generator)
        ema_context = (
            self.diffusion.ema_scope()
            if sample_with_ema and self.diffusion.use_ema
            else nullcontext()
        )
        with ema_context:
            sampled_latent = self._ddim_sample_latents(
                conditioning,
                steps=sample_steps,
                eta=sample_eta,
                temperature=sample_temperature,
                generator=generator,
            )
        decoded = self._decode_first_stage_float32(sampled_latent)
        decoded = self.linear_transform(decoded, stage="denorm")
        return decoded.clamp(0.0, 1.0)

    def _should_sample_validation(self, batch_idx: int) -> bool:
        if self.validation_num_samples == 0:
            return False
        if batch_idx >= self.validation_image_batches:
            return False
        if (self.current_epoch + 1) % self.validation_sample_every_n_epochs != 0:
            return False
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return True
        # Only rank zero constructs the five expensive full DDIM samples. The
        # remaining ranks still evaluate their share of the 25 validation
        # batches, but do not produce duplicate images that are discarded.
        return trainer.is_global_zero and not trainer.sanity_checking

    @torch.no_grad()
    def _validation_images(
        self,
        batch: Mapping[str, Any],
        hr: torch.Tensor,
        lr: torch.Tensor,
        *,
        batch_idx: int,
    ) -> tuple[
        OrderedDict[str, torch.Tensor],
        list[str],
        dict[str, torch.Tensor],
    ]:
        count = min(self.validation_num_samples, hr.shape[0])
        hr = hr[:count]
        lr = lr[:count]
        seed = self.validation_seed + batch_idx
        sr = self.sample_super_resolution(lr, seed=seed)

        # Stable posterior modes make AE reconstruction panels comparable
        # across epochs; diffusion conditioning itself remains a seeded sample.
        hr_latent = self._encode_hr(hr, sample=False)
        hr_reconstruction = self._decode_first_stage_float32(hr_latent)
        hr_reconstruction = self.linear_transform(
            hr_reconstruction, stage="denorm"
        ).clamp(0.0, 1.0)

        lr_up = F.interpolate(
            lr,
            size=(lr.shape[-2] * 4, lr.shape[-1] * 4),
            mode="bilinear",
            align_corners=False,
        )
        condition_latent = self._encode_conditioning(lr, sample=False)
        condition_reconstruction = self._decode_first_stage_float32(condition_latent)
        condition_reconstruction = self.linear_transform(
            condition_reconstruction, stage="denorm"
        ).clamp(0.0, 1.0)

        images = OrderedDict(
            low_resolution=lr.detach().clamp(0.0, 1.0).cpu(),
            low_resolution_upsampled=lr_up.detach().clamp(0.0, 1.0).cpu(),
            conditioning_reconstruction=condition_reconstruction.detach().cpu(),
            target=hr.detach().clamp(0.0, 1.0).cpu(),
            autoencoder_reconstruction=hr_reconstruction.detach().cpu(),
            super_resolution=sr.detach().cpu(),
            absolute_error=(sr - hr).abs().mean(dim=1, keepdim=True).detach().cpu(),
        )
        raw_ids = batch.get("sample_id")
        if raw_ids is None:
            sample_ids = [f"validation_{batch_idx}_{index}" for index in range(count)]
        else:
            sample_ids = [str(value) for value in list(raw_ids)[:count]]
        valid_mask = None
        if "valid_mask" in batch:
            raw_mask = batch["valid_mask"]
            mask_batch = {
                "valid_mask": (
                    raw_mask[:count] if isinstance(raw_mask, torch.Tensor) else raw_mask
                )
            }
            valid_mask = self._valid_mask(mask_batch, hr)
        image_metrics = compute_reconstruction_metrics(
            sr,
            hr,
            valid_mask,
            value_min=0.0,
            value_max=1.0,
        )
        image_metrics.update(
            compute_rgb_nir_reconstruction_metrics(
                sr,
                hr,
                valid_mask,
                value_min=0.0,
                value_max=1.0,
            )
        )
        image_metrics["downsample_consistency"] = downsample_consistency(
            sr, lr, valid_mask
        )
        total_channels = sr.shape[1]
        if total_channels >= 3:
            rgb_channels = slice(0, 3)
            image_metrics["rgb_downsample_consistency"] = downsample_consistency(
                sr[:, rgb_channels],
                lr[:, rgb_channels],
                select_channel_mask(
                    valid_mask, rgb_channels, total_channels=total_channels
                ),
            )
        if total_channels >= 4:
            nir_channels = slice(3, 4)
            image_metrics["nir_downsample_consistency"] = downsample_consistency(
                sr[:, nir_channels],
                lr[:, nir_channels],
                select_channel_mask(
                    valid_mask, nir_channels, total_channels=total_channels
                ),
            )
        return images, sample_ids, image_metrics

    def validation_step(
        self, batch: Mapping[str, Any], batch_idx: int
    ) -> dict[str, Any]:
        _, _, hr, lr, _, _ = self._shared_step(batch, stage="val", batch_idx=batch_idx)
        output: dict[str, Any] = {"images": OrderedDict(), "sample_ids": []}
        if self._should_sample_validation(batch_idx):
            images, sample_ids, image_metrics = self._validation_images(
                batch, hr, lr, batch_idx=batch_idx
            )
            output["images"] = images
            output["sample_ids"] = sample_ids
            for name, value in image_metrics.items():
                self.log(
                    f"val/sr_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=len(sample_ids),
                )
        return output

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers(use_pl_optimizer=False)
        if optimizer is not None:
            self.log(
                "train/lr",
                float(optimizer.param_groups[0]["lr"]),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def configure_optimizers(self) -> Any:
        parameters = [
            parameter
            for parameter in self.diffusion.model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_patience is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


__all__ = ["DiffusionTrainingModule"]
