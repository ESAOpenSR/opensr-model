"""Small, stateless image metrics used by both training stages."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

_NIR_RECONSTRUCTION_METRICS = frozenset({"mae", "mse", "rmse", "psnr", "out_of_range"})


def _expanded_mask(
    mask: Optional[torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference)
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask.float(), size=reference.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and reference.shape[1] != 1:
        mask = mask.expand(-1, reference.shape[1], -1, -1)
    return mask.to(device=reference.device, dtype=reference.dtype)


def masked_mean(
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a finite mean over valid pixels (and channels)."""

    if mask is None:
        return value.mean()
    expanded = _expanded_mask(mask, value)
    return (value * expanded).sum() / expanded.sum().clamp_min(eps)


def spectral_angle(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean spectral angle in radians for BCHW multispectral tensors."""

    dot = (prediction * target).sum(dim=1)
    pred_norm = prediction.square().sum(dim=1).sqrt()
    target_norm = target.square().sum(dim=1).sqrt()
    cosine = dot / (pred_norm * target_norm).clamp_min(eps)
    both_zero = (pred_norm <= eps) & (target_norm <= eps)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    angles = torch.acos(cosine.clamp(-1.0, 1.0))
    if mask is None:
        pixel_mask = None
    elif mask.ndim == 4:
        pixel_mask = mask[:, :1].squeeze(1)
    elif mask.ndim == 3:
        pixel_mask = mask
    else:
        raise ValueError("mask must have shape Bx1xHxW or BxHxW")
    return masked_mean(angles, pixel_mask, eps=eps)


def _ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
) -> Optional[torch.Tensor]:
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError:  # pragma: no cover - Lightning normally installs torchmetrics
        return None

    # TorchMetrics' default 11-pixel kernel requires reasonably sized validation tiles.
    if min(prediction.shape[-2:]) < 11:
        return None
    return structural_similarity_index_measure(
        prediction.float(), target.float(), data_range=float(data_range)
    )


def compute_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute reflectance-aware reconstruction metrics without keeping state."""

    data_range = float(value_max - value_min)
    if data_range <= 0:
        raise ValueError("value_max must be larger than value_min")

    prediction = prediction.float()
    target = target.float()
    error = prediction - target
    mae = masked_mean(error.abs(), mask)
    mse = masked_mean(error.square(), mask)
    rmse = mse.sqrt()
    psnr = 10.0 * torch.log10(
        torch.as_tensor(data_range**2, device=mse.device) / mse.clamp_min(1e-12)
    )

    pred_clamped = prediction.clamp(value_min, value_max)
    target_clamped = target.clamp(value_min, value_max)
    if mask is not None:
        expanded = _expanded_mask(mask, pred_clamped)
        # Invalid pixels are made identical. Valid-fraction is logged separately, so the
        # slight SSIM optimism at nodata boundaries is visible rather than silently hidden.
        pred_clamped = pred_clamped * expanded + target_clamped * (1.0 - expanded)

    metrics: Dict[str, torch.Tensor] = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "sam": spectral_angle(prediction, target, mask),
        "out_of_range": masked_mean(
            ((prediction < value_min) | (prediction > value_max)).float(), mask
        ),
    }
    if mask is not None:
        metrics["valid_fraction"] = mask.float().mean()

    ssim = _ssim(pred_clamped, target_clamped, data_range)
    if ssim is not None and torch.isfinite(ssim):
        metrics["ssim"] = ssim
    return metrics


def select_channel_mask(
    mask: Optional[torch.Tensor],
    channels: slice,
    *,
    total_channels: int,
) -> Optional[torch.Tensor]:
    """Select matching mask bands while preserving a shared single-band mask."""

    if mask is None or mask.ndim == 3:
        return mask
    if mask.ndim != 4:
        raise ValueError("mask must have shape Bx1xHxW, BxCxHxW, or BxHxW")
    if mask.shape[1] == 1:
        return mask
    if mask.shape[1] != total_channels:
        raise ValueError(
            "A channel-specific mask must have either one channel or the same "
            f"{total_channels} channels as the image; got {mask.shape[1]}"
        )
    return mask[:, channels]


def compute_rgb_nir_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return honest first-three-band RGB and fourth-band NIR metrics.

    RGB retains every metric supported by :func:`compute_reconstruction_metrics`.
    NIR intentionally excludes SAM (which is not meaningful for a one-dimensional
    spectrum) and SSIM, leaving only direct error, PSNR, and range telemetry.
    Inputs with fewer than the relevant bands simply omit that metric family.
    """

    if prediction.ndim != 4 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have the same BxCxHxW shape")
    total_channels = prediction.shape[1]
    metrics: Dict[str, torch.Tensor] = {}
    if total_channels >= 3:
        rgb_channels = slice(0, 3)
        rgb_metrics = compute_reconstruction_metrics(
            prediction[:, rgb_channels],
            target[:, rgb_channels],
            select_channel_mask(mask, rgb_channels, total_channels=total_channels),
            value_min=value_min,
            value_max=value_max,
        )
        metrics.update({f"rgb_{name}": value for name, value in rgb_metrics.items()})
    if total_channels >= 4:
        nir_channels = slice(3, 4)
        nir_metrics = compute_reconstruction_metrics(
            prediction[:, nir_channels],
            target[:, nir_channels],
            select_channel_mask(mask, nir_channels, total_channels=total_channels),
            value_min=value_min,
            value_max=value_max,
        )
        metrics.update(
            {
                f"nir_{name}": value
                for name, value in nir_metrics.items()
                if name in _NIR_RECONSTRUCTION_METRICS
            }
        )
    return metrics


def batch_clipped_fraction(
    batch: Mapping[str, Any], reference: torch.Tensor
) -> Optional[torch.Tensor]:
    """Reduce optional per-sample clipping fractions to a validated batch mean."""

    raw_value = batch.get("clipped_fraction")
    if raw_value is None:
        return None
    if isinstance(raw_value, torch.Tensor):
        values = raw_value.to(device=reference.device, dtype=torch.float32)
    else:
        values = torch.as_tensor(
            raw_value, device=reference.device, dtype=torch.float32
        )
    if values.numel() == 0:
        raise ValueError("clipped_fraction must contain at least one value")
    if not torch.isfinite(values).all():
        raise ValueError("clipped_fraction must contain only finite values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("clipped_fraction values must lie in [0, 1]")
    return values.mean()


def downsample_consistency(
    prediction_hr: torch.Tensor,
    condition_lr: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MAE between an SR image reduced to native LR size and its conditioning image."""

    reduced = F.interpolate(
        prediction_hr.float(),
        size=condition_lr.shape[-2:],
        mode="area",
    )
    lr_mask = None
    if mask is not None:
        lr_mask = F.interpolate(
            mask.float(), size=condition_lr.shape[-2:], mode="nearest"
        )
    return masked_mean((reduced - condition_lr.float()).abs(), lr_mask)


__all__ = [
    "batch_clipped_fraction",
    "compute_reconstruction_metrics",
    "compute_rgb_nir_reconstruction_metrics",
    "downsample_consistency",
    "masked_mean",
    "select_channel_mask",
    "spectral_angle",
]
