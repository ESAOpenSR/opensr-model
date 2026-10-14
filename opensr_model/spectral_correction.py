"""PyTorch measurement-consistency correction for decoded SR reflectances.

No model weights, NumPy conversions, or image transfers to CPU are required.
Gaussian blur is an illustrative periodic operator, not a calibrated S2 PSF.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BackProjectionConfig:
    scale: int = 4
    blur_sigma: float | tuple[float, ...] = 0.0
    max_iterations: int = 100
    tolerance: float = 1e-6
    relaxation: float = 1.0

    def __post_init__(self):
        for name, minimum in (("scale", 1), ("max_iterations", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("tolerance", "relaxation"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and numeric")
        if self.tolerance < 0 or not 0 < self.relaxation <= 1:
            raise ValueError("tolerance must be >= 0 and relaxation must be in (0, 1]")
        sigmas = (
            (self.blur_sigma,)
            if isinstance(self.blur_sigma, Real)
            else tuple(self.blur_sigma)
        )
        if not sigmas or any(
            isinstance(s, bool)
            or not isinstance(s, Real)
            or not math.isfinite(s)
            or s < 0
            for s in sigmas
        ):
            raise ValueError("blur_sigma must contain finite nonnegative numbers")
        if not isinstance(self.blur_sigma, Real):
            object.__setattr__(self, "blur_sigma", tuple(float(s) for s in sigmas))

    def sigmas(self, channels):
        values = (
            (float(self.blur_sigma),) * channels
            if isinstance(self.blur_sigma, Real)
            else self.blur_sigma
        )
        if len(values) != channels:
            raise ValueError("blur_sigma needs one value per image band")
        return values


class BlurDownsample(torch.nn.Module):
    """BCHW symmetric periodic Gaussian blur followed by block averaging.

    `adjoint` is the exact transpose, including the 1/scale**2 averaging weight.
    Move this small operator to the input dtype/device using `.to(image)`.
    Sigma is measured in HR pixels. Zero sigma selects pure block averaging.
    """

    def __init__(self, channels=4, scale=4, blur_sigma=0.0):
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, Integral)
            or channels < 1
        ):
            raise ValueError("channels must be a positive integer")
        config = BackProjectionConfig(scale=scale, blur_sigma=blur_sigma)
        sigmas = config.sigmas(channels)
        self.scale, self.channels = scale, channels
        radii = [int(4 * s + 0.5) for s in sigmas]
        self.radius = max(radii)
        positions = torch.arange(-self.radius, self.radius + 1, dtype=torch.float64)
        kernels = []
        for sigma, radius in zip(sigmas, radii):
            if sigma == 0:
                kernel = (positions == 0).double()
            else:
                kernel = torch.exp(-0.5 * (positions / sigma).square())
                kernel = kernel * (positions.abs() <= radius)
                kernel = kernel / kernel.sum()
            kernels.append(kernel)
        self.register_buffer("kernel", torch.stack(kernels)[:, None, None, :])

    def _check(self, image):
        if image.ndim != 4 or image.shape[1] != self.channels or min(image.shape) < 1:
            raise ValueError("expected nonempty BCHW with the configured channel count")

    def _blur(self, image):
        if self.radius == 0:
            return image
        # Index padding handles tiny images even when the kernel exceeds a side.
        h, w = image.shape[-2:]
        x = torch.arange(-self.radius, w + self.radius, device=image.device) % w
        y = torch.arange(-self.radius, h + self.radius, device=image.device) % h
        image = F.conv2d(image.index_select(-1, x), self.kernel, groups=self.channels)
        return F.conv2d(
            image.index_select(-2, y),
            self.kernel.transpose(-1, -2),
            groups=self.channels,
        )

    def forward(self, image):
        self._check(image)
        b, c, h, w = image.shape
        s = self.scale
        if h % s or w % s:
            raise ValueError("HR height and width must be divisible by scale")
        return self._blur(image).reshape(b, c, h // s, s, w // s, s).mean(dim=(3, 5))

    def adjoint(self, residual):
        self._check(residual)
        expanded = residual.repeat_interleave(self.scale, -2).repeat_interleave(
            self.scale, -1
        )
        return self._blur(expanded / self.scale**2)


def _mask(value, reference, name):
    if value is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")
    try:
        return torch.broadcast_to(value.to(reference.device), reference.shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to {tuple(reference.shape)}") from exc


@torch.no_grad()
def back_projection(
    source,
    observed,
    config=None,
    *,
    valid_mask=None,
    output_mask=None,
    return_info=False,
):
    """Correct CHW/BCHW SR using x <- x + alpha D* M(y-Dx).

    `valid_mask` marks usable LR measurements. Optional `output_mask` marks HR
    pixels permitted to change; other HR pixels are held at zero throughout
    optimization. Masks must be bool and broadcastable to BCHW (use B1HW for
    spatial batch masks). With no observations, a sample is left unchanged,
    apart from output masking; diagnostics mark it as unconverged.

    Each sample stops independently at masked LR RMSE <= tolerance. The safe
    step is relaxation * scale**2. Half/bfloat16 inputs are promoted and RETURNED
    in float32; float64 is preserved. Autocast is disabled during correction.
    Outputs are not clipped. This is an inference operation, not a training loss.
    Optional diagnostics contain device tensors, including residual history
    (iterations+1, batch), per-sample iteration counts and convergence flags.
    """
    config = BackProjectionConfig() if config is None else config
    if not isinstance(config, BackProjectionConfig):
        raise TypeError("config must be BackProjectionConfig")
    if source.ndim not in (3, 4) or observed.ndim != source.ndim:
        raise ValueError("source and observed must both be CHW or both BCHW")
    if not source.is_floating_point() or not observed.is_floating_point():
        raise ValueError("source and observed must be floating-point tensors")
    if source.device != observed.device:
        raise ValueError("source and observed must be on the same device")
    unbatched = source.ndim == 3
    if unbatched:
        source, observed = source.unsqueeze(0), observed.unsqueeze(0)
    if (
        min(source.shape) < 1
        or min(observed.shape) < 1
        or source.shape[:2] != observed.shape[:2]
    ):
        raise ValueError(
            "source and observed need matching nonempty batch/band dimensions"
        )
    if source.shape[-2:] != tuple(n * config.scale for n in observed.shape[-2:]):
        raise ValueError("source spatial shape must equal observed shape times scale")
    dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=source.device.type, enabled=False):
        x, y = source.to(dtype=dtype).clone(), observed.to(dtype=dtype)
        valid = _mask(valid_mask, y, "valid_mask")
        support = _mask(output_mask, x, "output_mask")
        if not torch.isfinite(x[support]).all() or not torch.isfinite(y[valid]).all():
            raise ValueError("valid source/observation values must be finite")
        x = torch.where(support, x, 0.0)
        y = torch.where(valid, y, 0.0)
        operator = BlurDownsample(x.shape[1], config.scale, config.blur_sigma).to(x)
        counts = valid.sum(dim=(1, 2, 3))
        iterations = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        history = []
        for step in range(config.max_iterations + 1):
            residual = torch.where(valid, y - operator(x), 0.0)
            errors = (residual.square().sum(dim=(1, 2, 3)) / counts.clamp_min(1)).sqrt()
            if return_info:
                history.append(errors)
            active = (counts > 0) & (errors > config.tolerance)
            if step == config.max_iterations or not active.any():
                break
            update = config.relaxation * config.scale**2 * operator.adjoint(residual)
            x = x + torch.where(support & active[:, None, None, None], update, 0.0)
            iterations += active.long()
        output = x.squeeze(0) if unbatched else x
        if not return_info:
            return output
        return output, {
            "lr_rmse": errors,
            "iterations": iterations,
            "converged": (counts > 0) & (errors <= config.tolerance),
            "valid_measurements": counts,
            "residual_history": torch.stack(history),
        }
