"""Render a controlled RGB comparison of DDIM eta and temperature.

Example:
    python -m opensr_model.ablations.temperature_eta \
        --input path/to/rgb_nir.tif --checkpoint opensr-ldsrs2_v1_0_0.ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

from opensr_model import SRLatentDiffusion

DEFAULT_ETAS = (0.0, 0.5, 0.95)
DEFAULT_TEMPERATURES = (0.0, 0.5, 1.0, 1.5)


def _parse_values(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative values")
    return values


def _read_input(path: Path) -> torch.Tensor:
    """Read RGB-NIR reflectance from a GeoTIFF or a saved tensor."""
    if path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            for key in ("LR_image", "image", "tensor"):
                if key in value:
                    value = value[key]
                    break
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path} does not contain a tensor")
        image = value.detach().to(dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)
    else:
        with rasterio.open(path) as dataset:
            if dataset.count < 4:
                raise ValueError(
                    "input raster must contain RGB and NIR (at least 4 bands)"
                )
            image = torch.from_numpy(dataset.read((1, 2, 3, 4))).unsqueeze(0).float()

    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] < 4:
        raise ValueError("input must have shape (1, 4, H, W) or (4, H, W)")
    image = image[:, :4]
    # Sentinel-2 products are commonly stored as uint16 reflectance * 10,000.
    if image.max().item() > 2:
        image = image / 10_000.0
    return image.clamp_min(0)


def _texture_crop(
    image: torch.Tensor, size: int = 128
) -> tuple[torch.Tensor, int, int]:
    """Select the valid crop with the most RGB edge content."""
    height, width = image.shape[-2:]
    if height < size or width < size:
        raise ValueError(f"input must be at least {size}x{size} pixels")
    if (height, width) == (size, size):
        return image, 0, 0

    gray = image[:, :3].mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    valid = (image[:, :3] > 0).all(dim=1, keepdim=True).float()
    kernel = torch.ones((1, 1, size, size), dtype=image.dtype)
    texture = F.conv2d((dx + dy) * valid, kernel, stride=max(1, size // 8))
    coverage = F.conv2d(valid, kernel, stride=max(1, size // 8)) / float(size * size)
    scores = texture.masked_fill(coverage < 0.98, -1)
    flat_index = int(scores.flatten().argmax())
    output_width = scores.shape[-1]
    stride = max(1, size // 8)
    top = (flat_index // output_width) * stride
    left = (flat_index % output_width) * stride
    return image[..., top : top + size, left : left + size], top, left


def _rgb_bounds(images: Sequence[torch.Tensor]) -> tuple[float, float]:
    pixels = torch.cat([item[0, :3].reshape(-1) for item in images])
    pixels = pixels[torch.isfinite(pixels) & (pixels > 0)]
    if not pixels.numel():
        return 0.0, 1.0
    # Keep large grids below torch.quantile its 2**24-element limit.
    max_pixels = 2_000_000
    if pixels.numel() > max_pixels:
        stride = max(1, pixels.numel() // max_pixels)
        pixels = pixels[::stride][:max_pixels]
    low, high = torch.quantile(pixels, torch.tensor([0.02, 0.98])).tolist()
    return float(low), float(max(high, low + 1e-6))


def _rgb(image: torch.Tensor, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    rgb = (image[0, :3].detach().float().cpu() - low) / (high - low)
    return rgb.clamp(0, 1).permute(1, 2, 0).numpy()


def _detail_crop(image: torch.Tensor, fraction: float = 0.5) -> torch.Tensor:
    height, width = image.shape[-2:]
    crop_height, crop_width = round(height * fraction), round(width * fraction)
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return image[..., top : top + crop_height, left : left + crop_width]


def _sample(
    model: SRLatentDiffusion,
    image: torch.Tensor,
    *,
    eta: float,
    temperature: float,
    steps: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if image.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return (
        model(
            image,
            sampling_eta=eta,
            sampling_temperature=temperature,
            sampling_steps=steps,
            histogram_matching=True,
        )
        .detach()
        .cpu()
    )


def render_ablation(
    *,
    input_path: Path,
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    etas: Sequence[float] = DEFAULT_ETAS,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    steps: int = 100,
    seed: int = 42,
    crop_size: int = 128,
    device: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _read_input(input_path)
    lr, crop_top, crop_left = _texture_crop(source, crop_size)
    config = OmegaConf.load(config_path)
    model = SRLatentDiffusion(config, device=device)
    model.load_pretrained(str(checkpoint))
    lr_device = lr.to(device)

    predictions: dict[tuple[float, float], torch.Tensor] = {}
    for eta in etas:
        for temperature in temperatures:
            print(f"Sampling eta={eta:g}, temperature={temperature:g}")
            predictions[(eta, temperature)] = _sample(
                model,
                lr_device,
                eta=eta,
                temperature=temperature,
                steps=steps,
                seed=seed,
            )

    ordered_predictions = [
        predictions[(eta, temperature)] for eta in etas for temperature in temperatures
    ]
    bounds = _rgb_bounds([lr, *ordered_predictions])
    figure, axes = plt.subplots(
        len(etas),
        len(temperatures) + 1,
        figsize=(3.5 * (len(temperatures) + 1), 3.5 * len(etas)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    lr_up = F.interpolate(lr, scale_factor=4, mode="nearest")
    lr_detail = _detail_crop(lr_up)
    for row, eta in enumerate(etas):
        axes[row, 0].imshow(_rgb(lr_detail, bounds), interpolation="nearest")
        axes[row, 0].set_title("LR input (nearest)", fontsize=11)
        axes[row, 0].set_ylabel(f"eta = {eta:g}", fontsize=11)
        for column, temperature in enumerate(temperatures, start=1):
            detail = _detail_crop(predictions[(eta, temperature)])
            axes[row, column].imshow(_rgb(detail, bounds), interpolation="nearest")
            axes[row, column].set_title(f"T={temperature:g}, eta={eta:g}", fontsize=11)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"OpenSR RGB sampling ablation — fixed seed {seed}, {steps} DDIM steps\n"
        "Shared 2–98% RGB stretch; center 50% detail crop",
        fontsize=14,
    )
    output_path = output_dir / "temperature_eta_rgb_closeup.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    overview = plt.figure(figsize=(6, 6))
    axis = overview.add_subplot(111)
    axis.imshow(_rgb(source, _rgb_bounds([source])))
    axis.add_patch(
        Rectangle(
            (crop_left, crop_top),
            crop_size,
            crop_size,
            fill=False,
            color="yellow",
            linewidth=2,
        )
    )
    axis.set_title("Selected LR source crop (RGB)")
    axis.axis("off")
    overview.savefig(
        output_dir / "selected_source_crop.png", dpi=160, bbox_inches="tight"
    )
    plt.close(overview)

    metadata = {
        "input": str(input_path),
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "device": device,
        "seed": seed,
        "sampling_steps": steps,
        "etas": list(etas),
        "temperatures": list(temperatures),
        "lr_crop": {"top": crop_top, "left": crop_left, "size": crop_size},
        "detail_fraction": 0.5,
        "rgb_stretch": {"low": bounds[0], "high": bounds[1]},
        "note": "Temperature has no effect at eta=0 because the DDIM sigma is zero.",
    }
    (output_dir / "temperature_eta_rgb_closeup.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="4-band RGB-NIR TIFF or tensor"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "config_10m.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "outputs"
    )
    parser.add_argument("--etas", type=_parse_values, default=DEFAULT_ETAS)
    parser.add_argument(
        "--temperatures", type=_parse_values, default=DEFAULT_TEMPERATURES
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = render_ablation(
        input_path=args.input,
        checkpoint=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        etas=args.etas,
        temperatures=args.temperatures,
        steps=args.steps,
        seed=args.seed,
        crop_size=args.crop_size,
        device=args.device,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
