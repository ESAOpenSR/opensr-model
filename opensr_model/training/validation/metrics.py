"""Full-resolution reflectance metrics and site-cluster paired uncertainty."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

from ..modules.metrics import masked_mean, spectral_angle, downsample_consistency


def reconstruction_scores(sr, hr, lr, mask, lpips):
    sr, hr, lr = sr.float(), hr.float(), lr.float()
    out = {}
    for name, channels in (("rgb", slice(0, 3)), ("nir", slice(3, 4)), ("rgbn", slice(0, 4))):
        pred, target = sr[:, channels], hr[:, channels]
        err = pred - target
        mse = masked_mean(err.square(), mask)
        out[name + "_mae"] = masked_mean(err.abs(), mask)
        out[name + "_rmse"] = mse.sqrt()
        out[name + "_bias"] = masked_mean(err, mask)
        out[name + "_psnr"] = -10 * torch.log10(mse.clamp_min(1e-12))
        out[name + "_lr_consistency"] = downsample_consistency(pred, lr[:, channels], mask)
        _, ssim_map = structural_similarity_index_measure(pred, target, data_range=1., return_full_image=True)
        out[name + "_ssim"] = masked_mean(ssim_map, mask)
    out["sam"] = spectral_angle(sr, hr, mask)
    pred = torch.where(mask, sr[:, :3], hr[:, :3])
    lpmap = lpips(pred * 2 - 1, hr[:, :3] * 2 - 1)
    invalid = 1 - mask.float()
    invalid = F.max_pool2d(invalid, (1, 65), stride=1, padding=(0, 32))
    invalid = F.max_pool2d(invalid, (65, 1), stride=1, padding=(32, 0))
    lp_mask = mask & (invalid < .5)
    out["rgb_lpips"] = masked_mean(lpmap, lp_mask) if lp_mask.any() else sr.new_tensor(float("nan"))
    out["lpips_valid_fraction"] = lp_mask.float().mean()
    a, b = hr[:, 3:4] + hr[:, 0:1], sr[:, 3:4] + sr[:, 0:1]
    ndvi_mask = mask & (a > .02) & (b > .02)
    ndvi_hr = (hr[:, 3:4] - hr[:, 0:1]) / a.clamp_min(1e-6)
    ndvi_sr = (sr[:, 3:4] - sr[:, 0:1]) / b.clamp_min(1e-6)
    out["ndvi_mae"] = masked_mean((ndvi_sr - ndvi_hr).abs(), ndvi_mask) if ndvi_mask.any() else sr.new_tensor(float("nan"))
    return {k: float(v.detach().cpu()) for k, v in out.items()}


def finite_json(scores):
    return {k: v if v is not None and math.isfinite(v) else None for k, v in scores.items()}


def aggregate(records: Sequence[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        for key, value in row["metrics"].items():
            if value is not None and math.isfinite(value):
                groups[key].append(value)
    return {key: float(np.mean(values)) for key, values in groups.items()}


def guardrail_decision(scores: dict, baseline: dict, tolerance: float = .10) -> tuple[bool, list[str]]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Spectral tolerance must be finite and nonnegative")
    reasons = []
    for name in ("rgb_lpips", "nir_mae", "sam"):
        if any(not isinstance(s.get(name), (int, float)) or not math.isfinite(s[name]) for s in (scores, baseline)):
            reasons.append(f"nonfinite_{name}")
    if reasons:
        return False, reasons
    for name in ("nir_mae", "sam"):
        limit = baseline[name] * (1 + tolerance)
        if scores[name] > limit + 1e-12:
            reasons.append(f"{name}_exceeds_limit")
    return not reasons, reasons


def paired_report(baseline: Sequence[dict], candidate: Sequence[dict], *, seed=42, resamples=2000):
    """Resample whole sites; adjacent tiles are not independent observations."""
    reference = {r["id"]: r for r in baseline}
    paired = [(reference[r["id"]], r) for r in candidate if r["id"] in reference]
    sites = sorted({r["site"] for _, r in paired})
    out = {"paired_tiles": len(paired), "sites": len(sites), "metrics": {}}
    if not paired:
        return out
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sites), size=(resamples, len(sites)))
    for metric in sorted(paired[0][0]["metrics"]):
        values = defaultdict(list)
        for a, b in paired:
            x, y = a["metrics"].get(metric), b["metrics"].get(metric)
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                values[b["site"]].append(y - x)
        sums = np.array([sum(values[s]) for s in sites])
        counts = np.array([len(values[s]) for s in sites])
        if not counts.sum():
            continue
        denominators = counts[draws].sum(1)
        valid = denominators > 0
        sampled = sums[draws].sum(1)[valid] / denominators[valid]
        out["metrics"][metric] = {
            "mean_delta": float(sums.sum() / counts.sum()),
            "site_balanced_delta": float(np.mean([np.mean(values[s]) for s in sites if values[s]])),
            "ci95": np.quantile(sampled, [.025, .975]).tolist(),
        }
    return out
