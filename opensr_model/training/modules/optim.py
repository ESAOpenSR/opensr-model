"""Shared optimizer and scheduler construction."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional

import torch


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    config: Optional[Mapping[str, Any]] = None,
) -> torch.optim.AdamW:
    config = dict(config or {})
    betas = tuple(config.get("betas", (0.9, 0.999)))
    if len(betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    return torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", config.get("lr", 1e-4))),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(config.get("eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Optional[Mapping[str, Any]],
    *,
    total_steps: int,
    monitor: str = "val/loss",
) -> Optional[dict[str, Any]]:
    """Build a Lightning scheduler dictionary.

    Supported names are ``constant``, ``cosine``, and ``plateau``. Cosine has an
    optional linear warmup and is stepped per optimizer update.
    """

    config = dict(config or {})
    name = str(config.get("name", "constant")).lower()
    if name in {"constant", "none", "null"}:
        return None
    if name == "cosine":
        warmup_steps = int(config.get("warmup_steps", 0))
        min_ratio = float(config.get("min_lr_ratio", 0.0))
        total_steps = max(int(total_steps), warmup_steps + 1)

        def multiplier(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(step + 1, 1) / float(warmup_steps)
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}
    if name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "min")),
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 5)),
            min_lr=float(config.get("min_lr", 0.0)),
        )
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": str(config.get("monitor", monitor)),
        }
    raise ValueError(
        f"Unknown scheduler {name!r}; expected constant, cosine, or plateau"
    )


__all__ = ["build_adamw", "build_scheduler"]
