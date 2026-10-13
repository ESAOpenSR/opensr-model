"""Native-resolution local/W&B previews with fixed, shared display stretches."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def wandb_run(trainer):
    from pytorch_lightning.loggers import WandbLogger
    if trainer.is_global_zero and isinstance(trainer.logger, WandbLogger):
        return trainer.logger.experiment
    return None


def image_columns() -> list[str]:
    return ["id", "site", "landcover"] + [
        f"{view}/{band}/{kind}"
        for view in ("full", "zoom64", "zoom256")
        for band in ("RGB", "NIR")
        for kind in ("LR", "SR", "HR")
    ]


def preview_row(directory: Path, row: dict, lr, sr, hr=None) -> list:
    """Bounds come from the reference, or input for unpaired previews, never SR."""
    directory.mkdir(parents=True, exist_ok=True)
    reference = hr if hr is not None else lr
    bounds = [np.quantile(reference[:3], [.02, .98]), np.quantile(reference[3], [.02, .98])]
    # Nearest-neighbour expansion exposes the original LR pixels in aligned zooms.
    up = np.repeat(np.repeat(lr, 4, axis=-2), 4, axis=-1)
    tensors = {"LR": up, "SR": sr, "HR": hr}
    h, w = sr.shape[-2:]
    center = row.get("zoom_center_hr", [h // 2, w // 2])
    if len(center) != 2 or not (0 <= center[0] < h and 0 <= center[1] < w):
        raise ValueError(f"Invalid zoom_center_hr for {row['id']}")
    result = [row["id"], row.get("site", ""), row.get("landcover", "")]
    for view, size in (("full", None), ("zoom64", 64), ("zoom256", 256)):
        for band, index in (("RGB", 0), ("NIR", 1)):
            for kind, tensor in tensors.items():
                if tensor is None:
                    result.append(None)
                    continue
                a = tensor
                if size is not None:
                    size_y, size_x = min(size, h), min(size, w)
                    top = max(0, min(h - size_y, int(center[0]) - size_y // 2))
                    left = max(0, min(w - size_x, int(center[1]) - size_x // 2))
                    a = a[:, top:top + size_y, left:left + size_x]
                a = a[:3].transpose(1, 2, 0) if index == 0 else a[3]
                lo, hi = bounds[index]
                display = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
                path = directory / f"{safe_id(row['id'])}_{view}_{band}_{kind}.png"
                Image.fromarray(display).save(path)
                result.append(str(path))
    return result


def log_outputs(trainer, directory: Path, table_rows: list[list], *, label: str, step: int):
    run = wandb_run(trainer)
    if run is None:
        return
    import wandb
    name = safe_id(f"{run.id}-{label}")
    artifact = wandb.Artifact(name, type="reconstructions", metadata={"optimizer_step": step})
    for path in sorted(directory.glob("*.npz")):
        artifact.add_file(str(path), skip_cache=True, policy="immutable")
    for path in sorted(directory.glob("*.json")):
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    if table_rows:
        data = [[*r[:3], *[wandb.Image(p) if p else None for p in r[3:]]] for r in table_rows]
        run.log({f"validation/{label}": wandb.Table(columns=image_columns(), data=data),
                 "optimizer_step": step})
