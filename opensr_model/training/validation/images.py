"""Native-resolution local/W&B previews with fixed, shared display stretches."""

from __future__ import annotations

import json
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
    bounds = [
        np.quantile(reference[:3], [0.02, 0.98]),
        np.quantile(reference[3], [0.02, 0.98]),
    ]
    # Nearest-neighbour expansion exposes the original LR pixels in aligned zooms.
    up = np.repeat(np.repeat(lr, 4, axis=-2), 4, axis=-1)
    tensors = {"LR": up, "SR": sr, "HR": hr}
    h, w = sr.shape[-2:]
    center = row.get("zoom_center_hr", [h // 2, w // 2])
    if len(center) != 2 or not (0 <= center[0] < h and 0 <= center[1] < w):
        raise ValueError(f"Invalid zoom_center_hr for {row['id']}")
    (directory / f"{safe_id(row['id'])}_geometry.json").write_text(
        json.dumps({"zoom_center_hr": [int(v) for v in center]})
    )
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
                    a = a[:, top : top + size_y, left : left + size_x]
                a = a[:3].transpose(1, 2, 0) if index == 0 else a[3]
                lo, hi = bounds[index]
                display = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(
                    np.uint8
                )
                path = directory / f"{safe_id(row['id'])}_{view}_{band}_{kind}.png"
                Image.fromarray(display).save(path)
                result.append(str(path))
    return result


def log_outputs(
    trainer, directory: Path, table_rows: list[list], *, label: str, step: int
):
    run = wandb_run(trainer)
    if run is None:
        return
    import wandb

    name = safe_id(f"{run.id}-{label}")
    artifact = wandb.Artifact(
        name, type="reconstructions", metadata={"optimizer_step": step}
    )
    for path in sorted(directory.glob("*.npz")):
        artifact.add_file(str(path), skip_cache=True, policy="immutable")
    for path in sorted(directory.glob("*.json")):
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    log_preview_figures(run, directory, table_rows, label=label, step=step)


def log_preview_figures(
    run, directory: Path, rows: list[list], *, label: str, step: int
):
    """One stable W&B gallery per cohort/band, including aligned detail crops."""
    import wandb

    cohort = label.split("_step", 1)[0].removesuffix("_validate_only")
    figures = {band: [] for band in ("RGB", "NIR")}
    for row in rows:
        for band in figures:
            path = preview_figure(directory / "figures", row, band=band)
            figures[band].append(wandb.Image(str(path), caption=str(row[0])))
    if rows:
        run.log(
            {
                **{
                    f"validation/{cohort}_{band}": gallery
                    for band, gallery in figures.items()
                },
                "optimizer_step": step,
            }
        )


def preview_figure(directory: Path, row: list, *, band: str) -> Path:
    """Render saved reference-stretched panels without independently rescaling them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    columns = dict(zip(image_columns()[3:], row[3:]))
    kinds = [kind for kind in ("LR", "SR", "HR") if columns.get(f"full/{band}/{kind}")]
    geometry = Path(columns[f"full/{band}/LR"]).with_name(
        f"{safe_id(row[0])}_geometry.json"
    )
    center = (
        json.loads(geometry.read_text())["zoom_center_hr"]
        if geometry.exists()
        else None
    )
    views = (
        ("full", "Full scene"),
        ("zoom256", "256 × 256 detail"),
        ("zoom64", "64 × 64 detail"),
    )
    figure, axes = plt.subplots(
        3, len(kinds), figsize=(4 * len(kinds), 12), squeeze=False
    )
    try:
        for ri, (view, title) in enumerate(views):
            for ci, kind in enumerate(kinds):
                axis = axes[ri, ci]
                with Image.open(columns[f"{view}/{band}/{kind}"]) as source:
                    pixels = np.asarray(source)
                axis.imshow(
                    pixels,
                    cmap="gray" if band == "NIR" else None,
                    vmin=0,
                    vmax=255,
                    interpolation="nearest",
                )
                if ri == 0:
                    axis.set_title(kind)
                    height, width = pixels.shape[:2]
                    cy, cx = center or (height // 2, width // 2)
                    for size, color in ((256, "#ffba49"), (64, "#00cbe6")):
                        h, w = min(size, height), min(size, width)
                        top = max(0, min(height - h, cy - h // 2))
                        left = max(0, min(width - w, cx - w // 2))
                        axis.add_patch(
                            Rectangle(
                                (left - 0.5, top - 0.5),
                                w,
                                h,
                                fill=False,
                                edgecolor=color,
                                linewidth=1.2,
                            )
                        )
                if ci == 0:
                    axis.set_ylabel(title)
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(" · ".join(str(v) for v in [row[0], band, *row[1:3]] if v))
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe_id(row[0])}_{band}.png"
        figure.savefig(path, dpi=140)
        return path
    finally:
        plt.close(figure)
