"""Callbacks for complete-image logging and inference-compatible exports."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only


class ReconstructionImageLogger(Callback):
    """Save full validation tiles and forward figures to supported loggers.

    A validation step opts in by returning ``{"images": {name: BCHW, ...},
    "sample_ids": [...]}``. The callback accumulates a fixed number of examples,
    writes RGB (and, for RGB-NIR inputs, NIR) PNGs every epoch, and never crops the
    supplied tensors for display.
    """

    def __init__(
        self,
        *,
        max_images: int = 4,
        every_n_epochs: int = 1,
        output_dir: Optional[str] = None,
        rgb_bands: tuple[int, int, int] = (0, 1, 2),
        nir_band: Optional[int] = 3,
        value_min: float = 0.0,
        value_max: float = 0.3,
        dpi: int = 140,
    ) -> None:
        super().__init__()
        if max_images < 1:
            raise ValueError("max_images must be positive")
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if value_max <= value_min:
            raise ValueError("value_max must be greater than value_min")
        self.max_images = max_images
        self.every_n_epochs = every_n_epochs
        self.output_dir = output_dir
        self.rgb_bands = tuple(rgb_bands)
        self.nir_band = nir_band
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.dpi = dpi
        self._images: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        self._sample_ids: list[str] = []

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: torch.nn.Module
    ) -> None:
        self._images.clear()
        self._sample_ids.clear()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.is_global_zero or not isinstance(outputs, Mapping):
            return
        image_mapping = outputs.get("images")
        if not isinstance(image_mapping, Mapping) or not image_mapping:
            return
        remaining = self.max_images - len(self._sample_ids)
        if remaining <= 0:
            return

        first = next(iter(image_mapping.values()))
        if not isinstance(first, torch.Tensor) or first.ndim != 4:
            raise ValueError("Validation image outputs must be BCHW tensors")
        take = min(remaining, first.shape[0])
        for name, tensor in image_mapping.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError(f"Image panel {name!r} is not a BCHW tensor")
            if tensor.shape[0] < take:
                raise ValueError(
                    "All validation image panels must have the same batch size"
                )
            self._images.setdefault(str(name), []).append(
                tensor[:take].detach().float().cpu()
            )

        ids = outputs.get("sample_ids")
        if ids is None:
            ids = [f"batch{batch_idx:04d}_{index:02d}" for index in range(take)]
        self._sample_ids.extend(str(item) for item in list(ids)[:take])

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if trainer.sanity_checking or not self._images:
            return
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        images = OrderedDict(
            (name, torch.cat(parts, dim=0)[: self.max_images])
            for name, parts in self._images.items()
        )
        count = min(tensor.shape[0] for tensor in images.values())
        sample_ids = self._sample_ids[:count]
        epoch_dir = self._resolve_output_dir(trainer) / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        rgb_figure = self._make_figure(images, sample_ids, mode="rgb")
        rgb_path = epoch_dir / "reconstructions_rgb.png"
        rgb_figure.savefig(rgb_path, dpi=self.dpi, bbox_inches="tight")
        self._log_figure(trainer, "validation/reconstructions_rgb", rgb_figure)
        plt.close(rgb_figure)

        if self.nir_band is not None and any(
            tensor.shape[1] > self.nir_band for tensor in images.values()
        ):
            nir_figure = self._make_figure(images, sample_ids, mode="nir")
            nir_path = epoch_dir / "reconstructions_nir.png"
            nir_figure.savefig(nir_path, dpi=self.dpi, bbox_inches="tight")
            self._log_figure(trainer, "validation/reconstructions_nir", nir_figure)
            plt.close(nir_figure)

        # Separate files make it easy to inspect or publish a complete example without
        # extracting it from a grid. They use the same uncropped source tensors.
        for index, sample_id in enumerate(sample_ids):
            single = OrderedDict(
                (name, tensor[index : index + 1]) for name, tensor in images.items()
            )
            figure = self._make_figure(single, [sample_id], mode="rgb")
            safe_id = _safe_name(sample_id)
            figure.savefig(
                epoch_dir / f"{index:02d}_{safe_id}_rgb.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close(figure)

    def _resolve_output_dir(self, trainer: Trainer) -> Path:
        if self.output_dir:
            path = Path(self.output_dir).expanduser()
            if not path.is_absolute():
                path = Path(trainer.default_root_dir) / path
            return path
        log_dir = getattr(trainer, "log_dir", None)
        root = Path(log_dir) if log_dir else Path(trainer.default_root_dir)
        return root / "images"

    def _make_figure(
        self,
        images: Mapping[str, torch.Tensor],
        sample_ids: list[str],
        *,
        mode: str,
    ) -> plt.Figure:
        rows = max(1, len(sample_ids))
        columns = len(images)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.0 * columns, 4.0 * rows),
            squeeze=False,
        )
        for row, sample_id in enumerate(sample_ids):
            for column, (name, tensor) in enumerate(images.items()):
                axis = axes[row, column]
                panel = tensor[row]
                if (
                    mode == "nir"
                    and self.nir_band is not None
                    and panel.shape[0] > self.nir_band
                ):
                    axis.imshow(
                        panel[self.nir_band].numpy(),
                        cmap="gray",
                        vmin=self.value_min,
                        vmax=self.value_max,
                    )
                elif panel.shape[0] >= 3 and max(self.rgb_bands) < panel.shape[0]:
                    rgb = panel[list(self.rgb_bands)].permute(1, 2, 0).numpy()
                    rgb = np.clip(
                        (rgb - self.value_min) / (self.value_max - self.value_min),
                        0.0,
                        1.0,
                    )
                    axis.imshow(rgb)
                else:
                    data = panel.abs().mean(dim=0).numpy()
                    axis.imshow(data, cmap="magma", vmin=0.0)
                axis.set_title(f"{sample_id}\n{name}" if column == 0 else name)
                axis.axis("off")
        figure.tight_layout()
        return figure

    def _log_figure(self, trainer: Trainer, tag: str, figure: plt.Figure) -> None:
        for logger in trainer.loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_figure"):
                experiment.add_figure(tag, figure, global_step=trainer.global_step)
                continue
            if logger.__class__.__name__.lower().startswith("wandb"):
                try:
                    import wandb

                    experiment.log({tag: wandb.Image(figure)}, step=trainer.global_step)
                except (ImportError, AttributeError):
                    pass


class NativeCheckpointCallback(Callback):
    """Export checkpoints accepted by the repository's strict inference loader.

    Normal ``ModelCheckpoint`` files remain available for exact Lightning resume.
    This callback deliberately creates a second artifact without wrapper, metric,
    perceptual-loss, optimizer, or discriminator state.
    """

    def __init__(
        self,
        *,
        dirpath: str,
        monitor: str = "val/loss",
        mode: str = "min",
        save_last: bool = True,
        save_best: bool = True,
        every_n_epochs: int = 1,
        autoencoder_base_checkpoint: Optional[str] = None,
        weight_source: str = "ema",
    ) -> None:
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if not save_last and not save_best:
            raise ValueError("At least one of save_last or save_best must be enabled")
        weight_source = str(weight_source).lower()
        if weight_source not in {"raw", "ema"}:
            raise ValueError("weight_source must be 'raw' or 'ema'")
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_best = save_best
        self.every_n_epochs = every_n_epochs
        self.autoencoder_base_checkpoint = autoencoder_base_checkpoint
        self.weight_source = weight_source
        self.best_score = math.inf if mode == "min" else -math.inf
        self._full_state_spec: Any = None

    @property
    def state_key(self) -> str:
        """Keep independent destinations and monitors separate on resume."""

        return self._generate_state_key(
            dirpath=os.path.normpath(os.fspath(Path(self.dirpath).expanduser())),
            monitor=self.monitor,
            mode=self.mode,
            save_last=self.save_last,
            save_best=self.save_best,
            every_n_epochs=self.every_n_epochs,
            weight_source=self.weight_source,
            autoencoder_base_checkpoint=(
                None
                if self.autoencoder_base_checkpoint is None
                else os.path.normpath(
                    os.fspath(Path(self.autoencoder_base_checkpoint).expanduser())
                )
            ),
        )

    def setup(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        stage: Optional[str] = None,
    ) -> None:
        """Fail early for strategies that expose only local state shards."""

        strategy_name = type(trainer.strategy).__name__.lower()
        if any(token in strategy_name for token in ("fsdp", "deepspeed")):
            raise RuntimeError(
                "NativeCheckpointCallback requires an unsharded state_dict and does "
                f"not support {type(trainer.strategy).__name__}. Keep Lightning's "
                "sharded resume checkpoints, then export with a single-device or "
                "DDP strategy."
            )

    def state_dict(self) -> dict[str, Any]:
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        value = float(state_dict.get("best_score", self.best_score))
        if not math.isnan(value):
            self.best_score = value

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        if not hasattr(pl_module, "native_state_dict"):
            raise TypeError(
                "NativeCheckpointCallback requires the Lightning module to implement "
                "native_state_dict()"
            )

        metric = trainer.callback_metrics.get(self.monitor)
        score = None if metric is None else float(metric.detach().cpu())
        improved = score is not None and (
            score < self.best_score if self.mode == "min" else score > self.best_score
        )

        output_dir = Path(self.dirpath).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path(trainer.default_root_dir) / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = (
            "inference"
            if getattr(pl_module, "checkpoint_stage", "") == "diffusion"
            or self.autoencoder_base_checkpoint
            else "autoencoder"
        )
        last_path = output_dir / f"last-{suffix}.ckpt"
        best_path = output_dir / f"best-{suffix}.ckpt"
        if self.save_last:
            self._save(last_path, trainer, pl_module, score)
        if self.save_best and improved:
            if self.save_last:
                _clone_checkpoint(last_path, best_path)
            else:
                self._save(best_path, trainer, pl_module, score)
            # Update only after the destination has been committed successfully.
            self.best_score = float(score)

    def _save(
        self,
        path: Path,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        score: Optional[float],
    ) -> None:
        from .checkpoints import (
            build_native_diffusion_state_spec,
            materialize_diffusion_ema_weights,
            merge_autoencoder_checkpoint,
            save_native_checkpoint,
        )

        state_dict = pl_module.native_state_dict()
        stage = getattr(pl_module, "checkpoint_stage", "unknown")
        if stage == "diffusion" and self.weight_source == "ema":
            state_dict = materialize_diffusion_ema_weights(state_dict)
        metadata = {
            "stage": stage,
            "monitor": self.monitor,
            "monitor_value": score,
        }
        if stage == "diffusion":
            metadata["inference_unet_weight_source"] = self.weight_source
            inference_config = getattr(pl_module, "opensr_inference_config", None)
            if not callable(inference_config):
                raise TypeError(
                    "Diffusion native export requires pl_module."
                    "opensr_inference_config() for sampling provenance"
                )
            metadata["opensr_inference_config"] = inference_config()
        generator_updates = getattr(pl_module, "generator_updates", None)
        if isinstance(generator_updates, torch.Tensor):
            metadata["generator_updates"] = int(generator_updates.detach().cpu())
        if (
            getattr(pl_module, "checkpoint_stage", "") == "autoencoder"
            and self.autoencoder_base_checkpoint
        ):
            if self._full_state_spec is None:
                model_config = getattr(pl_module, "model_config", None)
                if model_config is None:
                    raise TypeError(
                        "Autoencoder full-checkpoint export requires pl_module."
                        "model_config to validate the base architecture"
                    )
                self._full_state_spec = build_native_diffusion_state_spec(model_config)
            merge_autoencoder_checkpoint(
                self.autoencoder_base_checkpoint,
                state_dict,
                path,
                expected_full_state_dict=self._full_state_spec,
                epoch=int(trainer.current_epoch),
                global_step=int(trainer.global_step),
                metadata=metadata,
            )
        else:
            save_native_checkpoint(
                path,
                state_dict,
                epoch=int(trainer.current_epoch),
                global_step=int(trainer.global_step),
                metadata=metadata,
            )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:100] or "sample"


def _clone_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    temporary = temporary_dir / "checkpoint.tmp"
    try:
        try:
            os.link(source, temporary)
        except OSError:
            with source.open("rb") as source_handle, temporary.open("xb") as target:
                shutil.copyfileobj(source_handle, target, length=16 * 1024 * 1024)
                target.flush()
                _best_effort_fsync(target.fileno())
        try:
            with temporary.open("rb") as handle:
                _best_effort_fsync(handle.fileno())
        except OSError:
            pass
        os.replace(temporary, destination)
        _best_effort_sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
        try:
            temporary_dir.rmdir()
        except OSError:
            pass


def _best_effort_fsync(file_descriptor: int) -> None:
    try:
        os.fsync(file_descriptor)
    except OSError:
        pass


def _best_effort_sync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        return
    try:
        _best_effort_fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = ["NativeCheckpointCallback", "ReconstructionImageLogger"]
