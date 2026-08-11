"""Callbacks for complete-image logging and inference-compatible exports."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import warnings
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


class NvidiaPowerCostLogger(Callback):
    """Integrate NVIDIA board power and log estimated electricity cost to W&B.

    ``actual`` means the GPU board power reported by ``nvidia-smi``. It is not a
    wall-socket measurement. ``extrapolated`` adds a configurable fixed allowance
    for the CPU, motherboard, memory, storage, cooling and PSU conversion losses.
    """

    def __init__(
        self,
        *,
        electricity_eur_per_kwh: float = 0.30,
        overhead_watts: float = 200.0,
        every_n_steps: int = 20,
        query_timeout_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        if electricity_eur_per_kwh < 0:
            raise ValueError("electricity_eur_per_kwh must be non-negative")
        if overhead_watts < 0:
            raise ValueError("overhead_watts must be non-negative")
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        self.electricity_eur_per_kwh = float(electricity_eur_per_kwh)
        self.overhead_watts = float(overhead_watts)
        self.every_n_steps = int(every_n_steps)
        self.query_timeout_seconds = float(query_timeout_seconds)
        self.gpu_energy_kwh = 0.0
        self.extrapolated_energy_kwh = 0.0
        self._previous_time: float | None = None
        self._previous_gpu_power_watts: float | None = None
        self._last_logged_step: int | None = None
        self._warned = False

    def state_dict(self) -> dict[str, float]:
        return {
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "extrapolated_energy_kwh": self.extrapolated_energy_kwh,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.gpu_energy_kwh = float(state_dict.get("gpu_energy_kwh", 0.0))
        self.extrapolated_energy_kwh = float(
            state_dict.get("extrapolated_energy_kwh", self.gpu_energy_kwh)
        )
        # A resumed job must not charge for the time between processes.
        self._previous_time = None
        self._previous_gpu_power_watts = None

    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._sample_and_log(trainer)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step % self.every_n_steps == 0 and step != self._last_logged_step:
            self._sample_and_log(trainer)

    def on_validation_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._sample_and_log(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._sample_and_log(trainer)

    def on_fit_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._sample_and_log(trainer)

    @rank_zero_only
    def _sample_and_log(self, trainer: Trainer) -> None:
        try:
            per_gpu_watts = self._read_gpu_power_watts()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if not self._warned:
                warnings.warn(
                    f"GPU cost logging disabled until telemetry recovers: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            self._previous_time = None
            self._previous_gpu_power_watts = None
            return

        now = time.monotonic()
        gpu_power_watts = sum(per_gpu_watts)
        interval_gpu_energy_kwh = 0.0
        interval_extrapolated_energy_kwh = 0.0
        if (
            self._previous_time is not None
            and self._previous_gpu_power_watts is not None
        ):
            elapsed_hours = max(0.0, now - self._previous_time) / 3600.0
            average_gpu_kw = (self._previous_gpu_power_watts + gpu_power_watts) / 2000.0
            interval_gpu_energy_kwh = average_gpu_kw * elapsed_hours
            interval_extrapolated_energy_kwh = (
                average_gpu_kw + self.overhead_watts / 1000.0
            ) * elapsed_hours
            self.gpu_energy_kwh += interval_gpu_energy_kwh
            self.extrapolated_energy_kwh += interval_extrapolated_energy_kwh

        self._previous_time = now
        self._previous_gpu_power_watts = gpu_power_watts
        self._last_logged_step = int(trainer.global_step)
        metrics = {
            "cost/actual/gpu_power_kw": gpu_power_watts / 1000.0,
            "cost/actual/interval_energy_kwh": interval_gpu_energy_kwh,
            "cost/actual/energy_kwh": self.gpu_energy_kwh,
            "cost/actual/price_eur": (
                self.gpu_energy_kwh * self.electricity_eur_per_kwh
            ),
            "cost/extrapolated/system_power_kw": (gpu_power_watts + self.overhead_watts)
            / 1000.0,
            "cost/extrapolated/interval_energy_kwh": (interval_extrapolated_energy_kwh),
            "cost/extrapolated/energy_kwh": self.extrapolated_energy_kwh,
            "cost/extrapolated/price_eur": (
                self.extrapolated_energy_kwh * self.electricity_eur_per_kwh
            ),
        }
        metrics.update(
            {
                f"cost/actual/gpu_{index}_power_kw": watts / 1000.0
                for index, watts in enumerate(per_gpu_watts)
            }
        )
        for logger in trainer.loggers:
            if logger.__class__.__name__.lower().startswith("wandb"):
                logger.log_metrics(metrics, step=int(trainer.global_step))

    def _read_gpu_power_watts(self) -> list[float]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.query_timeout_seconds,
        )
        values = [
            float(line.strip()) for line in result.stdout.splitlines() if line.strip()
        ]
        if not values:
            raise ValueError("nvidia-smi returned no GPU power readings")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(
                f"nvidia-smi returned invalid GPU power readings: {values}"
            )
        return values


class ReconstructionImageLogger(Callback):
    """Save full validation tiles and optional detail crops to supported loggers.

    A validation step opts in by returning ``{"images": {name: BCHW, ...},
    "details": {name: BCHW, ...}, "sample_ids": [...]}``. The callback accumulates
    a fixed number of examples and writes RGB (and, for RGB-NIR inputs, NIR) PNGs
    for both mappings. Detail tensors are prepared by the module so LR and HR crops
    can remain exactly aligned.
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
        self._details: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        self._sample_ids: list[str] = []
        self._validation_run_index = 0

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: torch.nn.Module
    ) -> None:
        self._validation_run_index = 0

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: torch.nn.Module
    ) -> None:
        self._images.clear()
        self._details.clear()
        self._sample_ids.clear()
        if not trainer.sanity_checking:
            self._validation_run_index += 1

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

        detail_mapping = outputs.get("details")
        if detail_mapping is not None:
            if not isinstance(detail_mapping, Mapping):
                raise TypeError("Validation detail panels must be a mapping")
            for name, tensor in detail_mapping.items():
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    raise ValueError(f"Detail panel {name!r} is not a BCHW tensor")
                if tensor.shape[0] < take:
                    raise ValueError(
                        "All validation detail panels must match the image batch size"
                    )
                self._details.setdefault(str(name), []).append(
                    tensor[:take].detach().float().cpu()
                )

        ids = outputs.get("sample_ids")
        if ids is None:
            ids = [f"batch{batch_idx:04d}_{index:02d}" for index in range(take)]
        self._sample_ids.extend(str(item) for item in list(ids)[:take])

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if not self._images:
            return
        is_sanity = bool(trainer.sanity_checking)
        epoch = int(trainer.current_epoch)
        if not is_sanity and (epoch + 1) % self.every_n_epochs != 0:
            return

        images = OrderedDict(
            (name, torch.cat(parts, dim=0)[: self.max_images])
            for name, parts in self._images.items()
        )
        details = OrderedDict(
            (name, torch.cat(parts, dim=0)[: self.max_images])
            for name, parts in self._details.items()
        )
        count = min(tensor.shape[0] for tensor in images.values())
        sample_ids = self._sample_ids[:count]
        if is_sanity:
            validation_dir = self._resolve_output_dir(trainer) / "sanity"
            tag_prefix = "sanity"
        else:
            validation_dir = (
                self._resolve_output_dir(trainer)
                / f"epoch_{epoch:04d}"
                / (
                    f"validation_{self._validation_run_index:02d}"
                    f"_step_{int(trainer.global_step):08d}"
                )
            )
            tag_prefix = "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        figures: OrderedDict[str, plt.Figure] = OrderedDict()
        rgb_figure = self._make_figure(images, sample_ids, mode="rgb")
        rgb_figure.savefig(
            validation_dir / "reconstructions_rgb.png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        figures[f"{tag_prefix}/reconstructions_rgb"] = rgb_figure

        if self.nir_band is not None and any(
            tensor.shape[1] > self.nir_band for tensor in images.values()
        ):
            nir_figure = self._make_figure(images, sample_ids, mode="nir")
            nir_figure.savefig(
                validation_dir / "reconstructions_nir.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            figures[f"{tag_prefix}/reconstructions_nir"] = nir_figure

        if details:
            detail_rgb_figure = self._make_figure(details, sample_ids, mode="rgb")
            detail_rgb_figure.savefig(
                validation_dir / "details_rgb.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            figures[f"{tag_prefix}/details_rgb"] = detail_rgb_figure
            if self.nir_band is not None and any(
                tensor.shape[1] > self.nir_band for tensor in details.values()
            ):
                detail_nir_figure = self._make_figure(details, sample_ids, mode="nir")
                detail_nir_figure.savefig(
                    validation_dir / "details_nir.png",
                    dpi=self.dpi,
                    bbox_inches="tight",
                )
                figures[f"{tag_prefix}/details_nir"] = detail_nir_figure

        self._log_figures(trainer, figures)
        for figure in figures.values():
            plt.close(figure)

        # Separate files make it easy to inspect or publish a complete example without
        # extracting it from a grid. They use the same uncropped source tensors.
        for index, sample_id in enumerate(sample_ids):
            single = OrderedDict(
                (name, tensor[index : index + 1]) for name, tensor in images.items()
            )
            figure = self._make_figure(single, [sample_id], mode="rgb")
            safe_id = _safe_name(sample_id)
            figure.savefig(
                validation_dir / f"{index:02d}_{safe_id}_rgb.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close(figure)
            if details:
                detail = OrderedDict(
                    (name, tensor[index : index + 1])
                    for name, tensor in details.items()
                )
                detail_figure = self._make_figure(detail, [sample_id], mode="rgb")
                detail_figure.savefig(
                    validation_dir / f"{index:02d}_{safe_id}_detail_rgb.png",
                    dpi=self.dpi,
                    bbox_inches="tight",
                )
                plt.close(detail_figure)

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

    def _log_figures(self, trainer: Trainer, figures: Mapping[str, plt.Figure]) -> None:
        for logger in trainer.loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_figure"):
                for tag, figure in figures.items():
                    experiment.add_figure(tag, figure, global_step=trainer.global_step)
                continue
            if logger.__class__.__name__.lower().startswith("wandb"):
                try:
                    import wandb

                    experiment.log(
                        {tag: wandb.Image(figure) for tag, figure in figures.items()},
                        step=trainer.global_step,
                    )
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


__all__ = [
    "NativeCheckpointCallback",
    "NvidiaPowerCostLogger",
    "ReconstructionImageLogger",
]
