"""Command-line entrypoints for independent OpenSR training stages."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from .autoencoder_module import AutoencoderTrainingModule
from .callbacks import NativeCheckpointCallback, ReconstructionImageLogger
from .checkpoints import load_checkpoint_payload
from .config import (
    as_plain_dict,
    load_model_config,
    load_training_config,
    package_config_path,
    save_resolved_config,
)
from .data import OpenSRDataModule
from .diffusion_module import DiffusionTrainingModule


def run_training(
    config: DictConfig,
    *,
    validate_only: bool = False,
) -> None:
    """Construct and run one fully configured Lightning stage."""

    stage = str(config.stage)
    pl.seed_everything(int(config.seed), workers=True)
    matmul_precision = str(config.get("float32_matmul_precision", "high"))
    torch.set_float32_matmul_precision(matmul_precision)

    run_dir = Path(str(config.run_dir)).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config.run_dir = str(run_dir)

    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint:
        _validate_resume_checkpoint(resume_checkpoint)
    model_config = load_model_config(config.model.architecture_config)
    datamodule = OpenSRDataModule(**as_plain_dict(config.data))
    resolved_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    # Freeze directory discovery in the run record. Reusing this resolved config
    # on resume retains the exact parts even if later uploads have completed.
    resolved_config.data.taco_path = [str(path) for path in datamodule.taco_paths]
    save_resolved_config(resolved_config, run_dir / "resolved_config.yaml")
    if stage == "diffusion" and datamodule.factor != 4:
        raise ValueError(
            "The current OpenSR inference architecture requires data.factor=4"
        )
    module = _build_module(
        stage, model_config, config, resuming=bool(resume_checkpoint)
    )
    logger = _build_logger(config.logging, run_dir)
    callbacks = _build_callbacks(stage, config, run_dir, logger is not False)

    trainer_kwargs = as_plain_dict(config.trainer)
    _prepare_trainer_kwargs(stage, module, trainer_kwargs)
    trainer_kwargs["default_root_dir"] = str(run_dir)
    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = callbacks
    trainer = pl.Trainer(**trainer_kwargs)

    if validate_only:
        trainer.validate(module, datamodule=datamodule, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_checkpoint)


def _build_module(
    stage: str,
    model_config: DictConfig,
    config: DictConfig,
    *,
    resuming: bool = False,
) -> pl.LightningModule:
    pretrained = config.model.get("pretrained_checkpoint")
    if stage == "autoencoder":
        return AutoencoderTrainingModule(
            model_config,
            config.training,
            pretrained_checkpoint=None if resuming else pretrained,
        )

    autoencoder_checkpoint = config.model.get("autoencoder_checkpoint")
    allow_random = bool(config.model.get("allow_random_first_stage", False))
    if (
        not resuming
        and not pretrained
        and not autoencoder_checkpoint
        and not allow_random
    ):
        raise ValueError(
            "Diffusion training requires model.pretrained_checkpoint (full model) or "
            "model.autoencoder_checkpoint (stage-1 output). Set "
            "model.allow_random_first_stage=true only for architecture smoke tests."
        )
    optimizer = as_plain_dict(config.training.get("optimizer", {}))
    scheduler = as_plain_dict(config.training.get("scheduler", {}))
    sharpness = as_plain_dict(config.training.get("sharpness", {}))
    validation = as_plain_dict(config.training.get("validation", {}))
    scheduler_name = str(scheduler.get("name", "plateau")).lower()
    patience: Optional[int]
    if scheduler_name in {"constant", "none", "null"}:
        patience = None
    elif scheduler_name == "plateau":
        patience = int(scheduler.get("patience", 10))
    else:
        raise ValueError(
            "Diffusion scheduler must be 'constant' or 'plateau'; "
            f"got {scheduler_name!r}"
        )
    return DiffusionTrainingModule(
        model_config,
        learning_rate=float(optimizer.get("learning_rate", optimizer.get("lr", 1e-4))),
        weight_decay=float(optimizer.get("weight_decay", 0.0)),
        adam_betas=tuple(optimizer.get("betas", (0.9, 0.999))),
        scheduler_factor=float(scheduler.get("factor", 0.5)),
        scheduler_patience=patience,
        scheduler_min_lr=float(scheduler.get("min_lr", 0.0)),
        pretrained_checkpoint=None if resuming else pretrained,
        autoencoder_checkpoint=None if resuming else autoencoder_checkpoint,
        sharpness_config=sharpness,
        validation_num_samples=int(validation.get("num_samples", 4)),
        validation_image_batches=int(validation.get("image_batches", 1)),
        validation_sample_every_n_epochs=int(validation.get("every_n_epochs", 1)),
        validation_sampling_steps=int(validation.get("sampling_steps", 100)),
        validation_sampling_eta=float(validation.get("sampling_eta", 0.95)),
        validation_sampling_temperature=float(validation.get("temperature", 1.0)),
        validation_seed=int(validation.get("seed", int(config.seed))),
        validation_use_ema=bool(validation.get("use_ema", True)),
    )


def _prepare_trainer_kwargs(
    stage: str,
    module: pl.LightningModule,
    trainer_kwargs: dict[str, Any],
) -> None:
    """Reconcile Lightning's automatic settings with the manual GAN loop."""

    if module.automatic_optimization:
        if stage == "autoencoder":
            clip = float(getattr(module, "gradient_clip_val", 0.0))
            if clip > 0 and float(trainer_kwargs.get("gradient_clip_val", 0.0)) == 0:
                trainer_kwargs["gradient_clip_val"] = clip
        return

    trainer_accumulation = trainer_kwargs.get("accumulate_grad_batches", 1)
    if trainer_accumulation != 1:
        raise ValueError(
            "GAN training performs accumulation inside the LightningModule. Set "
            "trainer.accumulate_grad_batches=1 and configure "
            "training.accumulate_grad_batches instead."
        )
    max_steps = trainer_kwargs.get("max_steps", -1)
    if max_steps not in (None, -1):
        raise ValueError(
            "trainer.max_steps is ambiguous with two manual GAN optimizers; use "
            "trainer.max_epochs instead"
        )

    if not _uses_multiple_processes(trainer_kwargs):
        return
    strategy = trainer_kwargs.get("strategy", "auto")
    if strategy in (None, "auto"):
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
        return
    if isinstance(strategy, str) and "find_unused_parameters_true" not in strategy:
        raise ValueError(
            "Multi-process GAN training requires a DDP strategy with "
            "find_unused_parameters=True; use ddp_find_unused_parameters_true"
        )


def _uses_multiple_processes(trainer_kwargs: Mapping[str, Any]) -> bool:
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            return True
    except ValueError:
        pass
    devices = trainer_kwargs.get("devices", "auto")
    if isinstance(devices, int):
        return devices > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    if devices == "auto":
        accelerator = str(trainer_kwargs.get("accelerator", "auto")).lower()
        return accelerator in {"auto", "gpu", "cuda"} and torch.cuda.device_count() > 1
    return False


def _build_logger(config: DictConfig, run_dir: Path) -> Any:
    logger_type = str(config.get("type", "csv")).lower()
    if logger_type in {"none", "false", "disabled"}:
        return False
    save_dir = Path(str(config.get("save_dir", run_dir / "logs"))).expanduser()
    if not save_dir.is_absolute():
        save_dir = run_dir / save_dir
    name = str(config.get("name", "opensr"))
    version = config.get("version")
    common: dict[str, Any] = {"save_dir": str(save_dir), "name": name}
    if version is not None:
        common["version"] = str(version)
    if logger_type == "csv":
        return CSVLogger(**common)
    if logger_type == "tensorboard":
        try:
            from pytorch_lightning.loggers import TensorBoardLogger

            return TensorBoardLogger(**common)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install opensr-model[train] for TensorBoard logging"
            ) from exc
    if logger_type == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger

            return WandbLogger(
                project=str(config.get("project", "opensr-training")),
                name=name,
                save_dir=str(save_dir),
                log_model=bool(config.get("log_model", False)),
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install opensr-model[wandb] for W&B logging") from exc
    raise ValueError("logging.type must be csv, tensorboard, wandb, or none")


def _build_callbacks(
    stage: str,
    config: DictConfig,
    run_dir: Path,
    has_logger: bool,
) -> list[pl.Callback]:
    checkpoint = config.checkpoint
    resume_dir = _under_run_dir(
        run_dir, checkpoint.get("resume_dir", "checkpoints/resume")
    )
    resume_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[pl.Callback] = []
    if bool(config.trainer.get("enable_checkpointing", True)):
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(resume_dir),
                filename="epoch={epoch:04d}-step={step}",
                monitor=str(checkpoint.get("monitor", "val/loss")),
                mode=str(checkpoint.get("mode", "min")),
                save_top_k=int(checkpoint.get("save_top_k", 1)),
                save_last=bool(checkpoint.get("save_last", True)),
                every_n_epochs=int(checkpoint.get("every_n_epochs", 1)),
                save_weights_only=False,
                auto_insert_metric_name=False,
            )
        )
    if has_logger:
        callbacks.append(LoggingRateMonitorSafe(logging_interval="step"))

    images = config.logging.get("images", {})
    if bool(images.get("enabled", True)):
        callbacks.append(
            ReconstructionImageLogger(
                max_images=int(images.get("max_images", 4)),
                every_n_epochs=int(images.get("every_n_epochs", 1)),
                output_dir=images.get("output_dir"),
                rgb_bands=tuple(images.get("rgb_bands", (0, 1, 2))),
                nir_band=images.get("nir_band", 3),
                value_min=float(images.get("value_min", 0.0)),
                value_max=float(images.get("value_max", 0.3)),
            )
        )

    native = checkpoint.get("native", {})
    if bool(native.get("enabled", True)):
        native_dir = _under_run_dir(run_dir, native.get("dir", "checkpoints/native"))
        base_checkpoint = native.get("autoencoder_base_checkpoint")
        if stage == "autoencoder" and str(base_checkpoint).lower() == "auto":
            candidate = config.model.get("pretrained_checkpoint")
            base_checkpoint = (
                candidate
                if candidate and _checkpoint_has_full_diffusion_state(candidate)
                else None
            )
        callbacks.append(
            NativeCheckpointCallback(
                dirpath=str(native_dir),
                monitor=str(checkpoint.get("monitor", "val/loss")),
                mode=str(checkpoint.get("mode", "min")),
                save_last=bool(native.get("save_last", True)),
                save_best=bool(native.get("save_best", True)),
                every_n_epochs=int(native.get("every_n_epochs", 1)),
                autoencoder_base_checkpoint=(
                    str(base_checkpoint)
                    if stage == "autoencoder" and base_checkpoint
                    else None
                ),
                weight_source=str(
                    native.get(
                        "weight_source", "ema" if stage == "diffusion" else "raw"
                    )
                ),
            )
        )

    early = checkpoint.get("early_stopping", {})
    if bool(early.get("enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=str(checkpoint.get("monitor", "val/loss")),
                mode=str(checkpoint.get("mode", "min")),
                patience=int(early.get("patience", 20)),
                min_delta=float(early.get("min_delta", 0.0)),
            )
        )
    return callbacks


class LoggingRateMonitorSafe(LearningRateMonitor):
    """Named subclass so old resume metadata cannot collide with callbacks."""


def _under_run_dir(run_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else run_dir / path


def _validate_resume_checkpoint(path: str | Path) -> None:
    payload = load_checkpoint_payload(path)
    optimizer_states = payload.get("optimizer_states")
    loops = payload.get("loops")
    if (
        not isinstance(optimizer_states, list)
        or not optimizer_states
        or not all(isinstance(item, Mapping) for item in optimizer_states)
        or not isinstance(loops, Mapping)
    ):
        raise ValueError(
            f"{path} is a native/legacy weights checkpoint, not a complete Lightning "
            "resume checkpoint. Use model.pretrained_checkpoint instead."
        )


def _checkpoint_has_full_diffusion_state(path: str | Path) -> bool:
    try:
        keys = tuple(load_checkpoint_payload(path)["state_dict"])
    except Exception:
        return False

    def contains(fragment: str) -> bool:
        return any(key == fragment or key.endswith(f".{fragment}") for key in keys)

    return (
        contains("betas")
        and any("model.diffusion_model." in key for key in keys)
        and any("model_ema." in key for key in keys)
        and any("first_stage_model.encoder." in key for key in keys)
        and any("first_stage_model.decoder." in key for key in keys)
    )


def _parser(stage: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train OpenSR's autoencoder and diffusion stages independently."
    )
    if stage is None:
        parser.add_argument("stage", choices=("autoencoder", "diffusion"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dot-list override; repeat for multiple values.",
    )
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def _run_from_args(args: argparse.Namespace, stage: str) -> None:
    config_path = args.config or package_config_path(f"train_{stage}.yaml")
    overrides = list(args.overrides)
    if args.pretrained is not None:
        overrides.append(
            f"model.pretrained_checkpoint={args.pretrained.expanduser().resolve()}"
        )
    if args.resume is not None:
        overrides.append(f"resume_checkpoint={args.resume.expanduser().resolve()}")
    config = load_training_config(
        config_path,
        overrides=overrides,
        expected_stage=stage,
    )
    if args.print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    run_training(config, validate_only=bool(args.validate_only))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    _run_from_args(args, args.stage)


def autoencoder_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser("autoencoder").parse_args(argv)
    _run_from_args(args, "autoencoder")


def diffusion_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser("diffusion").parse_args(argv)
    _run_from_args(args, "diffusion")


if __name__ == "__main__":
    main()
