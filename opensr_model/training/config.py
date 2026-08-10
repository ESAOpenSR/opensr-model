"""Configuration loading and validation for both training entrypoints."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

_STAGES = {"autoencoder", "diffusion"}


def package_config_path(name: str) -> Path:
    candidate = files("opensr_model").joinpath("configs", name)
    return Path(str(candidate))


def load_training_config(
    path: str | Path,
    *,
    overrides: Optional[Iterable[str]] = None,
    expected_stage: Optional[str] = None,
) -> DictConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    package_directory = package_config_path(path.name).parent.resolve()
    # The built-in templates are commonly invoked directly with --set. In that
    # case relative user paths belong to the launch directory, not site-packages.
    base_directory = Path.cwd() if path.parent == package_directory else path.parent
    _resolve_known_paths(config, base_directory)
    validate_training_config(config, expected_stage=expected_stage)
    return config


def load_model_config(
    reference: str | Path | DictConfig | dict[str, Any],
) -> DictConfig:
    if OmegaConf.is_config(reference):
        return reference
    if isinstance(reference, dict):
        return OmegaConf.create(reference)

    path = Path(str(reference)).expanduser()
    if not path.is_file() and path.parent == Path("."):
        package_candidate = package_config_path(path.name)
        if package_candidate.is_file():
            path = package_candidate
    if not path.is_file():
        raise FileNotFoundError(
            f"Model architecture configuration not found: {reference}"
        )
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    required = ("first_stage_config", "cond_stage_config", "denoiser_settings", "other")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Model configuration is missing: {', '.join(missing)}")
    return config


def validate_training_config(
    config: DictConfig,
    *,
    expected_stage: Optional[str] = None,
) -> None:
    stage = str(config.get("stage", ""))
    if stage not in _STAGES:
        raise ValueError(f"stage must be one of {sorted(_STAGES)}, got {stage!r}")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(
            f"Configuration stage is {stage!r}, but the {expected_stage!r} entrypoint was used"
        )
    for section in ("model", "data", "training", "trainer", "checkpoint", "logging"):
        if section not in config:
            raise ValueError(f"Training configuration is missing section {section!r}")
    if not config.model.get("architecture_config"):
        raise ValueError("model.architecture_config is required")
    if int(config.get("seed", 0)) < 0:
        raise ValueError("seed must be non-negative")
    _validate_taco_data_config(config.data)


def save_resolved_config(config: DictConfig, destination: str | Path) -> Path:
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, destination, resolve=True)
    return destination


def as_plain_dict(config: Any) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        result = OmegaConf.to_container(config, resolve=True)
        if not isinstance(result, dict):
            raise TypeError("Expected a mapping configuration")
        return result
    return dict(config)


def _resolve_known_paths(config: DictConfig, base_dir: Path) -> None:
    path_fields = (
        "run_dir",
        "resume_checkpoint",
        "model.architecture_config",
        "model.pretrained_checkpoint",
        "model.autoencoder_checkpoint",
        "checkpoint.native.autoencoder_base_checkpoint",
        "data.synthetic_hr_nir_dir",
    )
    for dotted in path_fields:
        value = OmegaConf.select(config, dotted)
        if value in (None, ""):
            continue
        if (
            dotted == "checkpoint.native.autoencoder_base_checkpoint"
            and str(value).lower() == "auto"
        ):
            continue
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            continue
        candidate = (base_dir / path).resolve()
        # A bare packaged architecture name is deliberately left unresolved so
        # load_model_config can find it through importlib.resources.
        if dotted == "model.architecture_config" and not candidate.exists():
            continue
        OmegaConf.update(config, dotted, str(candidate), merge=False)

    taco_path = OmegaConf.select(config, "data.taco_path")
    if taco_path not in (None, ""):
        values = (
            list(taco_path)
            if isinstance(taco_path, (list, tuple, ListConfig))
            else [taco_path]
        )
        resolved = []
        for value in values:
            path = Path(str(value)).expanduser()
            resolved.append(
                str(path if path.is_absolute() else (base_dir / path).resolve())
            )
        OmegaConf.update(
            config,
            "data.taco_path",
            (
                resolved
                if isinstance(taco_path, (list, tuple, ListConfig))
                else resolved[0]
            ),
            merge=False,
        )


def _validate_taco_data_config(data: DictConfig) -> None:
    """Reject legacy file sources and validate the TACO-only data contract."""

    taco_path = data.get("taco_path")
    if taco_path in (None, "") or (
        isinstance(taco_path, (list, tuple, ListConfig)) and not taco_path
    ):
        raise ValueError("data.taco_path is required; training accepts TACO data only")

    legacy_fields = {
        "train",
        "val",
        "hr_load",
        "lr_load",
        "degradation",
        "manifest",
        "hr_root",
        "lr_root",
        "synthetic_samples",
        "synthesize_missing_lr",
    }
    present = sorted(field for field in legacy_fields if field in data)
    if present:
        raise ValueError(
            "Legacy data sources are not supported; remove data fields: "
            + ", ".join(present)
        )

    assets = ("lr_asset_id", "hr_asset_id", "raw_lr_asset_id")
    missing_assets = [field for field in assets if not str(data.get(field, "")).strip()]
    if missing_assets:
        raise ValueError(f"Missing TACO asset identifiers: {', '.join(missing_assets)}")

    rgb_bands = list(data.get("rgb_bands", []))
    if len(rgb_bands) != 3 or len(set(map(int, rgb_bands))) != 3:
        raise ValueError("data.rgb_bands must contain three distinct one-based bands")
    if any(int(band) < 1 for band in rgb_bands) or int(data.get("nir_band", 0)) < 1:
        raise ValueError(
            "data.rgb_bands and data.nir_band are one-based positive indices"
        )

    if float(data.get("reflectance_scale", 0.0)) <= 0.0:
        raise ValueError("data.reflectance_scale must be positive")
    if str(data.get("hr_nir_strategy", "")) != "synthetic_sidecar":
        raise ValueError(
            "data.hr_nir_strategy must be 'synthetic_sidecar'; interpolated LR NIR "
            "is not an allowed HR target"
        )
    if not str(data.get("synthetic_hr_nir_dir", "")).strip():
        raise ValueError("data.synthetic_hr_nir_dir is required")
    val_fraction = float(data.get("val_fraction", 0.0))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("data.val_fraction must be strictly between 0 and 1")
    factor = int(data.get("factor", 0))
    if factor != 4:
        raise ValueError(
            "data.factor must be 4 for the current OpenSR checkpoint contract"
        )
    patch_size = data.get("train_patch_size")
    if patch_size is not None and (
        int(patch_size) <= 0 or int(patch_size) % factor != 0
    ):
        raise ValueError(
            "data.train_patch_size must be positive and divisible by data.factor"
        )
    value_range = list(data.get("value_range", []))
    if len(value_range) != 2 or float(value_range[0]) >= float(value_range[1]):
        raise ValueError(
            "data.value_range must contain an increasing [minimum, maximum]"
        )


__all__ = [
    "as_plain_dict",
    "load_model_config",
    "load_training_config",
    "package_config_path",
    "save_resolved_config",
    "validate_training_config",
]
