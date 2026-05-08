from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RATE_LIMIT_RETRY_DELAYS_SECONDS = [15, 30, 60, 120, 120, 120]


@dataclass(slots=True)
class EnvironmentConfig:
    python_executable: str = "python"
    modules: list[str] = field(default_factory=list)
    conda_env: str | None = None


@dataclass(slots=True)
class ModelConfig:
    config_path: str | None = None
    checkpoint_path: str | None = None


@dataclass(slots=True)
class AoiConfig:
    path: str | None = None
    layer: str | None = None


@dataclass(slots=True)
class StagingConfig:
    collection: str = "sentinel-2-l2a"
    bands: list[str] = field(default_factory=lambda: ["B04", "B03", "B02", "B08"])
    image_index: int = 0
    edge_size: int = 4096
    resolution: int = 10
    nodata: int = 0
    output_dtype: str = "uint16"
    compression: str = "deflate"
    overlap_meters: float = 128.0
    retry_on_rate_limit: bool = True
    rate_limit_retry_delays_seconds: list[int] = field(
        default_factory=lambda: list(DEFAULT_RATE_LIMIT_RETRY_DELAYS_SECONDS)
    )


@dataclass(slots=True)
class InferenceConfig:
    factor: int = 4
    window_size: tuple[int, int] = (128, 128)
    batch_size: int = 2
    overlap: int = 12
    eliminate_border_px: int = 2
    gpus: int | list[int] = 0
    save_preview: bool = False


@dataclass(slots=True)
class SlurmConfig:
    partition: str | None = None
    gpu_type: str | None = None
    gpus: int = 1
    gres: str | None = None
    cpus_per_task: int = 8
    mem_gb: int = 128
    time: str = "01:00:00"
    account: str | None = None
    qos: str | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeConfig:
    project_name: str = "opensr"
    output_root: Path = Path("runs")
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aoi: AoiConfig = field(default_factory=AoiConfig)
    staging: StagingConfig = field(default_factory=StagingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    config_path: Path | None = None


def bundled_model_config() -> Path:
    import opensr_model

    return Path(opensr_model.__file__).resolve().parent / "configs" / "config_10m.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _runtime_from_mapping(data: dict[str, Any]) -> RuntimeConfig:
    environment = EnvironmentConfig(**data.get("environment", {}))
    model = ModelConfig(**data.get("model", {}))
    aoi = AoiConfig(**data.get("aoi", {}))
    staging = StagingConfig(**data.get("staging", {}))
    inference_data = dict(data.get("inference", {}))
    if "window_size" in inference_data:
        inference_data["window_size"] = tuple(inference_data["window_size"])
    inference = InferenceConfig(**inference_data)
    slurm = SlurmConfig(**data.get("slurm", {}))
    output_root = Path(data.get("output_root", "runs"))
    return RuntimeConfig(
        project_name=data.get("project_name", "opensr"),
        output_root=output_root,
        environment=environment,
        model=model,
        aoi=aoi,
        staging=staging,
        inference=inference,
        slurm=slurm,
    )


def resolve_model_config_path(config: RuntimeConfig) -> Path:
    if config.model.config_path is None:
        return bundled_model_config().resolve()
    return Path(config.model.config_path).expanduser().resolve()


def load_runtime_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> RuntimeConfig:
    path = Path(config_path).expanduser().resolve()
    data = _read_yaml(path)
    if overrides:
        data = _merge(data, overrides)
    config = _runtime_from_mapping(data)
    config.config_path = path
    base_dir = path.parent

    if not config.output_root.is_absolute():
        config.output_root = (base_dir / config.output_root).resolve()

    if config.model.config_path is not None:
        model_config_path = Path(config.model.config_path).expanduser()
        if not model_config_path.is_absolute():
            model_config_path = (base_dir / model_config_path).resolve()
        config.model.config_path = str(model_config_path)

    if config.model.checkpoint_path is not None:
        checkpoint_path = Path(config.model.checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (base_dir / checkpoint_path).resolve()
        config.model.checkpoint_path = str(checkpoint_path)

    if config.aoi.path is not None:
        aoi_path = Path(config.aoi.path).expanduser()
        if not aoi_path.is_absolute():
            aoi_path = (base_dir / aoi_path).resolve()
        config.aoi.path = str(aoi_path)

    validate_runtime_config(config)
    return config


def validate_runtime_config(config: RuntimeConfig) -> None:
    if config.staging.edge_size <= 0:
        raise ValueError("staging.edge_size must be positive")
    if config.staging.resolution <= 0:
        raise ValueError("staging.resolution must be positive")
    if config.staging.overlap_meters < 0:
        raise ValueError("staging.overlap_meters must be non-negative")
    if any(delay <= 0 for delay in config.staging.rate_limit_retry_delays_seconds):
        raise ValueError("staging.rate_limit_retry_delays_seconds must contain positive integers")
    if config.inference.factor <= 0:
        raise ValueError("inference.factor must be positive")
    if len(config.inference.window_size) != 2 or min(config.inference.window_size) <= 0:
        raise ValueError("inference.window_size must contain two positive integers")
    if config.inference.batch_size <= 0:
        raise ValueError("inference.batch_size must be positive")
    if config.slurm.gpus < 0:
        raise ValueError("slurm.gpus must be non-negative")
    if config.slurm.mem_gb <= 0:
        raise ValueError("slurm.mem_gb must be positive")
    if config.slurm.cpus_per_task <= 0:
        raise ValueError("slurm.cpus_per_task must be positive")
    if config.aoi.path is not None and not Path(config.aoi.path).exists():
        raise FileNotFoundError(f"AOI path not found: {config.aoi.path}")

    model_config_path = resolve_model_config_path(config)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    if config.model.checkpoint_path is not None:
        checkpoint_path = Path(config.model.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def runtime_config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_root"] = str(config.output_root)
    data["config_path"] = str(config.config_path) if config.config_path else None
    data["model"]["config_path"] = str(resolve_model_config_path(config))
    if config.model.checkpoint_path is not None:
        data["model"]["checkpoint_path"] = str(Path(config.model.checkpoint_path).expanduser().resolve())
    data["inference"]["window_size"] = list(config.inference.window_size)
    return data
