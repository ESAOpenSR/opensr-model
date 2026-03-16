from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from deployment.opensr_hpc.config import StagingConfig
from deployment.opensr_hpc.raster import ensure_proj_env, parse_epsg, scale_to_uint16


LOGGER = logging.getLogger("opensr-hpc")


class SkipTileError(RuntimeError):
    def __init__(self, reason: str, *, details: dict[str, int] | None = None) -> None:
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


def _cube_validity_stats(cube) -> dict[str, int]:
    data = np.asarray(cube.data)
    if data.size == 0:
        return {"total_pixels": 0, "valid_pixels": 0, "nonzero_pixels": 0}

    total_pixels = int(data.shape[-2] * data.shape[-1])
    finite_mask = np.isfinite(data)
    valid_pixels = int(finite_mask.any(axis=0).sum())
    nonzero_pixels = int((finite_mask & (data != 0)).any(axis=0).sum())
    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nonzero_pixels": nonzero_pixels,
    }


def ensure_cube_has_valid_data(cube) -> dict[str, int]:
    stats = _cube_validity_stats(cube)
    if stats["total_pixels"] == 0:
        raise SkipTileError("empty_cube", details=stats)
    if stats["valid_pixels"] == 0:
        raise SkipTileError("all_nan_cube", details=stats)
    if stats["nonzero_pixels"] == 0:
        raise SkipTileError("all_zero_cube", details=stats)
    return stats


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if status_code == 429 or response_status == 429:
        return True

    message = str(exc).lower()
    markers = ["429", "too many requests", "rate limit", "rate-limit"]
    return any(marker in message for marker in markers)


def create_cube_with_retry(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    config: StagingConfig,
):
    try:
        import cubo
        import rioxarray  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opensr-hpc staging requires optional dependencies. Install with `pip install \"opensr-model[hpc]\"`."
        ) from exc

    for attempt, delay in enumerate(config.rate_limit_retry_delays_seconds, start=1):
        try:
            return cubo.create(
                lat=latitude,
                lon=longitude,
                collection=config.collection,
                bands=config.bands,
                start_date=start_date,
                end_date=end_date,
                edge_size=config.edge_size,
                resolution=config.resolution,
            )
        except Exception as exc:  # pragma: no cover
            if not (config.retry_on_rate_limit and is_rate_limit_error(exc)):
                raise
            LOGGER.warning(
                "Rate limit detected during cubo staging for lat=%s lon=%s. Retrying in %s seconds (%s/%s).",
                latitude,
                longitude,
                delay,
                attempt,
                len(config.rate_limit_retry_delays_seconds),
            )
            time.sleep(delay)

    return cubo.create(
        lat=latitude,
        lon=longitude,
        collection=config.collection,
        bands=config.bands,
        start_date=start_date,
        end_date=end_date,
        edge_size=config.edge_size,
        resolution=config.resolution,
    )


def stage_cutout(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    config: StagingConfig,
    output_path: Path,
) -> Path:
    ensure_proj_env()
    LOGGER.info(
        "staging cubo cutout lat=%s lon=%s start_date=%s end_date=%s output=%s",
        latitude,
        longitude,
        start_date,
        end_date,
        output_path,
    )
    cube = create_cube_with_retry(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    if "time" in cube.dims:
        cube = cube.isel(time=config.image_index)
    cube = cube.transpose("band", "y", "x")
    stats = ensure_cube_has_valid_data(cube)
    LOGGER.info("validated staged cutout lat=%s lon=%s stats=%s", latitude, longitude, stats)

    epsg_text = str(cube.attrs.get("epsg", "") or cube.coords.get("epsg", ""))
    epsg_code = parse_epsg(epsg_text, latitude, longitude)
    cube = cube.rio.write_crs(epsg_code, inplace=False)

    if config.output_dtype == "uint16":
        cube = cube.copy(data=scale_to_uint16(cube.data))
    cube = cube.rio.write_nodata(config.nodata, encoded=True, inplace=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cube.rio.to_raster(
        output_path,
        compress=config.compression,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    )
    LOGGER.info("wrote staged cutout lat=%s lon=%s output=%s", latitude, longitude, output_path)
    return output_path.resolve()
