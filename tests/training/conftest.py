from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import tacoreader.v1
from rasterio.transform import from_origin


class _FakeILoc:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


class _FakeTacoTable:
    def __init__(self, rows: list[dict[str, Any]], reads: list[Any]) -> None:
        self._rows = rows
        self._reads = reads
        self.iloc = _FakeILoc(rows)
        self.columns = tuple(dict.fromkeys(key for row in rows for key in row))

    def __len__(self) -> int:
        return len(self._rows)

    def read(self, index: int) -> Any:
        return self._reads[index]


@dataclass
class FakeTacoBundle:
    root: Path
    taco_paths: tuple[Path, ...]
    raster_paths: dict[str, Path]
    load_calls: list[Path]


@pytest.fixture
def fake_taco_factory(tmp_path, monkeypatch):
    """Create fake TACO metadata whose assets are genuine tiny GeoTIFFs.

    The fake only replaces ``tacoreader.v1.load``. Production imports and
    raster I/O remain untouched, so tests exercise band, nodata, transform,
    normalization and mask behavior through rasterio exactly as real data do.
    """

    catalogs: dict[Path, _FakeTacoTable] = {}
    load_calls: list[Path] = []
    bundle_number = 0

    def fake_load(path: str | Path) -> _FakeTacoTable:
        resolved = Path(path).resolve()
        load_calls.append(resolved)
        try:
            return catalogs[resolved]
        except KeyError as exc:
            raise FileNotFoundError(f"No fake TACO catalog for {resolved}") from exc

    monkeypatch.setattr(tacoreader.v1, "load", fake_load)

    def make_bundle(
        *,
        sample_ids: list[str] | None = None,
        id_gees: list[str | None] | None = None,
        parts: int = 1,
        hr_size: int = 32,
        include_nodata: bool = True,
        include_outliers: bool = False,
        schema: str = "worldwide",
        child_order: tuple[str, ...] | None = None,
    ) -> FakeTacoBundle:
        nonlocal bundle_number
        bundle_number += 1
        root = tmp_path / f"taco_{bundle_number}"
        root.mkdir()
        if hr_size <= 0 or hr_size % 4:
            raise ValueError("hr_size must be positive and divisible by four")
        lr_size = hr_size // 4
        if sample_ids is None:
            sample_ids = [f"tile_{index:03d}_bing" for index in range(12)]
        if id_gees is None:
            id_gees = [f"scene-{index // 2:03d}" for index in range(len(sample_ids))]
        if len(id_gees) != len(sample_ids):
            raise ValueError("id_gees and sample_ids must have equal lengths")
        if parts <= 0:
            raise ValueError("parts must be positive")
        if schema not in {"worldwide", "crosssensor"}:
            raise ValueError("schema must be 'worldwide' or 'crosssensor'")
        if child_order is None:
            child_order = (
                ("hrharm", "lr", "hr", "lrharm")
                if schema == "worldwide"
                else ("hr", "lr")
            )

        lr_transform = from_origin(100.0, 200.0, 10.0, 10.0)
        hr_transform = from_origin(100.0, 200.0, 2.5, 2.5)
        raw_lr = np.full((12, lr_size, lr_size), 500, dtype=np.uint16)
        raw_lr[7] = 4000
        lrharm = np.stack(
            [
                np.full((lr_size, lr_size), 1000, dtype=np.uint16),
                np.full((lr_size, lr_size), 2000, dtype=np.uint16),
                np.full((lr_size, lr_size), 3000, dtype=np.uint16),
            ]
        )
        hrharm = np.stack(
            [
                np.full((hr_size, hr_size), 1100, dtype=np.uint16),
                np.full((hr_size, hr_size), 2100, dtype=np.uint16),
                np.full((hr_size, hr_size), 3100, dtype=np.uint16),
            ]
        )
        raw_hr = np.full((3, hr_size, hr_size), 100, dtype=np.uint16)
        # Zero is an ordinary valid reflectance and must survive mask creation.
        raw_lr[7, -1, -1] = 0
        if include_nodata:
            raw_lr[7, 0, 0] = 65535
            lrharm[:, 0, 0] = 65535
            hrharm[:, 4, 4] = 65535
        if include_outliers:
            raw_lr[7, 1, 1] = 12000
            hrharm[0, hr_size // 2, hr_size // 2] = 15000

        raster_paths: dict[str, Path] = {}

        def write_asset(asset_id: str, values: np.ndarray, transform) -> None:
            path = root / f"{asset_id}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=values.shape[1],
                width=values.shape[2],
                count=values.shape[0],
                dtype="uint16",
                crs="EPSG:32632",
                transform=transform,
                nodata=65535,
            ) as destination:
                destination.write(values)
            raster_paths[asset_id] = path

        if schema == "worldwide":
            write_asset("lr", raw_lr, lr_transform)
            write_asset("lrharm", lrharm, lr_transform)
            write_asset("hr", raw_hr, hr_transform)
            write_asset("hrharm", hrharm, hr_transform)
        else:
            crosssensor_lr = np.concatenate((lrharm, raw_lr[7:8]), axis=0)
            crosssensor_hr = np.concatenate(
                (
                    hrharm,
                    np.full((1, hr_size, hr_size), 4100, dtype=np.uint16),
                ),
                axis=0,
            )
            if include_nodata:
                crosssensor_hr[:, 4, 4] = 65535
            if include_outliers:
                crosssensor_hr[3, hr_size // 2, hr_size // 2] = 15000
            write_asset("lr", crosssensor_lr, lr_transform)
            write_asset("hr", crosssensor_hr, hr_transform)

        children = _FakeTacoTable(
            [
                {"tortilla:id": asset_id, "tortilla:file_format": "GTiff"}
                for asset_id in child_order
            ],
            [str(raster_paths[asset_id]) for asset_id in child_order],
        )
        taco_paths: list[Path] = []
        for part_index in range(parts):
            suffix = ".part.tortilla" if schema == "worldwide" else ".part.taco"
            path = root / f"dataset.{part_index:04d}{suffix}"
            footer = b"x"
            path.write_bytes(
                b"#y"
                + (18).to_bytes(8, "little")
                + len(footer).to_bytes(8, "little")
                + footer
            )
            part_rows: list[dict[str, Any]] = []
            part_reads: list[Any] = []
            for index in range(part_index, len(sample_ids), parts):
                product = sample_ids[index].rsplit("_", 1)[-1]
                part_rows.append(
                    {
                        "tortilla:id": sample_ids[index],
                        "tortilla:file_format": "TORTILLA",
                        "id_gee": id_gees[index],
                        "hr_product": product,
                    }
                )
                part_reads.append(children)
            catalogs[path.resolve()] = _FakeTacoTable(part_rows, part_reads)
            taco_paths.append(path)
        return FakeTacoBundle(root, tuple(taco_paths), raster_paths, load_calls)

    return make_bundle


@pytest.fixture
def tiny_model_config() -> dict:
    first_stage = {
        "embed_dim": 2,
        "double_z": True,
        "z_channels": 2,
        "resolution": 16,
        "in_channels": 4,
        "out_ch": 4,
        "ch": 32,
        "ch_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    unet = {
        "image_size": 8,
        "in_channels": 4,
        "model_channels": 32,
        "out_channels": 2,
        "num_res_blocks": 1,
        "attention_resolutions": [],
        "channel_mult": [1, 2],
        "num_head_channels": 16,
    }
    return {
        "apply_normalization": False,
        "encode_conditioning": True,
        "first_stage_config": first_stage,
        "cond_stage_config": unet,
        "denoiser_settings": {
            "timesteps": 10,
            "linear_start": 1e-4,
            "linear_end": 1e-2,
            "parameterization": "eps",
        },
        "other": {
            "concat_mode": True,
            "cond_stage_trainable": False,
            "first_stage_key": "image",
            "cond_stage_key": "LR_image",
        },
    }


@pytest.fixture
def tiny_autoencoder_training_config() -> dict:
    return {
        "loss": {
            "kl_weight": 1e-6,
            "logvar_init": 0.0,
            "perceptual_weight": 0.0,
            "discriminator_factor": 0.0,
            "discriminator_weight": 0.5,
            "discriminator_start": 0,
        },
        "optimizer": {"learning_rate": 1e-4, "betas": [0.5, 0.9]},
        "scheduler": {"name": "constant"},
        "metrics": {"value_min": 0.0, "value_max": 1.0},
    }
