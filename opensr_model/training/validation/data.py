"""Portable, frozen data preparation for cross-sensor transfer validation."""
from __future__ import annotations

import collections
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..runtime.config import as_plain_dict, package_config_path
from ..data.taco import (
    InvalidPairError, OpenSRDataModule, TacoPairedDataset, _row, _row_get,
    _tacoreader_v1, scan_taco_catalog, split_taco_records,
)

BANDS = (4, 3, 2, 8)
PROTOCOL_VERSION = 1
DATASET_KEYS = {
    "factor", "lr_asset_id", "hr_asset_id", "raw_lr_asset_id", "rgb_bands",
    "nir_band", "hr_nir_strategy", "synthetic_hr_nir_dir", "reflectance_scale",
    "nodata", "value_range", "strict_full_validity",
}


def digest_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def preparation_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def select_diverse(rows: Sequence[dict], count: int, seed: int = 42) -> list[dict]:
    """Include developed areas, cover sites, then round-robin site/cover strata."""
    if not 0 < count <= len(rows):
        raise ValueError(f"subset_size must be in [1, {len(rows)}], got {count}")
    key = lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()
    ordered = sorted(rows, key=key)
    selected: list[dict] = []
    ids: set[str] = set()

    def add(row):
        if row["id"] not in ids and len(selected) < count:
            selected.append(row)
            ids.add(row["id"])

    for row in ordered:
        if row["landcover"] == "Developed":
            add(row)
    sites = {r["site"] for r in selected}
    for row in ordered:
        if row["site"] not in sites:
            add(row)
            sites.add(row["site"])
    groups = collections.defaultdict(list)
    for row in ordered:
        if row["id"] not in ids:
            groups[(row["site"], row["landcover"])].append(row)
    while len(selected) < count:
        for group in sorted(groups):
            if groups[group]:
                add(groups[group].pop(0))
    return selected


def strict_pixels(source, bands: Sequence[int]) -> np.ndarray:
    values = source.read(bands)
    valid = (source.read_masks(bands) > 0) & np.isfinite(values) & (values > 0)
    valid &= values != 65535
    if source.nodata is not None:
        valid &= values != source.nodata
    if not valid.all():
        raise InvalidPairError(f"Invalid or zero pixel in full raster {source.name}")
    return values


def read_neon_pair(root: Path, row: Mapping) -> dict[str, Any]:
    if not row["lr"].startswith("s2_l2a_10m/") or not row["hr"].startswith("neon_2.5m_linearized/"):
        raise ValueError("SEN2NEON requires corrected observed LR and canonical 2.5 m HR")
    with rasterio.open(root / row["lr"]) as lr, rasterio.open(root / row["hr"]) as hr:
        if not (
            lr.count == hr.count == 12 and lr.shape == (256, 256)
            and hr.shape == (1024, 1024) and lr.crs == hr.crs and lr.crs
            and (lr.transform * Affine.scale(.25)).almost_equals(hr.transform)
            and lr.nodata == 65535 and hr.nodata == 0
            and set(lr.dtypes + hr.dtypes) == {"uint16"}
        ):
            raise ValueError(f"Invalid canonical SEN2NEON grid/bands/dtype: {row['id']}")
        a, b = strict_pixels(lr, BANDS), strict_pixels(hr, BANDS)
        bounds = transform_bounds(lr.crs, "EPSG:4326", *lr.bounds)
    a, b = torch.from_numpy(a.astype(np.float32) * 1e-4), torch.from_numpy(b.astype(np.float32) * 1e-4)
    mask = torch.ones((1, 1024, 1024), dtype=torch.bool)
    mask[:, :5] = False
    mask[:, -5:] = False
    mask[:, :, :5] = False
    mask[:, :, -5:] = False
    return {
        "LR_image": a.clamp(0, 1), "image": b.clamp(0, 1), "valid_mask": mask,
        "sample_id": row["id"], "bounds_wgs84": bounds,
        "clipped_fraction": float(((a > 1).sum() + (b > 1).sum()) / (a.numel() + b.numel())),
    }


class SEN2NEONDataset(Dataset):
    def __init__(self, root: str | Path, rows: Sequence[dict]):
        self.root, self.rows = Path(root), list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return read_neon_pair(self.root, self.rows[index])


def resolve_sen2neon(options: Mapping) -> tuple[Path | None, list[dict], Path | None]:
    enabled = options.get("enabled", "auto")
    if enabled not in (True, False, "auto"):
        raise ValueError("sen2neon.enabled must be true, false, or auto")
    if enabled is False:
        return None, [], None
    root = Path(options.get("root") or os.environ.get("SEN2NEON_ROOT") or "data/SEN2NEON").expanduser().resolve()
    if not root.exists():
        if enabled is True:
            raise FileNotFoundError(f"Required SEN2NEON root does not exist: {root}")
        return None, [], None
    manifest = Path(options.get("manifest") or package_config_path("sen2neon_clean192.json"))
    rows = json.loads(manifest.read_text())
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("SEN2NEON manifest must contain unique, nonempty sample IDs")
    metadata = root / "metadata.jsonl"
    catalog = {r["id"]: r for r in map(json.loads, metadata.read_text().splitlines())}
    expected_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    for row in rows:
        source = catalog.get(row["id"], {})
        if source.get("bands") != expected_bands or any(source.get(k) != row[k] for k in ("lr", "hr")):
            raise ValueError(f"SEN2NEON metadata mismatch: {row['id']}")
        for kind in ("lr", "hr"):
            path = root / row[kind]
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete SEN2NEON dataset: {path}")
            if not row.get(kind + "_sha256"):
                raise ValueError(f"Missing {kind} checksum for {row['id']}")
    return root, rows, manifest


def load_raw_manifest(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("raw_s2 manifest must be a list with unique IDs")
    for row in rows:
        p = Path(row["path"]).expanduser()
        row["path"] = str(p if p.is_absolute() else (path.parent / p).resolve())
        if len(row.get("bands", [])) != 4 or len(set(row["bands"])) != 4:
            raise ValueError("Raw S2 previews require four explicit one-based RGBN bands")
        if not np.isfinite(row.get("scale", 0)) or row["scale"] <= 0:
            raise ValueError("Raw S2 previews require a positive finite reflectance scale")
        # Validate the exact pixels now, before any optimizer updates.
        read_raw(row)
    return rows


def read_raw(row: Mapping) -> torch.Tensor:
    with rasterio.open(row["path"]) as source:
        values = strict_pixels(source, row["bands"])
    return torch.from_numpy(values.astype(np.float32) * row["scale"]).clamp(0, 1)


def pad_lr(lr: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preserve every source pixel; the denoiser needs LR dimensions divisible by eight."""
    h, w = lr.shape[-2:]
    ph, pw = max(128, (h + 7) // 8 * 8) - h, max(128, (w + 7) // 8 * 8) - w
    mode = "reflect" if ph < h and pw < w else "replicate"
    return F.pad(lr, (0, pw, 0, ph), mode=mode), (h * 4, w * 4)


def overlapping_groups(records, bounds: Sequence[Sequence[float]]) -> set[str]:
    if not bounds:
        return set()
    boxes = np.asarray(bounds)
    tables = {p: _tacoreader_v1().load(str(p)) for p in {r.taco_path for r in records}}
    excluded = set()
    for record in records:
        row = _row(tables[record.taco_path], record.local_index)
        transform = _row_get(row, "stac:geotransform")
        shape = _row_get(row, "stac:raster_shape")
        crs = _row_get(row, "stac:crs")
        if transform is None or shape is None or crs is None:
            raise ValueError(f"Cannot audit training geography: {record.sample_id}")
        h, w = map(int, shape)
        b = rasterio.transform.array_bounds(h, w, Affine.from_gdal(*transform))
        b = transform_bounds(crs, "EPSG:4326", *b)
        intersects = (boxes[:, 0] <= b[2]) & (boxes[:, 2] >= b[0]) & (boxes[:, 1] <= b[3]) & (boxes[:, 3] >= b[1])
        if intersects.any():
            excluded.add(record.split_group)
    return excluded


def prepare_transfer(config, run_dir: Path) -> dict[str, Any]:
    """Locked/cached audit. A changed input requires a new run directory."""
    validation = as_plain_dict(config.training.validation)
    options = validation.get("sen2neon", {})
    root, rows, manifest = resolve_sen2neon(options)
    raw = load_raw_manifest(validation.get("raw_s2", {}).get("manifest"))
    data_args = as_plain_dict(config.data)
    data_args.pop("preparation_manifest", None)
    dm = OpenSRDataModule(**data_args)
    data_args["taco_path"] = [str(p) for p in dm.taco_paths]
    signatures = [file_signature(p) for p in dm.taco_paths]
    validity_key = digest_json({"version": PROTOCOL_VERSION, "sources": signatures,
                               "reader": {k: v for k, v in data_args.items() if k in DATASET_KEYS}})
    validity_path = run_dir.parent / ".data_cache" / f"{validity_key}.json"
    neon_signatures = [file_signature(root / r[k]) for r in rows for k in ("lr", "hr")] if root else []
    raw_signatures = [file_signature(Path(r["path"])) for r in raw]
    protocol = {
        "version": PROTOCOL_VERSION, "sen2neon_enabled": bool(root),
        "manifest_sha256": digest_file(manifest) if manifest else None,
        "sampling": {k: validation.get(k) for k in ("sampling_steps", "sampling_eta", "temperature", "seed")},
        "sen2neon": options, "raw": raw, "data": data_args,
        "every_n_train_steps": validation.get("every_n_train_steps", 500),
    }
    request = digest_json({"protocol": protocol, "sources": signatures, "neon": neon_signatures, "raw": raw_signatures})
    path = run_dir / "data_manifest.json"
    with preparation_lock(run_dir / ".preparation.lock"):
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["request_hash"] != request:
                raise ValueError("Data/validation protocol changed; use the original configuration or a new run_dir")
            # A completed audit can also seed the shared per-source pixel cache.
            if not validity_path.exists():
                validity = {sample_id: None for sample_id in cached["eligible_ids"]}
                validity.update({r["id"]: r["reason"] for r in cached["excluded"]
                                 if r["reason"] != "sen2neon_geographic_overlap"})
                write_json(validity_path, validity)
            return cached
        if config.get("resume_checkpoint"):
            raise FileNotFoundError("Resume requires the original run's data_manifest.json")
        verified_rows = []
        for index, row in enumerate(rows):
            for kind in ("lr", "hr"):
                if digest_file(root / row[kind]) != row[kind + "_sha256"]:
                    raise ValueError(f"SEN2NEON checksum mismatch: {row['id']} {kind}")
            sample = read_neon_pair(root, row)
            verified_rows.append(dict(row, bounds_wgs84=list(sample["bounds_wgs84"])))
            if (index + 1) % 48 == 0:
                print(f"Verified SEN2NEON {index + 1}/{len(rows)}", flush=True)
        records = scan_taco_catalog(dm.taco_paths)
        excluded_groups = overlapping_groups(records, [r["bounds_wgs84"] for r in verified_rows])
        excluded = []
        candidates = []
        for record in records:
            if record.split_group in excluded_groups:
                excluded.append({"id": record.sample_id, "reason": "sen2neon_geographic_overlap"})
            else:
                candidates.append(record)
        dataset = TacoPairedDataset(dm.taco_paths, _records=candidates, _resolved_paths=dm.taco_paths,
                                    **{k: v for k, v in data_args.items() if k in DATASET_KEYS})
        validity = json.loads(validity_path.read_text()) if validity_path.exists() else {}
        eligible = []
        for index, record in enumerate(candidates):
            if record.sample_id not in validity:
                try:
                    dataset.validate_sample(index)
                    validity[record.sample_id] = None
                except InvalidPairError as exc:
                    validity[record.sample_id] = str(exc)
            reason = validity[record.sample_id]
            if reason is None:
                eligible.append(record)
            else:
                excluded.append({"id": record.sample_id, "reason": reason})
            if (index + 1) % 500 == 0:
                print(f"Audited cross-sensor {index + 1}/{len(candidates)}, eligible={len(eligible)}", flush=True)
        write_json(validity_path, validity)
        train, val = split_taco_records(eligible, val_fraction=dm.val_fraction, seed=dm.split_seed)
        result = {
            "request_hash": request, "protocol": protocol,
            "sources": [dict(s, sha256=digest_file(s["path"])) for s in signatures],
            "raw_sources": [dict(s, sha256=digest_file(s["path"])) for s in raw_signatures],
            "eligible_ids": [r.sample_id for r in eligible], "excluded": excluded,
            "train_ids": [r.sample_id for r in train], "val_ids": [r.sample_id for r in val],
            "sen2neon_root": str(root) if root else None, "sen2neon_rows": verified_rows,
            "selection_ids": [r["id"] for r in select_diverse(verified_rows, int(options.get("subset_size", 48)),
                                                           int(validation.get("seed", config.seed)))] if root else [],
            "raw_rows": raw,
        }
        write_json(path, result)
        print(f"Prepared {len(train)} train / {len(val)} internal validation; excluded {len(excluded)}; SEN2NEON={bool(root)}", flush=True)
        return result
