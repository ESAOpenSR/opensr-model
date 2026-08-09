"""TACO v1 data pipeline for OpenSR training.

Two paired-image schemas are supported and detected from asset IDs:

* the worldwide corpus stores ``lr``, ``hr``, ``lrharm`` and ``hrharm``;
  harmonized assets supply RGB and band 8 of raw ``lr`` supplies NIR;
* SEN2NAIPv2 cross-sensor stores direct four-band RGB-NIR ``lr`` and ``hr``
  assets at 10 m and 2.5 m respectively.

Both legacy ``.tortilla`` and newer ``.taco`` container suffixes are accepted.
Assets are always resolved by ``tortilla:id`` rather than child position. Since
worldwide ``hrharm`` has no NIR, its HR NIR is a conservative, mask-normalized
bilinear upsampling of LR NIR.

Digital numbers are scaled by ``1e-4``. Rare valid reflectance outliers are
clamped to the configured model range and reported as ``clipped_fraction``.
Nodata is filled with zero but zero itself remains a valid reflectance value.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from affine import Affine
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

TACO_SUFFIX = ".tortilla"
TACO_SUFFIXES = (TACO_SUFFIX, ".taco")
TACO_MAGIC = frozenset({b"#y", b"WX"})
EXPECTED_HARMONIZED_BANDS = 3
EXPECTED_RAW_LR_BANDS = 12
EXPECTED_ASSET_IDS = frozenset({"lr", "hr", "lrharm", "hrharm"})
CROSSSENSOR_ASSET_IDS = frozenset({"lr", "hr"})
CROSSSENSOR_BANDS = (1, 2, 3, 4)

TacoPathInput = str | Path | Sequence[str | Path]
TrainingSample = dict[str, Tensor | str]


@dataclass(frozen=True, slots=True)
class TacoSampleRecord:
    """A stable pointer to one outer sample in one TACO part."""

    taco_path: Path
    local_index: int
    sample_id: str
    split_group: str


@dataclass(frozen=True, slots=True)
class _RasterAsset:
    data: Tensor
    valid: Tensor
    shape: tuple[int, int]
    crs: Any
    transform: Affine


def _tacoreader_v1() -> Any:
    try:
        import tacoreader.v1 as taco_v1
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "TACO training requires tacoreader's legacy v1 adapter. Install "
            "training dependencies with `pip install -e '.[train]'`."
        ) from exc
    return taco_v1


def _validate_tortilla_header(path: Path) -> None:
    """Reject truncated/non-TACO files before tacoreader can silently skip them."""

    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(18)
    except OSError as exc:
        raise OSError(f"Cannot inspect TACO file {path}: {exc}") from exc
    if len(header) != 18:
        raise ValueError(f"Incomplete TACO header in {path} (size={size} bytes)")
    if header[:2] not in TACO_MAGIC:
        raise ValueError(f"Invalid TACO v1 magic in {path}: {header[:2]!r}")
    offset = int.from_bytes(header[2:10], "little")
    length = int.from_bytes(header[10:18], "little")
    if offset < 18 or length <= 0 or offset + length > size:
        raise ValueError(
            f"Incomplete TACO footer in {path}: offset={offset}, length={length}, "
            f"file_size={size}"
        )


def resolve_taco_paths(taco_path: TacoPathInput) -> tuple[Path, ...]:
    """Resolve file/directory/list input into a fixed, sorted TACO snapshot.

    Directory discovery is non-recursive and sees only visible ``*.tortilla``
    and ``*.taco`` files with complete headers/footers. In-progress entries are
    warned about and ignored. Explicit bad files always fail.
    """

    if isinstance(taco_path, (str, Path)):
        entries: list[str | Path] = [taco_path]
    elif isinstance(taco_path, Sequence):
        entries = list(taco_path)
        if not entries:
            raise ValueError("taco_path cannot be empty")
    else:
        raise TypeError("taco_path must be a file, directory, or sequence of them")

    result: list[Path] = []
    for raw in entries:
        if not isinstance(raw, (str, Path)):
            raise TypeError("Every taco_path entry must be a string or Path")
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"TACO path does not exist: {path}")
        if path.is_dir():
            candidates = sorted(
                (
                    candidate.resolve()
                    for candidate in path.iterdir()
                    if candidate.is_file()
                    and not candidate.name.startswith(".")
                    and candidate.suffix.lower() in TACO_SUFFIXES
                ),
                key=lambda candidate: candidate.name,
            )
            for candidate in candidates:
                try:
                    _validate_tortilla_header(candidate)
                except (OSError, ValueError) as exc:
                    warnings.warn(
                        f"Ignoring a not-yet-complete TACO upload: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    result.append(candidate)
            continue
        if not path.is_file():
            raise ValueError(f"TACO path is not a regular file or directory: {path}")
        if path.name.startswith("."):
            raise ValueError(f"Hidden upload temporary is not accepted: {path}")
        if path.suffix.lower() not in TACO_SUFFIXES:
            raise ValueError(
                f"Only TACO v1 .tortilla or .taco files are accepted: {path}"
            )
        _validate_tortilla_header(path)
        result.append(path)

    paths = tuple(sorted(set(result), key=str))
    if not paths:
        raise FileNotFoundError(
            "No complete, visible .tortilla or .taco files were found"
        )
    return paths


def _columns(table: Any) -> set[str]:
    columns = getattr(table, "columns", None)
    if columns is None:
        raise TypeError(f"tacoreader returned {type(table).__name__} without columns")
    return {str(column) for column in columns}


def _row(table: Any, index: int) -> Any:
    try:
        return table.iloc[index]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise TypeError(
            "tacoreader result does not support positional row access"
        ) from exc


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(np.asarray(value != value).item()):
            return None
    except (TypeError, ValueError):
        pass
    result = str(value).strip()
    return result or None


def _group_key(row: Any, sample_id: str) -> str:
    """Keep imagery-provider variants of one spatial tile in one split."""

    product = _text(_row_get(row, "hr_product"))
    if product and sample_id.endswith(f"_{product}"):
        return f"tile:{sample_id[: -(len(product) + 1)]}"
    for product in ("bing", "google", "esri"):
        suffix = f"_{product}"
        if sample_id.endswith(suffix):
            return f"tile:{sample_id[: -len(suffix)]}"
    # SEN2NAIPv2 IDs end in the acquisition date. Keep repeated observations
    # of the same spatial patch in one split to avoid temporal leakage.
    spatial_id = re.sub(r"_\d{8}$", "", sample_id)
    if spatial_id != sample_id:
        return f"tile:{spatial_id}"
    return f"sample:{sample_id}"


def scan_taco_catalog(paths: Sequence[Path]) -> tuple[TacoSampleRecord, ...]:
    """Load each part separately and validate unique outer sample IDs."""

    records: list[TacoSampleRecord] = []
    seen: dict[str, Path] = {}
    for path in paths:
        _validate_tortilla_header(path)
        try:
            table = _tacoreader_v1().load(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load TACO metadata from {path}") from exc
        missing = {"tortilla:id", "tortilla:file_format"} - _columns(table)
        if missing:
            raise ValueError(f"TACO part {path} is missing columns {sorted(missing)}")
        if len(table) == 0:
            raise ValueError(f"TACO part contains no samples: {path}")
        for local_index in range(len(table)):
            row = _row(table, local_index)
            sample_id = _text(_row_get(row, "tortilla:id"))
            if sample_id is None:
                raise ValueError(f"Empty outer tortilla:id in {path} row {local_index}")
            file_format = _text(_row_get(row, "tortilla:file_format"))
            if file_format != "TORTILLA":
                raise ValueError(
                    f"Outer sample {sample_id!r} must be TORTILLA, got {file_format!r}"
                )
            if sample_id in seen:
                raise ValueError(
                    f"Duplicate outer tortilla:id {sample_id!r} across {seen[sample_id]} "
                    f"and {path}"
                )
            seen[sample_id] = path
            records.append(
                TacoSampleRecord(
                    path, local_index, sample_id, _group_key(row, sample_id)
                )
            )
    return tuple(sorted(records, key=lambda record: record.sample_id))


def _stable_hash(group: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(
            f"{seed}\0{group}".encode(), digest_size=8, person=b"OpenSRSplit"
        ).digest(),
        "big",
    )


def split_taco_records(
    records: Sequence[TacoSampleRecord], *, val_fraction: float, seed: int
) -> tuple[tuple[TacoSampleRecord, ...], tuple[TacoSampleRecord, ...]]:
    """Stable hash split over whole spatial/provider groups."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between zero and one")
    groups: dict[str, list[TacoSampleRecord]] = defaultdict(list)
    for record in records:
        groups[record.split_group].append(record)
    if len(groups) < 2:
        raise ValueError("At least two independent TACO split groups are required")
    limit = int(val_fraction * (1 << 64))
    validation_groups = {group for group in groups if _stable_hash(group, seed) < limit}
    ranked = sorted(groups, key=lambda group: (_stable_hash(group, seed), group))
    if not validation_groups:
        validation_groups.add(ranked[0])
    elif len(validation_groups) == len(groups):
        validation_groups.remove(ranked[-1])
    train = tuple(r for r in records if r.split_group not in validation_groups)
    val = tuple(r for r in records if r.split_group in validation_groups)
    return train, val


def _coerce_rgb_bands(value: Sequence[int]) -> tuple[int, int, int]:
    bands = tuple(value)
    if len(bands) != 3 or any(not isinstance(band, int) for band in bands):
        raise ValueError("rgb_bands must contain exactly three integers")
    if set(bands) != {1, 2, 3}:
        raise ValueError("rgb_bands must be a permutation of one-based bands 1, 2, 3")
    return bands  # type: ignore[return-value]


def _coerce_patch(value: int | Sequence[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        patch = (value, value)
    elif isinstance(value, Sequence) and len(value) == 2:
        patch = (int(value[0]), int(value[1]))
    else:
        raise TypeError("train_patch_size must be an integer, pair, or None")
    if min(patch) <= 0 or any(dimension % 4 for dimension in patch):
        raise ValueError("train_patch_size must be positive and divisible by factor=4")
    return patch


def _coerce_range(value: Sequence[float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("value_range must contain (minimum, maximum)")
    lower, upper = float(value[0]), float(value[1])
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("value_range bounds must be finite and increasing")
    return lower, upper


def _asset_paths(
    table: Any,
    *,
    sample_id: str,
    required_ids: Sequence[str],
    expected_ids: frozenset[str] = EXPECTED_ASSET_IDS,
) -> dict[str, str]:
    missing_columns = {"tortilla:id", "tortilla:file_format"} - _columns(table)
    if missing_columns:
        raise ValueError(
            f"Sample {sample_id!r} is missing child columns {sorted(missing_columns)}"
        )
    positions: dict[str, int] = {}
    for index in range(len(table)):
        row = _row(table, index)
        asset_id = _text(_row_get(row, "tortilla:id"))
        if asset_id is None:
            raise ValueError(f"Sample {sample_id!r} contains an empty child id")
        if asset_id in positions:
            raise ValueError(
                f"Sample {sample_id!r} contains duplicate asset id {asset_id!r}"
            )
        positions[asset_id] = index
    if set(positions) != expected_ids:
        raise ValueError(
            f"Sample {sample_id!r} must contain asset ids {sorted(expected_ids)}, "
            f"found {sorted(positions)}"
        )
    missing_ids = sorted(set(required_ids) - set(positions))
    if missing_ids:
        raise ValueError(
            f"Sample {sample_id!r} is missing required assets {missing_ids}; "
            f"available={sorted(positions)}"
        )
    result: dict[str, str] = {}
    for asset_id in required_ids:
        index = positions[asset_id]
        file_format = _text(_row_get(_row(table, index), "tortilla:file_format"))
        if file_format != "GTiff":
            raise ValueError(
                f"Sample {sample_id!r} asset {asset_id!r} must be GTiff, "
                f"got {file_format!r}"
            )
        path = table.read(index)
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"Sample {sample_id!r} asset {asset_id!r} did not resolve to a path"
            )
        result[asset_id] = str(path)
    return result


def _asset_ids(table: Any, *, sample_id: str) -> frozenset[str]:
    """Read and validate child IDs without opening their raster payloads."""

    missing_columns = {"tortilla:id", "tortilla:file_format"} - _columns(table)
    if missing_columns:
        raise ValueError(
            f"Sample {sample_id!r} is missing child columns {sorted(missing_columns)}"
        )
    result: set[str] = set()
    for index in range(len(table)):
        asset_id = _text(_row_get(_row(table, index), "tortilla:id"))
        if asset_id is None:
            raise ValueError(f"Sample {sample_id!r} contains an empty child id")
        if asset_id in result:
            raise ValueError(
                f"Sample {sample_id!r} contains duplicate asset id {asset_id!r}"
            )
        result.add(asset_id)
    return frozenset(result)


def _read_asset(
    path: str,
    *,
    sample_id: str,
    asset_id: str,
    bands: tuple[int, ...],
    expected_count: int,
    nodata: int,
    scale: float,
) -> _RasterAsset:
    context = f"sample {sample_id!r} asset {asset_id!r}"
    try:
        with rasterio.open(path) as source:
            if source.driver != "GTiff":
                raise ValueError(f"{context} must be a GeoTIFF, got {source.driver!r}")
            if source.count != expected_count:
                raise ValueError(
                    f"{context} must contain {expected_count} bands, found {source.count}"
                )
            if any(dtype != "uint16" for dtype in source.dtypes):
                raise TypeError(f"{context} must be uint16, found {source.dtypes}")
            if source.nodata != float(nodata):
                raise ValueError(
                    f"{context} nodata must be {nodata}, found {source.nodata}"
                )
            if source.crs is None:
                raise ValueError(f"{context} has no CRS")
            if max(bands) > source.count:
                raise ValueError(f"{context} does not contain requested bands {bands}")
            values = source.read(list(bands))
            masks = source.read_masks(list(bands))
            shape = (source.height, source.width)
            crs = source.crs
            transform = source.transform
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise RuntimeError(f"Cannot read {context} from {path}") from exc

    valid_np = np.all(masks > 0, axis=0)
    valid_np &= np.all(values != nodata, axis=0)
    data = torch.from_numpy(values.astype(np.float32, copy=False)).mul_(scale)
    valid = torch.from_numpy(valid_np.copy()).unsqueeze(0)
    if valid.any() and not torch.isfinite(data[:, valid[0]]).all():
        raise ValueError(f"{context} contains non-finite valid values")
    data = data.masked_fill(~valid.expand_as(data), 0.0)
    return _RasterAsset(data, valid, shape, crs, transform)


def _affine_close(left: Affine, right: Affine, *, atol: float = 1e-6) -> bool:
    return bool(
        np.allclose(
            np.asarray(tuple(left)[:6]),
            np.asarray(tuple(right)[:6]),
            rtol=0.0,
            atol=atol,
        )
    )


def _validate_alignment(
    lr_rgb: _RasterAsset,
    raw_lr: _RasterAsset,
    hr_rgb: _RasterAsset,
    *,
    sample_id: str,
    factor: int,
) -> None:
    if lr_rgb.shape != raw_lr.shape:
        raise ValueError(
            f"Sample {sample_id!r} lrharm/raw lr shapes differ: "
            f"{lr_rgb.shape} vs {raw_lr.shape}"
        )
    expected_hr = (lr_rgb.shape[0] * factor, lr_rgb.shape[1] * factor)
    if hr_rgb.shape != expected_hr:
        raise ValueError(
            f"Sample {sample_id!r} is not exactly {factor}x: "
            f"LR={lr_rgb.shape}, HR={hr_rgb.shape}"
        )
    if lr_rgb.crs != raw_lr.crs or lr_rgb.crs != hr_rgb.crs:
        raise ValueError(f"Sample {sample_id!r} assets do not share a CRS")
    if not _affine_close(lr_rgb.transform, raw_lr.transform):
        raise ValueError(f"Sample {sample_id!r} LR transforms are not aligned")
    expected_lr_transform = hr_rgb.transform * Affine.scale(factor, factor)
    if not _affine_close(lr_rgb.transform, expected_lr_transform):
        raise ValueError(
            f"Sample {sample_id!r} HR/LR transforms are not on an exact {factor}x grid"
        )


def _validate_pair_alignment(
    lr: _RasterAsset, hr: _RasterAsset, *, sample_id: str, factor: int
) -> None:
    expected_hr = (lr.shape[0] * factor, lr.shape[1] * factor)
    if hr.shape != expected_hr:
        raise ValueError(
            f"Sample {sample_id!r} is not exactly {factor}x: "
            f"LR={lr.shape}, HR={hr.shape}"
        )
    if lr.crs != hr.crs:
        raise ValueError(f"Sample {sample_id!r} LR/HR assets do not share a CRS")
    expected_lr_transform = hr.transform * Affine.scale(factor, factor)
    # SEN2NAIPv2 contains sub-millimetre origin roundoff between paired files.
    if not _affine_close(lr.transform, expected_lr_transform, atol=1e-4):
        raise ValueError(
            f"Sample {sample_id!r} HR/LR transforms are not on an exact {factor}x grid"
        )


def _upsample_nir(
    nir: Tensor, valid: Tensor, size: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Mask-normalized bilinear NIR with conservative interpolation support."""

    weights = valid.to(nir.dtype)
    numerator = F.interpolate(
        (nir * weights).unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)
    denominator = F.interpolate(
        weights.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)
    output = numerator / denominator.clamp_min(torch.finfo(nir.dtype).eps)
    output_valid = denominator >= 1.0 - 1e-6
    return output.masked_fill(~output_valid, 0.0), output_valid


def _clamp_and_count(
    data: Tensor,
    valid: Tensor,
    value_range: tuple[float, float] | None,
    *,
    sample_id: str,
    name: str,
) -> tuple[Tensor, int, int]:
    if not valid.any():
        raise ValueError(f"Sample {sample_id!r} {name} has no valid pixels")
    expanded = valid.expand_as(data)
    values = data[expanded]
    if not torch.isfinite(values).all():
        raise ValueError(f"Sample {sample_id!r} {name} contains non-finite values")
    total = int(values.numel())
    if value_range is None:
        return data, 0, total
    lower, upper = value_range
    clipped = int(((values < lower) | (values > upper)).sum().item())
    # Only valid reflectance is clamped. Invalid support remains the explicit
    # zero fill and is never included in clipping telemetry.
    bounded = data.clamp(lower, upper)
    return torch.where(expanded, bounded, data), clipped, total


def _aligned_crop(
    hr: Tensor,
    lr: Tensor,
    mask: Tensor,
    *,
    patch: tuple[int, int] | None,
    factor: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if patch is None:
        return hr, lr, mask
    patch_h, patch_w = patch
    hr_h, hr_w = hr.shape[-2:]
    if patch_h > hr_h or patch_w > hr_w:
        raise ValueError(f"train_patch_size {patch} exceeds HR shape {(hr_h, hr_w)}")
    if hr_h % factor or hr_w % factor:
        raise ValueError(f"HR shape {(hr_h, hr_w)} is not divisible by factor={factor}")
    lr_patch_h, lr_patch_w = patch_h // factor, patch_w // factor
    max_top = lr.shape[-2] - lr_patch_h
    max_left = lr.shape[-1] - lr_patch_w
    top = left = 0
    found = False
    for _ in range(10):
        candidate_top = int(torch.randint(max_top + 1, ()).item()) if max_top else 0
        candidate_left = int(torch.randint(max_left + 1, ()).item()) if max_left else 0
        y, x = candidate_top * factor, candidate_left * factor
        if mask[:, y : y + patch_h, x : x + patch_w].any():
            top, left, found = candidate_top, candidate_left, True
            break
    if not found:
        valid_coordinates = torch.nonzero(mask[0], as_tuple=False)
        if not len(valid_coordinates):
            raise ValueError("Cannot crop a sample with no valid pixels")
        centre_y, centre_x = valid_coordinates[len(valid_coordinates) // 2].tolist()
        top = min(max((centre_y - patch_h // 2) // factor, 0), max_top)
        left = min(max((centre_x - patch_w // 2) // factor, 0), max_left)
    hr_top, hr_left = top * factor, left * factor
    return (
        hr[:, hr_top : hr_top + patch_h, hr_left : hr_left + patch_w],
        lr[:, top : top + lr_patch_h, left : left + lr_patch_w],
        mask[:, hr_top : hr_top + patch_h, hr_left : hr_left + patch_w],
    )


def _augment(
    hr: Tensor,
    lr: Tensor,
    mask: Tensor,
    *,
    horizontal: float,
    vertical: float,
    rotate_90: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    tensors = [hr, lr, mask]
    if torch.rand(()) < horizontal:
        tensors = [tensor.flip(-1) for tensor in tensors]
    if torch.rand(()) < vertical:
        tensors = [tensor.flip(-2) for tensor in tensors]
    if rotate_90:
        turns = (
            int(torch.randint(4, ()).item())
            if hr.shape[-2] == hr.shape[-1]
            else 2 * int(torch.randint(2, ()).item())
        )
        if turns:
            tensors = [torch.rot90(tensor, turns, (-2, -1)) for tensor in tensors]
    return tensors[0], tensors[1], tensors[2]


class TacoPairedDataset(Dataset[TrainingSample]):
    """Lazy TACO reader for worldwide and SEN2NAIPv2 paired imagery."""

    def __init__(
        self,
        taco_path: TacoPathInput,
        *,
        training: bool = False,
        train_patch_size: int | Sequence[int] | None = 512,
        factor: int = 4,
        lr_asset_id: str = "lrharm",
        hr_asset_id: str = "hrharm",
        raw_lr_asset_id: str = "lr",
        rgb_bands: Sequence[int] = (1, 2, 3),
        nir_band: int = 8,
        hr_nir_strategy: Literal["upsample_lr"] = "upsample_lr",
        reflectance_scale: float = 1e-4,
        nodata: int = 65535,
        value_range: Sequence[float] | None = (0.0, 1.0),
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        rotate_90: bool = True,
        _records: Sequence[TacoSampleRecord] | None = None,
        _resolved_paths: Sequence[Path] | None = None,
    ) -> None:
        super().__init__()
        if factor != 4:
            raise ValueError("The current OpenSR model and corpus require factor=4")
        asset_ids = (lr_asset_id, hr_asset_id, raw_lr_asset_id)
        if any(
            not isinstance(asset_id, str) or not asset_id.strip()
            for asset_id in asset_ids
        ):
            raise ValueError("TACO asset ids must be non-empty strings")
        if len(set(asset_ids)) != 3:
            raise ValueError(
                "lr_asset_id, hr_asset_id, and raw_lr_asset_id must differ"
            )
        if not isinstance(nir_band, int) or not 1 <= nir_band <= EXPECTED_RAW_LR_BANDS:
            raise ValueError("nir_band must be one-based within the 12-band raw LR")
        if hr_nir_strategy != "upsample_lr":
            raise ValueError(
                "Only hr_nir_strategy='upsample_lr' is supported; hrharm has no NIR"
            )
        if not math.isfinite(reflectance_scale) or reflectance_scale <= 0:
            raise ValueError("reflectance_scale must be finite and positive")
        if nodata != 65535:
            raise ValueError("The worldwide TACO corpus uses nodata=65535")
        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("vertical_flip_probability", vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

        self.taco_paths = (
            tuple(_resolved_paths)
            if _resolved_paths is not None
            else resolve_taco_paths(taco_path)
        )
        self.records = (
            tuple(_records)
            if _records is not None
            else scan_taco_catalog(self.taco_paths)
        )
        if not self.records:
            raise ValueError("TacoPairedDataset requires at least one sample")
        self.training = training
        self.train_patch_size = _coerce_patch(train_patch_size)
        self.factor = factor
        self.lr_asset_id = lr_asset_id
        self.hr_asset_id = hr_asset_id
        self.raw_lr_asset_id = raw_lr_asset_id
        self.rgb_bands = _coerce_rgb_bands(rgb_bands)
        self.nir_band = nir_band
        self.hr_nir_strategy = hr_nir_strategy
        self.reflectance_scale = reflectance_scale
        self.nodata = nodata
        self.value_range = _coerce_range(value_range)
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rotate_90 = rotate_90
        self._reader_pid: int | None = None
        self._readers: dict[Path, Any] = {}

    @property
    def num_samples(self) -> int:
        return len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reader_pid"] = None
        state["_readers"] = {}
        return state

    def _reader(self, path: Path) -> Any:
        pid = os.getpid()
        if self._reader_pid != pid:
            self._reader_pid = pid
            self._readers = {}
        reader = self._readers.get(path)
        if reader is None:
            _validate_tortilla_header(path)
            try:
                reader = _tacoreader_v1().load(str(path))
            except Exception as exc:
                raise RuntimeError(f"Failed to lazily open TACO part {path}") from exc
            self._readers[path] = reader
        return reader

    def _load_full(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, Tensor]:
        record = self.records[index]
        reader = self._reader(record.taco_path)
        if record.local_index >= len(reader):
            raise RuntimeError(
                f"TACO part changed after catalog snapshot: {record.taco_path}"
            )
        outer_row = _row(reader, record.local_index)
        current_id = _text(_row_get(outer_row, "tortilla:id"))
        if current_id != record.sample_id:
            raise RuntimeError(
                f"TACO part changed after snapshot: expected {record.sample_id!r}, "
                f"found {current_id!r}"
            )
        try:
            child_table = reader.read(record.local_index)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read nested sample {record.sample_id!r}"
            ) from exc
        asset_ids = _asset_ids(child_table, sample_id=record.sample_id)
        if asset_ids == EXPECTED_ASSET_IDS:
            paths = _asset_paths(
                child_table,
                sample_id=record.sample_id,
                required_ids=(
                    self.lr_asset_id,
                    self.hr_asset_id,
                    self.raw_lr_asset_id,
                ),
            )
            lr_rgb = _read_asset(
                paths[self.lr_asset_id],
                sample_id=record.sample_id,
                asset_id=self.lr_asset_id,
                bands=self.rgb_bands,
                expected_count=EXPECTED_HARMONIZED_BANDS,
                nodata=self.nodata,
                scale=self.reflectance_scale,
            )
            hr_rgb = _read_asset(
                paths[self.hr_asset_id],
                sample_id=record.sample_id,
                asset_id=self.hr_asset_id,
                bands=self.rgb_bands,
                expected_count=EXPECTED_HARMONIZED_BANDS,
                nodata=self.nodata,
                scale=self.reflectance_scale,
            )
            raw_lr = _read_asset(
                paths[self.raw_lr_asset_id],
                sample_id=record.sample_id,
                asset_id=self.raw_lr_asset_id,
                bands=(self.nir_band,),
                expected_count=EXPECTED_RAW_LR_BANDS,
                nodata=self.nodata,
                scale=self.reflectance_scale,
            )
            _validate_alignment(
                lr_rgb,
                raw_lr,
                hr_rgb,
                sample_id=record.sample_id,
                factor=self.factor,
            )

            lr_valid = lr_rgb.valid & raw_lr.valid
            lr = torch.cat((lr_rgb.data, raw_lr.data), dim=0)
            lr = lr.masked_fill(~lr_valid.expand_as(lr), 0.0)
            hr_nir, hr_nir_valid = _upsample_nir(raw_lr.data, lr_valid, hr_rgb.shape)
            hr_valid = hr_rgb.valid & hr_nir_valid
            hr = torch.cat((hr_rgb.data, hr_nir), dim=0)
            hr = hr.masked_fill(~hr_valid.expand_as(hr), 0.0)
        elif asset_ids == CROSSSENSOR_ASSET_IDS:
            paths = _asset_paths(
                child_table,
                sample_id=record.sample_id,
                required_ids=("lr", "hr"),
                expected_ids=CROSSSENSOR_ASSET_IDS,
            )
            lr_asset = _read_asset(
                paths["lr"],
                sample_id=record.sample_id,
                asset_id="lr",
                bands=CROSSSENSOR_BANDS,
                expected_count=len(CROSSSENSOR_BANDS),
                nodata=self.nodata,
                scale=self.reflectance_scale,
            )
            hr_asset = _read_asset(
                paths["hr"],
                sample_id=record.sample_id,
                asset_id="hr",
                bands=CROSSSENSOR_BANDS,
                expected_count=len(CROSSSENSOR_BANDS),
                nodata=self.nodata,
                scale=self.reflectance_scale,
            )
            _validate_pair_alignment(
                lr_asset, hr_asset, sample_id=record.sample_id, factor=self.factor
            )
            lr, lr_valid = lr_asset.data, lr_asset.valid
            hr, hr_valid = hr_asset.data, hr_asset.valid
        else:
            raise ValueError(
                f"Sample {record.sample_id!r} has unsupported asset ids "
                f"{sorted(asset_ids)}; expected {sorted(EXPECTED_ASSET_IDS)} or "
                f"{sorted(CROSSSENSOR_ASSET_IDS)}"
            )

        lr, lr_clipped, lr_total = _clamp_and_count(
            lr, lr_valid, self.value_range, sample_id=record.sample_id, name="LR"
        )
        hr, hr_clipped, hr_total = _clamp_and_count(
            hr, hr_valid, self.value_range, sample_id=record.sample_id, name="HR"
        )
        fraction = torch.tensor(
            (lr_clipped + hr_clipped) / (lr_total + hr_total), dtype=torch.float32
        )
        return hr, lr, hr_valid, record.sample_id, fraction

    def validate_sample(self, index: int = 0) -> None:
        """Validate one full sample without consuming crop/augmentation RNG."""

        self._load_full(index)

    def __getitem__(self, index: int) -> TrainingSample:
        hr, lr, mask, sample_id, clipped_fraction = self._load_full(index)
        if self.training:
            hr, lr, mask = _aligned_crop(
                hr, lr, mask, patch=self.train_patch_size, factor=self.factor
            )
            hr, lr, mask = _augment(
                hr,
                lr,
                mask,
                horizontal=self.horizontal_flip_probability,
                vertical=self.vertical_flip_probability,
                rotate_90=self.rotate_90,
            )
        return {
            "image": hr.float().contiguous(),
            "LR_image": lr.float().contiguous(),
            "valid_mask": mask.float().contiguous(),
            "sample_id": sample_id,
            "clipped_fraction": clipped_fraction,
        }


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class OpenSRDataModule(LightningDataModule):
    """Lightning data module for supported paired-image TACO corpora.

    The default training crop is the complete 512x512 HR tile. Validation is
    never cropped or augmented, so validation image logging reconstructs whole
    scenes on every epoch.
    """

    def __init__(
        self,
        taco_path: TacoPathInput,
        *,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        factor: int = 4,
        train_patch_size: int | Sequence[int] | None = 512,
        lr_asset_id: str = "lrharm",
        hr_asset_id: str = "hrharm",
        raw_lr_asset_id: str = "lr",
        rgb_bands: Sequence[int] = (1, 2, 3),
        nir_band: int = 8,
        hr_nir_strategy: Literal["upsample_lr"] = "upsample_lr",
        reflectance_scale: float = 1e-4,
        nodata: int = 65535,
        value_range: Sequence[float] | None = (0.0, 1.0),
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        rotate_90: bool = True,
        batch_size: int = 1,
        val_batch_size: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between zero and one")
        if batch_size <= 0 or (val_batch_size is not None and val_batch_size <= 0):
            raise ValueError("batch sizes must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0")
        if prefetch_factor is not None and (num_workers == 0 or prefetch_factor <= 0):
            raise ValueError(
                "prefetch_factor requires num_workers > 0 and a positive value"
            )

        # This path snapshot is fixed for the job. Newly completed upload parts
        # are included after restart, never halfway through an epoch.
        self.taco_paths = resolve_taco_paths(taco_path)
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.factor = factor
        self.train_patch_size = train_patch_size
        self.lr_asset_id = lr_asset_id
        self.hr_asset_id = hr_asset_id
        self.raw_lr_asset_id = raw_lr_asset_id
        self.rgb_bands = tuple(rgb_bands)
        self.nir_band = nir_band
        self.hr_nir_strategy = hr_nir_strategy
        self.reflectance_scale = reflectance_scale
        self.nodata = nodata
        self.value_range = None if value_range is None else tuple(value_range)
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rotate_90 = rotate_90
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.sample_records: tuple[TacoSampleRecord, ...] | None = None
        self.train_dataset: TacoPairedDataset | None = None
        self.val_dataset: TacoPairedDataset | None = None

    @property
    def num_samples(self) -> int | None:
        return None if self.sample_records is None else len(self.sample_records)

    @property
    def train_sample_count(self) -> int | None:
        return None if self.train_dataset is None else len(self.train_dataset)

    @property
    def val_sample_count(self) -> int | None:
        return None if self.val_dataset is None else len(self.val_dataset)

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit", "validate"}:
            return
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        self.sample_records = scan_taco_catalog(self.taco_paths)
        train, val = split_taco_records(
            self.sample_records, val_fraction=self.val_fraction, seed=self.split_seed
        )
        common: dict[str, Any] = {
            "taco_path": self.taco_paths,
            "train_patch_size": self.train_patch_size,
            "factor": self.factor,
            "lr_asset_id": self.lr_asset_id,
            "hr_asset_id": self.hr_asset_id,
            "raw_lr_asset_id": self.raw_lr_asset_id,
            "rgb_bands": self.rgb_bands,
            "nir_band": self.nir_band,
            "hr_nir_strategy": self.hr_nir_strategy,
            "reflectance_scale": self.reflectance_scale,
            "nodata": self.nodata,
            "value_range": self.value_range,
            "_resolved_paths": self.taco_paths,
        }
        self.train_dataset = TacoPairedDataset(
            training=True,
            horizontal_flip_probability=self.horizontal_flip_probability,
            vertical_flip_probability=self.vertical_flip_probability,
            rotate_90=self.rotate_90,
            _records=train,
            **common,
        )
        self.val_dataset = TacoPairedDataset(
            training=False,
            horizontal_flip_probability=0.0,
            vertical_flip_probability=0.0,
            rotate_90=False,
            _records=val,
            **common,
        )
        # Representative eager reads catch schema/normalization mistakes before
        # the optimizer starts. Remaining samples are validated lazily on use.
        self.train_dataset.validate_sample(0)
        self.val_dataset.validate_sample(0)

    def _loader(
        self,
        dataset: TacoPairedDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader[dict[str, Tensor | list[str]]]:
        generator = torch.Generator().manual_seed(self.split_seed + int(not shuffle))
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": drop_last,
            "worker_init_fn": _seed_worker,
            "generator": generator,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader[dict[str, Tensor | list[str]]]:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return self._loader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Tensor | list[str]]]:
        if self.val_dataset is None:
            self.setup("validate")
        assert self.val_dataset is not None
        return self._loader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
        )


__all__ = [
    "OpenSRDataModule",
    "TACO_SUFFIX",
    "TACO_SUFFIXES",
    "TacoPairedDataset",
    "TacoSampleRecord",
    "resolve_taco_paths",
    "scan_taco_catalog",
    "split_taco_records",
]
