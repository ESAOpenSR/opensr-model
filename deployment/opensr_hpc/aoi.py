from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

from deployment.opensr_hpc.patching import Patch, build_patches, meters_to_lat_deg, meters_to_lon_deg


@dataclass(frozen=True, slots=True)
class AoiSelection:
    aoi_path: Path
    aoi_layer: str | None
    geometry: BaseGeometry
    patches: list[Patch]


def resolve_aoi_source_path(path: str | Path) -> Path:
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"AOI path not found: {source_path}")

    if source_path.is_dir():
        candidates = sorted(candidate for candidate in source_path.iterdir() if candidate.suffix.lower() == ".shp")
        if not candidates:
            raise ValueError(f"No .shp file found in AOI directory: {source_path}")
        if len(candidates) > 1:
            raise ValueError(f"Expected exactly one .shp file in AOI directory: {source_path}")
        return candidates[0].resolve()

    if source_path.suffix.lower() != ".shp":
        raise ValueError(f"AOI path must be a .shp file or a directory containing one: {source_path}")
    return source_path


def _load_source_crs(shp_path: Path) -> CRS:
    prj_path = shp_path.with_suffix(".prj")
    if not prj_path.exists():
        raise ValueError(f"AOI shapefile is missing .prj sidecar: {prj_path}")
    prj_text = prj_path.read_text(encoding="utf-8").strip()
    if not prj_text:
        raise ValueError(f"AOI shapefile has an empty .prj sidecar: {prj_path}")
    return CRS.from_user_input(prj_text)


def load_aoi_geometry(path: str | Path) -> tuple[Path, BaseGeometry]:
    shp_path = resolve_aoi_source_path(path)
    source_crs = _load_source_crs(shp_path)

    reader = shapefile.Reader(str(shp_path))
    try:
        geometries: list[BaseGeometry] = []
        for record in reader.shapeRecords():
            geom = shape(record.shape.__geo_interface__)
            if geom.is_empty:
                continue
            if geom.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError("AOI shapefile must contain polygon geometries")
            geometries.append(geom)
    finally:
        reader.close()

    if not geometries:
        raise ValueError(f"AOI shapefile does not contain any polygon geometries: {shp_path}")

    geometry = unary_union(geometries)
    if geometry.is_empty:
        raise ValueError(f"AOI shapefile geometry is empty after union: {shp_path}")

    if source_crs != CRS.from_epsg(4326):
        transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
        geometry = transform(transformer.transform, geometry)

    if geometry.is_empty:
        raise ValueError(f"AOI shapefile geometry is empty after reprojection: {shp_path}")
    return shp_path, geometry


def patch_footprint(patch: Patch, resolution_m: float) -> BaseGeometry:
    patch_size_m = patch.edge_size * resolution_m
    half_lat = meters_to_lat_deg(patch_size_m) / 2.0
    half_lon = meters_to_lon_deg(patch_size_m, patch.latitude) / 2.0
    return box(
        patch.longitude - half_lon,
        patch.latitude - half_lat,
        patch.longitude + half_lon,
        patch.latitude + half_lat,
    )


def select_aoi_patches(
    *,
    aoi_path: str | Path,
    aoi_layer: str | None,
    edge_size: int,
    resolution_m: float,
    overlap_meters: float,
) -> AoiSelection:
    resolved_path, geometry = load_aoi_geometry(aoi_path)
    lat_min, lon_min, lat_max, lon_max = geometry.bounds[1], geometry.bounds[0], geometry.bounds[3], geometry.bounds[2]
    patches = build_patches(lat_min, lon_min, lat_max, lon_max, edge_size, resolution_m, overlap_meters)
    selected = [patch for patch in patches if patch_footprint(patch, resolution_m).intersects(geometry)]
    if not selected:
        raise ValueError(f"No SR cutouts intersect the AOI geometry: {resolved_path}")
    return AoiSelection(aoi_path=resolved_path, aoi_layer=aoi_layer, geometry=geometry, patches=selected)
