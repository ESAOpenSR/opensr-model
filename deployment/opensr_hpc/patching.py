from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Patch:
    patch_id: str
    latitude: float
    longitude: float
    edge_size: int
    row_index: int
    row_count: int
    column_index: int
    column_count: int


def meters_to_lat_deg(distance_m: float) -> float:
    return distance_m / 111_320.0


def meters_to_lon_deg(distance_m: float, latitude_deg: float) -> float:
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(latitude_deg))
    if meters_per_degree_lon <= 0:
        raise ValueError(
            f"Cannot compute longitude degrees at latitude {latitude_deg:.6f}"
        )
    return distance_m / meters_per_degree_lon


def clamp_center(
    suggested_center: float, min_value: float, max_value: float, half_extent: float
) -> float:
    lower_limit = min_value + half_extent
    upper_limit = max_value - half_extent
    if lower_limit > upper_limit:
        return (min_value + max_value) / 2.0
    return max(lower_limit, min(suggested_center, upper_limit))


def compute_centers(
    min_value: float, max_value: float, patch_deg: float, step_deg: float
) -> list[float]:
    if patch_deg <= 0 or step_deg <= 0:
        raise ValueError("patch_deg and step_deg must be positive")

    half_extent = patch_deg / 2.0
    span = max(0.0, max_value - min_value)

    if span <= patch_deg:
        center = (min_value + max_value) / 2.0
        return [clamp_center(center, min_value, max_value, half_extent)]

    count = int(math.ceil((span - patch_deg) / step_deg)) + 1
    centers: list[float] = []
    for index in range(count):
        candidate = min_value + half_extent + index * step_deg
        candidate = clamp_center(candidate, min_value, max_value, half_extent)
        if centers and abs(candidate - centers[-1]) < 1e-12:
            continue
        centers.append(candidate)

    if not centers:
        centers.append(
            clamp_center(
                (min_value + max_value) / 2.0, min_value, max_value, half_extent
            )
        )

    return centers


def build_patches(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    edge_size: int,
    resolution_m: float,
    overlap_meters: float,
) -> list[Patch]:
    lat_min, lat_max = sorted((lat1, lat2))
    if abs(lon2 - lon1) > 180.0:
        raise ValueError("Bounding boxes that cross the antimeridian are not supported")
    lon_min, lon_max = sorted((lon1, lon2))

    if edge_size <= 0:
        raise ValueError("edge_size must be positive")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    patch_size_m = edge_size * resolution_m
    if overlap_meters >= patch_size_m:
        raise ValueError("overlap_meters must be smaller than patch size")

    patch_step_m = patch_size_m - overlap_meters
    patch_lat_deg = meters_to_lat_deg(patch_size_m)
    lat_step_deg = meters_to_lat_deg(patch_step_m)
    lat_centers = compute_centers(lat_min, lat_max, patch_lat_deg, lat_step_deg)
    lat_rows = len(lat_centers)

    patches: list[Patch] = []
    patch_number = 1
    for row_index, lat_center in enumerate(lat_centers):
        patch_lon_deg = meters_to_lon_deg(patch_size_m, lat_center)
        lon_step_deg = meters_to_lon_deg(patch_step_m, lat_center)
        lon_centers = compute_centers(lon_min, lon_max, patch_lon_deg, lon_step_deg)
        lon_cols = len(lon_centers)
        for column_index, lon_center in enumerate(lon_centers):
            patches.append(
                Patch(
                    patch_id=f"patch_{patch_number:06d}",
                    latitude=lat_center,
                    longitude=lon_center,
                    edge_size=edge_size,
                    row_index=row_index,
                    row_count=lat_rows,
                    column_index=column_index,
                    column_count=lon_cols,
                )
            )
            patch_number += 1
    return patches
