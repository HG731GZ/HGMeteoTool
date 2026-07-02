from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .catalog_sources import default_catalog_dir


@dataclass(frozen=True)
class MilkyWayRing:
    ra_deg: np.ndarray
    dec_deg: np.ndarray


@dataclass(frozen=True)
class MilkyWayPolygon:
    rings: tuple[MilkyWayRing, ...]


@dataclass(frozen=True)
class MilkyWayCatalog:
    source_name: str
    polygons: tuple[MilkyWayPolygon, ...]

    @property
    def point_count(self) -> int:
        return int(sum(ring.ra_deg.size for polygon in self.polygons for ring in polygon.rings))


def default_milky_way_path() -> Path:
    return default_catalog_dir() / "d3_celestial" / "mw.json"


def _geojson_ra_to_deg(longitude_deg: float) -> float:
    return longitude_deg if longitude_deg >= 0.0 else longitude_deg + 360.0


def _ring_from_coordinates(coordinates: list[object]) -> MilkyWayRing | None:
    ra_values: list[float] = []
    dec_values: list[float] = []
    for point in coordinates:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            longitude = float(point[0])
            declination = float(point[1])
        except (TypeError, ValueError):
            continue
        ra_values.append(_geojson_ra_to_deg(longitude))
        dec_values.append(declination)

    if len(ra_values) < 3:
        return None
    return MilkyWayRing(
        ra_deg=np.asarray(ra_values, dtype=np.float64),
        dec_deg=np.asarray(dec_values, dtype=np.float64),
    )


def _polygons_from_geometry(geometry: dict[str, object]) -> list[MilkyWayPolygon]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return []

    raw_polygons: list[object]
    if geometry_type == "Polygon":
        raw_polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        raw_polygons = coordinates
    else:
        return []

    polygons: list[MilkyWayPolygon] = []
    for raw_polygon in raw_polygons:
        if not isinstance(raw_polygon, list):
            continue
        rings = tuple(
            ring
            for raw_ring in raw_polygon
            if isinstance(raw_ring, list)
            for ring in (_ring_from_coordinates(raw_ring),)
            if ring is not None
        )
        if rings:
            polygons.append(MilkyWayPolygon(rings=rings))
    return polygons


@lru_cache(maxsize=4)
def _load_milky_way_cached(path_text: str) -> MilkyWayCatalog:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"银河 GeoJSON 数据不存在：{path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError(f"银河数据不是 GeoJSON FeatureCollection：{path}")

    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError(f"银河数据缺少 features 数组：{path}")

    polygons: list[MilkyWayPolygon] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            polygons.extend(_polygons_from_geometry(geometry))

    if not polygons:
        raise ValueError(f"未能从银河 GeoJSON 中解析出多边形：{path}")

    return MilkyWayCatalog(
        source_name="d3-celestial Milky Way GeoJSON",
        polygons=tuple(polygons),
    )


def load_milky_way(path: Path | None = None) -> MilkyWayCatalog:
    return _load_milky_way_cached(str(path or default_milky_way_path()))
