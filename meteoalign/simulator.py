from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

_ASTROPY_ROOT = Path(os.environ.get("METEOALIGN_ASTROPY_CACHE", Path(tempfile.gettempdir()) / "meteoalign_astropy"))
(_ASTROPY_ROOT / "cache").mkdir(parents=True, exist_ok=True)
(_ASTROPY_ROOT / "config").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_ASTROPY_ROOT / "cache"))
os.environ.setdefault("XDG_CONFIG_HOME", str(_ASTROPY_ROOT / "config"))

try:
    (Path.home() / ".astropy" / "cache").mkdir(parents=True, exist_ok=True)
except OSError:
    fallback_home = _ASTROPY_ROOT / "home"
    fallback_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("METEOALIGN_ORIGINAL_HOME", str(Path.home()))
    os.environ["HOME"] = str(fallback_home)

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers
from skyfield.api import Loader, load_file, wgs84

from .catalog import StarCatalog, project_root
from .milky_way import MilkyWayCatalog


iers.conf.auto_download = False
iers.conf.auto_max_age = None
iers.conf.iers_degraded_accuracy = "warn"


RECTILINEAR_LENS_MODEL = "rectilinear"
FISHEYE_EQUIDISTANT = "fisheye_equidistant"
FISHEYE_EQUISOLID = "fisheye_equisolid"
MERCATOR_LENS_MODEL = "mercator"
CYLINDRICAL_EQUIDISTANT_LENS_MODEL = "cylindrical_equidistant"
FISHEYE_LENS_MODELS = {
    FISHEYE_EQUIDISTANT,
    FISHEYE_EQUISOLID,
}
CYLINDRICAL_LENS_MODELS = {
    MERCATOR_LENS_MODEL,
    CYLINDRICAL_EQUIDISTANT_LENS_MODEL,
}
SUPPORTED_LENS_MODELS = {RECTILINEAR_LENS_MODEL, *FISHEYE_LENS_MODELS, *CYLINDRICAL_LENS_MODELS}
Point2D = tuple[float, float]


@dataclass(frozen=True)
class SolarSystemObjectSpec:
    object_id: str
    display_name: str
    kernel_name: str
    mag_v: float
    color_rgb: tuple[int, int, int]
    radius_px: float
    reference_allowed: bool


SOLAR_SYSTEM_OBJECT_SPECS = (
    SolarSystemObjectSpec(
        object_id="sun",
        display_name="太阳",
        kernel_name="sun",
        mag_v=-26.7,
        color_rgb=(255, 232, 150),
        radius_px=9.5,
        reference_allowed=False,
    ),
    SolarSystemObjectSpec(
        object_id="mercury",
        display_name="水星",
        kernel_name="mercury barycenter",
        mag_v=-0.4,
        color_rgb=(205, 200, 185),
        radius_px=3.5,
        reference_allowed=True,
    ),
    SolarSystemObjectSpec(
        object_id="venus",
        display_name="金星",
        kernel_name="venus barycenter",
        mag_v=-4.0,
        color_rgb=(255, 238, 185),
        radius_px=5.2,
        reference_allowed=True,
    ),
    SolarSystemObjectSpec(
        object_id="mars",
        display_name="火星",
        kernel_name="mars barycenter",
        mag_v=-1.0,
        color_rgb=(255, 128, 92),
        radius_px=4.2,
        reference_allowed=True,
    ),
    SolarSystemObjectSpec(
        object_id="jupiter",
        display_name="木星",
        kernel_name="jupiter barycenter",
        mag_v=-2.5,
        color_rgb=(255, 214, 165),
        radius_px=5.5,
        reference_allowed=True,
    ),
    SolarSystemObjectSpec(
        object_id="saturn",
        display_name="土星",
        kernel_name="saturn barycenter",
        mag_v=0.5,
        color_rgb=(255, 226, 160),
        radius_px=4.8,
        reference_allowed=True,
    ),
    SolarSystemObjectSpec(
        object_id="moon",
        display_name="月亮",
        kernel_name="moon",
        mag_v=-12.0,
        color_rgb=(220, 220, 215),
        radius_px=8.5,
        reference_allowed=False,
    ),
)


@dataclass(frozen=True)
class ObserverSettings:
    observation_time_utc: datetime
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0


@dataclass(frozen=True)
class CameraSettings:
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    focal_length_mm: float
    lens_model: str = RECTILINEAR_LENS_MODEL
    fisheye_fov_deg: float = 180.0


@dataclass(frozen=True)
class ViewSettings:
    center_az_deg: float
    center_alt_deg: float
    roll_deg: float = 0.0


@dataclass(frozen=True)
class ProjectedGridLine:
    kind: str
    label: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ProjectedLabel:
    text: str
    x_px: float
    y_px: float
    kind: str = "direction"


@dataclass(frozen=True)
class HorizontalStarCatalog:
    source_name: str
    star_ids: np.ndarray
    display_names: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag_v: np.ndarray
    color_index_bv: np.ndarray
    spectral_type: np.ndarray
    common_names: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray

    def __len__(self) -> int:
        return int(self.ra_deg.size)


@dataclass(frozen=True)
class HorizontalSolarSystemCatalog:
    source_name: str
    object_ids: np.ndarray
    display_names: np.ndarray
    kernel_names: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag_v: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray
    color_rgb: np.ndarray
    radius_px: np.ndarray
    reference_allowed: np.ndarray

    def __len__(self) -> int:
        return int(self.ra_deg.size)


@dataclass(frozen=True)
class HorizontalMilkyWayRing:
    alt_deg: np.ndarray
    az_deg: np.ndarray


@dataclass(frozen=True)
class HorizontalMilkyWayPolygon:
    rings: tuple[HorizontalMilkyWayRing, ...]


@dataclass(frozen=True)
class HorizontalMilkyWayCatalog:
    source_name: str
    polygons: tuple[HorizontalMilkyWayPolygon, ...]

    @property
    def point_count(self) -> int:
        return int(sum(ring.alt_deg.size for polygon in self.polygons for ring in polygon.rings))


@dataclass(frozen=True)
class ProjectedMilkyWayPolygon:
    rings: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class ProjectedFillRect:
    x_px: float
    y_px: float
    width_px: float
    height_px: float


@dataclass(frozen=True)
class ProjectedSolarSystemObject:
    object_id: str
    display_name: str
    kernel_name: str
    ra_deg: float
    dec_deg: float
    mag_v: float
    sim_x: float
    sim_y: float
    alt_deg: float
    az_deg: float
    radius_px: float
    color_rgb: tuple[int, int, int]
    alpha: int
    above_horizon: bool
    reference_allowed: bool


@dataclass(frozen=True)
class ProjectedStarMap:
    width: int
    height: int
    source_name: str
    x_px: np.ndarray
    y_px: np.ndarray
    radius_px: np.ndarray
    intensity: np.ndarray
    alpha: np.ndarray
    above_horizon: np.ndarray
    star_ids: np.ndarray
    display_names: np.ndarray
    common_names: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray
    mag_v: np.ndarray
    color_index_bv: np.ndarray
    spectral_type: np.ndarray
    star_rgb: np.ndarray
    grid_lines: tuple[ProjectedGridLine, ...]
    direction_labels: tuple[ProjectedLabel, ...]
    catalog_count: int
    lens_model: str = RECTILINEAR_LENS_MODEL
    sky_circle_radius_px: float | None = None
    horizon_shadow_rects: tuple[ProjectedFillRect, ...] = ()
    milky_way_polygons: tuple[ProjectedMilkyWayPolygon, ...] = ()
    solar_system_objects: tuple[ProjectedSolarSystemObject, ...] = ()

    def __len__(self) -> int:
        return int(self.x_px.size)

    @property
    def above_horizon_count(self) -> int:
        return int(np.count_nonzero(self.above_horizon))


@dataclass(frozen=True)
class ReferenceStar:
    index: int
    star_id: str
    name: str
    display_name: str
    common_name: str
    ra_deg: float
    dec_deg: float
    mag_v: float
    sim_x: float
    sim_y: float
    alt_deg: float
    az_deg: float
    object_type: str = "star"
    index_label: str = ""


def horizontal_fov_deg(camera: CameraSettings) -> float:
    if camera.lens_model in FISHEYE_LENS_MODELS or camera.lens_model in CYLINDRICAL_LENS_MODELS:
        return float(camera.fisheye_fov_deg)
    return float(np.degrees(2.0 * np.arctan(camera.sensor_width_mm / (2.0 * camera.focal_length_mm))))


def vertical_fov_deg(camera: CameraSettings) -> float:
    if camera.lens_model in FISHEYE_LENS_MODELS or camera.lens_model in CYLINDRICAL_LENS_MODELS:
        aspect_scale = camera.image_height_px / max(float(camera.image_width_px), 1.0)
        return float(camera.fisheye_fov_deg * min(aspect_scale, 1.0))
    return float(np.degrees(2.0 * np.arctan(camera.sensor_height_mm / (2.0 * camera.focal_length_mm))))


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def default_solar_system_ephemeris_path() -> Path:
    return project_root() / "catalog" / "de440s.bsp"


@lru_cache(maxsize=1)
def _skyfield_timescale():
    # Skyfield 的时间尺度文件放进临时缓存根目录，避免写入系统或用户配置目录。
    loader = Loader(str(_ASTROPY_ROOT / "skyfield"))
    return loader.timescale()


@lru_cache(maxsize=4)
def _load_solar_system_ephemeris(path_text: str):
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 JPL DE440s 星历文件：{path}。请运行 scripts/download_catalogs.py 下载。"
        )
    return load_file(str(path))


def compute_altaz_from_radec(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray]:
    obstime = Time(_ensure_aware_utc(observer.observation_time_utc))
    location = EarthLocation.from_geodetic(
        lon=observer.longitude_deg * u.deg,
        lat=observer.latitude_deg * u.deg,
        height=observer.elevation_m * u.m,
    )
    sky_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = sky_coords.transform_to(AltAz(obstime=obstime, location=location))
    return altaz.alt.degree.astype(np.float64), altaz.az.degree.astype(np.float64)


def compute_altaz(catalog: StarCatalog, observer: ObserverSettings) -> tuple[np.ndarray, np.ndarray]:
    return compute_altaz_from_radec(catalog.ra_deg, catalog.dec_deg, observer)


def compute_horizontal_catalog(
    catalog: StarCatalog,
    observer: ObserverSettings,
    visible_mag_limit: float,
) -> HorizontalStarCatalog:
    limited_catalog = catalog.with_mag_limit(visible_mag_limit)
    alt_deg, az_deg = compute_altaz(limited_catalog, observer)
    return HorizontalStarCatalog(
        source_name=limited_catalog.source_name,
        star_ids=limited_catalog.star_ids,
        display_names=limited_catalog.display_names,
        ra_deg=limited_catalog.ra_deg,
        dec_deg=limited_catalog.dec_deg,
        mag_v=limited_catalog.mag_v,
        color_index_bv=limited_catalog.color_index_bv,
        spectral_type=limited_catalog.spectral_type,
        common_names=limited_catalog.common_names,
        alt_deg=alt_deg,
        az_deg=az_deg,
    )


def compute_horizontal_solar_system(
    observer: ObserverSettings,
    ephemeris_path: Path | None = None,
) -> HorizontalSolarSystemCatalog:
    ephemeris_file = (ephemeris_path or default_solar_system_ephemeris_path()).resolve()
    ephemeris = _load_solar_system_ephemeris(str(ephemeris_file))
    time = _skyfield_timescale().from_datetime(_ensure_aware_utc(observer.observation_time_utc))
    location = wgs84.latlon(
        latitude_degrees=observer.latitude_deg,
        longitude_degrees=observer.longitude_deg,
        elevation_m=observer.elevation_m,
    )
    observer_position = ephemeris["earth"] + location

    object_ids: list[str] = []
    display_names: list[str] = []
    kernel_names: list[str] = []
    ra_values: list[float] = []
    dec_values: list[float] = []
    mag_values: list[float] = []
    alt_values: list[float] = []
    az_values: list[float] = []
    color_values: list[tuple[int, int, int]] = []
    radius_values: list[float] = []
    reference_flags: list[bool] = []

    for spec in SOLAR_SYSTEM_OBJECT_SPECS:
        target = ephemeris[spec.kernel_name]
        apparent = observer_position.at(time).observe(target).apparent()
        ra, dec, _distance = apparent.radec(epoch="date")
        alt, az, _altaz_distance = apparent.altaz()
        object_ids.append(f"solar_system:{spec.object_id}")
        display_names.append(spec.display_name)
        kernel_names.append(spec.kernel_name)
        ra_values.append(float(ra.hours * 15.0) % 360.0)
        dec_values.append(float(dec.degrees))
        mag_values.append(spec.mag_v)
        alt_values.append(float(alt.degrees))
        az_values.append(float(az.degrees) % 360.0)
        color_values.append(spec.color_rgb)
        radius_values.append(spec.radius_px)
        reference_flags.append(spec.reference_allowed)

    return HorizontalSolarSystemCatalog(
        source_name=f"JPL DE440s ({ephemeris_file.name})",
        object_ids=np.asarray(object_ids, dtype="<U32"),
        display_names=np.asarray(display_names, dtype="<U8"),
        kernel_names=np.asarray(kernel_names, dtype="<U32"),
        ra_deg=np.asarray(ra_values, dtype=np.float64),
        dec_deg=np.asarray(dec_values, dtype=np.float64),
        mag_v=np.asarray(mag_values, dtype=np.float64),
        alt_deg=np.asarray(alt_values, dtype=np.float64),
        az_deg=np.asarray(az_values, dtype=np.float64),
        color_rgb=np.asarray(color_values, dtype=np.uint8),
        radius_px=np.asarray(radius_values, dtype=np.float64),
        reference_allowed=np.asarray(reference_flags, dtype=bool),
    )


def compute_horizontal_milky_way(
    milky_way: MilkyWayCatalog,
    observer: ObserverSettings,
) -> HorizontalMilkyWayCatalog:
    rings = [ring for polygon in milky_way.polygons for ring in polygon.rings]
    if not rings:
        return HorizontalMilkyWayCatalog(source_name=milky_way.source_name, polygons=())

    ring_lengths = [ring.ra_deg.size for ring in rings]
    all_ra = np.concatenate([ring.ra_deg for ring in rings])
    all_dec = np.concatenate([ring.dec_deg for ring in rings])
    all_alt, all_az = compute_altaz_from_radec(all_ra, all_dec, observer)

    horizontal_rings: list[HorizontalMilkyWayRing] = []
    offset = 0
    for length in ring_lengths:
        next_offset = offset + length
        horizontal_rings.append(
            HorizontalMilkyWayRing(
                alt_deg=all_alt[offset:next_offset],
                az_deg=all_az[offset:next_offset],
            )
        )
        offset = next_offset

    polygons: list[HorizontalMilkyWayPolygon] = []
    ring_offset = 0
    for polygon in milky_way.polygons:
        polygon_rings = tuple(horizontal_rings[ring_offset : ring_offset + len(polygon.rings)])
        ring_offset += len(polygon.rings)
        if polygon_rings:
            polygons.append(HorizontalMilkyWayPolygon(rings=polygon_rings))

    return HorizontalMilkyWayCatalog(source_name=milky_way.source_name, polygons=tuple(polygons))


def _local_vectors_from_altaz(alt_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)
    cos_alt = np.cos(alt)
    return np.column_stack(
        (
            cos_alt * np.sin(az),  # 东
            cos_alt * np.cos(az),  # 北
            np.sin(alt),  # 天顶方向
        )
    )


def local_vectors_from_altaz(alt_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    return _local_vectors_from_altaz(alt_deg, az_deg)


def _project_vectors_onto_camera_basis(
    vectors: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right, up, forward = basis
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    cam_x = x * right[0] + y * right[1] + z * right[2]
    cam_y = x * up[0] + y * up[1] + z * up[2]
    cam_z = x * forward[0] + y * forward[1] + z * forward[2]
    return cam_x.astype(np.float64), cam_y.astype(np.float64), cam_z.astype(np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def _camera_basis(view: ViewSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _local_vectors_from_altaz(
        np.asarray([view.center_alt_deg], dtype=np.float64),
        np.asarray([view.center_az_deg], dtype=np.float64),
    )[0]
    forward = _normalize(forward)

    reference_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, reference_up)
    if np.linalg.norm(right) < 1e-8:
        reference_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, reference_up)
    right = _normalize(right)
    up = _normalize(np.cross(right, forward))

    roll = np.deg2rad(view.roll_deg)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    rolled_right = right * cos_roll + up * sin_roll
    rolled_up = -right * sin_roll + up * cos_roll
    return rolled_right, rolled_up, forward


def camera_basis_from_view(view: ViewSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _camera_basis(view)


def _star_style(mag_v: np.ndarray, visible_mag_limit: float) -> tuple[np.ndarray, np.ndarray]:
    bright_mag = float(np.nanmin(mag_v)) if mag_v.size else -1.0
    denom = max(visible_mag_limit - bright_mag, 1e-6)
    normalized = np.clip((visible_mag_limit - mag_v) / denom, 0.0, 1.0)
    radius = 0.8 + 4.8 * np.sqrt(normalized)
    intensity = 70.0 + 185.0 * normalized
    return radius.astype(np.float64), intensity.astype(np.uint8)


def _rgb_from_bv(color_index: float) -> tuple[int, int, int]:
    if color_index < -0.05:
        return 145, 180, 255
    if color_index < 0.15:
        return 190, 215, 255
    if color_index < 0.35:
        return 245, 245, 255
    if color_index < 0.55:
        return 255, 245, 205
    if color_index < 0.85:
        return 255, 220, 135
    if color_index < 1.25:
        return 255, 165, 85
    return 255, 105, 80


def _rgb_from_spectral_type(spectral_type: str) -> tuple[int, int, int] | None:
    spectral_class = spectral_type.strip()[:1].upper()
    if spectral_class in {"O", "B"}:
        return 145, 180, 255
    if spectral_class == "A":
        return 220, 230, 255
    if spectral_class == "F":
        return 255, 245, 205
    if spectral_class == "G":
        return 255, 220, 135
    if spectral_class == "K":
        return 255, 165, 85
    if spectral_class in {"M", "C", "S"}:
        return 255, 105, 80
    return None


def _star_rgb(
    mag_v: np.ndarray,
    intensity: np.ndarray,
    color_index_bv: np.ndarray,
    spectral_type: np.ndarray,
) -> np.ndarray:
    rgb = np.column_stack((intensity, intensity, intensity)).astype(np.uint8)
    bright = mag_v <= 3.0
    for index in np.flatnonzero(bright):
        color_index = float(color_index_bv[index])
        if np.isfinite(color_index):
            rgb[index] = _rgb_from_bv(color_index)
            continue
        spectral_rgb = _rgb_from_spectral_type(str(spectral_type[index]))
        if spectral_rgb is not None:
            rgb[index] = spectral_rgb
    return rgb


def _project_altaz_points(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        return _project_altaz_points_rectilinear(alt_deg, az_deg, camera, basis)
    if camera.lens_model in FISHEYE_LENS_MODELS:
        return _project_altaz_points_fisheye(alt_deg, az_deg, camera, basis)
    if camera.lens_model in CYLINDRICAL_LENS_MODELS:
        return _project_altaz_points_cylindrical(alt_deg, az_deg, camera, basis)
    raise ValueError(f"Unsupported lens model: {camera.lens_model}")


def _project_altaz_points_rectilinear(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)

    finite_depth = np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z) & (cam_z > 1e-6)
    x_px = np.full_like(cam_z, np.nan, dtype=np.float64)
    y_px = np.full_like(cam_z, np.nan, dtype=np.float64)
    if np.any(finite_depth):
        x_mm = camera.focal_length_mm * cam_x[finite_depth] / cam_z[finite_depth]
        y_mm = camera.focal_length_mm * cam_y[finite_depth] / cam_z[finite_depth]
        x_px[finite_depth] = camera.image_width_px * 0.5 + (x_mm / camera.sensor_width_mm) * camera.image_width_px
        y_px[finite_depth] = camera.image_height_px * 0.5 - (y_mm / camera.sensor_height_mm) * camera.image_height_px

    margin_x = camera.image_width_px * 2.0
    margin_y = camera.image_height_px * 2.0
    valid = (
        finite_depth
        & np.isfinite(x_px)
        & np.isfinite(y_px)
        & (x_px >= -margin_x)
        & (x_px <= camera.image_width_px + margin_x)
        & (y_px >= -margin_y)
        & (y_px <= camera.image_height_px + margin_y)
    )
    return x_px.astype(np.float64), y_px.astype(np.float64), valid


def _fisheye_radius_ratio(theta: np.ndarray, theta_max: float, lens_model: str) -> np.ndarray:
    if theta_max <= 0.0:
        raise ValueError("Fisheye FOV must be positive")

    if lens_model == FISHEYE_EQUIDISTANT:
        return theta / theta_max
    if lens_model == FISHEYE_EQUISOLID:
        denominator = np.sin(theta_max / 2.0)
        if abs(float(denominator)) <= 1e-12:
            raise ValueError("Fisheye FOV is too small")
        return np.sin(theta / 2.0) / denominator
    raise ValueError(f"Unsupported fisheye lens model: {lens_model}")


def _fisheye_theta_from_radius_ratio(rho: np.ndarray, theta_max: float, lens_model: str) -> np.ndarray:
    if lens_model == FISHEYE_EQUIDISTANT:
        return rho * theta_max
    if lens_model == FISHEYE_EQUISOLID:
        return 2.0 * np.arcsin(np.clip(rho * np.sin(theta_max / 2.0), -1.0, 1.0))
    raise ValueError(f"Unsupported fisheye lens model: {lens_model}")


def _project_altaz_points_fisheye(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)

    theta = np.arccos(np.clip(cam_z, -1.0, 1.0))
    theta_max = np.deg2rad(camera.fisheye_fov_deg * 0.5)
    rho = _fisheye_radius_ratio(theta, theta_max, camera.lens_model)
    r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
    r_px = r_limit * rho

    plane_norm = np.hypot(cam_x, cam_y)
    unit_x = np.divide(cam_x, plane_norm, out=np.zeros_like(cam_x), where=plane_norm > 1e-12)
    unit_y = np.divide(cam_y, plane_norm, out=np.zeros_like(cam_y), where=plane_norm > 1e-12)
    x_px = camera.image_width_px * 0.5 + unit_x * r_px
    y_px = camera.image_height_px * 0.5 - unit_y * r_px

    margin = r_limit * 0.1
    valid = (
        (theta <= theta_max + 1e-9)
        & np.isfinite(r_px)
        & np.isfinite(x_px)
        & np.isfinite(y_px)
        & (r_px <= r_limit + margin)
        & (x_px >= -margin)
        & (x_px <= camera.image_width_px + margin)
        & (y_px >= -margin)
        & (y_px <= camera.image_height_px + margin)
    )
    return x_px.astype(np.float64), y_px.astype(np.float64), valid


def _projection_horizontal_scale_px(camera: CameraSettings) -> float:
    fov_deg = max(1.0, min(360.0, float(camera.fisheye_fov_deg)))
    fov_rad = max(np.deg2rad(fov_deg), 1e-6)
    return float(camera.image_width_px) / fov_rad


def _project_altaz_points_cylindrical(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)
    norm = np.sqrt(cam_x * cam_x + cam_y * cam_y + cam_z * cam_z)
    unit_y = np.divide(cam_y, norm, out=np.zeros_like(cam_y), where=norm > 1e-12)
    longitude = np.arctan2(cam_x, cam_z)
    latitude = np.arcsin(np.clip(unit_y, -1.0, 1.0))

    scale_px = _projection_horizontal_scale_px(camera)
    plane_y = latitude
    valid = np.isfinite(longitude) & np.isfinite(latitude) & (norm > 1e-12)
    if camera.lens_model == MERCATOR_LENS_MODEL:
        # 墨卡托在两极发散，渲染时略微避开极点。
        valid &= np.abs(latitude) < (np.pi * 0.5 - 1e-8)
        plane_y = np.arctanh(np.clip(np.sin(latitude), -1.0 + 1e-12, 1.0 - 1e-12))

    x_px = camera.image_width_px * 0.5 + scale_px * longitude
    y_px = camera.image_height_px * 0.5 - scale_px * plane_y
    margin_x = camera.image_width_px * 0.1
    margin_y = camera.image_height_px * 0.1
    valid &= (
        np.isfinite(x_px)
        & np.isfinite(y_px)
        & (x_px >= -margin_x)
        & (x_px <= camera.image_width_px + margin_x)
        & (y_px >= -margin_y)
        & (y_px <= camera.image_height_px + margin_y)
    )
    return x_px.astype(np.float64), y_px.astype(np.float64), valid.astype(bool)


def _alt_from_image_points(
    x_px: np.ndarray,
    y_px: np.ndarray,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = basis
    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        x_mm = (x_px - camera.image_width_px * 0.5) * camera.sensor_width_mm / camera.image_width_px
        y_mm = (camera.image_height_px * 0.5 - y_px) * camera.sensor_height_mm / camera.image_height_px
        cam_x = x_mm / camera.focal_length_mm
        cam_y = y_mm / camera.focal_length_mm
        cam_z = np.ones_like(cam_x, dtype=np.float64)
        valid = np.isfinite(cam_x) & np.isfinite(cam_y)
    elif camera.lens_model in FISHEYE_LENS_MODELS:
        center_x = camera.image_width_px * 0.5
        center_y = camera.image_height_px * 0.5
        screen_x = x_px - center_x
        screen_y = center_y - y_px
        r_px = np.hypot(screen_x, screen_y)
        r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
        rho = np.divide(r_px, r_limit, out=np.full_like(r_px, np.inf), where=r_limit > 1e-12)
        theta_max = np.deg2rad(camera.fisheye_fov_deg * 0.5)
        theta = _fisheye_theta_from_radius_ratio(np.clip(rho, 0.0, 1.0), theta_max, camera.lens_model)
        plane_norm = np.sin(theta)
        unit_x = np.divide(screen_x, r_px, out=np.zeros_like(screen_x), where=r_px > 1e-12)
        unit_y = np.divide(screen_y, r_px, out=np.zeros_like(screen_y), where=r_px > 1e-12)
        cam_x = unit_x * plane_norm
        cam_y = unit_y * plane_norm
        cam_z = np.cos(theta)
        valid = (rho <= 1.0 + 1e-9) & np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z)
    elif camera.lens_model in CYLINDRICAL_LENS_MODELS:
        center_x = camera.image_width_px * 0.5
        center_y = camera.image_height_px * 0.5
        scale_px = _projection_horizontal_scale_px(camera)
        longitude = (x_px - center_x) / max(scale_px, 1e-12)
        plane_y = (center_y - y_px) / max(scale_px, 1e-12)
        if camera.lens_model == MERCATOR_LENS_MODEL:
            latitude = np.arcsin(np.clip(np.tanh(plane_y), -1.0, 1.0))
            valid = np.isfinite(longitude) & np.isfinite(latitude)
        else:
            latitude = plane_y
            valid = np.isfinite(longitude) & np.isfinite(latitude) & (np.abs(latitude) <= np.pi * 0.5 + 1e-8)
        cos_lat = np.cos(latitude)
        cam_x = cos_lat * np.sin(longitude)
        cam_y = np.sin(latitude)
        cam_z = cos_lat * np.cos(longitude)
    else:
        raise ValueError(f"Unsupported lens model: {camera.lens_model}")

    local_x = cam_x * right[0] + cam_y * up[0] + cam_z * forward[0]
    local_y = cam_x * right[1] + cam_y * up[1] + cam_z * forward[1]
    local_z = cam_x * right[2] + cam_y * up[2] + cam_z * forward[2]
    norm = np.sqrt(local_x * local_x + local_y * local_y + local_z * local_z)
    alt_deg = np.rad2deg(np.arcsin(np.divide(local_z, norm, out=np.zeros_like(local_z), where=norm > 1e-12)))
    valid &= norm > 1e-12
    return alt_deg.astype(np.float64), valid


def _build_horizon_shadow_rects(
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[ProjectedFillRect, ...]:
    # 用屏幕网格反算高度角，生成地平线下的深灰遮罩。
    target_cell_px = 20.0
    columns = min(max(int(np.ceil(camera.image_width_px / target_cell_px)), 24), 180)
    rows = min(max(int(np.ceil(camera.image_height_px / target_cell_px)), 16), 140)
    x_edges = np.linspace(0.0, float(camera.image_width_px), columns + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, float(camera.image_height_px), rows + 1, dtype=np.float64)
    x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
    y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    alt_deg, valid = _alt_from_image_points(grid_x.ravel(), grid_y.ravel(), camera=camera, basis=basis)
    below = (valid & (alt_deg < 0.0)).reshape((rows, columns))

    rects: list[ProjectedFillRect] = []
    for row in range(rows):
        column = 0
        while column < columns:
            if not below[row, column]:
                column += 1
                continue
            start_column = column
            while column < columns and below[row, column]:
                column += 1
            rects.append(
                ProjectedFillRect(
                    x_px=float(x_edges[start_column]),
                    y_px=float(y_edges[row]),
                    width_px=float(x_edges[column] - x_edges[start_column]),
                    height_px=float(y_edges[row + 1] - y_edges[row]),
                )
            )
    return tuple(rects)


def _split_projected_line(
    x_px: np.ndarray,
    y_px: np.ndarray,
    valid: np.ndarray,
    kind: str,
    label: str,
) -> list[ProjectedGridLine]:
    lines: list[ProjectedGridLine] = []
    current: list[tuple[float, float]] = []
    for x_value, y_value, is_valid in zip(x_px, y_px, valid):
        if is_valid:
            current.append((float(x_value), float(y_value)))
            continue
        if len(current) >= 2:
            lines.append(ProjectedGridLine(kind=kind, label=label, points=tuple(current)))
        current = []
    if len(current) >= 2:
        lines.append(ProjectedGridLine(kind=kind, label=label, points=tuple(current)))
    return lines


def _clean_polygon_points(points: list[Point2D]) -> list[Point2D]:
    if len(points) < 2:
        return points

    cleaned: list[Point2D] = []
    for point in points:
        is_new_point = not cleaned or abs(cleaned[-1][0] - point[0]) > 1e-6 or abs(cleaned[-1][1] - point[1]) > 1e-6
        if is_new_point:
            cleaned.append(point)
    is_closed = (
        len(cleaned) >= 2
        and abs(cleaned[0][0] - cleaned[-1][0]) <= 1e-6
        and abs(cleaned[0][1] - cleaned[-1][1]) <= 1e-6
    )
    if is_closed:
        cleaned.pop()
    return cleaned


def _clip_polygon_edge(
    points: list[Point2D],
    is_inside: Callable[[Point2D], bool],
    intersect: Callable[[Point2D, Point2D], Point2D],
) -> list[Point2D]:
    if not points:
        return []

    clipped: list[Point2D] = []
    previous = points[-1]
    previous_inside = bool(is_inside(previous))
    for current in points:
        current_inside = bool(is_inside(current))
        if current_inside:
            if not previous_inside:
                clipped.append(intersect(previous, current))
            clipped.append(current)
        elif previous_inside:
            clipped.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return _clean_polygon_points(clipped)


def _polygon_area_px(points: tuple[Point2D, ...]) -> float:
    if len(points) < 3:
        return 0.0
    x_values = np.asarray([point[0] for point in points], dtype=np.float64)
    y_values = np.asarray([point[1] for point in points], dtype=np.float64)
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def _clip_polygon_to_image_rect(
    points: tuple[Point2D, ...],
    width_px: int,
    height_px: int,
) -> tuple[Point2D, ...]:
    # 对银河面片做屏幕空间裁剪，避免越过画面边缘时整块消失。
    clipped = _clean_polygon_points(list(points))
    if len(clipped) < 3:
        return ()

    left = 0.0
    top = 0.0
    right = float(width_px - 1)
    bottom = float(height_px - 1)

    def intersect_vertical(x_edge: float, start: Point2D, end: Point2D) -> Point2D:
        dx = end[0] - start[0]
        if abs(dx) <= 1e-12:
            return x_edge, end[1]
        t = (x_edge - start[0]) / dx
        return x_edge, start[1] + t * (end[1] - start[1])

    def intersect_horizontal(y_edge: float, start: Point2D, end: Point2D) -> Point2D:
        dy = end[1] - start[1]
        if abs(dy) <= 1e-12:
            return end[0], y_edge
        t = (y_edge - start[1]) / dy
        return start[0] + t * (end[0] - start[0]), y_edge

    clipped = _clip_polygon_edge(
        clipped,
        lambda point: point[0] >= left,
        lambda start, end: intersect_vertical(left, start, end),
    )
    clipped = _clip_polygon_edge(
        clipped,
        lambda point: point[0] <= right,
        lambda start, end: intersect_vertical(right, start, end),
    )
    clipped = _clip_polygon_edge(
        clipped,
        lambda point: point[1] >= top,
        lambda start, end: intersect_horizontal(top, start, end),
    )
    clipped = _clip_polygon_edge(
        clipped,
        lambda point: point[1] <= bottom,
        lambda start, end: intersect_horizontal(bottom, start, end),
    )

    result = tuple(clipped)
    if len(result) < 3 or _polygon_area_px(result) <= 1e-4:
        return ()
    return result


def _build_horizontal_grid(
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[ProjectedGridLine, ...], tuple[ProjectedLabel, ...]]:
    grid_lines: list[ProjectedGridLine] = []

    az_samples = np.linspace(0.0, 360.0, 361, dtype=np.float64)
    for alt_value in (-30.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        x_px, y_px, valid = _project_altaz_points(
            alt_deg=np.full_like(az_samples, alt_value),
            az_deg=az_samples,
            camera=camera,
            basis=basis,
        )
        kind = "horizon" if alt_value == 0.0 else "altitude"
        grid_lines.extend(_split_projected_line(x_px, y_px, valid, kind=kind, label=f"{alt_value:g} deg"))

    alt_samples = np.linspace(-30.0, 90.0, 121, dtype=np.float64)
    for az_value in range(0, 360, 30):
        x_px, y_px, valid = _project_altaz_points(
            alt_deg=alt_samples,
            az_deg=np.full_like(alt_samples, float(az_value)),
            camera=camera,
            basis=basis,
        )
        grid_lines.extend(_split_projected_line(x_px, y_px, valid, kind="azimuth", label=f"{az_value:g} deg"))

    labels: list[ProjectedLabel] = []
    direction_marks = (
        (0.0, "北 N"),
        (45.0, "东北 NE"),
        (90.0, "东 E"),
        (135.0, "东南 SE"),
        (180.0, "南 S"),
        (225.0, "西南 SW"),
        (270.0, "西 W"),
        (315.0, "西北 NW"),
    )
    label_alt = np.zeros(len(direction_marks), dtype=np.float64)
    label_az = np.asarray([mark[0] for mark in direction_marks], dtype=np.float64)
    x_px, y_px, valid = _project_altaz_points(label_alt, label_az, camera=camera, basis=basis)
    for index, (_, text) in enumerate(direction_marks):
        if not valid[index]:
            continue
        if 0.0 <= x_px[index] <= camera.image_width_px - 1 and 0.0 <= y_px[index] <= camera.image_height_px - 1:
            labels.append(ProjectedLabel(text=text, x_px=float(x_px[index]), y_px=float(y_px[index])))

    return tuple(grid_lines), tuple(labels)


def _camera_longitudes_from_altaz(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, _cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)
    return np.arctan2(cam_x, cam_z).astype(np.float64)


def _cylindrical_ring_crosses_projection_seam(
    ring: HorizontalMilkyWayRing,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    longitudes = _camera_longitudes_from_altaz(ring.alt_deg, ring.az_deg, basis)
    if longitudes.size < 3 or not np.all(np.isfinite(longitudes)):
        return True

    # 圆柱类投影在相机经度 +/-pi 处有断点。银河环如果跨过该断点，
    # 屏幕空间会出现一条贯穿画面的闭合边，QPainter 填充后就会变成大矩形面片。
    closed_longitudes = np.concatenate((longitudes, longitudes[:1]))
    return bool(np.any(np.abs(np.diff(closed_longitudes)) > np.pi))


def _project_milky_way_polygons(
    horizontal_milky_way: HorizontalMilkyWayCatalog | None,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[ProjectedMilkyWayPolygon, ...]:
    if horizontal_milky_way is None:
        return ()

    projected_polygons: list[ProjectedMilkyWayPolygon] = []
    for polygon in horizontal_milky_way.polygons:
        projected_rings: list[tuple[tuple[float, float], ...]] = []
        for ring in polygon.rings:
            x_px, y_px, valid = _project_altaz_points(
                alt_deg=ring.alt_deg,
                az_deg=ring.az_deg,
                camera=camera,
                basis=basis,
            )
            # 银河使用面填充渲染，必须先保证环在镜头投影上连续有效。
            # 画面外的部分随后会裁剪掉，不能因为碰到图像边界就丢弃整块面片。
            projectable = np.isfinite(x_px) & np.isfinite(y_px)
            if camera.lens_model in FISHEYE_LENS_MODELS:
                projectable &= valid
            if not bool(np.all(projectable)):
                continue
            if camera.lens_model in CYLINDRICAL_LENS_MODELS and _cylindrical_ring_crosses_projection_seam(ring, basis):
                continue
            points = tuple((float(x_value), float(y_value)) for x_value, y_value in zip(x_px, y_px))
            clipped_points = _clip_polygon_to_image_rect(points, camera.image_width_px, camera.image_height_px)
            if len(clipped_points) >= 3:
                projected_rings.append(clipped_points)
        if projected_rings:
            projected_polygons.append(ProjectedMilkyWayPolygon(rings=tuple(projected_rings)))

    return tuple(projected_polygons)


def _project_solar_system_objects(
    horizontal_solar_system: HorizontalSolarSystemCatalog | None,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[ProjectedSolarSystemObject, ...]:
    if horizontal_solar_system is None or len(horizontal_solar_system) == 0:
        return ()

    x_px, y_px, valid_projection = _project_altaz_points(
        horizontal_solar_system.alt_deg,
        horizontal_solar_system.az_deg,
        camera=camera,
        basis=basis,
    )
    inside = (
        valid_projection
        & np.isfinite(x_px)
        & np.isfinite(y_px)
        & (x_px >= 0.0)
        & (x_px <= camera.image_width_px - 1)
        & (y_px >= 0.0)
        & (y_px <= camera.image_height_px - 1)
    )

    projected_objects: list[ProjectedSolarSystemObject] = []
    for index in np.flatnonzero(inside):
        above_horizon = bool(horizontal_solar_system.alt_deg[index] >= 0.0)
        color = (
            int(horizontal_solar_system.color_rgb[index, 0]),
            int(horizontal_solar_system.color_rgb[index, 1]),
            int(horizontal_solar_system.color_rgb[index, 2]),
        )
        projected_objects.append(
            ProjectedSolarSystemObject(
                object_id=str(horizontal_solar_system.object_ids[index]),
                display_name=str(horizontal_solar_system.display_names[index]),
                kernel_name=str(horizontal_solar_system.kernel_names[index]),
                ra_deg=float(horizontal_solar_system.ra_deg[index]),
                dec_deg=float(horizontal_solar_system.dec_deg[index]),
                mag_v=float(horizontal_solar_system.mag_v[index]),
                sim_x=float(x_px[index]),
                sim_y=float(y_px[index]),
                alt_deg=float(horizontal_solar_system.alt_deg[index]),
                az_deg=float(horizontal_solar_system.az_deg[index]),
                radius_px=float(horizontal_solar_system.radius_px[index]),
                color_rgb=color,
                alpha=255 if above_horizon else 128,
                above_horizon=above_horizon,
                reference_allowed=bool(horizontal_solar_system.reference_allowed[index]),
            )
        )
    return tuple(projected_objects)


def project_horizontal_catalog(
    horizontal_catalog: HorizontalStarCatalog,
    camera: CameraSettings,
    view: ViewSettings,
    visible_mag_limit: float = 6.5,
    horizontal_milky_way: HorizontalMilkyWayCatalog | None = None,
    horizontal_solar_system: HorizontalSolarSystemCatalog | None = None,
) -> ProjectedStarMap:
    if camera.sensor_width_mm <= 0 or camera.sensor_height_mm <= 0:
        raise ValueError("Sensor dimensions must be positive")
    if camera.image_width_px <= 0 or camera.image_height_px <= 0:
        raise ValueError("Image dimensions must be positive")
    if camera.focal_length_mm <= 0:
        raise ValueError("Focal length must be positive")
    if camera.lens_model not in SUPPORTED_LENS_MODELS:
        raise ValueError(f"Unsupported lens model: {camera.lens_model}")
    if camera.lens_model in FISHEYE_LENS_MODELS and not (1.0 <= camera.fisheye_fov_deg <= 300.0):
        raise ValueError("Fisheye FOV must be between 1 and 300 degrees")
    if camera.lens_model in CYLINDRICAL_LENS_MODELS and not (1.0 <= camera.fisheye_fov_deg <= 360.0):
        raise ValueError("Cylindrical projection FOV must be between 1 and 360 degrees")

    basis = _camera_basis(view)

    x_px, y_px, valid_projection = _project_altaz_points(
        horizontal_catalog.alt_deg,
        horizontal_catalog.az_deg,
        camera=camera,
        basis=basis,
    )

    inside = (
        valid_projection
        & np.isfinite(x_px)
        & np.isfinite(y_px)
        & (x_px >= 0.0)
        & (x_px <= camera.image_width_px - 1)
        & (y_px >= 0.0)
        & (y_px <= camera.image_height_px - 1)
    )

    radius, intensity = _star_style(horizontal_catalog.mag_v[inside], visible_mag_limit)
    star_rgb = _star_rgb(
        mag_v=horizontal_catalog.mag_v[inside],
        intensity=intensity,
        color_index_bv=horizontal_catalog.color_index_bv[inside],
        spectral_type=horizontal_catalog.spectral_type[inside],
    )
    above_horizon = horizontal_catalog.alt_deg[inside] >= 0.0
    alpha = np.where(above_horizon, 255, 128).astype(np.uint8)
    milky_way_polygons = _project_milky_way_polygons(horizontal_milky_way, camera=camera, basis=basis)
    solar_system_objects = _project_solar_system_objects(horizontal_solar_system, camera=camera, basis=basis)
    grid_lines, direction_labels = _build_horizontal_grid(camera=camera, basis=basis)
    horizon_shadow_rects = _build_horizon_shadow_rects(camera=camera, basis=basis)
    sky_circle_radius_px = None
    if camera.lens_model in FISHEYE_LENS_MODELS:
        sky_circle_radius_px = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5

    return ProjectedStarMap(
        width=camera.image_width_px,
        height=camera.image_height_px,
        source_name=horizontal_catalog.source_name,
        x_px=x_px[inside].astype(np.float64),
        y_px=y_px[inside].astype(np.float64),
        radius_px=radius,
        intensity=intensity,
        alpha=alpha,
        above_horizon=above_horizon,
        star_ids=horizontal_catalog.star_ids[inside],
        display_names=horizontal_catalog.display_names[inside],
        common_names=horizontal_catalog.common_names[inside],
        ra_deg=horizontal_catalog.ra_deg[inside],
        dec_deg=horizontal_catalog.dec_deg[inside],
        alt_deg=horizontal_catalog.alt_deg[inside].astype(np.float64),
        az_deg=horizontal_catalog.az_deg[inside].astype(np.float64),
        mag_v=horizontal_catalog.mag_v[inside],
        color_index_bv=horizontal_catalog.color_index_bv[inside],
        spectral_type=horizontal_catalog.spectral_type[inside],
        star_rgb=star_rgb,
        grid_lines=grid_lines,
        direction_labels=direction_labels,
        catalog_count=len(horizontal_catalog),
        lens_model=camera.lens_model,
        sky_circle_radius_px=sky_circle_radius_px,
        horizon_shadow_rects=horizon_shadow_rects,
        milky_way_polygons=milky_way_polygons,
        solar_system_objects=solar_system_objects,
    )


def project_catalog(
    catalog: StarCatalog,
    observer: ObserverSettings,
    camera: CameraSettings,
    view: ViewSettings,
    visible_mag_limit: float = 6.5,
) -> ProjectedStarMap:
    horizontal_catalog = compute_horizontal_catalog(catalog, observer, visible_mag_limit)
    return project_horizontal_catalog(
        horizontal_catalog=horizontal_catalog,
        camera=camera,
        view=view,
        visible_mag_limit=visible_mag_limit,
    )


def _reference_star_name(star_map: ProjectedStarMap, index: int) -> tuple[str, str, str]:
    display_name = str(star_map.display_names[index]).strip()
    common_name = str(star_map.common_names[index]).strip()
    star_id = str(star_map.star_ids[index]).strip()
    name = common_name or display_name or star_id
    return name, display_name, common_name


def select_reference_stars(
    star_map: ProjectedStarMap,
    max_count: int | None = 12,
    mag_limit: float | None = None,
    edge_margin_ratio: float = 0.05,
    min_distance_ratio: float = 0.06,
) -> tuple[ReferenceStar, ...]:
    if len(star_map) == 0:
        return ()
    if max_count is not None and max_count <= 0:
        max_count = 0

    width = float(star_map.width)
    height = float(star_map.height)
    min_dimension = min(width, height)
    edge_margin = max(18.0, min_dimension * edge_margin_ratio)
    min_distance = max(32.0, min_dimension * min_distance_ratio)

    selected: list[int] = []
    selected_positions: list[tuple[float, float]] = []
    if len(star_map) > 0 and max_count != 0:
        candidate_mask = (
            np.isfinite(star_map.x_px)
            & np.isfinite(star_map.y_px)
            & (star_map.x_px >= edge_margin)
            & (star_map.x_px <= width - edge_margin)
            & (star_map.y_px >= edge_margin)
            & (star_map.y_px <= height - edge_margin)
        )
        if mag_limit is not None:
            candidate_mask &= star_map.mag_v <= float(mag_limit)

        candidate_indices = np.flatnonzero(candidate_mask)
    else:
        candidate_indices = np.asarray([], dtype=np.int64)

    sorted_indices = candidate_indices[np.argsort(star_map.mag_v[candidate_indices], kind="stable")]
    for candidate_index in sorted_indices:
        x_value = float(star_map.x_px[candidate_index])
        y_value = float(star_map.y_px[candidate_index])
        if any(np.hypot(x_value - old_x, y_value - old_y) < min_distance for old_x, old_y in selected_positions):
            continue
        selected.append(int(candidate_index))
        selected_positions.append((x_value, y_value))
        if max_count is not None and len(selected) >= max_count:
            break

    reference_stars: list[ReferenceStar] = []
    for output_index, star_index in enumerate(selected, start=1):
        name, display_name, common_name = _reference_star_name(star_map, star_index)
        reference_stars.append(
            ReferenceStar(
                index=output_index,
                star_id=str(star_map.star_ids[star_index]),
                name=name,
                display_name=display_name,
                common_name=common_name,
                ra_deg=float(star_map.ra_deg[star_index]),
                dec_deg=float(star_map.dec_deg[star_index]),
                mag_v=float(star_map.mag_v[star_index]),
                sim_x=float(star_map.x_px[star_index]),
                sim_y=float(star_map.y_px[star_index]),
                alt_deg=float(star_map.alt_deg[star_index]),
                az_deg=float(star_map.az_deg[star_index]),
            )
        )
    return tuple(reference_stars)
