from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import StarMapUiConfig
from .simulator import (
    CYLINDRICAL_LENS_MODELS,
    FISHEYE_LENS_MODELS,
    RECTILINEAR_LENS_MODEL,
    CameraSettings,
    ObserverSettings,
    ProjectedStarMap,
    ViewSettings,
    _camera_basis,
    _project_altaz_points,
    compute_altaz_from_radec,
    local_vectors_from_altaz,
)


MAXIMUM_METEOR_COUNT_MULTIPLIER = 10.0


@dataclass(frozen=True)
class MeteorShowerSpec:
    """流星雨的年度活动参数，辐射点坐标取活动峰值附近。"""

    shower_id: str
    chinese_name: str
    english_name: str
    activity_start: tuple[int, int]
    activity_end: tuple[int, int]
    peak_date: tuple[int, int]
    radiant_ra_deg: float
    radiant_dec_deg: float
    zhr: int | None
    color_hex: str = "#FFFFFF"
    color_note: str = "未查到可靠的固定颜色资料，使用白色"

    @property
    def display_name(self) -> str:
        return f"{self.chinese_name}（{self.english_name}）"


# 活动期、峰值辐射点与 ZHR 来自 IMO《2026 Meteor Shower Calendar》表 5。
# 颜色只在可靠资料明确描述典型色彩时填写，其余严格按白色处理。
METEOR_SHOWER_SPECS: tuple[MeteorShowerSpec, ...] = (
    MeteorShowerSpec("QUA", "象限仪座流星雨", "Quadrantids", (12, 28), (1, 12), (1, 3), 230, 49, 80),
    MeteorShowerSpec("GUM", "小熊座γ流星雨", "Gamma Ursae Minorids", (1, 10), (1, 22), (1, 18), 228, 67, 3),
    MeteorShowerSpec("ACE", "半人马座α流星雨", "Alpha Centaurids", (1, 31), (2, 20), (2, 8), 211, -58, 6),
    MeteorShowerSpec("LYR", "天琴座流星雨", "April Lyrids", (4, 14), (4, 30), (4, 22), 271, 34, 18),
    MeteorShowerSpec("PPU", "船尾座π流星雨", "Pi Puppids", (4, 15), (4, 28), (4, 24), 110, -45, None),
    MeteorShowerSpec("ETA", "宝瓶座η流星雨", "Eta Aquariids", (4, 19), (5, 28), (5, 6), 338, -1, 50),
    MeteorShowerSpec("ELY", "天琴座η流星雨", "Eta Lyrids", (5, 3), (5, 14), (5, 11), 291, 43, 3),
    MeteorShowerSpec("ARI", "白昼白羊座流星雨", "Daytime Arietids", (5, 14), (6, 24), (6, 7), 43, 24, 30),
    MeteorShowerSpec("JBO", "六月牧夫座流星雨", "June Bootids", (6, 22), (7, 2), (6, 22), 221, 48, None),
    MeteorShowerSpec("JPE", "七月飞马座流星雨", "July Pegasids", (7, 1), (7, 20), (7, 10), 347, 11, 3),
    MeteorShowerSpec("GDR", "七月天龙座γ流星雨", "July Gamma Draconids", (7, 25), (7, 31), (7, 28), 280, 51, 5),
    MeteorShowerSpec("SDA", "南宝瓶座δ流星雨", "Southern Delta Aquariids", (7, 12), (8, 23), (7, 31), 340, -16, 25),
    MeteorShowerSpec("CAP", "摩羯座α流星雨", "Alpha Capricornids", (7, 3), (8, 15), (7, 31), 307, -10, 5),
    MeteorShowerSpec("ERI", "波江座η流星雨", "Eta Eridanids", (7, 31), (8, 19), (8, 7), 41, -11, 3),
    MeteorShowerSpec(
        "PER",
        "英仙座流星雨",
        "Perseids",
        (7, 17),
        (8, 24),
        (8, 13),
        48,
        58,
        100,
        "#85FFB0",
        "NASA 资料指出明亮英仙座流星常见绿色初始辉光",
    ),
    MeteorShowerSpec("KCG", "天鹅座κ流星雨", "Kappa Cygnids", (8, 3), (8, 28), (8, 17), 286, 59, 3),
    MeteorShowerSpec("AUR", "御夫座流星雨", "Aurigids", (8, 28), (9, 5), (9, 1), 91, 39, 6),
    MeteorShowerSpec("SPE", "九月英仙座ε流星雨", "September Epsilon Perseids", (9, 5), (9, 21), (9, 9), 48, 40, 8),
    MeteorShowerSpec("SLY", "九月天猫座流星雨", "September Lyncids", (9, 10), (10, 8), (9, 13), 113, 56, 3),
    MeteorShowerSpec("DSX", "白昼六分仪座流星雨", "Daytime Sextantids", (9, 20), (10, 6), (10, 1), 156, -2, 5),
    MeteorShowerSpec("OCT", "十月鹿豹座流星雨", "October Camelopardalids", (10, 5), (10, 6), (10, 6), 164, 79, 5),
    MeteorShowerSpec("DRA", "天龙座流星雨", "Draconids", (10, 6), (10, 10), (10, 9), 262, 54, 5),
    MeteorShowerSpec("EGE", "双子座ε流星雨", "Epsilon Geminids", (10, 14), (10, 27), (10, 18), 102, 27, 3),
    MeteorShowerSpec(
        "ORI",
        "猎户座流星雨",
        "Orionids",
        (10, 2),
        (11, 7),
        (10, 21),
        95,
        16,
        20,
        "#E5FF85",
        "NASA/JPL 资料将猎户座流星描述为黄色和绿色",
    ),
    MeteorShowerSpec("LMI", "小狮座流星雨", "Leonis Minorids", (10, 19), (10, 27), (10, 24), 162, 37, 2),
    MeteorShowerSpec("STA", "南金牛座流星雨", "Southern Taurids", (9, 20), (11, 20), (11, 5), 52, 15, 7),
    MeteorShowerSpec("NTA", "北金牛座流星雨", "Northern Taurids", (10, 20), (12, 10), (11, 12), 58, 22, 5),
    MeteorShowerSpec("LEO", "狮子座流星雨", "Leonids", (11, 6), (11, 30), (11, 17), 152, 22, 15),
    MeteorShowerSpec("AMO", "麒麟座α流星雨", "Alpha Monocerotids", (11, 15), (11, 25), (11, 22), 117, 1, None),
    MeteorShowerSpec("NOO", "十一月猎户座流星雨", "November Orionids", (11, 13), (12, 6), (11, 28), 91, 16, 3),
    MeteorShowerSpec("PHO", "凤凰座流星雨", "Phoenicids", (12, 1), (12, 5), (12, 2), 8, -27, None),
    MeteorShowerSpec("PUP", "船尾-船帆座流星雨", "Puppid-Velids", (12, 1), (12, 15), (12, 7), 123, -45, 10),
    MeteorShowerSpec("MON", "麒麟座流星雨", "Monocerotids", (12, 1), (12, 19), (12, 9), 100, 8, 3),
    MeteorShowerSpec("HYD", "长蛇座σ流星雨", "Sigma Hydrids", (12, 3), (12, 20), (12, 9), 125, 2, 7),
    MeteorShowerSpec("GEM", "双子座流星雨", "Geminids", (12, 4), (12, 20), (12, 14), 112, 33, 150),
    MeteorShowerSpec("COM", "后发座流星雨", "Comae Berenicids", (12, 4), (1, 30), (12, 23), 164, 29, 3),
    MeteorShowerSpec("URS", "小熊座流星雨", "Ursids", (12, 17), (12, 26), (12, 22), 217, 76, 10),
)

METEOR_SHOWER_BY_ID = {spec.shower_id: spec for spec in METEOR_SHOWER_SPECS}


@dataclass(frozen=True)
class ProjectedMeteor:
    """一颗已投影流星的采样点及显示属性。"""

    shower_id: str
    color_hex: str
    brightness: float
    points: tuple[tuple[float, float, float, bool], ...]


@dataclass(frozen=True)
class ProjectedMeteorRadiant:
    """一个已投影到画面的流星雨辐射点。"""

    shower_id: str
    label: str
    x_px: float
    y_px: float


@dataclass(frozen=True)
class MeteorSkyPool:
    """按 10× 数量预生成的天球流星主池。"""

    shower_ids: tuple[str, ...]
    color_hexes: tuple[str, ...]
    brightness: np.ndarray
    zhr: np.ndarray
    rank_in_shower: np.ndarray
    start_equatorial_vectors: np.ndarray
    end_equatorial_vectors: np.ndarray
    angular_length_deg: np.ndarray

    def __len__(self) -> int:
        return len(self.shower_ids)


@dataclass(frozen=True)
class MeteorHorizontalPool:
    """主池在指定观测时间与地点下的地平坐标向量缓存。"""

    sky_pool: MeteorSkyPool
    start_local_vectors: np.ndarray
    tangent_local_vectors: np.ndarray
    angular_length_rad: np.ndarray


def projected_meteors(star_map: ProjectedStarMap, config: StarMapUiConfig) -> tuple[ProjectedMeteor, ...]:
    """根据星图元数据和配置生成可缓存、可复现的流星轨迹。"""

    if not config.show_meteor_showers or not config.selected_meteor_shower_ids:
        return ()
    if star_map.observer is None or star_map.camera is None or star_map.view is None:
        return ()
    selected_ids = tuple(str(shower_id) for shower_id in config.selected_meteor_shower_ids)
    return _projected_meteors_cached(
        star_map.observer,
        star_map.camera,
        star_map.view,
        selected_ids,
        config.meteor_count_multiplier,
        config.meteor_min_length_deg,
        config.meteor_max_length_deg,
        config.meteor_random_seed,
    )


def projected_meteor_radiants(
    star_map: ProjectedStarMap,
    config: StarMapUiConfig,
) -> tuple[ProjectedMeteorRadiant, ...]:
    """把当前所选流星雨的辐射点投影到可见画面。"""

    if not config.show_meteor_showers or not config.selected_meteor_shower_ids:
        return ()
    if star_map.observer is None or star_map.camera is None or star_map.view is None:
        return ()
    selected_ids = tuple(dict.fromkeys(str(value) for value in config.selected_meteor_shower_ids))
    return _projected_meteor_radiants_cached(
        star_map.observer,
        star_map.camera,
        star_map.view,
        selected_ids,
    )


@lru_cache(maxsize=6)
def _projected_meteor_radiants_cached(
    observer: ObserverSettings,
    camera: CameraSettings,
    view: ViewSettings,
    selected_ids: tuple[str, ...],
) -> tuple[ProjectedMeteorRadiant, ...]:
    """批量转换并缓存所选辐射点在当前取景画面中的位置。"""

    specs = tuple(
        spec
        for shower_id in selected_ids
        if (spec := METEOR_SHOWER_BY_ID.get(shower_id)) is not None
    )
    if not specs:
        return ()
    ra_deg = np.asarray([spec.radiant_ra_deg for spec in specs], dtype=np.float64)
    dec_deg = np.asarray([spec.radiant_dec_deg for spec in specs], dtype=np.float64)
    alt_deg, az_deg = compute_altaz_from_radec(ra_deg, dec_deg, observer)
    x_px, y_px, inside_projection = _project_altaz_points(
        alt_deg,
        az_deg,
        camera=camera,
        basis=_camera_basis(view),
    )
    visible = (
        (alt_deg >= 0.0)
        & inside_projection
        & np.isfinite(x_px)
        & np.isfinite(y_px)
        & (x_px >= 0.0)
        & (x_px <= float(camera.image_width_px))
        & (y_px >= 0.0)
        & (y_px <= float(camera.image_height_px))
    )
    return tuple(
        ProjectedMeteorRadiant(
            shower_id=spec.shower_id,
            label=spec.chinese_name,
            x_px=float(x_value),
            y_px=float(y_value),
        )
        for spec, x_value, y_value, is_visible in zip(specs, x_px, y_px, visible)
        if bool(is_visible)
    )


@lru_cache(maxsize=16)
def _meteor_sky_pool_cached(
    selected_ids: tuple[str, ...],
    minimum_length_deg: float,
    maximum_length_deg: float,
    random_seed: int,
) -> MeteorSkyPool:
    """为每个已选流星雨生成固定顺序的 10× 主池。"""

    shower_ids: list[str] = []
    color_hexes: list[str] = []
    brightness_parts: list[np.ndarray] = []
    zhr_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    start_vector_parts: list[np.ndarray] = []
    end_vector_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []

    for shower_id in selected_ids:
        spec = METEOR_SHOWER_BY_ID.get(shower_id)
        if spec is None or spec.zhr is None:
            continue
        maximum_count = int(round(float(spec.zhr) * MAXIMUM_METEOR_COUNT_MULTIPLIER))
        if maximum_count <= 0:
            continue
        digest = hashlib.sha256(f"{random_seed}:{shower_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))

        radiant_ra = np.deg2rad(spec.radiant_ra_deg)
        radiant_dec = np.deg2rad(spec.radiant_dec_deg)
        radiant = np.asarray(
            [
                np.cos(radiant_dec) * np.cos(radiant_ra),
                np.cos(radiant_dec) * np.sin(radiant_ra),
                np.sin(radiant_dec),
            ],
            dtype=np.float64,
        )
        east = np.asarray([-np.sin(radiant_ra), np.cos(radiant_ra), 0.0], dtype=np.float64)
        north = np.cross(radiant, east)
        bearings = rng.uniform(0.0, np.pi * 2.0, maximum_count)
        tangents = np.cos(bearings)[:, None] * east + np.sin(bearings)[:, None] * north

        max_start_distance = max(4.0, min(120.0, 177.0 - minimum_length_deg))
        cosine_distances = rng.uniform(
            np.cos(np.deg2rad(max_start_distance)),
            np.cos(np.deg2rad(3.0)),
            maximum_count,
        )
        start_distances_deg = np.rad2deg(np.arccos(np.clip(cosine_distances, -1.0, 1.0)))
        distance_factors = np.clip((start_distances_deg - 3.0) / 87.0, 0.0, 1.0)
        limited_maximums = minimum_length_deg + (maximum_length_deg - minimum_length_deg) * distance_factors
        limited_maximums = np.minimum(limited_maximums, 178.0 - start_distances_deg)
        lower_bounds = np.minimum(minimum_length_deg, limited_maximums)
        lengths_deg = lower_bounds + rng.random(maximum_count) * np.maximum(limited_maximums - lower_bounds, 0.0)

        start_distances = np.deg2rad(start_distances_deg)
        end_distances = np.deg2rad(start_distances_deg + lengths_deg)
        start_vectors = np.cos(start_distances)[:, None] * radiant + np.sin(start_distances)[:, None] * tangents
        end_vectors = np.cos(end_distances)[:, None] * radiant + np.sin(end_distances)[:, None] * tangents

        shower_ids.extend([spec.shower_id] * maximum_count)
        color_hexes.extend([spec.color_hex] * maximum_count)
        brightness_parts.append(rng.uniform(0.5, 1.0, maximum_count).astype(np.float64))
        zhr_parts.append(np.full(maximum_count, spec.zhr, dtype=np.int32))
        rank_parts.append(np.arange(maximum_count, dtype=np.int32))
        start_vector_parts.append(start_vectors.astype(np.float64))
        end_vector_parts.append(end_vectors.astype(np.float64))
        length_parts.append(lengths_deg.astype(np.float64))

    if not shower_ids:
        empty_float = np.empty(0, dtype=np.float64)
        return MeteorSkyPool(
            shower_ids=(),
            color_hexes=(),
            brightness=empty_float,
            zhr=np.empty(0, dtype=np.int32),
            rank_in_shower=np.empty(0, dtype=np.int32),
            start_equatorial_vectors=np.empty((0, 3), dtype=np.float64),
            end_equatorial_vectors=np.empty((0, 3), dtype=np.float64),
            angular_length_deg=empty_float,
        )
    return MeteorSkyPool(
        shower_ids=tuple(shower_ids),
        color_hexes=tuple(color_hexes),
        brightness=np.concatenate(brightness_parts),
        zhr=np.concatenate(zhr_parts),
        rank_in_shower=np.concatenate(rank_parts),
        start_equatorial_vectors=np.vstack(start_vector_parts),
        end_equatorial_vectors=np.vstack(end_vector_parts),
        angular_length_deg=np.concatenate(length_parts),
    )


def _radec_from_unit_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把赤道直角坐标单位向量批量转换为 RA/Dec。"""

    ra_deg = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360.0
    dec_deg = np.rad2deg(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)))
    return ra_deg.astype(np.float64), dec_deg.astype(np.float64)


@lru_cache(maxsize=16)
def _meteor_horizontal_pool_cached(
    observer: ObserverSettings,
    selected_ids: tuple[str, ...],
    minimum_length_deg: float,
    maximum_length_deg: float,
    random_seed: int,
) -> MeteorHorizontalPool:
    """一次性批量转换整个 10× 主池，取景角度变化时直接复用。"""

    sky_pool = _meteor_sky_pool_cached(
        selected_ids,
        minimum_length_deg,
        maximum_length_deg,
        random_seed,
    )
    if len(sky_pool) == 0:
        return MeteorHorizontalPool(
            sky_pool=sky_pool,
            start_local_vectors=np.empty((0, 3), dtype=np.float64),
            tangent_local_vectors=np.empty((0, 3), dtype=np.float64),
            angular_length_rad=np.empty(0, dtype=np.float64),
        )

    all_equatorial = np.vstack((sky_pool.start_equatorial_vectors, sky_pool.end_equatorial_vectors))
    ra_deg, dec_deg = _radec_from_unit_vectors(all_equatorial)
    alt_deg, az_deg = compute_altaz_from_radec(ra_deg, dec_deg, observer)
    local_vectors = local_vectors_from_altaz(alt_deg, az_deg)
    meteor_count = len(sky_pool)
    start_local = local_vectors[:meteor_count]
    end_local = local_vectors[meteor_count:]
    cos_lengths = np.clip(np.einsum("ij,ij->i", start_local, end_local), -1.0, 1.0)
    angular_lengths = np.arccos(cos_lengths)
    tangent_local = end_local - cos_lengths[:, None] * start_local
    tangent_norms = np.linalg.norm(tangent_local, axis=1)
    tangent_local = np.divide(
        tangent_local,
        tangent_norms[:, None],
        out=np.zeros_like(tangent_local),
        where=tangent_norms[:, None] > 1e-12,
    )
    return MeteorHorizontalPool(
        sky_pool=sky_pool,
        start_local_vectors=start_local.astype(np.float64),
        tangent_local_vectors=tangent_local.astype(np.float64),
        angular_length_rad=angular_lengths.astype(np.float64),
    )


def _adaptive_segment_counts(length_deg: np.ndarray, camera: CameraSettings) -> np.ndarray:
    """按投影曲率风险与天球长度选择 4 到 24 个分段，并确保包含 3/4 位置。"""

    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        desired = np.full(length_deg.shape, 4, dtype=np.int32)
    elif camera.lens_model in FISHEYE_LENS_MODELS:
        desired = np.ceil(np.maximum(length_deg, 1.0) / 7.5).astype(np.int32)
    else:
        desired = np.ceil(np.maximum(length_deg, 1.0) / 6.0).astype(np.int32)
    desired = np.clip(desired, 4, 24)
    return (np.ceil(desired / 4.0).astype(np.int32) * 4).astype(np.int32)


def _track_may_intersect_image(
    x_px: np.ndarray,
    y_px: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> bool:
    """用连续段包围盒快速排除完全位于画面外的轨迹。"""

    start = 0
    for index in range(len(valid) + 1):
        if index < len(valid) and valid[index]:
            continue
        if index - start >= 2:
            run_x = x_px[start:index]
            run_y = y_px[start:index]
            if (
                float(np.max(run_x)) >= 0.0
                and float(np.min(run_x)) <= width
                and float(np.max(run_y)) >= 0.0
                and float(np.min(run_y)) <= height
            ):
                return True
        start = index + 1
    return False


@lru_cache(maxsize=6)
def _projected_meteors_cached(
    observer: ObserverSettings,
    camera: CameraSettings,
    view: ViewSettings,
    selected_ids: tuple[str, ...],
    count_multiplier: float,
    minimum_length_deg: float,
    maximum_length_deg: float,
    random_seed: int,
) -> tuple[ProjectedMeteor, ...]:
    """从 10× 地平主池取倍率前缀，并批量投影到当前取景画面。"""

    horizontal_pool = _meteor_horizontal_pool_cached(
        observer,
        selected_ids,
        minimum_length_deg,
        maximum_length_deg,
        random_seed,
    )
    sky_pool = horizontal_pool.sky_pool
    if len(sky_pool) == 0:
        return ()
    safe_multiplier = min(max(float(count_multiplier), 0.0), MAXIMUM_METEOR_COUNT_MULTIPLIER)
    count_limits = np.rint(sky_pool.zhr.astype(np.float64) * safe_multiplier).astype(np.int32)
    selected_indexes = np.flatnonzero(sky_pool.rank_in_shower < count_limits)
    if selected_indexes.size == 0:
        return ()

    basis = _camera_basis(view)
    segment_counts = _adaptive_segment_counts(sky_pool.angular_length_deg[selected_indexes], camera)
    point_counts = segment_counts + 1
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(point_counts, dtype=np.int64)))
    repeated_indexes = np.repeat(selected_indexes, point_counts)
    repeated_segments = np.repeat(segment_counts, point_counts)
    repeated_offsets = np.repeat(offsets[:-1], point_counts)
    local_point_indexes = np.arange(int(offsets[-1]), dtype=np.int64) - repeated_offsets
    progress = local_point_indexes.astype(np.float64) / repeated_segments.astype(np.float64)
    angles = progress * horizontal_pool.angular_length_rad[repeated_indexes]
    local_vectors = (
        np.cos(angles)[:, None] * horizontal_pool.start_local_vectors[repeated_indexes]
        + np.sin(angles)[:, None] * horizontal_pool.tangent_local_vectors[repeated_indexes]
    )
    alt_deg = np.rad2deg(np.arcsin(np.clip(local_vectors[:, 2], -1.0, 1.0)))
    az_deg = np.rad2deg(np.arctan2(local_vectors[:, 0], local_vectors[:, 1])) % 360.0
    x_px, y_px, _inside_projection = _project_altaz_points(alt_deg, az_deg, camera=camera, basis=basis)
    valid = (alt_deg >= 0.0) & np.isfinite(x_px) & np.isfinite(y_px)

    if camera.lens_model in CYLINDRICAL_LENS_MODELS and len(x_px) > 1:
        same_meteor = repeated_indexes[1:] == repeated_indexes[:-1]
        jumps = same_meteor & (np.abs(np.diff(x_px)) > camera.image_width_px * 0.5)
        jump_indexes = np.flatnonzero(jumps)
        valid[jump_indexes] = False
        valid[jump_indexes + 1] = False
    x_px = np.clip(x_px, -4.0 * camera.image_width_px, 5.0 * camera.image_width_px)
    y_px = np.clip(y_px, -4.0 * camera.image_height_px, 5.0 * camera.image_height_px)

    projected: list[ProjectedMeteor] = []
    for local_index, master_index in enumerate(selected_indexes):
        start = int(offsets[local_index])
        end = int(offsets[local_index + 1])
        meteor_valid = valid[start:end]
        if np.count_nonzero(meteor_valid) < 2:
            continue
        meteor_x = x_px[start:end]
        meteor_y = y_px[start:end]
        if not _track_may_intersect_image(
            meteor_x,
            meteor_y,
            meteor_valid,
            camera.image_width_px,
            camera.image_height_px,
        ):
            continue
        points = tuple(
            (float(x_value), float(y_value), float(t_value), bool(is_valid))
            for x_value, y_value, t_value, is_valid in zip(
                meteor_x,
                meteor_y,
                progress[start:end],
                meteor_valid,
            )
        )
        projected.append(
            ProjectedMeteor(
                shower_id=sky_pool.shower_ids[int(master_index)],
                color_hex=sky_pool.color_hexes[int(master_index)],
                brightness=float(sky_pool.brightness[int(master_index)]),
                points=points,
            )
        )
    return tuple(projected)


__all__ = [
    "METEOR_SHOWER_BY_ID",
    "METEOR_SHOWER_SPECS",
    "MAXIMUM_METEOR_COUNT_MULTIPLIER",
    "MeteorHorizontalPool",
    "MeteorSkyPool",
    "MeteorShowerSpec",
    "ProjectedMeteor",
    "ProjectedMeteorRadiant",
    "projected_meteor_radiants",
    "projected_meteors",
]
