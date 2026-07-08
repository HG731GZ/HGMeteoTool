from __future__ import annotations

import numpy as np

from .frame_astrometry import FrameAstrometricModel
from .mosaic_model_io import MosaicCoverageCache
from .projection_grid import build_pixel_radec_grid, grid_shape_for_long_side, radec_grid_to_altaz
from .simulator import (
    ObserverSettings,
    compute_altaz_from_radec,
    local_vectors_from_altaz,
)


def build_coverage_cache(
    model: FrameAstrometricModel,
    *,
    grid_precision: int,
    min_minor_cells: int = 3,
) -> MosaicCoverageCache:
    """从源图模型构建像素→天球的覆盖网格缓存。

    参数
    ----
    model : FrameAstrometricModel
        已求解的源图天文测量模型。
    grid_precision : int
        长边网格点数，决定覆盖网格的精细度。
    min_minor_cells : int
        短边最少网格数。

    返回
    ----
    MosaicCoverageCache
        包含像素网格坐标和对应 RA/Dec 的缓存。
    """
    width = max(1, int(model.image_width_px))
    height = max(1, int(model.image_height_px))
    rows, columns = grid_shape_for_long_side(width, height, grid_precision, min_minor_cells=min_minor_cells)
    sky_grid = build_pixel_radec_grid(model, width, height, rows, columns)
    return MosaicCoverageCache(
        grid_rows=rows,
        grid_columns=columns,
        grid_x_px=sky_grid.pixel_grid.x_px,
        grid_y_px=sky_grid.pixel_grid.y_px,
        ra_deg=sky_grid.first_deg,
        dec_deg=sky_grid.second_deg,
        valid=sky_grid.valid,
    )


def coverage_altaz(
    cache: MosaicCoverageCache,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将覆盖缓存的 RA/Dec 网格转换为当前观测者的地平坐标。

    返回 (alt_deg, az_deg, valid) 三个同形数组。
    """
    return radec_grid_to_altaz(cache.ra_deg, cache.dec_deg, cache.valid, observer)


def compute_center_from_model(
    model: FrameAstrometricModel,
    cache: MosaicCoverageCache,
    observer: ObserverSettings,
    image_width_px: int,
    image_height_px: int,
) -> tuple[float, float]:
    """从源图模型计算初始视图中心（az_deg, alt_deg）。

    优先使用图像中心的像素反解；若失败则回退到覆盖网格的几何中心。
    """
    center_point = np.asarray(
        [[image_width_px * 0.5, image_height_px * 0.5]],
        dtype=np.float64,
    )
    center_radec = model.pixel_to_sky_points(center_point)
    if np.all(np.isfinite(center_radec)):
        alt_deg, az_deg = compute_altaz_from_radec(
            center_radec[:, 0], center_radec[:, 1], observer,
        )
        return float(az_deg[0]) % 360.0, float(alt_deg[0])

    # 回退：使用覆盖网格中有效点的平均方向
    cache_alt, cache_az, cache_valid = coverage_altaz(cache, observer)
    valid_mask = cache_valid & np.isfinite(cache_alt) & np.isfinite(cache_az)
    if np.any(valid_mask):
        vectors = local_vectors_from_altaz(cache_alt[valid_mask], cache_az[valid_mask])
        mean_vector = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean_vector))
        if norm > 1e-12:
            mean_vector = mean_vector / norm
            alt = float(np.rad2deg(np.arcsin(np.clip(mean_vector[2], -1.0, 1.0))))
            az = float(np.rad2deg(np.arctan2(mean_vector[0], mean_vector[1]))) % 360.0
            return az, alt
    return 0.0, 20.0


def suggest_fov_from_coverage(
    cache: MosaicCoverageCache,
    observer: ObserverSettings,
    center_az_deg: float,
    center_alt_deg: float,
    *,
    min_fov_deg: float = 25.0,
    max_fov_deg: float = 360.0,
    percentile: float = 98.0,
    scale: float = 2.35,
) -> float:
    """根据覆盖网格的天球张角建议合适的初始 FOV。

    计算覆盖网格各点到视图中心的天球角距，取指定分位数后乘以缩放系数。
    """
    cache_alt, cache_az, cache_valid = coverage_altaz(cache, observer)
    valid = cache_valid & np.isfinite(cache_alt) & np.isfinite(cache_az)
    if not np.any(valid):
        return min_fov_deg

    center_vector = local_vectors_from_altaz(
        np.asarray([center_alt_deg], dtype=np.float64),
        np.asarray([center_az_deg], dtype=np.float64),
    )[0]
    vectors = local_vectors_from_altaz(cache_alt[valid], cache_az[valid])
    dots = np.sum(vectors * center_vector[None, :], axis=1)
    angles_deg = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    if angles_deg.size == 0:
        return min_fov_deg

    suggested = float(np.percentile(angles_deg, percentile) * scale)
    return max(min_fov_deg, min(max_fov_deg, suggested))


__all__ = [
    "build_coverage_cache",
    "compute_center_from_model",
    "coverage_altaz",
    "suggest_fov_from_coverage",
]
