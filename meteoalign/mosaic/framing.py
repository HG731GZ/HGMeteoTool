from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..alignment.constants import (
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from ..coordinates import normalize_vector, radec_to_unit_vectors


MOSAIC_FRAMING_SCHEMA = "hgmeteo_free_projection_framing"
MOSAIC_FRAMING_VERSION = 1
MOSAIC_RESOLUTION_METHOD = "ptgui_center_jacobian"
MOSAIC_RESOLUTION_SOURCE_STEP_PX = 1.0


def _ceil_output_dimension_px(value: float) -> int:
    # 避免理论整数尺寸因浮点误差被 ceil 多加 1 像素。
    epsilon = max(1e-9, abs(float(value)) * 1e-12)
    return int(math.ceil(max(1.0, float(value) - epsilon)))


@dataclass(frozen=True)
class MosaicResolutionEstimate:
    """自由投影取景的最优输出边界。"""

    boundary_width_px: int
    boundary_height_px: int
    center_angular_resolution_rad_per_px: float
    target_center_px_per_rad: float
    source_center_x_px: float
    source_center_y_px: float
    source_jacobian_rad_per_px: tuple[tuple[float, float], tuple[float, float]]
    viewport_width_px: int
    viewport_height_px: int
    height_over_width: float


def _center_tangent_basis(center_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = normalize_vector(center_vector)
    celestial_north = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(celestial_north, center)
    if float(np.linalg.norm(east)) <= 1e-8:
        east = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    east = normalize_vector(east)
    north = normalize_vector(np.cross(center, east))
    return center, east, north


def _vectors_to_center_tangent_radians(vectors: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """把中心附近的天球方向映射到以弧度计的局部等距切平面。"""

    direction = np.asarray(vectors, dtype=np.float64)
    center_component = np.clip(direction @ center, -1.0, 1.0)
    theta = np.arccos(center_component)
    sin_theta = np.sin(theta)
    scale = np.divide(theta, sin_theta, out=np.ones_like(theta), where=np.abs(sin_theta) > 1e-12)
    tangent = np.column_stack((direction @ east, direction @ north)) * scale[:, None]
    tangent[~np.all(np.isfinite(tangent), axis=1)] = np.nan
    return tangent.astype(np.float64)


def _finite_diff_step(center_px: float, max_px: int) -> float:
    edge_room = min(float(center_px), max(float(max_px - 1) - float(center_px), 0.0))
    if edge_room <= 0.0:
        raise ValueError("源图尺寸过小，无法在中心计算局部 Jacobian。")
    return min(MOSAIC_RESOLUTION_SOURCE_STEP_PX, edge_room)


def source_center_angular_resolution_rad_per_px(
    model: object,
    *,
    image_width_px: int,
    image_height_px: int,
) -> tuple[float, np.ndarray, float, float]:
    """在源图中心用有限差分估计 pixel_to_sky 的局部角分辨率。"""

    safe_width = int(image_width_px)
    safe_height = int(image_height_px)
    if safe_width < 2 or safe_height < 2:
        raise ValueError("源图尺寸至少需要 2 x 2 像素才能计算最优分辨率。")

    center_x = (safe_width - 1) * 0.5
    center_y = (safe_height - 1) * 0.5
    step_x = _finite_diff_step(center_x, safe_width)
    step_y = _finite_diff_step(center_y, safe_height)
    points = np.asarray(
        [
            [center_x, center_y],
            [center_x + step_x, center_y],
            [center_x - step_x, center_y],
            [center_x, center_y + step_y],
            [center_x, center_y - step_y],
        ],
        dtype=np.float64,
    )

    radec = np.asarray(model.pixel_to_sky_points(points), dtype=np.float64)
    if radec.shape != (5, 2) or not np.all(np.isfinite(radec)):
        raise ValueError("源图中心附近无法稳定反解天球坐标，不能计算最优分辨率。")

    vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])
    center, east, north = _center_tangent_basis(vectors[0])
    tangent = _vectors_to_center_tangent_radians(vectors, center, east, north)
    if not np.all(np.isfinite(tangent)):
        raise ValueError("源图中心局部切平面包含无效数值，不能计算最优分辨率。")

    derivative_x = (tangent[1] - tangent[2]) / (2.0 * step_x)
    derivative_y = (tangent[3] - tangent[4]) / (2.0 * step_y)
    jacobian = np.column_stack((derivative_x, derivative_y)).astype(np.float64)
    determinant = float(np.linalg.det(jacobian))
    if np.isfinite(determinant) and abs(determinant) > 1e-24:
        resolution = math.sqrt(abs(determinant))
    else:
        column_norms = np.linalg.norm(jacobian, axis=0)
        valid_norms = column_norms[np.isfinite(column_norms) & (column_norms > 1e-12)]
        if valid_norms.size == 0:
            raise ValueError("源图中心局部 Jacobian 退化，不能计算最优分辨率。")
        resolution = float(np.exp(np.mean(np.log(valid_norms))))

    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("源图中心角分辨率无效。")
    return float(resolution), jacobian, float(center_x), float(center_y)


def target_output_dimensions_for_resolution(
    *,
    angular_resolution_rad_per_px: float,
    projection_model: str,
    fov_deg: float,
    height_over_width: float,
) -> tuple[int, int, float]:
    """按目标投影中心相同角分辨率计算完整取景边界 W/H。"""

    resolution = float(angular_resolution_rad_per_px)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("角分辨率必须是正数。")
    px_per_rad = 1.0 / resolution
    aspect = max(float(height_over_width), 1e-9)
    fov_rad = math.radians(max(1.0, min(360.0, float(fov_deg))))

    if projection_model == SKY_MATCHING_MODEL_RECTILINEAR:
        half_fov = min(fov_rad * 0.5, math.radians(89.999))
        boundary_width = 2.0 * px_per_rad * math.tan(half_fov)
        boundary_height = boundary_width * aspect
    elif projection_model in (SKY_MATCHING_MODEL_MERCATOR, SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT):
        boundary_width = px_per_rad * fov_rad
        boundary_height = boundary_width * aspect
    elif projection_model in (SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID):
        theta_max = fov_rad * 0.5
        if projection_model == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT:
            radius_px = px_per_rad * theta_max
        else:
            radius_px = px_per_rad * 2.0 * math.sin(theta_max * 0.5)
        min_side = max(1.0, radius_px * 2.0 + 1.0)
        if aspect <= 1.0:
            boundary_height = min_side
            boundary_width = boundary_height / aspect
        else:
            boundary_width = min_side
            boundary_height = boundary_width * aspect
    else:
        raise ValueError(f"不支持的目标投影模型：{projection_model}")

    if not all(np.isfinite(value) and value > 0.0 for value in (boundary_width, boundary_height)):
        raise ValueError("最优输出边界计算得到无效尺寸。")
    return _ceil_output_dimension_px(boundary_width), _ceil_output_dimension_px(boundary_height), float(px_per_rad)


def estimate_mosaic_optimal_resolution(
    model: object,
    *,
    source_image_width_px: int,
    source_image_height_px: int,
    projection_model: str,
    fov_deg: float,
    viewport_width_px: int,
    viewport_height_px: int,
) -> MosaicResolutionEstimate:
    """计算自由投影当前取景的 PTGui 风格最优输出边界。"""

    viewport_width = max(1, int(viewport_width_px))
    viewport_height = max(1, int(viewport_height_px))
    height_over_width = viewport_height / float(viewport_width)
    resolution, jacobian, center_x, center_y = source_center_angular_resolution_rad_per_px(
        model,
        image_width_px=source_image_width_px,
        image_height_px=source_image_height_px,
    )
    boundary_width, boundary_height, px_per_rad = target_output_dimensions_for_resolution(
        angular_resolution_rad_per_px=resolution,
        projection_model=projection_model,
        fov_deg=fov_deg,
        height_over_width=height_over_width,
    )
    return MosaicResolutionEstimate(
        boundary_width_px=boundary_width,
        boundary_height_px=boundary_height,
        center_angular_resolution_rad_per_px=float(resolution),
        target_center_px_per_rad=float(px_per_rad),
        source_center_x_px=float(center_x),
        source_center_y_px=float(center_y),
        source_jacobian_rad_per_px=(
            (float(jacobian[0, 0]), float(jacobian[0, 1])),
            (float(jacobian[1, 0]), float(jacobian[1, 1])),
        ),
        viewport_width_px=viewport_width,
        viewport_height_px=viewport_height,
        height_over_width=float(height_over_width),
    )


__all__ = [name for name in globals() if not name.startswith("__")]
