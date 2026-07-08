from __future__ import annotations

import numpy as np

from .constants import (
    FIT_WEIGHT_MAX,
    FIT_WEIGHT_MIN,
    MAX_TRANSFORM_ABS_PX,
    MIN_ALIGNMENT_PAIRS,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .validation import _array_is_finite


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("无法归一化零长度向量。")
    return vector / norm


def _radec_to_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra_rad = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec_rad = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec_rad)
    return np.column_stack((cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad))).astype(np.float64)


def _sky_plane_basis(ra_dec_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _radec_to_unit_vectors(ra_dec_points[:, 0], ra_dec_points[:, 1])
    center = vectors.mean(axis=0)
    if float(np.linalg.norm(center)) <= 1e-8:
        center = vectors[0]
    center = _normalize_vector(center)

    celestial_north = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(celestial_north, center)
    if float(np.linalg.norm(east)) <= 1e-8:
        east = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    east = _normalize_vector(east)
    north = _normalize_vector(np.cross(center, east))
    return center, east, north


def _project_radec_to_sky_plane(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_vector: np.ndarray,
    east_vector: np.ndarray,
    north_vector: np.ndarray,
) -> np.ndarray:
    ra_array = np.asarray(ra_deg, dtype=np.float64)
    dec_array = np.asarray(dec_deg, dtype=np.float64)
    if not (_array_is_finite(center_vector) and _array_is_finite(east_vector) and _array_is_finite(north_vector)):
        return np.full((ra_array.size, 2), np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        vectors = _radec_to_unit_vectors(ra_array, dec_array)
        center_component = np.clip(vectors @ center_vector, -1.0, 1.0)
        east_component = vectors @ east_vector
        north_component = vectors @ north_vector
        theta = np.arccos(center_component)
        sin_theta = np.sin(theta)
        scale = np.divide(theta, sin_theta, out=np.ones_like(theta), where=np.abs(sin_theta) > 1e-12)
        radians_to_degrees = 180.0 / np.pi
        projected = np.column_stack(
            (east_component * scale * radians_to_degrees, north_component * scale * radians_to_degrees)
        )
    projected[~np.all(np.isfinite(projected), axis=1)] = np.nan
    return projected


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if not np.isfinite(angle):
        raise ValueError("旋转参数包含无效数值。")
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = vector / angle
    x_value, y_value, z_value = axis
    skew = np.asarray(
        (
            (0.0, -z_value, y_value),
            (z_value, 0.0, -x_value),
            (-y_value, x_value, 0.0),
        ),
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def _orthonormalized_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("投影初始旋转矩阵必须是有限的 3x3 数组。")
    input_determinant = float(np.linalg.det(rotation))
    if not np.isfinite(input_determinant) or abs(input_determinant) <= 1e-12:
        raise ValueError("投影初始旋转矩阵接近奇异。")
    target_handedness = -1.0 if input_determinant < 0.0 else 1.0

    try:
        u_matrix, _values, vt_matrix = np.linalg.svd(rotation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("投影初始旋转矩阵无法正交化。") from exc
    orthonormal = u_matrix @ vt_matrix
    if float(np.linalg.det(orthonormal)) * target_handedness < 0.0:
        u_matrix[:, -1] *= -1.0
        orthonormal = u_matrix @ vt_matrix
    return orthonormal.astype(np.float64)


def _initial_rectilinear_from_rotation(
    vectors: np.ndarray,
    target_points: np.ndarray,
    rotation_matrix: np.ndarray,
    point_weights: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    rotation = _orthonormalized_rotation_matrix(rotation_matrix)
    camera_vectors = np.asarray(vectors, dtype=np.float64) @ rotation.T
    cam_x = camera_vectors[:, 0]
    cam_y = camera_vectors[:, 1]
    cam_z = camera_vectors[:, 2]
    valid = (
        np.all(np.isfinite(camera_vectors), axis=1)
        & np.all(np.isfinite(target_points), axis=1)
        & np.isfinite(point_weights)
        & (cam_z > 1e-6)
    )
    if np.count_nonzero(valid) < MIN_ALIGNMENT_PAIRS:
        raise ValueError("普通广角初值中可见配对星不足。")

    x_over_z = cam_x[valid] / cam_z[valid]
    y_over_z = -cam_y[valid] / cam_z[valid]
    valid_targets = target_points[valid]
    sqrt_weights = np.sqrt(np.clip(point_weights[valid], FIT_WEIGHT_MIN, FIT_WEIGHT_MAX))

    design = np.zeros((int(np.count_nonzero(valid)) * 2, 3), dtype=np.float64)
    observed = np.zeros(design.shape[0], dtype=np.float64)
    design[0::2, 0] = 1.0
    design[0::2, 2] = x_over_z
    observed[0::2] = valid_targets[:, 0]
    design[1::2, 1] = 1.0
    design[1::2, 2] = y_over_z
    observed[1::2] = valid_targets[:, 1]
    row_weights = np.repeat(sqrt_weights, 2)

    try:
        solution, _residuals, _rank, _singular_values = np.linalg.lstsq(
            design * row_weights[:, None],
            observed * row_weights,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("普通广角主点与尺度初值无法稳定求解。") from exc

    center_x, center_y, scale_px = (float(value) for value in solution)
    if not all(np.isfinite(value) for value in (center_x, center_y, scale_px)):
        raise ValueError("普通广角主点与尺度初值包含无效数值。")
    if abs(scale_px) <= 1e-9:
        raise ValueError("普通广角尺度初值过小。")
    if scale_px < 0.0:
        # 负尺度等价于绕光轴翻转 180 度；转成正尺度可避免 log(scale) 退化。
        rotation = rotation.copy()
        rotation[:2] *= -1.0
        scale_px = -scale_px
    return rotation.astype(np.float64), center_x, center_y, float(scale_px)


def _initial_projection_from_rotation(
    vectors: np.ndarray,
    target_points: np.ndarray,
    rotation_matrix: np.ndarray,
    lens_model: str,
    point_weights: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    rotation = _orthonormalized_rotation_matrix(rotation_matrix)
    camera_vectors = np.asarray(vectors, dtype=np.float64) @ rotation.T
    projection_points, valid_projection = _projection_plane_coordinates(
        camera_vectors,
        lens_model=lens_model,
        strict_visibility=True,
    )
    valid = (
        valid_projection
        & np.all(np.isfinite(projection_points), axis=1)
        & np.all(np.isfinite(target_points), axis=1)
        & np.isfinite(point_weights)
    )
    if np.count_nonzero(valid) < MIN_ALIGNMENT_PAIRS:
        raise ValueError("投影初值中有效配对星不足。")

    projected_x = projection_points[valid, 0]
    projected_y = projection_points[valid, 1]
    valid_targets = target_points[valid]
    sqrt_weights = np.sqrt(np.clip(point_weights[valid], FIT_WEIGHT_MIN, FIT_WEIGHT_MAX))

    design = np.zeros((int(np.count_nonzero(valid)) * 2, 3), dtype=np.float64)
    observed = np.zeros(design.shape[0], dtype=np.float64)
    design[0::2, 0] = 1.0
    design[0::2, 2] = projected_x
    observed[0::2] = valid_targets[:, 0]
    design[1::2, 1] = 1.0
    design[1::2, 2] = -projected_y
    observed[1::2] = valid_targets[:, 1]
    row_weights = np.repeat(sqrt_weights, 2)

    try:
        solution, _residuals, _rank, _singular_values = np.linalg.lstsq(
            design * row_weights[:, None],
            observed * row_weights,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("投影主点与尺度初值无法稳定求解。") from exc

    center_x, center_y, scale_px = (float(value) for value in solution)
    if not all(np.isfinite(value) for value in (center_x, center_y, scale_px)):
        raise ValueError("投影主点与尺度初值包含无效数值。")
    if abs(scale_px) <= 1e-9:
        raise ValueError("投影尺度初值过小。")
    if scale_px < 0.0:
        # 负尺度等价于把投影平面绕光轴旋转 180 度，转成正尺度可避免 log(scale) 退化。
        rotation = rotation.copy()
        rotation[:2] *= -1.0
        scale_px = -scale_px
    return rotation.astype(np.float64), center_x, center_y, float(scale_px)


def _fixed_fisheye_scale_px(
    lens_model: str,
    image_size: tuple[int, int],
    fisheye_fov_deg: float | None,
) -> float | None:
    if lens_model not in (SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID):
        return None
    if fisheye_fov_deg is None:
        return None
    if not np.isfinite(float(fisheye_fov_deg)) or not (1.0 <= float(fisheye_fov_deg) <= 300.0):
        raise ValueError("鱼眼视场必须在 1 到 300 度之间。")

    radius_limit_px = min(float(image_size[0]), float(image_size[1])) * 0.5 - 0.5
    if radius_limit_px <= 0.0:
        raise ValueError("真实图像尺寸过小，无法计算鱼眼投影尺度。")
    theta_max = np.deg2rad(float(fisheye_fov_deg) * 0.5)
    if lens_model == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT:
        return float(radius_limit_px / theta_max)

    denominator = 2.0 * np.sin(theta_max * 0.5)
    if abs(float(denominator)) <= 1e-12:
        raise ValueError("鱼眼视场过小，无法计算等立体角投影尺度。")
    return float(radius_limit_px / denominator)


def _sky_basis_from_center(center_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = _normalize_vector(center_vector)
    celestial_north = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(celestial_north, center)
    if float(np.linalg.norm(east)) <= 1e-8:
        east = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    east = _normalize_vector(east)
    north = _normalize_vector(np.cross(center, east))
    return center, east, north


def _theta_from_fisheye_radius(
    radius_px: np.ndarray,
    lens_model: str,
    scale_px: float,
) -> np.ndarray:
    if lens_model == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT:
        return radius_px / scale_px
    if lens_model == SKY_MATCHING_MODEL_FISHEYE_EQUISOLID:
        return 2.0 * np.arcsin(np.clip(radius_px / (2.0 * scale_px), -1.0, 1.0))
    raise ValueError(f"不支持的鱼眼投影模型：{lens_model}")


def _initial_fisheye_rotation_from_pixels(
    sky_radec: np.ndarray,
    target_points: np.ndarray,
    lens_model: str,
    image_size: tuple[int, int],
    scale_px: float,
) -> np.ndarray:
    vectors = _radec_to_unit_vectors(sky_radec[:, 0], sky_radec[:, 1])
    center_x = float(image_size[0]) * 0.5
    center_y = float(image_size[1]) * 0.5
    screen_x = target_points[:, 0] - center_x
    screen_y = center_y - target_points[:, 1]
    radius = np.hypot(screen_x, screen_y)
    theta = _theta_from_fisheye_radius(radius, lens_model, scale_px)
    valid = np.isfinite(theta) & (theta >= 0.0) & (theta <= np.pi)
    if np.count_nonzero(valid) < MIN_ALIGNMENT_PAIRS:
        raise ValueError("用于估计鱼眼光轴的有效星点不足。")

    try:
        forward, _residuals, _rank, _singular_values = np.linalg.lstsq(
            vectors[valid],
            np.cos(theta[valid]),
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("鱼眼光轴初值无法稳定求解。") from exc
    forward = _normalize_vector(forward)
    center, east, north = _sky_basis_from_center(forward)

    base_x = vectors @ east
    base_y = vectors @ north
    base_radius = np.hypot(base_x, base_y)
    observed_radius = np.hypot(screen_x, screen_y)
    valid_roll = valid & (base_radius > 1e-8) & (observed_radius > 1e-8)
    if np.count_nonzero(valid_roll) < 2:
        roll_rad = 0.0
    else:
        base_angle = np.arctan2(base_y[valid_roll], base_x[valid_roll])
        observed_angle = np.arctan2(screen_y[valid_roll], screen_x[valid_roll])
        roll_samples = base_angle - observed_angle
        weights = np.clip(observed_radius[valid_roll], 1.0, None)
        roll_rad = float(
            np.arctan2(
                np.sum(weights * np.sin(roll_samples)),
                np.sum(weights * np.cos(roll_samples)),
            )
        )

    cos_roll = float(np.cos(roll_rad))
    sin_roll = float(np.sin(roll_rad))
    right = _normalize_vector(east * cos_roll + north * sin_roll)
    up = _normalize_vector(-east * sin_roll + north * cos_roll)
    return np.vstack((right, up, center)).astype(np.float64)


def _projection_plane_coordinates(
    camera_vectors: np.ndarray,
    lens_model: str,
    strict_visibility: bool,
) -> tuple[np.ndarray, np.ndarray]:
    camera_array = np.asarray(camera_vectors, dtype=np.float64)
    cam_x = camera_array[:, 0]
    cam_y = camera_array[:, 1]
    cam_z = camera_array[:, 2]
    norm = np.linalg.norm(camera_array, axis=1)
    valid = np.all(np.isfinite(camera_array), axis=1) & (norm > 1e-12)

    projected_x = np.full(camera_array.shape[0], np.nan, dtype=np.float64)
    projected_y = np.full(camera_array.shape[0], np.nan, dtype=np.float64)
    if lens_model == SKY_MATCHING_MODEL_RECTILINEAR:
        visible = cam_z > 1e-8
        if strict_visibility:
            valid &= visible
        safe_z = np.where(np.abs(cam_z) > 1e-8, cam_z, np.where(cam_z >= 0.0, 1e-8, -1e-8))
        projected_x = cam_x / safe_z
        projected_y = cam_y / safe_z
    elif lens_model in (SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID):
        unit_z = np.clip(cam_z / np.where(norm > 1e-12, norm, 1.0), -1.0, 1.0)
        theta = np.arccos(unit_z)
        plane_norm = np.hypot(cam_x, cam_y)
        unit_x = np.divide(cam_x, plane_norm, out=np.zeros_like(cam_x), where=plane_norm > 1e-12)
        unit_y = np.divide(cam_y, plane_norm, out=np.zeros_like(cam_y), where=plane_norm > 1e-12)
        if lens_model == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT:
            radius = theta
        else:
            radius = 2.0 * np.sin(theta * 0.5)
        projected_x = radius * unit_x
        projected_y = radius * unit_y
    elif lens_model in (SKY_MATCHING_MODEL_MERCATOR, SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT):
        unit_y = np.clip(cam_y / np.where(norm > 1e-12, norm, 1.0), -1.0, 1.0)
        longitude = np.arctan2(cam_x, cam_z)
        latitude = np.arcsin(unit_y)
        projected_x = longitude
        if lens_model == SKY_MATCHING_MODEL_MERCATOR:
            # 墨卡托在两极发散，避开极点附近的无穷大。
            valid &= np.abs(latitude) < (np.pi * 0.5 - 1e-8)
            projected_y = np.arctanh(np.clip(np.sin(latitude), -1.0 + 1e-12, 1.0 - 1e-12))
        else:
            projected_y = latitude
    else:
        raise ValueError(f"不支持的投影匹配模型：{lens_model}")

    projected = np.column_stack((projected_x, projected_y)).astype(np.float64)
    valid &= np.all(np.isfinite(projected), axis=1)
    return projected, valid


def _project_unit_vectors_with_known_projection(
    *,
    vectors: np.ndarray,
    rotation_matrix: np.ndarray,
    center_x_px: float,
    center_y_px: float,
    scale_px: float,
    lens_model: str,
    strict_visibility: bool,
) -> tuple[np.ndarray, np.ndarray]:
    vector_array = np.asarray(vectors, dtype=np.float64)
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    camera_vectors = vector_array @ rotation.T
    projection_points, valid = _projection_plane_coordinates(
        camera_vectors,
        lens_model=lens_model,
        strict_visibility=strict_visibility,
    )
    valid &= np.isfinite(scale_px) & (scale_px > 0.0)
    x_px = center_x_px + scale_px * projection_points[:, 0]
    y_px = center_y_px - scale_px * projection_points[:, 1]

    projected = np.column_stack((x_px, y_px)).astype(np.float64)
    valid &= np.all(np.isfinite(projected), axis=1) & (np.abs(projected[:, 0]) < MAX_TRANSFORM_ABS_PX) & (
        np.abs(projected[:, 1]) < MAX_TRANSFORM_ABS_PX
    )
    return projected, valid


def _projection_points_from_params(
    vectors: np.ndarray,
    initial_rotation_matrix: np.ndarray,
    params: np.ndarray,
    lens_model: str,
    fixed_scale_px: float | None = None,
    fixed_center_px: tuple[float, float] | None = None,
    strict_visibility: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    delta_rotation = _rotation_matrix_from_rotvec(params[:3])
    rotation_matrix = delta_rotation @ initial_rotation_matrix
    scale_px = float(fixed_scale_px) if fixed_scale_px is not None else float(np.exp(params[3]))
    if fixed_center_px is None:
        center_x_px = float(params[4])
        center_y_px = float(params[5])
    else:
        center_x_px = float(fixed_center_px[0])
        center_y_px = float(fixed_center_px[1])
    return _project_unit_vectors_with_known_projection(
        vectors=vectors,
        rotation_matrix=rotation_matrix,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        scale_px=scale_px,
        lens_model=lens_model,
        strict_visibility=strict_visibility,
    )

__all__ = [name for name in globals() if not name.startswith("__")]
