from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


MIN_ALIGNMENT_PAIRS = 4
QUADRATIC_ALIGNMENT_PAIRS = 6
MAX_ALIGNMENT_CONDITION_NUMBER = 1e10
MAX_TRANSFORM_ABS_PX = 1e9
SKY_MATCHING_MODEL_POLYNOMIAL = "polynomial"
SKY_MATCHING_MODEL_RECTILINEAR = "rectilinear"
SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT = "fisheye_equidistant"
SKY_MATCHING_MODEL_FISHEYE_EQUISOLID = "fisheye_equisolid"
SKY_KNOWN_PROJECTION_MODELS = (
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
)
SKY_MATCHING_MODELS = (SKY_MATCHING_MODEL_POLYNOMIAL, *SKY_KNOWN_PROJECTION_MODELS)
PROJECTION_FIT_MAX_NFEV = 2000
PROJECTION_INVALID_RESIDUAL_PX = 1e6
RESIDUAL_CORRECTION_CONDITION_NUMBER = 1e8
RESIDUAL_CORRECTION_POLYNOMIAL = "polynomial"
RESIDUAL_CORRECTION_TPS = "thin_plate_spline"


@dataclass(frozen=True)
class ReferenceAlignmentTransform:
    degree: int
    pair_count: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    coeff_x: np.ndarray
    coeff_y: np.ndarray
    rms_px: float

    @property
    def display_name(self) -> str:
        if self.degree >= 2:
            return "二阶多项式"
        return "一次仿射"

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, 2)
        if point_array.shape[1] != 2:
            raise ValueError("参考图配准点必须是 Nx2 数组。")

        if not _alignment_coefficients_are_valid(self.coeff_x, self.coeff_y):
            return np.full((point_array.shape[0], 2), np.nan, dtype=np.float64)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            terms = _polynomial_terms(point_array[:, 0], point_array[:, 1], self.degree)
            x_values = terms @ self.coeff_x
            y_values = terms @ self.coeff_y
            transformed = np.column_stack((x_values, y_values)).astype(np.float64)
        transformed[~np.all(np.isfinite(transformed), axis=1)] = np.nan
        return transformed

    def transform_point(self, x_value: float, y_value: float) -> tuple[float, float]:
        transformed = self.transform_points(np.asarray([[x_value, y_value]], dtype=np.float64))[0]
        return float(transformed[0]), float(transformed[1])


@dataclass(frozen=True)
class SkyAlignmentTransform:
    degree: int
    pair_count: int
    center_vector: np.ndarray
    east_vector: np.ndarray
    north_vector: np.ndarray
    coeff_x: np.ndarray
    coeff_y: np.ndarray
    rms_px: float

    @property
    def display_name(self) -> str:
        if self.degree >= 2:
            return "二阶多项式"
        return "一次仿射"

    def transform_radec_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
        if ra_dec_array.ndim == 1:
            ra_dec_array = ra_dec_array.reshape(1, 2)
        if ra_dec_array.shape[1] != 2:
            raise ValueError("天球配准点必须是 Nx2 的 RA/Dec 数组。")

        if not _sky_transform_components_are_valid(self):
            return np.full((ra_dec_array.shape[0], 2), np.nan, dtype=np.float64)

        sky_points = _project_radec_to_sky_plane(
            ra_dec_array[:, 0],
            ra_dec_array[:, 1],
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            terms = _polynomial_terms(sky_points[:, 0], sky_points[:, 1], self.degree)
            x_values = terms @ self.coeff_x
            y_values = terms @ self.coeff_y
            transformed = np.column_stack((x_values, y_values)).astype(np.float64)
        transformed[~np.all(np.isfinite(transformed), axis=1)] = np.nan
        return transformed

    def transform_radec(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        transformed = self.transform_radec_points(np.asarray([[ra_deg, dec_deg]], dtype=np.float64))[0]
        return float(transformed[0]), float(transformed[1])


@dataclass(frozen=True)
class ProjectionSkyAlignmentTransform:
    lens_model: str
    pair_count: int
    image_width_px: int
    image_height_px: int
    fov_deg: float | None
    rotation_matrix: np.ndarray
    center_x_px: float
    center_y_px: float
    scale_px: float
    residual_kind: str
    residual_degree: int
    residual_origin_x_px: float
    residual_origin_y_px: float
    residual_scale_x_px: float
    residual_scale_y_px: float
    residual_coeff_x: np.ndarray
    residual_coeff_y: np.ndarray
    residual_anchor_points: np.ndarray
    residual_tps_weights_x: np.ndarray
    residual_tps_weights_y: np.ndarray
    residual_tps_affine_x: np.ndarray
    residual_tps_affine_y: np.ndarray
    projection_rms_px: float
    rms_px: float

    @property
    def display_name(self) -> str:
        projection_names = {
            SKY_MATCHING_MODEL_RECTILINEAR: "普通广角透视",
            SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT: "等距鱼眼",
            SKY_MATCHING_MODEL_FISHEYE_EQUISOLID: "等立体角鱼眼",
        }
        projection_name = projection_names.get(self.lens_model, self.lens_model)
        if self.residual_kind == RESIDUAL_CORRECTION_TPS:
            return f"{projection_name}+锚点插值"
        if self.residual_degree >= 2:
            return f"{projection_name}+二阶残差"
        if self.residual_degree >= 1:
            return f"{projection_name}+一阶残差"
        return f"{projection_name}+常量残差"

    def raw_project_radec_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
        if ra_dec_array.ndim == 1:
            ra_dec_array = ra_dec_array.reshape(1, 2)
        if ra_dec_array.shape[1] != 2:
            raise ValueError("天球投影点必须是 Nx2 的 RA/Dec 数组。")
        vectors = _radec_to_unit_vectors(ra_dec_array[:, 0], ra_dec_array[:, 1])
        projected, valid = _project_unit_vectors_with_known_projection(
            vectors=vectors,
            rotation_matrix=self.rotation_matrix,
            center_x_px=self.center_x_px,
            center_y_px=self.center_y_px,
            scale_px=self.scale_px,
            lens_model=self.lens_model,
            strict_visibility=True,
        )
        projected[~valid] = np.nan
        return projected

    def transform_radec_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        projected = self.raw_project_radec_points(ra_dec_points)
        corrected = _apply_residual_correction(
            projected_points=projected,
            degree=self.residual_degree,
            origin_x_px=self.residual_origin_x_px,
            origin_y_px=self.residual_origin_y_px,
            scale_x_px=self.residual_scale_x_px,
            scale_y_px=self.residual_scale_y_px,
            coeff_x=self.residual_coeff_x,
            coeff_y=self.residual_coeff_y,
            residual_kind=self.residual_kind,
            anchor_points=self.residual_anchor_points,
            tps_weights_x=self.residual_tps_weights_x,
            tps_weights_y=self.residual_tps_weights_y,
            tps_affine_x=self.residual_tps_affine_x,
            tps_affine_y=self.residual_tps_affine_y,
        )
        corrected[~np.all(np.isfinite(corrected), axis=1)] = np.nan
        return corrected.astype(np.float64)

    def transform_radec(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        transformed = self.transform_radec_points(np.asarray([[ra_deg, dec_deg]], dtype=np.float64))[0]
        return float(transformed[0]), float(transformed[1])


def _polynomial_terms(x_values: np.ndarray, y_values: np.ndarray, degree: int) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if degree >= 2:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.column_stack(
                (
                    np.ones_like(x_values),
                    x_values,
                    y_values,
                    x_values * x_values,
                    x_values * y_values,
                    y_values * y_values,
                )
            )
    return np.column_stack((np.ones_like(x_values), x_values, y_values))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("无法归一化零长度向量。")
    return vector / norm


def _array_is_finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def _alignment_coefficients_are_valid(coeff_x: np.ndarray, coeff_y: np.ndarray) -> bool:
    return _array_is_finite(coeff_x) and _array_is_finite(coeff_y)


def _sky_transform_components_are_valid(transform: SkyAlignmentTransform) -> bool:
    return (
        _array_is_finite(transform.center_vector)
        and _array_is_finite(transform.east_vector)
        and _array_is_finite(transform.north_vector)
        and _alignment_coefficients_are_valid(transform.coeff_x, transform.coeff_y)
    )


def _residual_polynomial_terms(points: np.ndarray, degree: int) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("残差修正需要 Nx2 点数组。")
    x_values = point_array[:, 0]
    y_values = point_array[:, 1]
    if degree >= 2:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.column_stack(
                (
                    np.ones_like(x_values),
                    x_values,
                    y_values,
                    x_values * x_values,
                    x_values * y_values,
                    y_values * y_values,
                )
            )
    if degree >= 1:
        return np.column_stack((np.ones_like(x_values), x_values, y_values))
    return np.ones((point_array.shape[0], 1), dtype=np.float64)


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


def _initial_projection_basis(ra_dec_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    center, east, north = _sky_plane_basis(ra_dec_points)
    sky_points = _project_radec_to_sky_plane(ra_dec_points[:, 0], ra_dec_points[:, 1], center, east, north)
    center_x = float(np.nanmedian(target_points[:, 0]))
    center_y = float(np.nanmedian(target_points[:, 1]))
    scale_px = float("nan")
    roll_rad = 0.0

    try:
        degree, coeff_x, coeff_y, _rms_px = _fit_image_polynomial(sky_points, target_points, 1)
        if degree == 1 and coeff_x.size >= 3 and coeff_y.size >= 3:
            center_x = float(coeff_x[0])
            center_y = float(coeff_y[0])
            derivative = np.asarray(
                (
                    (float(coeff_x[1]), float(coeff_x[2])),
                    (float(coeff_y[1]), float(coeff_y[2])),
                ),
                dtype=np.float64,
            )
            # 小角度下投影导数近似为 s * [[cos r, sin r], [sin r, -cos r]]。
            cos_part = 0.5 * (derivative[0, 0] - derivative[1, 1])
            sin_part = 0.5 * (derivative[0, 1] + derivative[1, 0])
            if np.isfinite(cos_part) and np.isfinite(sin_part) and abs(cos_part) + abs(sin_part) > 1e-9:
                roll_rad = float(np.arctan2(sin_part, cos_part))
            column_scales = np.asarray(
                (np.hypot(derivative[0, 0], derivative[1, 0]), np.hypot(derivative[0, 1], derivative[1, 1])),
                dtype=np.float64,
            )
            finite_scales = column_scales[np.isfinite(column_scales) & (column_scales > 1e-9)]
            if finite_scales.size:
                scale_px = float(np.median(finite_scales) * 180.0 / np.pi)
    except ValueError:
        pass

    if not np.isfinite(scale_px) or scale_px <= 1e-9:
        vectors = _radec_to_unit_vectors(ra_dec_points[:, 0], ra_dec_points[:, 1])
        theta = np.arccos(np.clip(vectors @ center, -1.0, 1.0))
        pixel_radius = np.hypot(target_points[:, 0] - center_x, target_points[:, 1] - center_y)
        valid = np.isfinite(theta) & np.isfinite(pixel_radius) & (theta > 1e-8)
        if np.any(valid):
            scale_px = float(np.median(pixel_radius[valid] / theta[valid]))
    if not np.isfinite(scale_px) or scale_px <= 1e-9:
        scale_px = max(float(np.nanmax(np.ptp(target_points, axis=0))), 1.0)

    cos_roll = float(np.cos(roll_rad))
    sin_roll = float(np.sin(roll_rad))
    right = _normalize_vector(east * cos_roll + north * sin_roll)
    up = _normalize_vector(-east * sin_roll + north * cos_roll)
    rotation_matrix = np.vstack((right, up, center)).astype(np.float64)
    return rotation_matrix, center_x, center_y, scale_px


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
    cam_x = camera_vectors[:, 0]
    cam_y = camera_vectors[:, 1]
    cam_z = camera_vectors[:, 2]
    valid = np.all(np.isfinite(camera_vectors), axis=1) & np.isfinite(scale_px) & (scale_px > 0.0)

    x_px = np.full(vector_array.shape[0], np.nan, dtype=np.float64)
    y_px = np.full(vector_array.shape[0], np.nan, dtype=np.float64)
    if lens_model == SKY_MATCHING_MODEL_RECTILINEAR:
        visible = cam_z > 1e-8
        if strict_visibility:
            valid &= visible
        safe_z = np.where(np.abs(cam_z) > 1e-8, cam_z, np.where(cam_z >= 0.0, 1e-8, -1e-8))
        x_px = center_x_px + scale_px * cam_x / safe_z
        y_px = center_y_px - scale_px * cam_y / safe_z
    elif lens_model in (SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID):
        norm = np.linalg.norm(camera_vectors, axis=1)
        safe_norm = np.where(norm > 1e-12, norm, 1.0)
        unit_z = np.clip(cam_z / safe_norm, -1.0, 1.0)
        theta = np.arccos(unit_z)
        plane_norm = np.hypot(cam_x, cam_y)
        unit_x = np.divide(cam_x, plane_norm, out=np.zeros_like(cam_x), where=plane_norm > 1e-12)
        unit_y = np.divide(cam_y, plane_norm, out=np.zeros_like(cam_y), where=plane_norm > 1e-12)
        if lens_model == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT:
            radius = scale_px * theta
        else:
            radius = scale_px * 2.0 * np.sin(theta * 0.5)
        x_px = center_x_px + radius * unit_x
        y_px = center_y_px - radius * unit_y
        valid &= norm > 1e-12
    else:
        raise ValueError(f"不支持的投影匹配模型：{lens_model}")

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


def _fit_known_projection_parameters(
    sky_radec: np.ndarray,
    target_points: np.ndarray,
    lens_model: str,
    image_size: tuple[int, int],
    fisheye_fov_deg: float | None,
    initial_rotation_matrix: np.ndarray | None,
) -> tuple[np.ndarray, float, float, float, np.ndarray, float]:
    vectors = _radec_to_unit_vectors(sky_radec[:, 0], sky_radec[:, 1])
    width_px = max(float(image_size[0]), 1.0)
    height_px = max(float(image_size[1]), 1.0)
    fixed_scale = _fixed_fisheye_scale_px(lens_model, image_size, fisheye_fov_deg)
    fixed_center: tuple[float, float] | None = None
    if fixed_scale is not None:
        if initial_rotation_matrix is None:
            initial_rotation = _initial_fisheye_rotation_from_pixels(
                sky_radec=sky_radec,
                target_points=target_points,
                lens_model=lens_model,
                image_size=image_size,
                scale_px=fixed_scale,
            )
        else:
            initial_rotation = np.asarray(initial_rotation_matrix, dtype=np.float64)
        fixed_center = (width_px * 0.5, height_px * 0.5)
        initial = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
        lower = np.asarray((-np.pi, -np.pi, -np.pi), dtype=np.float64)
        upper = np.asarray((np.pi, np.pi, np.pi), dtype=np.float64)
    else:
        initial_rotation, initial_cx, initial_cy, initial_scale = _initial_projection_basis(sky_radec, target_points)
        initial = np.asarray(
            (0.0, 0.0, 0.0, np.log(max(initial_scale, 1e-6)), initial_cx, initial_cy),
            dtype=np.float64,
        )
        lower = np.asarray((-np.pi, -np.pi, -np.pi, np.log(1e-6), -2.0 * width_px, -2.0 * height_px), dtype=np.float64)
        upper = np.asarray((np.pi, np.pi, np.pi, np.log(1e9), 3.0 * width_px, 3.0 * height_px), dtype=np.float64)
    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)

    def residual(params: np.ndarray) -> np.ndarray:
        projected, valid = _projection_points_from_params(
            vectors,
            initial_rotation,
            params,
            lens_model,
            fixed_scale_px=fixed_scale,
            fixed_center_px=fixed_center,
        )
        residual_values = projected - target_points
        invalid = ~valid | ~np.all(np.isfinite(residual_values), axis=1)
        if np.any(invalid):
            residual_values[invalid] = PROJECTION_INVALID_RESIDUAL_PX
        residual_values = np.clip(residual_values, -PROJECTION_INVALID_RESIDUAL_PX, PROJECTION_INVALID_RESIDUAL_PX)
        return residual_values.ravel()

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="linear",
        max_nfev=PROJECTION_FIT_MAX_NFEV,
    )
    if not result.success:
        raise ValueError(f"已知投影参数拟合未收敛：{result.message}")

    projected, valid = _projection_points_from_params(
        vectors,
        initial_rotation,
        result.x,
        lens_model,
        fixed_scale_px=fixed_scale,
        fixed_center_px=fixed_center,
        strict_visibility=True,
    )
    if not np.all(valid):
        raise ValueError("已知投影拟合后仍有配对星无法投影，请检查投影类型或配对星。")
    residual_vectors = projected - target_points
    if not _array_is_finite(residual_vectors):
        raise ValueError("已知投影拟合残差包含无效数值。")
    rms_px = float(np.sqrt(np.mean(np.sum(residual_vectors * residual_vectors, axis=1))))
    if not np.isfinite(rms_px):
        raise ValueError("已知投影拟合 RMS 包含无效数值。")

    rotation_matrix = _rotation_matrix_from_rotvec(result.x[:3]) @ initial_rotation
    if fixed_scale is not None and fixed_center is not None:
        scale_px = float(fixed_scale)
        center_x, center_y = fixed_center
    else:
        scale_px = float(np.exp(result.x[3]))
        center_x = float(result.x[4])
        center_y = float(result.x[5])
    return rotation_matrix, float(center_x), float(center_y), scale_px, projected, rms_px


def _normalize_residual_points(
    projected_points: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, float, float, float, float]:
    points = np.asarray(projected_points, dtype=np.float64)
    origin_x = float(image_size[0]) * 0.5
    origin_y = float(image_size[1]) * 0.5
    scale_x = max(float(image_size[0]), 1.0)
    scale_y = max(float(image_size[1]), 1.0)
    normalized = np.column_stack(((points[:, 0] - origin_x) / scale_x, (points[:, 1] - origin_y) / scale_y))
    return normalized.astype(np.float64), origin_x, origin_y, scale_x, scale_y


def _fit_residual_coefficients(
    normalized_points: np.ndarray,
    residual_values: np.ndarray,
    degree: int,
) -> np.ndarray:
    design = _residual_polynomial_terms(normalized_points, degree)
    if not _array_is_finite(design) or not _array_is_finite(residual_values):
        raise ValueError("残差修正点包含无效数值。")
    try:
        condition_number = float(np.linalg.cond(design))
    except np.linalg.LinAlgError as exc:
        raise ValueError("残差修正矩阵无法稳定求解。") from exc
    if not np.isfinite(condition_number) or condition_number > RESIDUAL_CORRECTION_CONDITION_NUMBER:
        raise ValueError("残差修正点分布过于集中，无法稳定求解。")
    try:
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(design, residual_values, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("残差修正矩阵无法稳定求解。") from exc
    if rank < design.shape[1]:
        raise ValueError("残差修正矩阵秩不足。")
    if not _array_is_finite(coefficients):
        raise ValueError("残差修正系数包含无效数值。")
    return coefficients.astype(np.float64)


def _thin_plate_spline_kernel_from_squared_distance(distance_squared: np.ndarray) -> np.ndarray:
    distance_squared = np.asarray(distance_squared, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = 0.5 * distance_squared * np.log(distance_squared)
    values[~np.isfinite(values)] = 0.0
    values[distance_squared <= 0.0] = 0.0
    return values.astype(np.float64)


def _thin_plate_spline_kernel(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    anchor_array = np.asarray(anchors, dtype=np.float64)
    delta = point_array[:, None, :] - anchor_array[None, :, :]
    distance_squared = np.sum(delta * delta, axis=2)
    return _thin_plate_spline_kernel_from_squared_distance(distance_squared)


def _fit_thin_plate_spline_coefficients(
    anchor_points: np.ndarray,
    residual_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.asarray(anchor_points, dtype=np.float64)
    values = np.asarray(residual_values, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("薄板样条锚点必须是 Nx2 数组。")
    if values.ndim != 1 or values.shape[0] != anchors.shape[0]:
        raise ValueError("薄板样条残差数量必须与锚点数量一致。")
    if anchors.shape[0] < 3:
        raise ValueError("薄板样条至少需要 3 个锚点。")
    if not _array_is_finite(anchors) or not _array_is_finite(values):
        raise ValueError("薄板样条锚点包含无效数值。")

    polynomial_terms = np.column_stack((np.ones(anchors.shape[0]), anchors[:, 0], anchors[:, 1]))
    try:
        polynomial_rank = int(np.linalg.matrix_rank(polynomial_terms))
    except np.linalg.LinAlgError as exc:
        raise ValueError("薄板样条锚点几何分布无法稳定求解。") from exc
    if polynomial_rank < 3:
        raise ValueError("薄板样条锚点接近共线，无法稳定插值。")

    kernel = _thin_plate_spline_kernel(anchors, anchors)
    matrix_size = anchors.shape[0] + 3
    system = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    system[: anchors.shape[0], : anchors.shape[0]] = kernel
    system[: anchors.shape[0], anchors.shape[0] :] = polynomial_terms
    system[anchors.shape[0] :, : anchors.shape[0]] = polynomial_terms.T
    rhs = np.zeros(matrix_size, dtype=np.float64)
    rhs[: anchors.shape[0]] = values
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("薄板样条矩阵无法稳定求解。") from exc

    weights = solution[: anchors.shape[0]].astype(np.float64)
    affine = solution[anchors.shape[0] :].astype(np.float64)
    fitted = _evaluate_thin_plate_spline(anchors, anchors, weights, affine)
    max_anchor_error = float(np.max(np.abs(fitted - values)))
    if not np.isfinite(max_anchor_error) or max_anchor_error > 1e-5:
        raise ValueError("薄板样条未能精确穿过锚点。")
    return weights, affine


def _evaluate_thin_plate_spline(
    points: np.ndarray,
    anchor_points: np.ndarray,
    weights: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    anchor_array = np.asarray(anchor_points, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    affine_array = np.asarray(affine, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("薄板样条求值点必须是 Nx2 数组。")
    if anchor_array.ndim != 2 or anchor_array.shape[1] != 2:
        raise ValueError("薄板样条锚点必须是 Nx2 数组。")
    if weight_array.shape[0] != anchor_array.shape[0] or affine_array.shape[0] != 3:
        raise ValueError("薄板样条系数数量不匹配。")
    kernel = _thin_plate_spline_kernel(point_array, anchor_array)
    polynomial_terms = np.column_stack((np.ones(point_array.shape[0]), point_array[:, 0], point_array[:, 1]))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        values = kernel @ weight_array + polynomial_terms @ affine_array
    values[~np.isfinite(values)] = np.nan
    return values.astype(np.float64)


def _residual_degree_candidates(pair_count: int) -> tuple[int, ...]:
    if pair_count >= 6:
        return 1, 0
    return (0,)


@dataclass(frozen=True)
class ResidualCorrectionResult:
    residual_kind: str
    degree: int
    origin_x_px: float
    origin_y_px: float
    scale_x_px: float
    scale_y_px: float
    coeff_x: np.ndarray
    coeff_y: np.ndarray
    anchor_points: np.ndarray
    tps_weights_x: np.ndarray
    tps_weights_y: np.ndarray
    tps_affine_x: np.ndarray
    tps_affine_y: np.ndarray
    corrected_projected: np.ndarray


def _fit_residual_correction(
    projected_points: np.ndarray,
    target_points: np.ndarray,
    image_size: tuple[int, int],
) -> ResidualCorrectionResult:
    normalized, origin_x, origin_y, scale_x, scale_y = _normalize_residual_points(projected_points, image_size)
    residual = target_points - projected_points

    try:
        weights_x, affine_x = _fit_thin_plate_spline_coefficients(normalized, residual[:, 0])
        weights_y, affine_y = _fit_thin_plate_spline_coefficients(normalized, residual[:, 1])
        corrected = projected_points + np.column_stack(
            (
                _evaluate_thin_plate_spline(normalized, normalized, weights_x, affine_x),
                _evaluate_thin_plate_spline(normalized, normalized, weights_y, affine_y),
            )
        )
        if not _array_is_finite(corrected):
            raise ValueError("薄板样条锚点插值结果包含无效数值。")
        return ResidualCorrectionResult(
            residual_kind=RESIDUAL_CORRECTION_TPS,
            degree=-1,
            origin_x_px=origin_x,
            origin_y_px=origin_y,
            scale_x_px=scale_x,
            scale_y_px=scale_y,
            coeff_x=np.asarray([], dtype=np.float64),
            coeff_y=np.asarray([], dtype=np.float64),
            anchor_points=normalized.astype(np.float64),
            tps_weights_x=weights_x,
            tps_weights_y=weights_y,
            tps_affine_x=affine_x,
            tps_affine_y=affine_y,
            corrected_projected=corrected.astype(np.float64),
        )
    except ValueError as exc:
        last_error: ValueError | None = exc

    for degree in _residual_degree_candidates(projected_points.shape[0]):
        try:
            coeff_x = _fit_residual_coefficients(normalized, residual[:, 0], degree)
            coeff_y = _fit_residual_coefficients(normalized, residual[:, 1], degree)
        except ValueError as exc:
            last_error = exc
            continue
        corrected = projected_points + np.column_stack(
            (
                _residual_polynomial_terms(normalized, degree) @ coeff_x,
                _residual_polynomial_terms(normalized, degree) @ coeff_y,
            )
        )
        if not _array_is_finite(corrected):
            last_error = ValueError("残差修正结果包含无效数值。")
            continue
        return ResidualCorrectionResult(
            residual_kind=RESIDUAL_CORRECTION_POLYNOMIAL,
            degree=degree,
            origin_x_px=origin_x,
            origin_y_px=origin_y,
            scale_x_px=scale_x,
            scale_y_px=scale_y,
            coeff_x=coeff_x,
            coeff_y=coeff_y,
            anchor_points=np.empty((0, 2), dtype=np.float64),
            tps_weights_x=np.asarray([], dtype=np.float64),
            tps_weights_y=np.asarray([], dtype=np.float64),
            tps_affine_x=np.asarray([], dtype=np.float64),
            tps_affine_y=np.asarray([], dtype=np.float64),
            corrected_projected=corrected.astype(np.float64),
        )
    if last_error is not None:
        raise last_error
    raise ValueError("无法计算残差修正。")


def _apply_residual_correction(
    *,
    projected_points: np.ndarray,
    residual_kind: str,
    degree: int,
    origin_x_px: float,
    origin_y_px: float,
    scale_x_px: float,
    scale_y_px: float,
    coeff_x: np.ndarray,
    coeff_y: np.ndarray,
    anchor_points: np.ndarray,
    tps_weights_x: np.ndarray,
    tps_weights_y: np.ndarray,
    tps_affine_x: np.ndarray,
    tps_affine_y: np.ndarray,
) -> np.ndarray:
    points = np.asarray(projected_points, dtype=np.float64)
    normalized = np.column_stack(
        (
            (points[:, 0] - origin_x_px) / max(float(scale_x_px), 1e-12),
            (points[:, 1] - origin_y_px) / max(float(scale_y_px), 1e-12),
        )
    )
    if residual_kind == RESIDUAL_CORRECTION_TPS:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            corrected = points + np.column_stack(
                (
                    _evaluate_thin_plate_spline(normalized, anchor_points, tps_weights_x, tps_affine_x),
                    _evaluate_thin_plate_spline(normalized, anchor_points, tps_weights_y, tps_affine_y),
                )
            )
    else:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            terms = _residual_polynomial_terms(normalized, degree)
            corrected = points + np.column_stack((terms @ coeff_x, terms @ coeff_y))
    corrected[~np.all(np.isfinite(points), axis=1)] = np.nan
    return corrected.astype(np.float64)


def _fit_polynomial_coefficients(
    source_points: np.ndarray,
    target_values: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, int, int]:
    design = _polynomial_terms(source_points[:, 0], source_points[:, 1], degree)
    if not _array_is_finite(design) or not _array_is_finite(target_values):
        raise ValueError("配准点包含无效数值，无法稳定求解配准。")
    try:
        condition_number = float(np.linalg.cond(design))
    except np.linalg.LinAlgError as exc:
        raise ValueError("配准矩阵无法稳定求解，请检查配对星位置。") from exc
    if not np.isfinite(condition_number) or condition_number > MAX_ALIGNMENT_CONDITION_NUMBER:
        raise ValueError("配对星几何分布过于集中，无法稳定求解配准。")
    try:
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(design, target_values, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("配准矩阵无法稳定求解，请检查配对星位置。") from exc
    if not _array_is_finite(coefficients):
        raise ValueError("配准结果包含无效系数，请检查配对星位置。")
    return coefficients.astype(np.float64), int(rank), int(design.shape[1])


def _fit_image_polynomial(
    source_points: np.ndarray,
    target_points: np.ndarray,
    preferred_degree: int,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    candidate_degrees = (preferred_degree, 1) if preferred_degree > 1 else (1,)
    last_error: ValueError | None = None
    for degree in candidate_degrees:
        try:
            coeff_x, rank_x, term_count = _fit_polynomial_coefficients(source_points, target_points[:, 0], degree)
            coeff_y, rank_y, _term_count_y = _fit_polynomial_coefficients(source_points, target_points[:, 1], degree)
            if rank_x < term_count or rank_y < term_count:
                raise ValueError("配对星几何分布过于集中，无法稳定求解配准。")
        except ValueError as exc:
            last_error = exc
            continue

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            predicted = _polynomial_terms(source_points[:, 0], source_points[:, 1], degree) @ np.column_stack(
                (coeff_x, coeff_y)
            )
        if not _array_is_finite(predicted) or float(np.nanmax(np.abs(predicted))) > MAX_TRANSFORM_ABS_PX:
            last_error = ValueError("配准结果数值过大，请检查配对星是否匹配正确。")
            continue
        residual = predicted - target_points
        rms_px = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if not np.isfinite(rms_px):
            last_error = ValueError("配准残差包含无效数值，请检查配对星位置。")
            continue
        return degree, coeff_x, coeff_y, rms_px

    if last_error is not None:
        raise last_error
    raise ValueError("无法计算配准。")


def fit_reference_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> ReferenceAlignmentTransform:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("参考图配准需要 Nx2 的源点与目标点。")
    if source.shape[0] != target.shape[0]:
        raise ValueError("参考图配准的源点与目标点数量不一致。")

    finite_mask = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source = source[finite_mask]
    target = target[finite_mask]
    pair_count = int(source.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对参考星才能计算自动配准。")

    preferred_degree = 2 if pair_count >= QUADRATIC_ALIGNMENT_PAIRS else 1
    degree, coeff_x, coeff_y, rms_px = _fit_image_polynomial(source, target, preferred_degree)
    return ReferenceAlignmentTransform(
        degree=degree,
        pair_count=pair_count,
        source_width=int(source_size[0]),
        source_height=int(source_size[1]),
        target_width=int(target_size[0]),
        target_height=int(target_size[1]),
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        rms_px=rms_px,
    )


def fit_sky_alignment(
    ra_dec_points: np.ndarray,
    target_points: np.ndarray,
    matching_model: str = SKY_MATCHING_MODEL_POLYNOMIAL,
    image_size: tuple[int, int] | None = None,
    fisheye_fov_deg: float | None = None,
    initial_rotation_matrix: np.ndarray | None = None,
) -> SkyAlignmentTransform | ProjectionSkyAlignmentTransform:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if sky_radec.ndim != 2 or target.ndim != 2 or sky_radec.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("天球配准需要 Nx2 的 RA/Dec 点与目标点。")
    if sky_radec.shape[0] != target.shape[0]:
        raise ValueError("天球配准的 RA/Dec 点与目标点数量不一致。")
    if matching_model not in SKY_MATCHING_MODELS:
        raise ValueError(f"不支持的匹配模型：{matching_model}")

    finite_mask = np.all(np.isfinite(sky_radec), axis=1) & np.all(np.isfinite(target), axis=1)
    sky_radec = sky_radec[finite_mask]
    target = target[finite_mask]
    pair_count = int(sky_radec.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对参考星才能计算天球残差。")

    if matching_model in SKY_KNOWN_PROJECTION_MODELS:
        if image_size is None:
            raise ValueError("已知投影匹配需要真实图像尺寸。")
        return fit_projection_sky_alignment(
            ra_dec_points=sky_radec,
            target_points=target,
            lens_model=matching_model,
            image_size=image_size,
            fisheye_fov_deg=fisheye_fov_deg,
            initial_rotation_matrix=initial_rotation_matrix,
        )

    center, east, north = _sky_plane_basis(sky_radec)
    sky_points = _project_radec_to_sky_plane(sky_radec[:, 0], sky_radec[:, 1], center, east, north)
    if not _array_is_finite(sky_points):
        raise ValueError("天球投影包含无效数值，请检查配对星坐标。")
    preferred_degree = 2 if pair_count >= QUADRATIC_ALIGNMENT_PAIRS else 1
    degree, coeff_x, coeff_y, rms_px = _fit_image_polynomial(sky_points, target, preferred_degree)
    return SkyAlignmentTransform(
        degree=degree,
        pair_count=pair_count,
        center_vector=center,
        east_vector=east,
        north_vector=north,
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        rms_px=rms_px,
    )


def fit_projection_sky_alignment(
    ra_dec_points: np.ndarray,
    target_points: np.ndarray,
    lens_model: str,
    image_size: tuple[int, int],
    fisheye_fov_deg: float | None = None,
    initial_rotation_matrix: np.ndarray | None = None,
) -> ProjectionSkyAlignmentTransform:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if sky_radec.ndim != 2 or target.ndim != 2 or sky_radec.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("已知投影匹配需要 Nx2 的 RA/Dec 点与目标点。")
    if sky_radec.shape[0] != target.shape[0]:
        raise ValueError("已知投影匹配的 RA/Dec 点与目标点数量不一致。")
    if lens_model not in SKY_KNOWN_PROJECTION_MODELS:
        raise ValueError(f"不支持的已知投影模型：{lens_model}")

    finite_mask = np.all(np.isfinite(sky_radec), axis=1) & np.all(np.isfinite(target), axis=1)
    sky_radec = sky_radec[finite_mask]
    target = target[finite_mask]
    pair_count = int(sky_radec.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对参考星才能拟合已知投影。")

    image_width = int(image_size[0])
    image_height = int(image_size[1])
    if image_width <= 0 or image_height <= 0:
        raise ValueError("真实图像尺寸无效，无法拟合已知投影。")

    rotation_matrix, center_x, center_y, scale_px, raw_projected, projection_rms = _fit_known_projection_parameters(
        sky_radec=sky_radec,
        target_points=target,
        lens_model=lens_model,
        image_size=(image_width, image_height),
        fisheye_fov_deg=fisheye_fov_deg,
        initial_rotation_matrix=initial_rotation_matrix,
    )
    residual_correction = _fit_residual_correction(raw_projected, target, (image_width, image_height))
    residual_vectors = residual_correction.corrected_projected - target
    rms_px = float(np.sqrt(np.mean(np.sum(residual_vectors * residual_vectors, axis=1))))
    if not np.isfinite(rms_px):
        raise ValueError("已知投影残差修正 RMS 包含无效数值。")

    return ProjectionSkyAlignmentTransform(
        lens_model=lens_model,
        pair_count=pair_count,
        image_width_px=image_width,
        image_height_px=image_height,
        fov_deg=None if fisheye_fov_deg is None else float(fisheye_fov_deg),
        rotation_matrix=rotation_matrix.astype(np.float64),
        center_x_px=float(center_x),
        center_y_px=float(center_y),
        scale_px=float(scale_px),
        residual_kind=residual_correction.residual_kind,
        residual_degree=int(residual_correction.degree),
        residual_origin_x_px=float(residual_correction.origin_x_px),
        residual_origin_y_px=float(residual_correction.origin_y_px),
        residual_scale_x_px=float(residual_correction.scale_x_px),
        residual_scale_y_px=float(residual_correction.scale_y_px),
        residual_coeff_x=residual_correction.coeff_x.astype(np.float64),
        residual_coeff_y=residual_correction.coeff_y.astype(np.float64),
        residual_anchor_points=residual_correction.anchor_points.astype(np.float64),
        residual_tps_weights_x=residual_correction.tps_weights_x.astype(np.float64),
        residual_tps_weights_y=residual_correction.tps_weights_y.astype(np.float64),
        residual_tps_affine_x=residual_correction.tps_affine_x.astype(np.float64),
        residual_tps_affine_y=residual_correction.tps_affine_y.astype(np.float64),
        projection_rms_px=float(projection_rms),
        rms_px=rms_px,
    )
