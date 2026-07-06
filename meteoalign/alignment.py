from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


MIN_ALIGNMENT_PAIRS = 4
QUADRATIC_ALIGNMENT_PAIRS = 6
MAX_ALIGNMENT_CONDITION_NUMBER = 1e10
MAX_TRANSFORM_ABS_PX = 1e9
SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION = "anchor_interpolation"
SKY_MATCHING_MODEL_POLYNOMIAL = "polynomial"
SKY_MATCHING_MODEL_RECTILINEAR = "rectilinear"
SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT = "fisheye_equidistant"
SKY_MATCHING_MODEL_FISHEYE_EQUISOLID = "fisheye_equisolid"
SKY_MATCHING_MODEL_MERCATOR = "mercator"
SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT = "cylindrical_equidistant"
SKY_KNOWN_PROJECTION_MODELS = (
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
)
SKY_SKY_PLANE_INTERPOLATION_MODELS = (SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION, SKY_MATCHING_MODEL_POLYNOMIAL)
SKY_MATCHING_MODELS = (*SKY_SKY_PLANE_INTERPOLATION_MODELS, *SKY_KNOWN_PROJECTION_MODELS)
PROJECTION_FIT_MAX_NFEV = 2000
PROJECTION_INVALID_RESIDUAL_PX = 1e6
RESIDUAL_CORRECTION_TPS = "thin_plate_spline"
ANCHOR_INTERPOLATION_TPS = "thin_plate_spline"
FIT_WEIGHT_MIN = 1e-6
FIT_WEIGHT_MAX = 1.0
SOFT_CONSTRAINT_TPS_SMOOTHING_BASE = 0.03
SKY_KNOWN_PROJECTION_DISPLAY_NAMES = {
    SKY_MATCHING_MODEL_RECTILINEAR: "普通透视镜头(TAN)",
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT: "等距鱼眼(ARC)",
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID: "等立体角鱼眼(ZEA)",
    SKY_MATCHING_MODEL_MERCATOR: "墨卡托(MER)",
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT: "等距圆柱(CAR)",
}
SKY_KNOWN_PROJECTION_CODES = {
    SKY_MATCHING_MODEL_RECTILINEAR: "TAN",
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT: "ARC",
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID: "ZEA",
    SKY_MATCHING_MODEL_MERCATOR: "MER",
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT: "CAR",
}


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
class AnchorInterpolation2D:
    kind: str
    origin_x: float
    origin_y: float
    scale_x: float
    scale_y: float
    anchor_points: np.ndarray
    tps_weights_x: np.ndarray
    tps_weights_y: np.ndarray
    tps_affine_x: np.ndarray
    tps_affine_y: np.ndarray

    def normalized_points(self, points: np.ndarray) -> np.ndarray:
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, 2)
        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise ValueError("锚点插值输入必须是 Nx2 点数组。")
        return np.column_stack(
            (
                (point_array[:, 0] - self.origin_x) / max(float(self.scale_x), 1e-12),
                (point_array[:, 1] - self.origin_y) / max(float(self.scale_y), 1e-12),
            )
        ).astype(np.float64)

    def evaluate_points(self, points: np.ndarray) -> np.ndarray:
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, 2)
        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise ValueError("锚点插值输入必须是 Nx2 点数组。")
        if self.kind != ANCHOR_INTERPOLATION_TPS:
            raise ValueError(f"不支持的锚点插值类型：{self.kind}")

        normalized = self.normalized_points(point_array)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            values = np.column_stack(
                (
                    _evaluate_thin_plate_spline(
                        normalized,
                        self.anchor_points,
                        self.tps_weights_x,
                        self.tps_affine_x,
                    ),
                    _evaluate_thin_plate_spline(
                        normalized,
                        self.anchor_points,
                        self.tps_weights_y,
                        self.tps_affine_y,
                    ),
                )
            )
        values[~np.all(np.isfinite(point_array), axis=1)] = np.nan
        values[~np.all(np.isfinite(values), axis=1)] = np.nan
        return values.astype(np.float64)


@dataclass(frozen=True)
class SkyAlignmentTransform:
    pair_count: int
    center_vector: np.ndarray
    east_vector: np.ndarray
    north_vector: np.ndarray
    interpolation: AnchorInterpolation2D
    rms_px: float
    residual_soft_constraint_count: int = 0
    residual_soft_weight_min: float = 1.0
    residual_soft_weight_max: float = 1.0

    @property
    def display_name(self) -> str:
        if self.residual_soft_constraint_count > 0:
            return "普适平滑插值"
        return "普适锚点插值"

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
        transformed = self.interpolation.evaluate_points(sky_points)
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
    residual_origin_x_px: float
    residual_origin_y_px: float
    residual_scale_x_px: float
    residual_scale_y_px: float
    residual_anchor_points: np.ndarray
    residual_tps_weights_x: np.ndarray
    residual_tps_weights_y: np.ndarray
    residual_tps_affine_x: np.ndarray
    residual_tps_affine_y: np.ndarray
    residual_hard_anchor_count: int
    residual_soft_constraint_count: int
    residual_soft_weight_min: float
    residual_soft_weight_max: float
    projection_rms_px: float
    rms_px: float

    @property
    def display_name(self) -> str:
        projection_name = SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(self.lens_model, self.lens_model)
        if self.residual_soft_constraint_count > 0:
            return f"{projection_name}+平滑残差"
        if self.residual_kind == RESIDUAL_CORRECTION_TPS:
            return f"{projection_name}+锚点插值"
        return f"{projection_name}+残差插值"

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
            origin_x_px=self.residual_origin_x_px,
            origin_y_px=self.residual_origin_y_px,
            scale_x_px=self.residual_scale_x_px,
            scale_y_px=self.residual_scale_y_px,
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


def _anchor_interpolation_is_valid(interpolation: AnchorInterpolation2D) -> bool:
    return (
        interpolation.kind == ANCHOR_INTERPOLATION_TPS
        and np.isfinite(float(interpolation.origin_x))
        and np.isfinite(float(interpolation.origin_y))
        and np.isfinite(float(interpolation.scale_x))
        and np.isfinite(float(interpolation.scale_y))
        and float(interpolation.scale_x) > 0.0
        and float(interpolation.scale_y) > 0.0
        and _array_is_finite(interpolation.anchor_points)
        and _array_is_finite(interpolation.tps_weights_x)
        and _array_is_finite(interpolation.tps_weights_y)
        and _array_is_finite(interpolation.tps_affine_x)
        and _array_is_finite(interpolation.tps_affine_y)
    )


def _sky_transform_components_are_valid(transform: SkyAlignmentTransform) -> bool:
    return (
        _array_is_finite(transform.center_vector)
        and _array_is_finite(transform.east_vector)
        and _array_is_finite(transform.north_vector)
        and _anchor_interpolation_is_valid(transform.interpolation)
    )


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


def _fit_point_weights(point_count: int, point_weights: np.ndarray | None) -> np.ndarray:
    if point_weights is None:
        return np.ones(point_count, dtype=np.float64)
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != point_count:
        raise ValueError("拟合权重数量必须与配对星数量一致。")
    if not np.all(np.isfinite(weights)):
        raise ValueError("拟合权重包含无效数值。")
    return np.clip(weights, FIT_WEIGHT_MIN, FIT_WEIGHT_MAX).astype(np.float64)


def _residual_anchor_mask(point_count: int, anchor_mask: np.ndarray | None) -> np.ndarray:
    if anchor_mask is None:
        return np.ones(point_count, dtype=bool)
    mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != point_count:
        raise ValueError("残差锚点标记数量必须与配对星数量一致。")
    return mask.astype(bool)


def _tps_smoothing_from_constraints(anchor_mask: np.ndarray, point_weights: np.ndarray) -> np.ndarray:
    mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    if mask.shape[0] != weights.shape[0]:
        raise ValueError("软约束权重数量必须与锚点标记数量一致。")

    smoothing = np.zeros(mask.shape[0], dtype=np.float64)
    soft_mask = ~mask
    if np.any(soft_mask):
        # 硬锚点保持精确穿过；软约束只通过对角正则项拉住曲面，权重越低越平滑。
        smoothing[soft_mask] = SOFT_CONSTRAINT_TPS_SMOOTHING_BASE / np.clip(
            weights[soft_mask],
            FIT_WEIGHT_MIN,
            FIT_WEIGHT_MAX,
        )
    return smoothing.astype(np.float64)


def _fit_known_projection_parameters(
    sky_radec: np.ndarray,
    target_points: np.ndarray,
    lens_model: str,
    image_size: tuple[int, int],
    fisheye_fov_deg: float | None,
    initial_rotation_matrix: np.ndarray | None,
    fit_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, np.ndarray, float]:
    vectors = _radec_to_unit_vectors(sky_radec[:, 0], sky_radec[:, 1])
    point_weights = _fit_point_weights(sky_radec.shape[0], fit_weights)
    sqrt_weights = np.sqrt(point_weights).reshape(-1, 1)
    width_px = max(float(image_size[0]), 1.0)
    height_px = max(float(image_size[1]), 1.0)
    fixed_scale = _fixed_fisheye_scale_px(lens_model, image_size, fisheye_fov_deg)
    initial_candidates: list[tuple[np.ndarray, np.ndarray, float | None, tuple[float, float] | None]] = []
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
            initial_rotation = _orthonormalized_rotation_matrix(initial_rotation_matrix)
        fixed_center = (width_px * 0.5, height_px * 0.5)
        initial = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
        lower = np.asarray((-np.pi, -np.pi, -np.pi), dtype=np.float64)
        upper = np.asarray((np.pi, np.pi, np.pi), dtype=np.float64)
        initial_candidates.append((initial_rotation, initial, float(fixed_scale), fixed_center))
    else:
        lower = np.asarray((-np.pi, -np.pi, -np.pi, np.log(1e-6), -2.0 * width_px, -2.0 * height_px), dtype=np.float64)
        upper = np.asarray((np.pi, np.pi, np.pi, np.log(1e9), 3.0 * width_px, 3.0 * height_px), dtype=np.float64)

        if initial_rotation_matrix is not None:
            try:
                if lens_model == SKY_MATCHING_MODEL_RECTILINEAR:
                    initial_rotation, initial_cx, initial_cy, initial_scale = _initial_rectilinear_from_rotation(
                        vectors,
                        target_points,
                        initial_rotation_matrix,
                        point_weights,
                    )
                else:
                    initial_rotation, initial_cx, initial_cy, initial_scale = _initial_projection_from_rotation(
                        vectors,
                        target_points,
                        initial_rotation_matrix,
                        lens_model,
                        point_weights,
                    )
                initial_candidates.append(
                    (
                        initial_rotation,
                        np.asarray(
                            (0.0, 0.0, 0.0, np.log(max(initial_scale, 1e-6)), initial_cx, initial_cy),
                            dtype=np.float64,
                        ),
                        None,
                        None,
                    )
                )
            except ValueError:
                pass

        initial_rotation, initial_cx, initial_cy, initial_scale = _initial_projection_basis(sky_radec, target_points)
        initial_candidates.append(
            (
                initial_rotation,
                np.asarray(
                    (0.0, 0.0, 0.0, np.log(max(initial_scale, 1e-6)), initial_cx, initial_cy),
                    dtype=np.float64,
                ),
                None,
                None,
            )
        )

    best_result: tuple[float, np.ndarray, np.ndarray, float | None, tuple[float, float] | None, np.ndarray, float] | None = None
    failure_messages: list[str] = []
    for candidate_rotation, candidate_initial, candidate_fixed_scale, candidate_fixed_center in initial_candidates:
        candidate_initial = np.clip(candidate_initial, lower + 1e-9, upper - 1e-9)

        def residual(params: np.ndarray) -> np.ndarray:
            projected, valid = _projection_points_from_params(
                vectors,
                candidate_rotation,
                params,
                lens_model,
                fixed_scale_px=candidate_fixed_scale,
                fixed_center_px=candidate_fixed_center,
            )
            residual_values = projected - target_points
            invalid = ~valid | ~np.all(np.isfinite(residual_values), axis=1)
            if np.any(invalid):
                residual_values[invalid] = PROJECTION_INVALID_RESIDUAL_PX
            residual_values = np.clip(residual_values, -PROJECTION_INVALID_RESIDUAL_PX, PROJECTION_INVALID_RESIDUAL_PX)
            return (residual_values * sqrt_weights).ravel()

        result = least_squares(
            residual,
            candidate_initial,
            bounds=(lower, upper),
            loss="linear",
            max_nfev=PROJECTION_FIT_MAX_NFEV,
        )
        if not result.success:
            failure_messages.append(str(result.message))
            continue

        projected, valid = _projection_points_from_params(
            vectors,
            candidate_rotation,
            result.x,
            lens_model,
            fixed_scale_px=candidate_fixed_scale,
            fixed_center_px=candidate_fixed_center,
            strict_visibility=True,
        )
        if not np.all(valid):
            failure_messages.append("拟合后仍有配对星无法投影")
            continue
        residual_vectors = projected - target_points
        if not _array_is_finite(residual_vectors):
            failure_messages.append("拟合残差包含无效数值")
            continue
        squared_distances = np.sum(residual_vectors * residual_vectors, axis=1)
        rms_px = float(np.sqrt(np.mean(squared_distances)))
        weighted_rms_px = float(np.sqrt(np.sum(point_weights * squared_distances) / np.sum(point_weights)))
        if not np.isfinite(rms_px) or not np.isfinite(weighted_rms_px):
            failure_messages.append("拟合 RMS 包含无效数值")
            continue
        if best_result is None or weighted_rms_px < best_result[0]:
            best_result = (
                weighted_rms_px,
                candidate_rotation,
                result.x.astype(np.float64),
                candidate_fixed_scale,
                candidate_fixed_center,
                projected.astype(np.float64),
                rms_px,
            )

    if best_result is None:
        detail = f"：{failure_messages[-1]}" if failure_messages else ""
        raise ValueError(f"已知投影参数拟合未收敛{detail}")

    _weighted_rms, initial_rotation, result_params, result_fixed_scale, result_fixed_center, projected, rms_px = best_result
    if not np.isfinite(rms_px):
        raise ValueError("已知投影拟合 RMS 包含无效数值。")
    rotation_matrix = _rotation_matrix_from_rotvec(result_params[:3]) @ initial_rotation
    if result_fixed_scale is not None and result_fixed_center is not None:
        scale_px = float(result_fixed_scale)
        center_x, center_y = result_fixed_center
    else:
        scale_px = float(np.exp(result_params[3]))
        center_x = float(result_params[4])
        center_y = float(result_params[5])
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
    smoothing: np.ndarray | None = None,
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
    if smoothing is None:
        smoothing_values = np.zeros(anchors.shape[0], dtype=np.float64)
    else:
        smoothing_values = np.asarray(smoothing, dtype=np.float64).reshape(-1)
        if smoothing_values.shape[0] != anchors.shape[0]:
            raise ValueError("薄板样条平滑参数数量必须与锚点数量一致。")
        if not np.all(np.isfinite(smoothing_values)) or np.any(smoothing_values < 0.0):
            raise ValueError("薄板样条平滑参数包含无效数值。")

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
    if np.any(smoothing_values > 0.0):
        system[: anchors.shape[0], : anchors.shape[0]] += np.diag(smoothing_values)
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("薄板样条矩阵无法稳定求解。") from exc

    weights = solution[: anchors.shape[0]].astype(np.float64)
    affine = solution[anchors.shape[0] :].astype(np.float64)
    if not np.any(smoothing_values > 0.0):
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


def _default_interpolation_normalization(points: np.ndarray) -> tuple[float, float, float, float]:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("锚点插值归一化需要 Nx2 点数组。")
    if not _array_is_finite(point_array):
        raise ValueError("锚点插值输入点包含无效数值。")

    origin_x = float(np.median(point_array[:, 0]))
    origin_y = float(np.median(point_array[:, 1]))
    span_x = float(np.ptp(point_array[:, 0]))
    span_y = float(np.ptp(point_array[:, 1]))
    scale_x = span_x if np.isfinite(span_x) and span_x > 1e-12 else 1.0
    scale_y = span_y if np.isfinite(span_y) and span_y > 1e-12 else 1.0
    return origin_x, origin_y, scale_x, scale_y


def _normalized_interpolation_points(
    points: np.ndarray,
    origin_x: float,
    origin_y: float,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    normalized = np.column_stack(
        (
            (point_array[:, 0] - origin_x) / max(float(scale_x), 1e-12),
            (point_array[:, 1] - origin_y) / max(float(scale_y), 1e-12),
        )
    )
    return normalized.astype(np.float64)


def fit_anchor_interpolation(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    origin_x: float | None = None,
    origin_y: float | None = None,
    scale_x: float | None = None,
    scale_y: float | None = None,
    anchor_mask: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
) -> AnchorInterpolation2D:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("锚点插值需要 Nx2 的源点与目标点。")
    if source.shape[0] != target.shape[0]:
        raise ValueError("锚点插值源点与目标点数量不一致。")
    if source.shape[0] < 3:
        raise ValueError("锚点插值至少需要 3 个锚点。")
    if not _array_is_finite(source) or not _array_is_finite(target):
        raise ValueError("锚点插值点包含无效数值。")
    hard_anchor_mask = _residual_anchor_mask(source.shape[0], anchor_mask)
    fit_weights = _fit_point_weights(source.shape[0], point_weights)
    smoothing = _tps_smoothing_from_constraints(hard_anchor_mask, fit_weights)

    default_origin_x, default_origin_y, default_scale_x, default_scale_y = _default_interpolation_normalization(source)
    input_origin_x = default_origin_x if origin_x is None else float(origin_x)
    input_origin_y = default_origin_y if origin_y is None else float(origin_y)
    input_scale_x = default_scale_x if scale_x is None else float(scale_x)
    input_scale_y = default_scale_y if scale_y is None else float(scale_y)
    if (
        not np.isfinite(input_origin_x)
        or not np.isfinite(input_origin_y)
        or not np.isfinite(input_scale_x)
        or not np.isfinite(input_scale_y)
        or input_scale_x <= 0.0
        or input_scale_y <= 0.0
    ):
        raise ValueError("锚点插值归一化参数无效。")

    normalized = _normalized_interpolation_points(source, input_origin_x, input_origin_y, input_scale_x, input_scale_y)
    weights_x, affine_x = _fit_thin_plate_spline_coefficients(normalized, target[:, 0], smoothing=smoothing)
    weights_y, affine_y = _fit_thin_plate_spline_coefficients(normalized, target[:, 1], smoothing=smoothing)
    interpolation = AnchorInterpolation2D(
        kind=ANCHOR_INTERPOLATION_TPS,
        origin_x=input_origin_x,
        origin_y=input_origin_y,
        scale_x=input_scale_x,
        scale_y=input_scale_y,
        anchor_points=normalized.astype(np.float64),
        tps_weights_x=weights_x,
        tps_weights_y=weights_y,
        tps_affine_x=affine_x,
        tps_affine_y=affine_y,
    )
    interpolated = interpolation.evaluate_points(source)
    if not _array_is_finite(interpolated):
        raise ValueError("锚点插值结果包含无效数值。")
    if np.any(hard_anchor_mask):
        max_anchor_error = float(np.max(np.abs(interpolated[hard_anchor_mask] - target[hard_anchor_mask])))
        if not np.isfinite(max_anchor_error) or max_anchor_error > 1e-5:
            raise ValueError("锚点插值未能精确穿过硬锚点。")
    return interpolation


@dataclass(frozen=True)
class ResidualCorrectionResult:
    residual_kind: str
    origin_x_px: float
    origin_y_px: float
    scale_x_px: float
    scale_y_px: float
    anchor_points: np.ndarray
    tps_weights_x: np.ndarray
    tps_weights_y: np.ndarray
    tps_affine_x: np.ndarray
    tps_affine_y: np.ndarray
    hard_anchor_count: int
    soft_constraint_count: int
    soft_weight_min: float
    soft_weight_max: float
    corrected_projected: np.ndarray


def _fit_residual_correction(
    projected_points: np.ndarray,
    target_points: np.ndarray,
    image_size: tuple[int, int],
    anchor_mask: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
) -> ResidualCorrectionResult:
    point_count = int(np.asarray(projected_points, dtype=np.float64).shape[0])
    hard_anchor_mask = _residual_anchor_mask(point_count, anchor_mask)
    fit_weights = _fit_point_weights(point_count, point_weights)
    _normalized, origin_x, origin_y, scale_x, scale_y = _normalize_residual_points(projected_points, image_size)
    residual = target_points - projected_points
    interpolation = fit_anchor_interpolation(
        projected_points,
        residual,
        origin_x=origin_x,
        origin_y=origin_y,
        scale_x=scale_x,
        scale_y=scale_y,
        anchor_mask=hard_anchor_mask,
        point_weights=fit_weights,
    )
    corrected = projected_points + interpolation.evaluate_points(projected_points)
    if not _array_is_finite(corrected):
        raise ValueError("薄板样条锚点插值结果包含无效数值。")
    return ResidualCorrectionResult(
        residual_kind=RESIDUAL_CORRECTION_TPS,
        origin_x_px=origin_x,
        origin_y_px=origin_y,
        scale_x_px=scale_x,
        scale_y_px=scale_y,
        anchor_points=interpolation.anchor_points.astype(np.float64),
        tps_weights_x=interpolation.tps_weights_x.astype(np.float64),
        tps_weights_y=interpolation.tps_weights_y.astype(np.float64),
        tps_affine_x=interpolation.tps_affine_x.astype(np.float64),
        tps_affine_y=interpolation.tps_affine_y.astype(np.float64),
        hard_anchor_count=int(np.count_nonzero(hard_anchor_mask)),
        soft_constraint_count=int(np.count_nonzero(~hard_anchor_mask)),
        soft_weight_min=float(np.min(fit_weights[~hard_anchor_mask])) if np.any(~hard_anchor_mask) else 1.0,
        soft_weight_max=float(np.max(fit_weights[~hard_anchor_mask])) if np.any(~hard_anchor_mask) else 1.0,
        corrected_projected=corrected.astype(np.float64),
    )


def _apply_residual_correction(
    *,
    projected_points: np.ndarray,
    residual_kind: str,
    origin_x_px: float,
    origin_y_px: float,
    scale_x_px: float,
    scale_y_px: float,
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
        raise ValueError(f"不支持的残差插值类型：{residual_kind}")
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
    matching_model: str = SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    image_size: tuple[int, int] | None = None,
    fisheye_fov_deg: float | None = None,
    initial_rotation_matrix: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
    residual_anchor_mask: np.ndarray | None = None,
) -> SkyAlignmentTransform | ProjectionSkyAlignmentTransform:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if sky_radec.ndim != 2 or target.ndim != 2 or sky_radec.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("天球配准需要 Nx2 的 RA/Dec 点与目标点。")
    if sky_radec.shape[0] != target.shape[0]:
        raise ValueError("天球配准的 RA/Dec 点与目标点数量不一致。")
    if matching_model not in SKY_MATCHING_MODELS:
        raise ValueError(f"不支持的匹配模型：{matching_model}")
    raw_point_weights = _fit_point_weights(sky_radec.shape[0], point_weights)
    raw_anchor_mask = _residual_anchor_mask(sky_radec.shape[0], residual_anchor_mask)

    finite_mask = np.all(np.isfinite(sky_radec), axis=1) & np.all(np.isfinite(target), axis=1)
    sky_radec = sky_radec[finite_mask]
    target = target[finite_mask]
    fit_weights = raw_point_weights[finite_mask]
    anchor_mask = raw_anchor_mask[finite_mask]
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
            point_weights=fit_weights,
            residual_anchor_mask=anchor_mask,
        )

    center, east, north = _sky_plane_basis(sky_radec)
    sky_points = _project_radec_to_sky_plane(sky_radec[:, 0], sky_radec[:, 1], center, east, north)
    if not _array_is_finite(sky_points):
        raise ValueError("天球投影包含无效数值，请检查配对星坐标。")
    interpolation = fit_anchor_interpolation(
        sky_points,
        target,
        anchor_mask=anchor_mask,
        point_weights=fit_weights,
    )
    predicted = interpolation.evaluate_points(sky_points)
    residual = predicted - target
    rms_px = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if not np.isfinite(rms_px):
        raise ValueError("天球锚点插值残差包含无效数值，请检查配对星位置。")
    return SkyAlignmentTransform(
        pair_count=pair_count,
        center_vector=center,
        east_vector=east,
        north_vector=north,
        interpolation=interpolation,
        rms_px=rms_px,
        residual_soft_constraint_count=int(np.count_nonzero(~anchor_mask)),
        residual_soft_weight_min=float(np.min(fit_weights[~anchor_mask])) if np.any(~anchor_mask) else 1.0,
        residual_soft_weight_max=float(np.max(fit_weights[~anchor_mask])) if np.any(~anchor_mask) else 1.0,
    )


def fit_projection_sky_alignment(
    ra_dec_points: np.ndarray,
    target_points: np.ndarray,
    lens_model: str,
    image_size: tuple[int, int],
    fisheye_fov_deg: float | None = None,
    initial_rotation_matrix: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
    residual_anchor_mask: np.ndarray | None = None,
) -> ProjectionSkyAlignmentTransform:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if sky_radec.ndim != 2 or target.ndim != 2 or sky_radec.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("已知投影匹配需要 Nx2 的 RA/Dec 点与目标点。")
    if sky_radec.shape[0] != target.shape[0]:
        raise ValueError("已知投影匹配的 RA/Dec 点与目标点数量不一致。")
    if lens_model not in SKY_KNOWN_PROJECTION_MODELS:
        raise ValueError(f"不支持的已知投影模型：{lens_model}")
    raw_point_weights = _fit_point_weights(sky_radec.shape[0], point_weights)
    raw_anchor_mask = _residual_anchor_mask(sky_radec.shape[0], residual_anchor_mask)

    finite_mask = np.all(np.isfinite(sky_radec), axis=1) & np.all(np.isfinite(target), axis=1)
    sky_radec = sky_radec[finite_mask]
    target = target[finite_mask]
    fit_weights = raw_point_weights[finite_mask]
    anchor_mask = raw_anchor_mask[finite_mask]
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
        fit_weights=fit_weights,
    )
    residual_correction = _fit_residual_correction(
        raw_projected,
        target,
        (image_width, image_height),
        anchor_mask=anchor_mask,
        point_weights=fit_weights,
    )
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
        residual_origin_x_px=float(residual_correction.origin_x_px),
        residual_origin_y_px=float(residual_correction.origin_y_px),
        residual_scale_x_px=float(residual_correction.scale_x_px),
        residual_scale_y_px=float(residual_correction.scale_y_px),
        residual_anchor_points=residual_correction.anchor_points.astype(np.float64),
        residual_tps_weights_x=residual_correction.tps_weights_x.astype(np.float64),
        residual_tps_weights_y=residual_correction.tps_weights_y.astype(np.float64),
        residual_tps_affine_x=residual_correction.tps_affine_x.astype(np.float64),
        residual_tps_affine_y=residual_correction.tps_affine_y.astype(np.float64),
        residual_hard_anchor_count=residual_correction.hard_anchor_count,
        residual_soft_constraint_count=residual_correction.soft_constraint_count,
        residual_soft_weight_min=residual_correction.soft_weight_min,
        residual_soft_weight_max=residual_correction.soft_weight_max,
        projection_rms_px=float(projection_rms),
        rms_px=rms_px,
    )
