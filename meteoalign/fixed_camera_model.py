from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from .alignment.constants import (
    FIT_WEIGHT_MAX,
    FIT_WEIGHT_MIN,
    SKY_KNOWN_PROJECTION_CODES,
    SKY_KNOWN_PROJECTION_DISPLAY_NAMES,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .alignment.fitting import fit_projection_sky_alignment
from .alignment.models import ProjectionSkyAlignmentTransform
from .alignment.projections import _project_unit_vectors_with_known_projection
from .alignment.residuals import _apply_residual_correction
from .coordinates import unit_vectors_to_radec
from .simulator import ObserverSettings, compute_altaz_from_radec, local_vectors_from_altaz


FIXED_CAMERA_MODEL_FORMAT = "meteoalign_fixed_camera_model"
FIXED_CAMERA_MODEL_VERSION = 1
FIXED_CAMERA_TIME_FIT_EPSILON_SECONDS = 1.0
FIXED_CAMERA_TIME_FIT_MAX_ITERATIONS = 4
FIXED_CAMERA_TIME_FIT_MAX_STEP_SECONDS = 180.0
FIXED_CAMERA_TIME_FIT_MIN_MOTION_PX_PER_SECOND = 1e-5
FIXED_CAMERA_INVERSE_MAX_ITERATIONS = 10
FIXED_CAMERA_INVERSE_FINITE_DIFF_STEP_PX = 0.5
FIXED_CAMERA_INVERSE_MAX_STEP_PX = 48.0
FIXED_CAMERA_INVERSE_TOLERANCE_PX = 3.0


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def _json_float(value: object, field_name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"固定相机模型缺少字段：{field_name}")
        return float(default)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"固定相机模型字段 {field_name} 不是有效数值。")
    return result


def _json_int(value: object, field_name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"固定相机模型缺少字段：{field_name}")
        return int(default)
    return int(value)


def _json_float_array(
    value: object,
    field_name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"固定相机模型字段 {field_name} 形状应为 {shape}，实际为 {array.shape}。")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"固定相机模型字段 {field_name} 包含无效数值。")
    return array.astype(np.float64)


def _json_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"固定相机模型字段 {field_name} 必须是对象。")
    return value


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _fit_point_weights(point_count: int, point_weights: np.ndarray | None) -> np.ndarray:
    if point_weights is None:
        return np.ones(point_count, dtype=np.float64)
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != point_count:
        raise ValueError("固定相机模型权重数量必须与匹配星数量一致。")
    if not np.all(np.isfinite(weights)):
        raise ValueError("固定相机模型权重包含无效数值。")
    return np.clip(weights, FIT_WEIGHT_MIN, FIT_WEIGHT_MAX).astype(np.float64)


@dataclass(frozen=True)
class FixedCameraModel:
    """固定三脚架序列共享的 ENU -> Pixel 静态模型。"""

    projection_transform: ProjectionSkyAlignmentTransform

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> "FixedCameraModel":
        """从导出的 JSON 字段恢复固定相机模型。"""
        if not isinstance(payload, dict):
            raise ValueError("固定相机模型 JSON 必须是对象。")
        if payload.get("kind") != "fixed_camera_enu_model":
            raise ValueError("JSON 中的 fixed_camera_model 不是固定相机 ENU 模型。")

        intrinsics = _json_mapping(payload.get("camera_intrinsics"), "camera_intrinsics")
        principal_point = _json_mapping(intrinsics.get("principal_point_px"), "camera_intrinsics.principal_point_px")
        pose = _json_mapping(payload.get("fixed_camera_pose"), "fixed_camera_pose")
        residual = _json_mapping(payload.get("static_residual_distortion"), "static_residual_distortion")
        normalization = _json_mapping(
            residual.get("normalization"),
            "static_residual_distortion.normalization",
        )
        diagnostics = payload.get("diagnostics")
        diagnostics_mapping = diagnostics if isinstance(diagnostics, dict) else {}

        image_width = _json_int(payload.get("image_width_px"), "image_width_px")
        image_height = _json_int(payload.get("image_height_px"), "image_height_px")
        lens_model = str(intrinsics.get("base_projection", "")).strip()
        if lens_model not in SKY_KNOWN_PROJECTION_MODELS:
            projection_code = str(intrinsics.get("projection_code", "")).strip().upper()
            code_to_model = {code: model for model, code in SKY_KNOWN_PROJECTION_CODES.items()}
            lens_model = code_to_model.get(projection_code, lens_model)
        if lens_model not in SKY_KNOWN_PROJECTION_MODELS:
            raise ValueError(f"固定相机模型包含不支持的基础投影：{lens_model}")

        fov_value = intrinsics.get("fov_deg")
        fov_deg = None if fov_value is None else _json_float(fov_value, "camera_intrinsics.fov_deg")
        transform = ProjectionSkyAlignmentTransform(
            lens_model=lens_model,
            pair_count=_json_int(
                diagnostics_mapping.get("fit_pair_count", residual.get("anchor_count")),
                "diagnostics.fit_pair_count",
                default=0,
            ),
            image_width_px=image_width,
            image_height_px=image_height,
            fov_deg=fov_deg,
            rotation_matrix=_json_float_array(
                pose.get("rotation_matrix_enu_to_camera"),
                "fixed_camera_pose.rotation_matrix_enu_to_camera",
                shape=(3, 3),
            ),
            center_x_px=_json_float(principal_point.get("x"), "camera_intrinsics.principal_point_px.x"),
            center_y_px=_json_float(principal_point.get("y"), "camera_intrinsics.principal_point_px.y"),
            scale_px=_json_float(intrinsics.get("scale_px"), "camera_intrinsics.scale_px"),
            residual_kind=str(residual.get("kind", "")),
            residual_origin_x_px=_json_float(
                normalization.get("origin_x_px"),
                "static_residual_distortion.normalization.origin_x_px",
            ),
            residual_origin_y_px=_json_float(
                normalization.get("origin_y_px"),
                "static_residual_distortion.normalization.origin_y_px",
            ),
            residual_scale_x_px=_json_float(
                normalization.get("scale_x_px"),
                "static_residual_distortion.normalization.scale_x_px",
            ),
            residual_scale_y_px=_json_float(
                normalization.get("scale_y_px"),
                "static_residual_distortion.normalization.scale_y_px",
            ),
            residual_anchor_points=_json_float_array(
                residual.get("anchor_points_normalized", []),
                "static_residual_distortion.anchor_points_normalized",
            ).reshape((-1, 2)),
            residual_tps_weights_x=_json_float_array(
                residual.get("tps_weights_dx_px", []),
                "static_residual_distortion.tps_weights_dx_px",
            ).reshape(-1),
            residual_tps_weights_y=_json_float_array(
                residual.get("tps_weights_dy_px", []),
                "static_residual_distortion.tps_weights_dy_px",
            ).reshape(-1),
            residual_tps_affine_x=_json_float_array(
                residual.get("tps_affine_dx_px", []),
                "static_residual_distortion.tps_affine_dx_px",
            ).reshape(-1),
            residual_tps_affine_y=_json_float_array(
                residual.get("tps_affine_dy_px", []),
                "static_residual_distortion.tps_affine_dy_px",
            ).reshape(-1),
            residual_hard_anchor_count=_json_int(residual.get("hard_anchor_count"), "hard_anchor_count", default=0),
            residual_soft_constraint_count=_json_int(
                residual.get("soft_constraint_count"),
                "soft_constraint_count",
                default=0,
            ),
            residual_soft_weight_min=_json_float(
                residual.get("soft_constraint_weight_min"),
                "soft_constraint_weight_min",
                default=1.0,
            ),
            residual_soft_weight_max=_json_float(
                residual.get("soft_constraint_weight_max"),
                "soft_constraint_weight_max",
                default=1.0,
            ),
            projection_rms_px=_json_float(
                diagnostics_mapping.get("projection_rms_px_before_residual"),
                "diagnostics.projection_rms_px_before_residual",
                default=float("nan"),
            ),
            rms_px=_json_float(diagnostics_mapping.get("rms_px"), "diagnostics.rms_px", default=float("nan")),
        )
        return cls(projection_transform=transform)

    @property
    def lens_model(self) -> str:
        return self.projection_transform.lens_model

    @property
    def image_width_px(self) -> int:
        return int(self.projection_transform.image_width_px)

    @property
    def image_height_px(self) -> int:
        return int(self.projection_transform.image_height_px)

    def project_enu_vectors(self, enu_vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(enu_vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, 3)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("固定相机模型需要 Nx3 的 ENU 单位向量。")

        transform = self.projection_transform
        projected, valid = _project_unit_vectors_with_known_projection(
            vectors=vectors,
            rotation_matrix=transform.rotation_matrix,
            center_x_px=transform.center_x_px,
            center_y_px=transform.center_y_px,
            scale_px=transform.scale_px,
            lens_model=transform.lens_model,
            strict_visibility=True,
        )
        corrected = _apply_residual_correction(
            projected_points=projected,
            residual_kind=transform.residual_kind,
            origin_x_px=transform.residual_origin_x_px,
            origin_y_px=transform.residual_origin_y_px,
            scale_x_px=transform.residual_scale_x_px,
            scale_y_px=transform.residual_scale_y_px,
            anchor_points=transform.residual_anchor_points,
            tps_weights_x=transform.residual_tps_weights_x,
            tps_weights_y=transform.residual_tps_weights_y,
            tps_affine_x=transform.residual_tps_affine_x,
            tps_affine_y=transform.residual_tps_affine_y,
        )
        corrected[~valid] = np.nan
        corrected[~np.all(np.isfinite(corrected), axis=1)] = np.nan
        return corrected.astype(np.float64)

    def project_radec_points(
        self,
        ra_dec_points: np.ndarray,
        observer: ObserverSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radec = np.asarray(ra_dec_points, dtype=np.float64)
        if radec.ndim == 1:
            radec = radec.reshape(1, 2)
        if radec.ndim != 2 or radec.shape[1] != 2:
            raise ValueError("固定相机模型需要 Nx2 的 RA/Dec 数组。")

        alt_deg, az_deg = compute_altaz_from_radec(radec[:, 0], radec[:, 1], observer)
        enu_vectors = local_vectors_from_altaz(alt_deg, az_deg)
        return self.project_enu_vectors(enu_vectors), alt_deg.astype(np.float64), az_deg.astype(np.float64)

    def project_radec_at_time(
        self,
        ra_dec_points: np.ndarray,
        *,
        observation_time_utc: datetime,
        latitude_deg: float,
        longitude_deg: float,
        elevation_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observer = ObserverSettings(
            observation_time_utc=_ensure_utc(observation_time_utc),
            latitude_deg=float(latitude_deg),
            longitude_deg=float(longitude_deg),
            elevation_m=float(elevation_m),
        )
        return self.project_radec_points(ra_dec_points, observer)

    def _apply_static_residual_correction(self, projected_pixels: np.ndarray) -> np.ndarray:
        transform = self.projection_transform
        return _apply_residual_correction(
            projected_points=projected_pixels,
            residual_kind=transform.residual_kind,
            origin_x_px=transform.residual_origin_x_px,
            origin_y_px=transform.residual_origin_y_px,
            scale_x_px=transform.residual_scale_x_px,
            scale_y_px=transform.residual_scale_y_px,
            anchor_points=transform.residual_anchor_points,
            tps_weights_x=transform.residual_tps_weights_x,
            tps_weights_y=transform.residual_tps_weights_y,
            tps_affine_x=transform.residual_tps_affine_x,
            tps_affine_y=transform.residual_tps_affine_y,
        )

    def _uncorrect_projection_pixels(self, pixel_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(pixel_points, dtype=np.float64)
        if target.ndim == 1:
            target = target.reshape(1, 2)
        if target.ndim != 2 or target.shape[1] != 2:
            raise ValueError("固定相机反解需要 Nx2 的像素坐标。")

        raw = target.copy()
        finite_target = np.all(np.isfinite(target), axis=1)
        raw[~finite_target] = np.nan
        for _iteration_index in range(FIXED_CAMERA_INVERSE_MAX_ITERATIONS):
            corrected = self._apply_static_residual_correction(raw)
            residual = corrected - target
            residual_norm = np.linalg.norm(residual, axis=1)
            active = finite_target & np.isfinite(residual_norm) & (residual_norm > 1e-4)
            if not np.any(active):
                break

            eps = FIXED_CAMERA_INVERSE_FINITE_DIFF_STEP_PX
            raw_dx = raw.copy()
            raw_dy = raw.copy()
            raw_dx[active, 0] += eps
            raw_dy[active, 1] += eps
            corrected_dx = self._apply_static_residual_correction(raw_dx)
            corrected_dy = self._apply_static_residual_correction(raw_dy)
            jac_x = (corrected_dx - corrected) / eps
            jac_y = (corrected_dy - corrected) / eps
            j11 = jac_x[:, 0]
            j21 = jac_x[:, 1]
            j12 = jac_y[:, 0]
            j22 = jac_y[:, 1]
            determinant = j11 * j22 - j12 * j21
            solvable = active & np.isfinite(determinant) & (np.abs(determinant) > 1e-8)
            if not np.any(solvable):
                break

            delta_x = np.zeros(target.shape[0], dtype=np.float64)
            delta_y = np.zeros(target.shape[0], dtype=np.float64)
            delta_x[solvable] = (
                residual[solvable, 0] * j22[solvable] - residual[solvable, 1] * j12[solvable]
            ) / determinant[solvable]
            delta_y[solvable] = (
                -residual[solvable, 0] * j21[solvable] + residual[solvable, 1] * j11[solvable]
            ) / determinant[solvable]
            step_norm = np.hypot(delta_x, delta_y)
            step_scale = np.minimum(1.0, FIXED_CAMERA_INVERSE_MAX_STEP_PX / np.maximum(step_norm, 1e-9))
            raw[solvable, 0] -= delta_x[solvable] * step_scale[solvable]
            raw[solvable, 1] -= delta_y[solvable] * step_scale[solvable]
            raw[~np.all(np.isfinite(raw), axis=1)] = np.nan

        final_residual = self._apply_static_residual_correction(raw) - target
        final_norm = np.linalg.norm(final_residual, axis=1)
        valid = finite_target & np.isfinite(final_norm) & (final_norm <= FIXED_CAMERA_INVERSE_TOLERANCE_PX)
        raw[~valid] = np.nan
        return raw.astype(np.float64), valid.astype(bool)

    def _camera_vectors_from_base_projection_pixels(self, projected_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = self.projection_transform
        points = np.asarray(projected_pixels, dtype=np.float64)
        plane_x = (points[:, 0] - float(transform.center_x_px)) / max(float(transform.scale_px), 1e-12)
        plane_y = (float(transform.center_y_px) - points[:, 1]) / max(float(transform.scale_px), 1e-12)
        vectors = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
        valid = np.all(np.isfinite(points), axis=1)

        if transform.lens_model == SKY_MATCHING_MODEL_RECTILINEAR:
            vectors[valid] = np.column_stack((plane_x[valid], plane_y[valid], np.ones(np.count_nonzero(valid))))
        elif transform.lens_model in (SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID):
            radius = np.hypot(plane_x, plane_y)
            if transform.lens_model == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT:
                theta = radius
                valid &= theta <= np.pi + 1e-8
            else:
                valid &= radius <= 2.0 + 1e-8
                theta = 2.0 * np.arcsin(np.clip(radius * 0.5, -1.0, 1.0))
            sin_theta = np.sin(theta)
            scale = np.divide(sin_theta, radius, out=np.ones_like(radius), where=radius > 1e-12)
            vectors[valid] = np.column_stack(
                (
                    plane_x[valid] * scale[valid],
                    plane_y[valid] * scale[valid],
                    np.cos(theta[valid]),
                )
            )
        elif transform.lens_model in (SKY_MATCHING_MODEL_MERCATOR, SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT):
            longitude = plane_x
            if transform.lens_model == SKY_MATCHING_MODEL_MERCATOR:
                latitude = np.arcsin(np.clip(np.tanh(plane_y), -1.0, 1.0))
            else:
                latitude = plane_y
                valid &= np.abs(latitude) <= np.pi * 0.5 + 1e-8
            cos_lat = np.cos(latitude)
            vectors[valid] = np.column_stack(
                (
                    cos_lat[valid] * np.sin(longitude[valid]),
                    np.sin(latitude[valid]),
                    cos_lat[valid] * np.cos(longitude[valid]),
                )
            )
        else:
            raise ValueError(f"不支持的固定相机反投影模型：{transform.lens_model}")

        norm = np.linalg.norm(vectors, axis=1)
        valid &= np.isfinite(norm) & (norm > 1e-12)
        vectors[valid] /= norm[valid, None]
        vectors[~valid] = np.nan
        return vectors.astype(np.float64), valid.astype(bool)

    def pixel_to_altaz_points(self, pixel_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_pixels, residual_valid = self._uncorrect_projection_pixels(pixel_points)
        camera_vectors, projection_valid = self._camera_vectors_from_base_projection_pixels(raw_pixels)
        enu_vectors = camera_vectors @ np.asarray(self.projection_transform.rotation_matrix, dtype=np.float64)
        norm = np.linalg.norm(enu_vectors, axis=1)
        valid = residual_valid & projection_valid & np.isfinite(norm) & (norm > 1e-12)
        unit = np.full_like(enu_vectors, np.nan, dtype=np.float64)
        unit[valid] = enu_vectors[valid] / norm[valid, None]
        alt_deg = np.full(unit.shape[0], np.nan, dtype=np.float64)
        az_deg = np.full(unit.shape[0], np.nan, dtype=np.float64)
        alt_deg[valid] = np.rad2deg(np.arcsin(np.clip(unit[valid, 2], -1.0, 1.0)))
        az_deg[valid] = np.rad2deg(np.arctan2(unit[valid, 0], unit[valid, 1])) % 360.0
        return alt_deg.astype(np.float64), az_deg.astype(np.float64), valid.astype(bool)

    def to_json_payload(self) -> dict[str, Any]:
        transform = self.projection_transform
        return {
            "kind": "fixed_camera_enu_model",
            "coordinate_input": "local ENU unit vector",
            "coordinate_axis_order": ["east", "north", "up"],
            "image_width_px": int(transform.image_width_px),
            "image_height_px": int(transform.image_height_px),
            "camera_intrinsics": {
                "base_projection": transform.lens_model,
                "projection_code": SKY_KNOWN_PROJECTION_CODES.get(transform.lens_model, transform.lens_model),
                "display_name": SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(transform.lens_model, transform.lens_model),
                "fov_deg": None if transform.fov_deg is None else float(transform.fov_deg),
                "principal_point_px": {
                    "x": float(transform.center_x_px),
                    "y": float(transform.center_y_px),
                },
                "scale_px": float(transform.scale_px),
            },
            "fixed_camera_pose": {
                "rotation_matrix_enu_to_camera": [
                    _as_float_list(row) for row in np.asarray(transform.rotation_matrix, dtype=np.float64)
                ],
            },
            "static_residual_distortion": {
                "kind": transform.residual_kind,
                "input": "base projection pixel coordinates",
                "normalization": {
                    "origin_x_px": float(transform.residual_origin_x_px),
                    "origin_y_px": float(transform.residual_origin_y_px),
                    "scale_x_px": float(transform.residual_scale_x_px),
                    "scale_y_px": float(transform.residual_scale_y_px),
                },
                "anchor_count": int(transform.residual_anchor_points.shape[0]),
                "hard_anchor_count": int(transform.residual_hard_anchor_count),
                "soft_constraint_count": int(transform.residual_soft_constraint_count),
                "soft_constraint_weight_min": float(transform.residual_soft_weight_min),
                "soft_constraint_weight_max": float(transform.residual_soft_weight_max),
                "anchor_points_normalized": [
                    _as_float_list(row) for row in np.asarray(transform.residual_anchor_points, dtype=np.float64)
                ],
                "tps_weights_dx_px": _as_float_list(transform.residual_tps_weights_x),
                "tps_weights_dy_px": _as_float_list(transform.residual_tps_weights_y),
                "tps_affine_dx_px": _as_float_list(transform.residual_tps_affine_x),
                "tps_affine_dy_px": _as_float_list(transform.residual_tps_affine_y),
            },
            "diagnostics": {
                "fit_pair_count": int(transform.pair_count),
                "projection_rms_px_before_residual": float(transform.projection_rms_px),
                "rms_px": float(transform.rms_px),
            },
        }


@dataclass(frozen=True)
class FixedCameraTimeFitResult:
    delta_t_seconds: float
    iteration_count: int
    accepted_mask: np.ndarray
    predicted_pixels: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray
    rms_px: float
    median_residual_px: float
    max_residual_px: float
    inlier_ratio: float
    projected_motion_px_per_s_median: float

    @property
    def accepted_count(self) -> int:
        return int(np.count_nonzero(self.accepted_mask))

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "delta_t_seconds": float(self.delta_t_seconds),
            "iteration_count": int(self.iteration_count),
            "accepted_count": self.accepted_count,
            "inlier_ratio": float(self.inlier_ratio),
            "rms_px": float(self.rms_px),
            "median_residual_px": float(self.median_residual_px),
            "max_residual_px": float(self.max_residual_px),
            "projected_motion_px_per_s_median": float(self.projected_motion_px_per_s_median),
        }


def fit_fixed_camera_model(
    *,
    enu_vectors: np.ndarray,
    pixel_points: np.ndarray,
    image_size: tuple[int, int],
    lens_model: str,
    initial_rotation_matrix: np.ndarray | None = None,
    fisheye_fov_deg: float | None = None,
    point_weights: np.ndarray | None = None,
    residual_anchor_mask: np.ndarray | None = None,
) -> FixedCameraModel:
    vectors = np.asarray(enu_vectors, dtype=np.float64)
    pixels = np.asarray(pixel_points, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("固定相机模型需要 Nx3 的 ENU 单位向量。")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("固定相机模型需要 Nx2 的像素坐标。")
    if vectors.shape[0] != pixels.shape[0]:
        raise ValueError("固定相机模型的 ENU 向量与像素点数量不一致。")
    if lens_model not in SKY_KNOWN_PROJECTION_MODELS:
        raise ValueError("固定相机序列模型需要选择普通透视、鱼眼或其他已知基础投影模型。")

    finite = np.all(np.isfinite(vectors), axis=1) & np.all(np.isfinite(pixels), axis=1)
    vector_norm = np.linalg.norm(vectors, axis=1)
    finite &= np.isfinite(vector_norm) & (vector_norm > 1e-12)
    if np.count_nonzero(finite) < 4:
        raise ValueError("固定相机模型至少需要 4 个有效恒星观测。")

    normalized_vectors = vectors[finite] / vector_norm[finite, None]
    weights = _fit_point_weights(vectors.shape[0], point_weights)[finite]
    if residual_anchor_mask is None:
        anchor_mask = None
    else:
        raw_anchor_mask = np.asarray(residual_anchor_mask, dtype=bool).reshape(-1)
        if raw_anchor_mask.shape[0] != vectors.shape[0]:
            raise ValueError("固定相机模型锚点标记数量必须与匹配星数量一致。")
        anchor_mask = raw_anchor_mask[finite]
    pseudo_radec = unit_vectors_to_radec(normalized_vectors)
    transform = fit_projection_sky_alignment(
        ra_dec_points=pseudo_radec,
        target_points=pixels[finite],
        lens_model=lens_model,
        image_size=image_size,
        fisheye_fov_deg=fisheye_fov_deg,
        initial_rotation_matrix=initial_rotation_matrix,
        point_weights=weights,
        residual_anchor_mask=anchor_mask,
    )
    return FixedCameraModel(projection_transform=transform)


def estimate_frame_time_correction(
    *,
    fixed_model: FixedCameraModel,
    ra_dec_points: np.ndarray,
    observed_pixels: np.ndarray,
    nominal_time_utc: datetime,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float,
    initial_delta_seconds: float = 0.0,
    point_weights: np.ndarray | None = None,
    epsilon_seconds: float = FIXED_CAMERA_TIME_FIT_EPSILON_SECONDS,
    max_iterations: int = FIXED_CAMERA_TIME_FIT_MAX_ITERATIONS,
) -> FixedCameraTimeFitResult:
    radec = np.asarray(ra_dec_points, dtype=np.float64)
    observed = np.asarray(observed_pixels, dtype=np.float64)
    if radec.ndim != 2 or radec.shape[1] != 2 or observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("时间修正需要 Nx2 的 RA/Dec 与观测像素数组。")
    if radec.shape[0] != observed.shape[0]:
        raise ValueError("时间修正的恒星数量与观测像素数量不一致。")

    weights = _fit_point_weights(radec.shape[0], point_weights)
    finite_input = np.all(np.isfinite(radec), axis=1) & np.all(np.isfinite(observed), axis=1) & np.isfinite(weights)
    delta = float(initial_delta_seconds) if np.isfinite(initial_delta_seconds) else 0.0
    iteration_count = 0
    accepted_mask = finite_input.copy()
    epsilon = max(float(epsilon_seconds), 1e-3)
    nominal_time = _ensure_utc(nominal_time_utc)

    for iteration_index in range(max(0, int(max_iterations))):
        current_time = nominal_time + timedelta(seconds=delta)
        predicted, _alt_deg, _az_deg = fixed_model.project_radec_at_time(
            radec,
            observation_time_utc=current_time,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            elevation_m=elevation_m,
        )
        forward, _forward_alt, _forward_az = fixed_model.project_radec_at_time(
            radec,
            observation_time_utc=current_time + timedelta(seconds=epsilon),
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            elevation_m=elevation_m,
        )
        backward, _backward_alt, _backward_az = fixed_model.project_radec_at_time(
            radec,
            observation_time_utc=current_time - timedelta(seconds=epsilon),
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            elevation_m=elevation_m,
        )
        velocity = (forward - backward) / (2.0 * epsilon)
        residual = observed - predicted
        residual_norm = np.linalg.norm(residual, axis=1)
        velocity_norm = np.linalg.norm(velocity, axis=1)
        valid = (
            finite_input
            & np.all(np.isfinite(predicted), axis=1)
            & np.all(np.isfinite(velocity), axis=1)
            & np.isfinite(residual_norm)
            & np.isfinite(velocity_norm)
            & (velocity_norm > FIXED_CAMERA_TIME_FIT_MIN_MOTION_PX_PER_SECOND)
        )
        if np.count_nonzero(valid) < 2:
            accepted_mask = valid
            break

        valid_residual = residual_norm[valid]
        robust_scale = 1.4826 * float(np.median(np.abs(valid_residual - np.median(valid_residual))))
        if not np.isfinite(robust_scale) or robust_scale <= 1e-6:
            robust_scale = max(float(np.median(valid_residual)), 1.0)
        cutoff = max(3.0 * robust_scale, 1.0)
        robust_weights = np.ones(radec.shape[0], dtype=np.float64)
        robust_weights[valid] = np.minimum(1.0, cutoff / np.maximum(valid_residual, 1e-9))
        active = valid & (robust_weights > 0.05)
        if np.count_nonzero(active) < 2:
            active = valid

        numerator = np.sum(weights[active] * robust_weights[active] * np.sum(velocity[active] * residual[active], axis=1))
        denominator = np.sum(weights[active] * robust_weights[active] * np.sum(velocity[active] * velocity[active], axis=1))
        accepted_mask = active
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 1e-12:
            break

        step = float(numerator / denominator)
        step = float(np.clip(step, -FIXED_CAMERA_TIME_FIT_MAX_STEP_SECONDS, FIXED_CAMERA_TIME_FIT_MAX_STEP_SECONDS))
        if not np.isfinite(step):
            break
        delta += step
        iteration_count = iteration_index + 1
        if abs(step) < 0.01:
            break

    final_time = nominal_time + timedelta(seconds=delta)
    predicted, alt_deg, az_deg = fixed_model.project_radec_at_time(
        radec,
        observation_time_utc=final_time,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        elevation_m=elevation_m,
    )
    forward, _forward_alt, _forward_az = fixed_model.project_radec_at_time(
        radec,
        observation_time_utc=final_time + timedelta(seconds=epsilon),
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        elevation_m=elevation_m,
    )
    backward, _backward_alt, _backward_az = fixed_model.project_radec_at_time(
        radec,
        observation_time_utc=final_time - timedelta(seconds=epsilon),
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        elevation_m=elevation_m,
    )
    velocity_norm = np.linalg.norm((forward - backward) / (2.0 * epsilon), axis=1)
    final_residual = observed - predicted
    residual_norm = np.linalg.norm(final_residual, axis=1)
    finite_final = (
        finite_input
        & np.all(np.isfinite(predicted), axis=1)
        & np.isfinite(residual_norm)
        & np.isfinite(velocity_norm)
        & (velocity_norm > FIXED_CAMERA_TIME_FIT_MIN_MOTION_PX_PER_SECOND)
    )
    if np.any(accepted_mask):
        finite_final &= accepted_mask
    final_distances = residual_norm[finite_final]
    if final_distances.size:
        rms_px = float(np.sqrt(np.mean(final_distances * final_distances)))
        median_px = float(np.median(final_distances))
        max_px = float(np.max(final_distances))
    else:
        rms_px = median_px = max_px = float("nan")
    valid_motion = velocity_norm[np.isfinite(velocity_norm)]
    motion_median = float(np.median(valid_motion)) if valid_motion.size else float("nan")
    inlier_ratio = float(np.count_nonzero(finite_final) / max(int(np.count_nonzero(finite_input)), 1))

    return FixedCameraTimeFitResult(
        delta_t_seconds=float(delta),
        iteration_count=int(iteration_count),
        accepted_mask=finite_final.astype(bool),
        predicted_pixels=predicted.astype(np.float64),
        alt_deg=alt_deg.astype(np.float64),
        az_deg=az_deg.astype(np.float64),
        rms_px=rms_px,
        median_residual_px=median_px,
        max_residual_px=max_px,
        inlier_ratio=inlier_ratio,
        projected_motion_px_per_s_median=motion_median,
    )
