from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .alignment import (
    AnchorInterpolation2D,
    ProjectionSkyAlignmentTransform,
    RESIDUAL_CORRECTION_TPS,
    SKY_KNOWN_PROJECTION_CODES,
    SKY_KNOWN_PROJECTION_DISPLAY_NAMES,
    SKY_KNOWN_PROJECTION_MODELS,
    _apply_residual_correction,
    _project_unit_vectors_with_known_projection,
)


CAMERA_CALIBRATION_PROFILE_VERSION = 1
BASE_PROJECTION_TANGENT_AEQD = "azimuthal_equidistant_tangent"
GLOBAL_DISTORTION_TPS_RESIDUAL_FIELD = "tps_residual_field"
GLOBAL_DISTORTION_TANGENT_PLANE_TPS = "tangent_plane_to_pixel_tps"
INVERSE_MODEL_ANALYTIC_WITH_RESIDUAL_UNWARP = "analytic_projection_with_iterative_residual_unwarp"
INVERSE_MODEL_TANGENT_TPS_NUMERICAL = "tangent_plane_tps_seed_with_numerical_forward_inverse"
CALIBRATION_INVERSE_MAX_ITERATIONS = 14
CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG = 1e-5
CALIBRATION_INVERSE_MAX_STEP_DEG = 2.0
CALIBRATION_INVERSE_TOLERANCE_PX = 1e-5
CALIBRATION_RESIDUAL_UNWARP_MAX_ITERATIONS = 10
CALIBRATION_RESIDUAL_UNWARP_FINITE_DIFF_STEP_PX = 0.5
CALIBRATION_RESIDUAL_UNWARP_MAX_STEP_PX = 48.0
CALIBRATION_RESIDUAL_UNWARP_TOLERANCE_PX = 3.0


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def _json_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"CameraCalibrationProfile 字段 {field_name} 必须是对象。")
    return value


def _json_float(value: object, field_name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"CameraCalibrationProfile 缺少字段：{field_name}")
        return float(default)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"CameraCalibrationProfile 字段 {field_name} 不是有效数值。")
    return result


def _json_int(value: object, field_name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"CameraCalibrationProfile 缺少字段：{field_name}")
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
        raise ValueError(f"CameraCalibrationProfile 字段 {field_name} 形状应为 {shape}，实际为 {array.shape}。")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"CameraCalibrationProfile 字段 {field_name} 包含无效数值。")
    return array.astype(np.float64)


def _interpolation_payload(
    interpolation: AnchorInterpolation2D,
    *,
    input_units: str,
    output_units: str,
    input_axis_order: list[str],
    output_axis_order: list[str],
    weight_names: tuple[str, str],
    affine_names: tuple[str, str],
) -> dict[str, Any]:
    return {
        "kind": interpolation.kind,
        "input_units": input_units,
        "output_units": output_units,
        "input_axis_order": input_axis_order,
        "output_axis_order": output_axis_order,
        "normalization": {
            "origin_x": float(interpolation.origin_x),
            "origin_y": float(interpolation.origin_y),
            "scale_x": float(interpolation.scale_x),
            "scale_y": float(interpolation.scale_y),
        },
        "anchor_count": int(interpolation.anchor_points.shape[0]),
        "anchor_points_normalized": [
            _as_float_list(row) for row in np.asarray(interpolation.anchor_points, dtype=np.float64)
        ],
        weight_names[0]: _as_float_list(interpolation.tps_weights_x),
        weight_names[1]: _as_float_list(interpolation.tps_weights_y),
        affine_names[0]: _as_float_list(interpolation.tps_affine_x),
        affine_names[1]: _as_float_list(interpolation.tps_affine_y),
    }


def _interpolation_from_payload(
    payload: dict[str, Any],
    field_name: str,
    *,
    weight_names: tuple[str, str],
    affine_names: tuple[str, str],
) -> AnchorInterpolation2D:
    normalization = _json_mapping(payload.get("normalization"), f"{field_name}.normalization")
    return AnchorInterpolation2D(
        kind=str(payload.get("kind", "")),
        origin_x=_json_float(normalization.get("origin_x"), f"{field_name}.normalization.origin_x"),
        origin_y=_json_float(normalization.get("origin_y"), f"{field_name}.normalization.origin_y"),
        scale_x=_json_float(normalization.get("scale_x"), f"{field_name}.normalization.scale_x"),
        scale_y=_json_float(normalization.get("scale_y"), f"{field_name}.normalization.scale_y"),
        anchor_points=_json_float_array(
            payload.get("anchor_points_normalized", []),
            f"{field_name}.anchor_points_normalized",
        ).reshape((-1, 2)),
        tps_weights_x=_json_float_array(payload.get(weight_names[0], []), f"{field_name}.{weight_names[0]}").reshape(-1),
        tps_weights_y=_json_float_array(payload.get(weight_names[1], []), f"{field_name}.{weight_names[1]}").reshape(-1),
        tps_affine_x=_json_float_array(payload.get(affine_names[0], []), f"{field_name}.{affine_names[0]}").reshape(-1),
        tps_affine_y=_json_float_array(payload.get(affine_names[1], []), f"{field_name}.{affine_names[1]}").reshape(-1),
    )


def camera_vectors_to_tangent_plane_deg(camera_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(camera_vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 3)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("相机射线必须是 Nx3 数组。")

    norm = np.linalg.norm(vectors, axis=1)
    valid = np.all(np.isfinite(vectors), axis=1) & np.isfinite(norm) & (norm > 1e-12)
    unit = np.full_like(vectors, np.nan, dtype=np.float64)
    unit[valid] = vectors[valid] / norm[valid, None]
    center_component = np.clip(unit[:, 2], -1.0, 1.0)
    theta = np.arccos(center_component)
    sin_theta = np.sin(theta)
    scale = np.divide(theta, sin_theta, out=np.ones_like(theta), where=np.abs(sin_theta) > 1e-12)
    plane = np.column_stack((unit[:, 0] * scale, unit[:, 1] * scale)) * (180.0 / np.pi)
    valid &= np.isfinite(theta) & (theta < np.pi - 1e-8) & np.all(np.isfinite(plane), axis=1)
    plane[~valid] = np.nan
    return plane.astype(np.float64), valid.astype(bool)


def tangent_plane_deg_to_camera_vectors(plane_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(plane_points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("相机切平面点必须是 Nx2 数组。")

    radius_deg = np.linalg.norm(points, axis=1)
    radius_rad = np.deg2rad(radius_deg)
    valid = np.all(np.isfinite(points), axis=1) & np.isfinite(radius_rad) & (radius_rad < np.pi - 1e-8)
    vectors = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
    nonzero = valid & (radius_deg > 1e-12)
    if np.any(nonzero):
        scale = np.sin(radius_rad[nonzero]) / radius_deg[nonzero]
        vectors[nonzero, 0] = points[nonzero, 0] * scale
        vectors[nonzero, 1] = points[nonzero, 1] * scale
        vectors[nonzero, 2] = np.cos(radius_rad[nonzero])
    zero = valid & ~nonzero
    vectors[zero] = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return vectors.astype(np.float64), valid.astype(bool)


@dataclass(frozen=True)
class CameraCalibrationProfile:
    """相机自身的 Camera Ray ↔ Pixel 几何，不包含姿态、RA/Dec、时间或地点。"""

    image_width_px: int
    image_height_px: int
    base_projection_type: str
    principal_point_x_px: float
    principal_point_y_px: float
    scale_x_px: float
    scale_y_px: float
    fov_deg: float | None = None
    global_distortion_type: str = GLOBAL_DISTORTION_TPS_RESIDUAL_FIELD
    residual_kind: str = RESIDUAL_CORRECTION_TPS
    residual_origin_x_px: float = 0.0
    residual_origin_y_px: float = 0.0
    residual_scale_x_px: float = 1.0
    residual_scale_y_px: float = 1.0
    residual_anchor_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    residual_tps_weights_x: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    residual_tps_weights_y: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    residual_tps_affine_x: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    residual_tps_affine_y: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    residual_hard_anchor_count: int = 0
    residual_soft_constraint_count: int = 0
    residual_soft_weight_min: float = 1.0
    residual_soft_weight_max: float = 1.0
    tangent_sky_to_pixel_interpolation: AnchorInterpolation2D | None = None
    tangent_pixel_to_plane_interpolation: AnchorInterpolation2D | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_projection_transform(cls, transform: ProjectionSkyAlignmentTransform) -> "CameraCalibrationProfile":
        return cls(
            image_width_px=int(transform.image_width_px),
            image_height_px=int(transform.image_height_px),
            base_projection_type=transform.lens_model,
            principal_point_x_px=float(transform.center_x_px),
            principal_point_y_px=float(transform.center_y_px),
            scale_x_px=float(transform.scale_px),
            scale_y_px=float(transform.scale_px),
            fov_deg=None if transform.fov_deg is None else float(transform.fov_deg),
            global_distortion_type=GLOBAL_DISTORTION_TPS_RESIDUAL_FIELD,
            residual_kind=transform.residual_kind,
            residual_origin_x_px=float(transform.residual_origin_x_px),
            residual_origin_y_px=float(transform.residual_origin_y_px),
            residual_scale_x_px=float(transform.residual_scale_x_px),
            residual_scale_y_px=float(transform.residual_scale_y_px),
            residual_anchor_points=np.asarray(transform.residual_anchor_points, dtype=np.float64),
            residual_tps_weights_x=np.asarray(transform.residual_tps_weights_x, dtype=np.float64),
            residual_tps_weights_y=np.asarray(transform.residual_tps_weights_y, dtype=np.float64),
            residual_tps_affine_x=np.asarray(transform.residual_tps_affine_x, dtype=np.float64),
            residual_tps_affine_y=np.asarray(transform.residual_tps_affine_y, dtype=np.float64),
            residual_hard_anchor_count=int(transform.residual_hard_anchor_count),
            residual_soft_constraint_count=int(transform.residual_soft_constraint_count),
            residual_soft_weight_min=float(transform.residual_soft_weight_min),
            residual_soft_weight_max=float(transform.residual_soft_weight_max),
            diagnostics={
                "projection_rms_px_before_global_distortion": float(transform.projection_rms_px),
                "rms_px": float(transform.rms_px),
                "fit_pair_count": int(transform.pair_count),
            },
        )

    @classmethod
    def from_tangent_anchor_interpolation(
        cls,
        *,
        image_width_px: int,
        image_height_px: int,
        sky_to_pixel_interpolation: AnchorInterpolation2D,
        pixel_to_plane_interpolation: AnchorInterpolation2D,
        diagnostics: dict[str, Any] | None = None,
    ) -> "CameraCalibrationProfile":
        return cls(
            image_width_px=int(image_width_px),
            image_height_px=int(image_height_px),
            base_projection_type=BASE_PROJECTION_TANGENT_AEQD,
            principal_point_x_px=float(image_width_px) * 0.5,
            principal_point_y_px=float(image_height_px) * 0.5,
            scale_x_px=float(max(image_width_px, 1)) / 90.0,
            scale_y_px=float(max(image_height_px, 1)) / 90.0,
            global_distortion_type=GLOBAL_DISTORTION_TANGENT_PLANE_TPS,
            tangent_sky_to_pixel_interpolation=sky_to_pixel_interpolation,
            tangent_pixel_to_plane_interpolation=pixel_to_plane_interpolation,
            diagnostics=dict(diagnostics or {}),
        )

    @property
    def uses_known_projection(self) -> bool:
        return self.base_projection_type in SKY_KNOWN_PROJECTION_MODELS

    @property
    def uses_tangent_interpolation(self) -> bool:
        return self.base_projection_type == BASE_PROJECTION_TANGENT_AEQD

    def camera_ray_to_pixel_points(self, camera_vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(camera_vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, 3)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("camera_ray_to_pixel_points 需要 Nx3 的相机射线数组。")

        if self.uses_known_projection:
            projected, valid = _project_unit_vectors_with_known_projection(
                vectors=vectors,
                rotation_matrix=np.eye(3, dtype=np.float64),
                center_x_px=float(self.principal_point_x_px),
                center_y_px=float(self.principal_point_y_px),
                scale_px=float(self.scale_x_px),
                lens_model=self.base_projection_type,
                strict_visibility=True,
            )
            corrected = self._apply_global_distortion(projected)
            corrected[~valid] = np.nan
            corrected[~np.all(np.isfinite(corrected), axis=1)] = np.nan
            return corrected.astype(np.float64)

        if self.uses_tangent_interpolation:
            if self.tangent_sky_to_pixel_interpolation is None:
                raise ValueError("切平面 profile 缺少 camera ray → pixel 插值。")
            plane_points, valid = camera_vectors_to_tangent_plane_deg(vectors)
            pixels = self.tangent_sky_to_pixel_interpolation.evaluate_points(plane_points)
            pixels[~valid] = np.nan
            pixels[~np.all(np.isfinite(pixels), axis=1)] = np.nan
            return pixels.astype(np.float64)

        raise ValueError(f"不支持的 CameraCalibrationProfile 基础投影：{self.base_projection_type}")

    def pixel_to_camera_ray_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixel_to_camera_ray_points 需要 Nx2 的像素坐标数组。")

        if self.uses_known_projection:
            raw_pixels, residual_valid = self._uncorrect_global_distortion(pixels)
            vectors, projection_valid = self._camera_vectors_from_base_projection_pixels(raw_pixels)
            valid = residual_valid & projection_valid
            vectors[~valid] = np.nan
            return vectors.astype(np.float64)

        if self.uses_tangent_interpolation:
            if self.tangent_pixel_to_plane_interpolation is None:
                raise ValueError("切平面 profile 缺少 pixel → camera ray 反向初值。")
            seed_plane = self.tangent_pixel_to_plane_interpolation.evaluate_points(pixels)
            plane_points = self._refine_pixel_to_tangent_plane_points(pixels, seed_plane)
            vectors, valid = tangent_plane_deg_to_camera_vectors(plane_points)
            vectors[~valid] = np.nan
            return vectors.astype(np.float64)

        raise ValueError(f"不支持的 CameraCalibrationProfile 基础投影：{self.base_projection_type}")

    def _tangent_plane_to_pixel_points(self, plane_points: np.ndarray) -> np.ndarray:
        vectors, valid = tangent_plane_deg_to_camera_vectors(plane_points)
        pixels = self.camera_ray_to_pixel_points(vectors)
        pixels[~valid] = np.nan
        return pixels.astype(np.float64)

    def _refine_pixel_to_tangent_plane_points(
        self,
        pixel_points: np.ndarray,
        seed_plane_points: np.ndarray,
    ) -> np.ndarray:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        seeds = np.asarray(seed_plane_points, dtype=np.float64)
        refined = np.full_like(seeds, np.nan, dtype=np.float64)
        valid = np.all(np.isfinite(pixels), axis=1) & np.all(np.isfinite(seeds), axis=1)
        if not np.any(valid):
            return refined

        target_pixels = pixels[valid]
        current_plane = seeds[valid].copy()
        best_plane = current_plane.copy()
        best_norm = self._pixel_inverse_residual_norms(best_plane, target_pixels)
        active = np.isfinite(best_norm)
        step_u = np.asarray([CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG, 0.0], dtype=np.float64)
        step_v = np.asarray([0.0, CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG], dtype=np.float64)

        for _iteration in range(CALIBRATION_INVERSE_MAX_ITERATIONS):
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break
            plane = current_plane[active_indices]
            projected = self._tangent_plane_to_pixel_points(plane)
            residual = projected - target_pixels[active_indices]
            residual_norm = np.linalg.norm(residual, axis=1)
            finite_residual = np.all(np.isfinite(residual), axis=1) & np.isfinite(residual_norm)
            if not np.any(finite_residual):
                active[active_indices] = False
                continue

            finite_indices = active_indices[finite_residual]
            finite_plane = plane[finite_residual]
            finite_norm = residual_norm[finite_residual]
            improved = finite_norm < best_norm[finite_indices]
            if np.any(improved):
                best_plane[finite_indices[improved]] = finite_plane[improved]
                best_norm[finite_indices[improved]] = finite_norm[improved]

            converged = finite_norm <= CALIBRATION_INVERSE_TOLERANCE_PX
            if np.any(converged):
                active[finite_indices[converged]] = False
            solve_indices = finite_indices[~converged]
            if solve_indices.size == 0:
                continue

            solve_plane = current_plane[solve_indices]
            base_projected = projected[finite_residual][~converged]
            projected_u = self._tangent_plane_to_pixel_points(solve_plane + step_u)
            projected_v = self._tangent_plane_to_pixel_points(solve_plane + step_v)
            j00 = (projected_u[:, 0] - base_projected[:, 0]) / CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG
            j10 = (projected_u[:, 1] - base_projected[:, 1]) / CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG
            j01 = (projected_v[:, 0] - base_projected[:, 0]) / CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG
            j11 = (projected_v[:, 1] - base_projected[:, 1]) / CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG
            determinant = j00 * j11 - j01 * j10
            solve_residual = residual[finite_residual][~converged]
            finite_jacobian = (
                np.isfinite(determinant)
                & (np.abs(determinant) > 1e-14)
                & np.all(np.isfinite(projected_u), axis=1)
                & np.all(np.isfinite(projected_v), axis=1)
            )
            if not np.any(finite_jacobian):
                active[solve_indices] = False
                continue

            delta_u = (solve_residual[:, 0] * j11 - j01 * solve_residual[:, 1]) / determinant
            delta_v = (j00 * solve_residual[:, 1] - solve_residual[:, 0] * j10) / determinant
            delta = np.column_stack((delta_u, delta_v))
            finite_delta = finite_jacobian & np.all(np.isfinite(delta), axis=1)
            if not np.any(finite_delta):
                active[solve_indices] = False
                continue

            accepted_indices = solve_indices[finite_delta]
            accepted_delta = delta[finite_delta]
            delta_norm = np.linalg.norm(accepted_delta, axis=1)
            damping = np.divide(
                CALIBRATION_INVERSE_MAX_STEP_DEG,
                delta_norm,
                out=np.ones_like(delta_norm),
                where=delta_norm > CALIBRATION_INVERSE_MAX_STEP_DEG,
            )
            current_plane[accepted_indices] = current_plane[accepted_indices] - accepted_delta * damping[:, None]
            rejected_indices = solve_indices[~finite_delta]
            if rejected_indices.size:
                active[rejected_indices] = False

        final_norm = self._pixel_inverse_residual_norms(current_plane, target_pixels)
        improved = np.isfinite(final_norm) & (final_norm < best_norm)
        if np.any(improved):
            best_plane[improved] = current_plane[improved]
        refined[valid] = best_plane
        return refined.astype(np.float64)

    def _pixel_inverse_residual_norms(self, plane_points: np.ndarray, target_pixels: np.ndarray) -> np.ndarray:
        projected = self._tangent_plane_to_pixel_points(plane_points)
        residual = projected - target_pixels
        finite = np.all(np.isfinite(residual), axis=1)
        norms = np.full(projected.shape[0], np.inf, dtype=np.float64)
        if np.any(finite):
            norms[finite] = np.linalg.norm(residual[finite], axis=1)
        return norms

    def _apply_global_distortion(self, projected_pixels: np.ndarray) -> np.ndarray:
        if self.global_distortion_type != GLOBAL_DISTORTION_TPS_RESIDUAL_FIELD:
            return np.asarray(projected_pixels, dtype=np.float64)
        return _apply_residual_correction(
            projected_points=projected_pixels,
            residual_kind=self.residual_kind,
            origin_x_px=self.residual_origin_x_px,
            origin_y_px=self.residual_origin_y_px,
            scale_x_px=self.residual_scale_x_px,
            scale_y_px=self.residual_scale_y_px,
            anchor_points=self.residual_anchor_points,
            tps_weights_x=self.residual_tps_weights_x,
            tps_weights_y=self.residual_tps_weights_y,
            tps_affine_x=self.residual_tps_affine_x,
            tps_affine_y=self.residual_tps_affine_y,
        )

    def _uncorrect_global_distortion(self, pixel_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(pixel_points, dtype=np.float64)
        raw = target.copy()
        finite_target = np.all(np.isfinite(target), axis=1)
        raw[~finite_target] = np.nan
        if self.global_distortion_type != GLOBAL_DISTORTION_TPS_RESIDUAL_FIELD:
            return raw.astype(np.float64), finite_target.astype(bool)

        for _iteration_index in range(CALIBRATION_RESIDUAL_UNWARP_MAX_ITERATIONS):
            corrected = self._apply_global_distortion(raw)
            residual = corrected - target
            residual_norm = np.linalg.norm(residual, axis=1)
            active = finite_target & np.isfinite(residual_norm) & (residual_norm > 1e-4)
            if not np.any(active):
                break

            eps = CALIBRATION_RESIDUAL_UNWARP_FINITE_DIFF_STEP_PX
            raw_dx = raw.copy()
            raw_dy = raw.copy()
            raw_dx[active, 0] += eps
            raw_dy[active, 1] += eps
            corrected_dx = self._apply_global_distortion(raw_dx)
            corrected_dy = self._apply_global_distortion(raw_dy)
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
            step_scale = np.minimum(
                1.0,
                CALIBRATION_RESIDUAL_UNWARP_MAX_STEP_PX / np.maximum(step_norm, 1e-9),
            )
            raw[solvable, 0] -= delta_x[solvable] * step_scale[solvable]
            raw[solvable, 1] -= delta_y[solvable] * step_scale[solvable]
            raw[~np.all(np.isfinite(raw), axis=1)] = np.nan

        final_residual = self._apply_global_distortion(raw) - target
        final_norm = np.linalg.norm(final_residual, axis=1)
        valid = finite_target & np.isfinite(final_norm) & (final_norm <= CALIBRATION_RESIDUAL_UNWARP_TOLERANCE_PX)
        raw[~valid] = np.nan
        return raw.astype(np.float64), valid.astype(bool)

    def _camera_vectors_from_base_projection_pixels(self, projected_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(projected_pixels, dtype=np.float64)
        plane_x = (points[:, 0] - float(self.principal_point_x_px)) / max(float(self.scale_x_px), 1e-12)
        plane_y = (float(self.principal_point_y_px) - points[:, 1]) / max(float(self.scale_y_px), 1e-12)
        vectors = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
        valid = np.all(np.isfinite(points), axis=1)

        if self.base_projection_type == "rectilinear":
            vectors[valid] = np.column_stack((plane_x[valid], plane_y[valid], np.ones(np.count_nonzero(valid))))
        elif self.base_projection_type in ("fisheye_equidistant", "fisheye_equisolid"):
            radius = np.hypot(plane_x, plane_y)
            if self.base_projection_type == "fisheye_equidistant":
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
        elif self.base_projection_type in ("mercator", "cylindrical_equidistant"):
            longitude = plane_x
            if self.base_projection_type == "mercator":
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
            raise ValueError(f"不支持的 CameraCalibrationProfile 反投影模型：{self.base_projection_type}")

        norm = np.linalg.norm(vectors, axis=1)
        valid &= np.isfinite(norm) & (norm > 1e-12)
        vectors[valid] /= norm[valid, None]
        vectors[~valid] = np.nan
        return vectors.astype(np.float64), valid.astype(bool)

    def to_json_payload(self) -> dict[str, Any]:
        base_parameters: dict[str, Any] = {}
        if self.base_projection_type in SKY_KNOWN_PROJECTION_MODELS:
            base_parameters["projection_code"] = SKY_KNOWN_PROJECTION_CODES.get(
                self.base_projection_type,
                self.base_projection_type,
            )
            base_parameters["display_name"] = SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(
                self.base_projection_type,
                self.base_projection_type,
            )
            base_parameters["fov_deg"] = None if self.fov_deg is None else float(self.fov_deg)

        payload: dict[str, Any] = {
            "version": CAMERA_CALIBRATION_PROFILE_VERSION,
            "base_projection": {
                "type": self.base_projection_type,
                "parameters": base_parameters,
            },
            "intrinsics": {
                "principal_point_x_px": float(self.principal_point_x_px),
                "principal_point_y_px": float(self.principal_point_y_px),
                "scale_x_px": float(self.scale_x_px),
                "scale_y_px": float(self.scale_y_px),
            },
            "coverage": {
                "image_width_px": int(self.image_width_px),
                "image_height_px": int(self.image_height_px),
                "pixel_convention": "0-based_pixel_center",
            },
            "diagnostics": dict(self.diagnostics),
        }

        if self.uses_tangent_interpolation:
            if self.tangent_sky_to_pixel_interpolation is None or self.tangent_pixel_to_plane_interpolation is None:
                raise ValueError("切平面 CameraCalibrationProfile 缺少双向插值数据。")
            payload["global_distortion"] = {
                "type": GLOBAL_DISTORTION_TANGENT_PLANE_TPS,
                "parameters": {
                    "camera_ray_to_tangent_plane": "azimuthal_equidistant_deg",
                    "tangent_plane_to_pixel": _interpolation_payload(
                        self.tangent_sky_to_pixel_interpolation,
                        input_units="deg in camera tangent plane",
                        output_units="px",
                        input_axis_order=["u_deg", "v_deg"],
                        output_axis_order=["x_px", "y_px"],
                        weight_names=("tps_weights_x_px", "tps_weights_y_px"),
                        affine_names=("tps_affine_x_px", "tps_affine_y_px"),
                    ),
                },
            }
            payload["inverse_model"] = {
                "type": INVERSE_MODEL_TANGENT_TPS_NUMERICAL,
                "parameters": {
                    "initial_estimate": _interpolation_payload(
                        self.tangent_pixel_to_plane_interpolation,
                        input_units="px",
                        output_units="deg in camera tangent plane",
                        input_axis_order=["x_px", "y_px"],
                        output_axis_order=["u_deg", "v_deg"],
                        weight_names=("tps_weights_u_deg", "tps_weights_v_deg"),
                        affine_names=("tps_affine_u_deg", "tps_affine_v_deg"),
                    ),
                    "forward_residual_tolerance_px": CALIBRATION_INVERSE_TOLERANCE_PX,
                    "max_iterations": CALIBRATION_INVERSE_MAX_ITERATIONS,
                    "finite_difference_step_deg": CALIBRATION_INVERSE_FINITE_DIFF_STEP_DEG,
                },
            }
        else:
            payload["global_distortion"] = {
                "type": self.global_distortion_type,
                "parameters": {
                    "kind": self.residual_kind,
                    "normalization": {
                        "origin_x_px": float(self.residual_origin_x_px),
                        "origin_y_px": float(self.residual_origin_y_px),
                        "scale_x_px": float(self.residual_scale_x_px),
                        "scale_y_px": float(self.residual_scale_y_px),
                    },
                    "anchor_count": int(self.residual_anchor_points.shape[0]),
                    "hard_anchor_count": int(self.residual_hard_anchor_count),
                    "soft_constraint_count": int(self.residual_soft_constraint_count),
                    "soft_constraint_weight_min": float(self.residual_soft_weight_min),
                    "soft_constraint_weight_max": float(self.residual_soft_weight_max),
                    "anchor_points_normalized": [
                        _as_float_list(row) for row in np.asarray(self.residual_anchor_points, dtype=np.float64)
                    ],
                    "tps_weights_dx_px": _as_float_list(self.residual_tps_weights_x),
                    "tps_weights_dy_px": _as_float_list(self.residual_tps_weights_y),
                    "tps_affine_dx_px": _as_float_list(self.residual_tps_affine_x),
                    "tps_affine_dy_px": _as_float_list(self.residual_tps_affine_y),
                },
            }
            payload["inverse_model"] = {
                "type": INVERSE_MODEL_ANALYTIC_WITH_RESIDUAL_UNWARP,
                "parameters": {
                    "residual_unwarp_tolerance_px": CALIBRATION_RESIDUAL_UNWARP_TOLERANCE_PX,
                    "max_iterations": CALIBRATION_RESIDUAL_UNWARP_MAX_ITERATIONS,
                },
            }

        if self.notes:
            payload["notes"] = dict(self.notes)
        return payload

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> "CameraCalibrationProfile":
        if not isinstance(payload, dict):
            raise ValueError("CameraCalibrationProfile JSON 必须是对象。")
        version = _json_int(payload.get("version"), "version")
        if version != CAMERA_CALIBRATION_PROFILE_VERSION:
            raise ValueError(f"不支持的 CameraCalibrationProfile version: {version}")

        base_projection = _json_mapping(payload.get("base_projection"), "base_projection")
        base_type = str(base_projection.get("type", "")).strip()
        base_parameters = base_projection.get("parameters")
        base_parameters = base_parameters if isinstance(base_parameters, dict) else {}
        intrinsics = _json_mapping(payload.get("intrinsics"), "intrinsics")
        coverage = _json_mapping(payload.get("coverage"), "coverage")
        global_distortion = _json_mapping(payload.get("global_distortion"), "global_distortion")
        inverse_model = _json_mapping(payload.get("inverse_model"), "inverse_model")
        diagnostics = payload.get("diagnostics")
        notes = payload.get("notes")

        if base_type == BASE_PROJECTION_TANGENT_AEQD:
            distortion_parameters = _json_mapping(global_distortion.get("parameters"), "global_distortion.parameters")
            tangent_payload = _json_mapping(
                distortion_parameters.get("tangent_plane_to_pixel"),
                "global_distortion.parameters.tangent_plane_to_pixel",
            )
            inverse_parameters = _json_mapping(inverse_model.get("parameters"), "inverse_model.parameters")
            inverse_payload = _json_mapping(
                inverse_parameters.get("initial_estimate"),
                "inverse_model.parameters.initial_estimate",
            )
            return cls(
                image_width_px=_json_int(coverage.get("image_width_px"), "coverage.image_width_px"),
                image_height_px=_json_int(coverage.get("image_height_px"), "coverage.image_height_px"),
                base_projection_type=base_type,
                principal_point_x_px=_json_float(
                    intrinsics.get("principal_point_x_px"),
                    "intrinsics.principal_point_x_px",
                ),
                principal_point_y_px=_json_float(
                    intrinsics.get("principal_point_y_px"),
                    "intrinsics.principal_point_y_px",
                ),
                scale_x_px=_json_float(intrinsics.get("scale_x_px"), "intrinsics.scale_x_px"),
                scale_y_px=_json_float(intrinsics.get("scale_y_px"), "intrinsics.scale_y_px"),
                global_distortion_type=str(global_distortion.get("type", "")),
                tangent_sky_to_pixel_interpolation=_interpolation_from_payload(
                    tangent_payload,
                    "global_distortion.parameters.tangent_plane_to_pixel",
                    weight_names=("tps_weights_x_px", "tps_weights_y_px"),
                    affine_names=("tps_affine_x_px", "tps_affine_y_px"),
                ),
                tangent_pixel_to_plane_interpolation=_interpolation_from_payload(
                    inverse_payload,
                    "inverse_model.parameters.initial_estimate",
                    weight_names=("tps_weights_u_deg", "tps_weights_v_deg"),
                    affine_names=("tps_affine_u_deg", "tps_affine_v_deg"),
                ),
                diagnostics=dict(diagnostics) if isinstance(diagnostics, dict) else {},
                notes=dict(notes) if isinstance(notes, dict) else {},
            )

        if base_type not in SKY_KNOWN_PROJECTION_MODELS:
            raise ValueError(f"不支持的 CameraCalibrationProfile 基础投影：{base_type}")

        distortion_parameters = _json_mapping(global_distortion.get("parameters"), "global_distortion.parameters")
        normalization = _json_mapping(
            distortion_parameters.get("normalization"),
            "global_distortion.parameters.normalization",
        )
        return cls(
            image_width_px=_json_int(coverage.get("image_width_px"), "coverage.image_width_px"),
            image_height_px=_json_int(coverage.get("image_height_px"), "coverage.image_height_px"),
            base_projection_type=base_type,
            principal_point_x_px=_json_float(intrinsics.get("principal_point_x_px"), "intrinsics.principal_point_x_px"),
            principal_point_y_px=_json_float(intrinsics.get("principal_point_y_px"), "intrinsics.principal_point_y_px"),
            scale_x_px=_json_float(intrinsics.get("scale_x_px"), "intrinsics.scale_x_px"),
            scale_y_px=_json_float(intrinsics.get("scale_y_px"), "intrinsics.scale_y_px"),
            fov_deg=None if base_parameters.get("fov_deg") is None else float(base_parameters.get("fov_deg")),
            global_distortion_type=str(global_distortion.get("type", "")),
            residual_kind=str(distortion_parameters.get("kind", "")),
            residual_origin_x_px=_json_float(
                normalization.get("origin_x_px"),
                "global_distortion.parameters.normalization.origin_x_px",
            ),
            residual_origin_y_px=_json_float(
                normalization.get("origin_y_px"),
                "global_distortion.parameters.normalization.origin_y_px",
            ),
            residual_scale_x_px=_json_float(
                normalization.get("scale_x_px"),
                "global_distortion.parameters.normalization.scale_x_px",
            ),
            residual_scale_y_px=_json_float(
                normalization.get("scale_y_px"),
                "global_distortion.parameters.normalization.scale_y_px",
            ),
            residual_anchor_points=_json_float_array(
                distortion_parameters.get("anchor_points_normalized", []),
                "global_distortion.parameters.anchor_points_normalized",
            ).reshape((-1, 2)),
            residual_tps_weights_x=_json_float_array(
                distortion_parameters.get("tps_weights_dx_px", []),
                "global_distortion.parameters.tps_weights_dx_px",
            ).reshape(-1),
            residual_tps_weights_y=_json_float_array(
                distortion_parameters.get("tps_weights_dy_px", []),
                "global_distortion.parameters.tps_weights_dy_px",
            ).reshape(-1),
            residual_tps_affine_x=_json_float_array(
                distortion_parameters.get("tps_affine_dx_px", []),
                "global_distortion.parameters.tps_affine_dx_px",
            ).reshape(-1),
            residual_tps_affine_y=_json_float_array(
                distortion_parameters.get("tps_affine_dy_px", []),
                "global_distortion.parameters.tps_affine_dy_px",
            ).reshape(-1),
            residual_hard_anchor_count=_json_int(
                distortion_parameters.get("hard_anchor_count"),
                "global_distortion.parameters.hard_anchor_count",
                default=0,
            ),
            residual_soft_constraint_count=_json_int(
                distortion_parameters.get("soft_constraint_count"),
                "global_distortion.parameters.soft_constraint_count",
                default=0,
            ),
            residual_soft_weight_min=_json_float(
                distortion_parameters.get("soft_constraint_weight_min"),
                "global_distortion.parameters.soft_constraint_weight_min",
                default=1.0,
            ),
            residual_soft_weight_max=_json_float(
                distortion_parameters.get("soft_constraint_weight_max"),
                "global_distortion.parameters.soft_constraint_weight_max",
                default=1.0,
            ),
            diagnostics=dict(diagnostics) if isinstance(diagnostics, dict) else {},
            notes=dict(notes) if isinstance(notes, dict) else {},
        )
