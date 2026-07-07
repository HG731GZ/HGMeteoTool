from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .alignment import (
    AnchorInterpolation2D,
    FIT_WEIGHT_MAX,
    FIT_WEIGHT_MIN,
    MIN_ALIGNMENT_PAIRS,
    ProjectionSkyAlignmentTransform,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODELS,
    fit_anchor_interpolation,
    fit_projection_sky_alignment,
)
from .coordinates import (
    project_radec_to_sky_plane,
    radec_to_unit_vectors,
    sky_plane_basis,
    sky_plane_to_radec,
)
from .camera_calibration import CameraCalibrationProfile
from .frame_astrometry import (
    FrameAstrometricModel,
    FrameLocalResidual,
    FramePose,
    SOURCE_MODEL_SCHEMA,
    SOURCE_MODEL_VERSION as FRAME_SOURCE_MODEL_VERSION,
)


SOURCE_MODEL_FORMAT = SOURCE_MODEL_SCHEMA
SOURCE_MODEL_VERSION = FRAME_SOURCE_MODEL_VERSION
INVERSE_SOLVER_MAX_NFEV = 80
INVERSE_SOLVER_INVALID_RESIDUAL_PX = 1e6
INVERSE_SOLVER_MAX_ITERATIONS = 14
INVERSE_SOLVER_TOLERANCE_PX = 1e-5
INVERSE_SOLVER_FINITE_DIFF_STEP_DEG = 1e-5
INVERSE_SOLVER_MAX_STEP_DEG = 2.0
INVERSE_SOLVER_SCALAR_FALLBACK_LIMIT = 8


def _finite_point_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=bool)
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.all(np.isfinite(array), axis=1)
    return mask


def _fit_point_weights(point_count: int, point_weights: np.ndarray | None) -> np.ndarray:
    if point_weights is None:
        return np.ones(point_count, dtype=np.float64)
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != point_count:
        raise ValueError("源图模型拟合权重数量必须与配对星数量一致。")
    if not np.all(np.isfinite(weights)):
        raise ValueError("源图模型拟合权重包含无效数值。")
    return np.clip(weights, FIT_WEIGHT_MIN, FIT_WEIGHT_MAX).astype(np.float64)


def _fit_anchor_mask(point_count: int, anchor_mask: np.ndarray | None) -> np.ndarray:
    if anchor_mask is None:
        return np.ones(point_count, dtype=bool)
    mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != point_count:
        raise ValueError("源图模型锚点标记数量必须与配对星数量一致。")
    return mask.astype(bool)


def _residual_summary(residual_vectors: np.ndarray) -> tuple[float, float, float]:
    if residual_vectors.size == 0:
        return float("nan"), float("nan"), float("nan")
    distances = np.linalg.norm(residual_vectors, axis=1)
    return (
        float(np.sqrt(np.mean(distances * distances))),
        float(np.median(distances)),
        float(np.max(distances)),
    )


def _angular_residual_summary_arcsec(reference_radec: np.ndarray, measured_radec: np.ndarray) -> tuple[float, float, float]:
    reference = np.asarray(reference_radec, dtype=np.float64)
    measured = np.asarray(measured_radec, dtype=np.float64)
    if reference.size == 0 or measured.size == 0:
        return float("nan"), float("nan"), float("nan")
    finite = np.all(np.isfinite(reference), axis=1) & np.all(np.isfinite(measured), axis=1)
    reference = reference[finite]
    measured = measured[finite]
    if reference.size == 0:
        return float("nan"), float("nan"), float("nan")
    reference_vectors = radec_to_unit_vectors(reference[:, 0], reference[:, 1])
    measured_vectors = radec_to_unit_vectors(measured[:, 0], measured[:, 1])
    dots = np.sum(reference_vectors * measured_vectors, axis=1)
    distances = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0))) * 3600.0
    if distances.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.sqrt(np.mean(distances * distances))),
        float(np.median(distances)),
        float(np.max(distances)),
    )


@dataclass(frozen=True)
class SourceAstrometricModel:
    image_width_px: int
    image_height_px: int
    pair_count: int
    center_vector: np.ndarray
    east_vector: np.ndarray
    north_vector: np.ndarray
    sky_to_pixel_interpolation: AnchorInterpolation2D | None
    pixel_to_sky_plane_interpolation: AnchorInterpolation2D
    rms_px: float
    median_residual_px: float
    max_residual_px: float
    inverse_seed_rms_arcsec: float
    inverse_seed_median_arcsec: float
    inverse_seed_max_arcsec: float
    inverse_fit_rms_arcsec: float
    inverse_fit_median_arcsec: float
    inverse_fit_max_arcsec: float
    inverse_roundtrip_rms_px: float
    inverse_roundtrip_median_px: float
    inverse_roundtrip_max_px: float
    model_type: str = "local_sky_plane_anchor_interpolation"
    projection_transform: ProjectionSkyAlignmentTransform | None = None

    def _sky_plane_to_pixel_points(self, plane_points: np.ndarray) -> np.ndarray:
        plane_array = np.asarray(plane_points, dtype=np.float64)
        if plane_array.ndim == 1:
            plane_array = plane_array.reshape(1, 2)
        if plane_array.ndim != 2 or plane_array.shape[1] != 2:
            raise ValueError("天球平面点必须是 Nx2 数组。")

        radec = sky_plane_to_radec(
            plane_array,
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )
        return self.direction_to_pixel_points(radec)

    def direction_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
        if ra_dec_array.ndim == 1:
            ra_dec_array = ra_dec_array.reshape(1, 2)
        if ra_dec_array.ndim != 2 or ra_dec_array.shape[1] != 2:
            raise ValueError("direction_to_pixel_points 需要 Nx2 的 RA/Dec 数组。")

        if self.projection_transform is not None:
            return self.projection_transform.transform_radec_points(ra_dec_array)
        if self.sky_to_pixel_interpolation is None:
            return np.full((ra_dec_array.shape[0], 2), np.nan, dtype=np.float64)

        plane_points = project_radec_to_sky_plane(
            ra_dec_array[:, 0],
            ra_dec_array[:, 1],
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )
        return self.sky_to_pixel_interpolation.evaluate_points(plane_points)

    def pixel_to_sky_plane_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixel_array = np.asarray(pixel_points, dtype=np.float64)
        if pixel_array.ndim == 1:
            pixel_array = pixel_array.reshape(1, 2)
        if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
            raise ValueError("pixel_to_sky_plane_points 需要 Nx2 的像素坐标数组。")

        initial_plane = self.pixel_to_sky_plane_interpolation.evaluate_points(pixel_array)
        if pixel_array.shape[0] > INVERSE_SOLVER_SCALAR_FALLBACK_LIMIT:
            return self._refine_pixel_to_sky_plane_points(pixel_array, initial_plane)

        refined = np.full_like(initial_plane, np.nan, dtype=np.float64)
        for index, (pixel, seed) in enumerate(zip(pixel_array, initial_plane, strict=True)):
            if not np.all(np.isfinite(pixel)) or not np.all(np.isfinite(seed)):
                continue
            refined[index] = self._refine_pixel_to_sky_plane(pixel, seed)
        return refined

    def _refine_pixel_to_sky_plane_points(self, pixel_points: np.ndarray, seed_plane_points: np.ndarray) -> np.ndarray:
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
        if not np.any(active):
            refined[valid] = best_plane
            return refined

        finite_diff_step = INVERSE_SOLVER_FINITE_DIFF_STEP_DEG
        step_u = np.asarray([finite_diff_step, 0.0], dtype=np.float64)
        step_v = np.asarray([0.0, finite_diff_step], dtype=np.float64)
        for _iteration in range(INVERSE_SOLVER_MAX_ITERATIONS):
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break

            plane = current_plane[active_indices]
            projected = self._sky_plane_to_pixel_points(plane)
            residual = projected - target_pixels[active_indices]
            residual_norm = np.linalg.norm(residual, axis=1)
            finite_residual = np.all(np.isfinite(residual), axis=1) & np.isfinite(residual_norm)
            if not np.any(finite_residual):
                active[active_indices] = False
                continue

            finite_indices = active_indices[finite_residual]
            finite_plane = plane[finite_residual]
            finite_residual_vectors = residual[finite_residual]
            finite_norm = residual_norm[finite_residual]
            improved = finite_norm < best_norm[finite_indices]
            if np.any(improved):
                improved_indices = finite_indices[improved]
                best_plane[improved_indices] = finite_plane[improved]
                best_norm[improved_indices] = finite_norm[improved]

            converged = finite_norm <= INVERSE_SOLVER_TOLERANCE_PX
            if np.any(converged):
                active[finite_indices[converged]] = False

            solve_indices = finite_indices[~converged]
            if solve_indices.size == 0:
                continue

            solve_plane = current_plane[solve_indices]
            base_projected = projected[finite_residual][~converged]
            projected_u = self._sky_plane_to_pixel_points(solve_plane + step_u)
            projected_v = self._sky_plane_to_pixel_points(solve_plane + step_v)
            j00 = (projected_u[:, 0] - base_projected[:, 0]) / finite_diff_step
            j10 = (projected_u[:, 1] - base_projected[:, 1]) / finite_diff_step
            j01 = (projected_v[:, 0] - base_projected[:, 0]) / finite_diff_step
            j11 = (projected_v[:, 1] - base_projected[:, 1]) / finite_diff_step
            determinant = j00 * j11 - j01 * j10
            solve_residual = finite_residual_vectors[~converged]

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
                INVERSE_SOLVER_MAX_STEP_DEG,
                delta_norm,
                out=np.ones_like(delta_norm),
                where=delta_norm > INVERSE_SOLVER_MAX_STEP_DEG,
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
        return refined

    def _pixel_inverse_residual_norms(self, plane_points: np.ndarray, target_pixels: np.ndarray) -> np.ndarray:
        projected = self._sky_plane_to_pixel_points(plane_points)
        residual = projected - target_pixels
        finite = np.all(np.isfinite(residual), axis=1)
        norms = np.full(projected.shape[0], np.inf, dtype=np.float64)
        if np.any(finite):
            norms[finite] = np.linalg.norm(residual[finite], axis=1)
        return norms

    def _refine_pixel_to_sky_plane(self, pixel_point: np.ndarray, seed_plane: np.ndarray) -> np.ndarray:
        target_pixel = np.asarray(pixel_point, dtype=np.float64)
        best_plane = np.asarray(seed_plane, dtype=np.float64)
        best_residual = self._pixel_inverse_residual(best_plane, target_pixel)
        best_norm = float(np.linalg.norm(best_residual)) if np.all(np.isfinite(best_residual)) else float("inf")

        def residual_function(plane_values: np.ndarray) -> np.ndarray:
            return self._pixel_inverse_residual(np.asarray(plane_values, dtype=np.float64), target_pixel)

        try:
            result = least_squares(
                residual_function,
                best_plane,
                max_nfev=INVERSE_SOLVER_MAX_NFEV,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
        except ValueError:
            result = None

        if result is not None and np.all(np.isfinite(result.x)):
            result_residual = self._pixel_inverse_residual(result.x, target_pixel)
            result_norm = (
                float(np.linalg.norm(result_residual)) if np.all(np.isfinite(result_residual)) else float("inf")
            )
            if result_norm <= best_norm:
                best_plane = result.x.astype(np.float64)
        return best_plane.astype(np.float64)

    def _pixel_inverse_residual(self, plane_point: np.ndarray, target_pixel: np.ndarray) -> np.ndarray:
        projected = self._sky_plane_to_pixel_points(np.asarray(plane_point, dtype=np.float64).reshape(1, 2))[0]
        if not np.all(np.isfinite(projected)):
            return np.asarray(
                [INVERSE_SOLVER_INVALID_RESIDUAL_PX, INVERSE_SOLVER_INVALID_RESIDUAL_PX],
                dtype=np.float64,
            )
        return (projected - target_pixel).astype(np.float64)

    def pixel_to_radec_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixel_array = np.asarray(pixel_points, dtype=np.float64)
        if pixel_array.ndim == 1:
            pixel_array = pixel_array.reshape(1, 2)
        if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
            raise ValueError("pixel_to_radec_points 需要 Nx2 的像素坐标数组。")

        plane_points = self.pixel_to_sky_plane_points(pixel_array)
        return sky_plane_to_radec(
            plane_points,
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )

    def pixel_to_radec(self, x_px: float, y_px: float) -> tuple[float, float]:
        result = self.pixel_to_radec_points(np.asarray([[x_px, y_px]], dtype=np.float64))[0]
        return float(result[0]), float(result[1])

    def to_frame_astrometric_model(
        self,
        *,
        fit_metadata: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> FrameAstrometricModel:
        metadata: dict[str, Any] = {
            "model_type": self.model_type,
            "control_point_count": int(self.pair_count),
        }
        if fit_metadata is not None:
            metadata.update(fit_metadata)

        model_diagnostics: dict[str, Any] = {
            "pair_count": int(self.pair_count),
            "rms_px": float(self.rms_px),
            "median_residual_px": float(self.median_residual_px),
            "max_residual_px": float(self.max_residual_px),
            "sky_to_pixel_rms_px": float(self.rms_px),
            "sky_to_pixel_median_px": float(self.median_residual_px),
            "sky_to_pixel_max_px": float(self.max_residual_px),
            "pixel_to_sky_seed_rms_arcsec": float(self.inverse_seed_rms_arcsec),
            "pixel_to_sky_seed_median_arcsec": float(self.inverse_seed_median_arcsec),
            "pixel_to_sky_seed_max_arcsec": float(self.inverse_seed_max_arcsec),
            "pixel_to_sky_rms_arcsec": float(self.inverse_fit_rms_arcsec),
            "pixel_to_sky_median_arcsec": float(self.inverse_fit_median_arcsec),
            "pixel_to_sky_max_arcsec": float(self.inverse_fit_max_arcsec),
            "round_trip_rms_px": float(self.inverse_roundtrip_rms_px),
            "round_trip_median_px": float(self.inverse_roundtrip_median_px),
            "round_trip_max_px": float(self.inverse_roundtrip_max_px),
        }
        if self.projection_transform is not None:
            model_diagnostics["projection_rms_px_before_global_distortion"] = float(
                self.projection_transform.projection_rms_px
            )
        if diagnostics is not None:
            model_diagnostics.update(diagnostics)

        if self.projection_transform is not None:
            frame_pose = FramePose(np.asarray(self.projection_transform.rotation_matrix, dtype=np.float64))
            calibration_profile = CameraCalibrationProfile.from_projection_transform(self.projection_transform)
        else:
            if self.sky_to_pixel_interpolation is None:
                raise ValueError("普适源图模型缺少 sky→pixel 锚点插值。")
            frame_pose = FramePose(
                np.vstack(
                    (
                        np.asarray(self.east_vector, dtype=np.float64),
                        np.asarray(self.north_vector, dtype=np.float64),
                        np.asarray(self.center_vector, dtype=np.float64),
                    )
                )
            )
            calibration_profile = CameraCalibrationProfile.from_tangent_anchor_interpolation(
                image_width_px=int(self.image_width_px),
                image_height_px=int(self.image_height_px),
                sky_to_pixel_interpolation=self.sky_to_pixel_interpolation,
                pixel_to_plane_interpolation=self.pixel_to_sky_plane_interpolation,
                diagnostics={
                    "rms_px": float(self.rms_px),
                    "fit_pair_count": int(self.pair_count),
                },
            )

        return FrameAstrometricModel(
            image_width_px=int(self.image_width_px),
            image_height_px=int(self.image_height_px),
            frame_pose=frame_pose,
            camera_calibration_profile=calibration_profile,
            frame_local_residual=FrameLocalResidual(),
            fit_metadata=metadata,
            diagnostics=model_diagnostics,
        )

    def to_json_payload(
        self,
        *,
        source_image: dict[str, Any] | None = None,
        fit_pairs: list[dict[str, Any]] | None = None,
        mask: dict[str, Any] | None = None,
        matching: dict[str, Any] | None = None,
        reference_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.to_frame_astrometric_model().to_json_payload(
            source_image=source_image,
            fit_pairs=fit_pairs,
            mask=mask,
            matching=matching,
            reference_payload=reference_payload,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )


def fit_source_astrometric_model(
    ra_dec_points: np.ndarray,
    pixel_points: np.ndarray,
    image_size: tuple[int, int],
    matching_model: str = SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    fisheye_fov_deg: float | None = None,
    initial_rotation_matrix: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
    residual_anchor_mask: np.ndarray | None = None,
) -> SourceAstrometricModel:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    pixels = np.asarray(pixel_points, dtype=np.float64)
    if sky_radec.ndim != 2 or sky_radec.shape[1] != 2:
        raise ValueError("源图模型需要 Nx2 的 RA/Dec 点。")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("源图模型需要 Nx2 的像素点。")
    if sky_radec.shape[0] != pixels.shape[0]:
        raise ValueError("源图模型的 RA/Dec 点与像素点数量不一致。")
    if matching_model not in SKY_MATCHING_MODELS:
        raise ValueError(f"不支持的源图匹配模型：{matching_model}")
    raw_point_weights = _fit_point_weights(sky_radec.shape[0], point_weights)
    raw_anchor_mask = _fit_anchor_mask(sky_radec.shape[0], residual_anchor_mask)

    finite_mask = _finite_point_mask(sky_radec, pixels)
    sky_radec = sky_radec[finite_mask]
    pixels = pixels[finite_mask]
    fit_weights = raw_point_weights[finite_mask]
    anchor_mask = raw_anchor_mask[finite_mask]
    pair_count = int(sky_radec.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对星点才能生成源图映射。")
    image_width = int(image_size[0])
    image_height = int(image_size[1])
    if image_width <= 0 or image_height <= 0:
        raise ValueError("源图尺寸无效，无法生成源图映射。")

    center, east, north = sky_plane_basis(sky_radec)
    sky_plane = project_radec_to_sky_plane(sky_radec[:, 0], sky_radec[:, 1], center, east, north)
    if not np.all(np.isfinite(sky_plane)):
        raise ValueError("源图模型的天球平面坐标包含无效数值。")

    pixel_to_sky_plane_interpolation = fit_anchor_interpolation(
        pixels,
        sky_plane,
        anchor_mask=anchor_mask,
        point_weights=fit_weights,
    )
    sky_to_pixel_interpolation: AnchorInterpolation2D | None = None
    projection_transform: ProjectionSkyAlignmentTransform | None = None
    if matching_model in SKY_KNOWN_PROJECTION_MODELS:
        projection_transform = fit_projection_sky_alignment(
            ra_dec_points=sky_radec,
            target_points=pixels,
            lens_model=matching_model,
            image_size=(image_width, image_height),
            fisheye_fov_deg=fisheye_fov_deg,
            initial_rotation_matrix=initial_rotation_matrix,
            point_weights=fit_weights,
            residual_anchor_mask=anchor_mask,
        )
        predicted_pixels = projection_transform.transform_radec_points(sky_radec)
        model_type = "known_projection_with_residual_interpolation"
    else:
        sky_to_pixel_interpolation = fit_anchor_interpolation(
            sky_plane,
            pixels,
            anchor_mask=anchor_mask,
            point_weights=fit_weights,
        )
        predicted_pixels = sky_to_pixel_interpolation.evaluate_points(sky_plane)
        model_type = "local_sky_plane_anchor_interpolation"
    rms_px, median_px, max_px = _residual_summary(predicted_pixels - pixels)

    seed_plane = pixel_to_sky_plane_interpolation.evaluate_points(pixels)
    seed_radec = sky_plane_to_radec(seed_plane, center, east, north)
    seed_rms_arcsec, seed_median_arcsec, seed_max_arcsec = _angular_residual_summary_arcsec(sky_radec, seed_radec)
    model = SourceAstrometricModel(
        image_width_px=image_width,
        image_height_px=image_height,
        pair_count=pair_count,
        center_vector=center,
        east_vector=east,
        north_vector=north,
        sky_to_pixel_interpolation=sky_to_pixel_interpolation,
        pixel_to_sky_plane_interpolation=pixel_to_sky_plane_interpolation,
        rms_px=float(rms_px),
        median_residual_px=float(median_px),
        max_residual_px=float(max_px),
        inverse_seed_rms_arcsec=float(seed_rms_arcsec),
        inverse_seed_median_arcsec=float(seed_median_arcsec),
        inverse_seed_max_arcsec=float(seed_max_arcsec),
        inverse_fit_rms_arcsec=float("nan"),
        inverse_fit_median_arcsec=float("nan"),
        inverse_fit_max_arcsec=float("nan"),
        inverse_roundtrip_rms_px=float("nan"),
        inverse_roundtrip_median_px=float("nan"),
        inverse_roundtrip_max_px=float("nan"),
        model_type=model_type,
        projection_transform=projection_transform,
    )
    inverse_radec = model.pixel_to_radec_points(pixels)
    inverse_rms_arcsec, inverse_median_arcsec, inverse_max_arcsec = _angular_residual_summary_arcsec(
        sky_radec,
        inverse_radec,
    )
    inverse_pixels = model.direction_to_pixel_points(inverse_radec)
    inverse_roundtrip_rms_px, inverse_roundtrip_median_px, inverse_roundtrip_max_px = _residual_summary(
        inverse_pixels - pixels
    )
    object.__setattr__(model, "inverse_fit_rms_arcsec", float(inverse_rms_arcsec))
    object.__setattr__(model, "inverse_fit_median_arcsec", float(inverse_median_arcsec))
    object.__setattr__(model, "inverse_fit_max_arcsec", float(inverse_max_arcsec))
    object.__setattr__(model, "inverse_roundtrip_rms_px", float(inverse_roundtrip_rms_px))
    object.__setattr__(model, "inverse_roundtrip_median_px", float(inverse_roundtrip_median_px))
    object.__setattr__(model, "inverse_roundtrip_max_px", float(inverse_roundtrip_max_px))
    return model
