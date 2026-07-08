from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .constants import (
    MAX_ALIGNMENT_CONDITION_NUMBER,
    MAX_TRANSFORM_ABS_PX,
    MIN_ALIGNMENT_PAIRS,
    PROJECTION_FIT_MAX_NFEV,
    PROJECTION_INVALID_RESIDUAL_PX,
    QUADRATIC_ALIGNMENT_PAIRS,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODELS,
)
from .interpolation import fit_anchor_interpolation
from .models import (
    ProjectionSkyAlignmentTransform,
    ReferenceAlignmentTransform,
    SkyAlignmentTransform,
)
from .projections import (
    _fixed_fisheye_scale_px,
    _initial_fisheye_rotation_from_pixels,
    _initial_projection_from_rotation,
    _initial_rectilinear_from_rotation,
    _normalize_vector,
    _orthonormalized_rotation_matrix,
    _project_radec_to_sky_plane,
    _projection_points_from_params,
    _radec_to_unit_vectors,
    _rotation_matrix_from_rotvec,
    _sky_plane_basis,
)
from .residuals import _fit_residual_correction
from .validation import (
    _array_is_finite,
    _fit_point_weights,
    _residual_anchor_mask,
)


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

__all__ = [name for name in globals() if not name.startswith("__")]
