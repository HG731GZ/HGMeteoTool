from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import RESIDUAL_CORRECTION_TPS
from .interpolation import _evaluate_thin_plate_spline, fit_anchor_interpolation
from .validation import _array_is_finite, _fit_point_weights, _residual_anchor_mask


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

__all__ = [name for name in globals() if not name.startswith("__")]
