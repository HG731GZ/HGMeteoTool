from __future__ import annotations

import numpy as np

from .constants import (
    ANCHOR_INTERPOLATION_TPS,
    FIT_WEIGHT_MAX,
    FIT_WEIGHT_MIN,
    SOFT_CONSTRAINT_TPS_SMOOTHING_BASE,
)


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


def _fit_point_weights(point_count: int, point_weights: np.ndarray | None) -> np.ndarray:
    if point_weights is None:
        return np.ones(point_count, dtype=np.float64)
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != point_count:
        raise ValueError("拟合权重数量必须与匹配星数量一致。")
    if not np.all(np.isfinite(weights)):
        raise ValueError("拟合权重包含无效数值。")
    return np.clip(weights, FIT_WEIGHT_MIN, FIT_WEIGHT_MAX).astype(np.float64)


def _residual_anchor_mask(point_count: int, anchor_mask: np.ndarray | None) -> np.ndarray:
    if anchor_mask is None:
        return np.ones(point_count, dtype=bool)
    mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != point_count:
        raise ValueError("残差锚点标记数量必须与匹配星数量一致。")
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

__all__ = [name for name in globals() if not name.startswith("__")]
