from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ANCHOR_INTERPOLATION_TPS
from .validation import (
    _array_is_finite,
    _fit_point_weights,
    _residual_anchor_mask,
    _tps_smoothing_from_constraints,
)


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

__all__ = [name for name in globals() if not name.startswith("__")]
