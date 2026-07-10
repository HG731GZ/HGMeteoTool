"""正向累积 remap 的空洞检测、快速修补与收尾。"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 是 remap 修补依赖。
    cv2 = None


MOSAIC_FORWARD_REMAP_LOW_WEIGHT_THRESHOLD = 0.25


def finalize_forward_inverse_map(
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    tile_size: int,
    *,
    exact_remap_repair: bool,
    exact_repair: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    fast_progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """将正向累计坐标转换为 OpenCV 可用的逆向 map。"""

    map_x = np.full(weights.shape, -1.0, dtype=np.float32)
    map_y = np.full(weights.shape, -1.0, dtype=np.float32)
    covered = weights > 1e-6
    if np.any(covered):
        map_x[covered] = accum_x[covered] / weights[covered]
        map_y[covered] = accum_y[covered] / weights[covered]
        target_mask = forward_inverse_map_target_mask(covered, tile_size)
        if exact_remap_repair and exact_repair is not None:
            low_weight = covered & (weights < MOSAIC_FORWARD_REMAP_LOW_WEIGHT_THRESHOLD)
            covered |= exact_repair(map_x, map_y, target_mask & (~covered | low_weight))
        elif not exact_remap_repair:
            covered = fill_forward_inverse_map_holes_fast(
                map_x,
                map_y,
                covered,
                target_mask,
                progress_callback=fast_progress_callback,
            )
    map_x[~covered] = -1.0
    map_y[~covered] = -1.0
    return np.ascontiguousarray(map_x, dtype=np.float32), np.ascontiguousarray(map_y, dtype=np.float32)


def forward_inverse_map_target_mask(covered: np.ndarray, tile_size: int) -> np.ndarray:
    """从正向覆盖区域估计源图在目标图上的内部投影范围。"""

    if cv2 is None or not np.any(covered):
        return covered
    radius = max(1, min(32, int(tile_size) * 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def fill_forward_inverse_map_holes_fast(
    map_x: np.ndarray,
    map_y: np.ndarray,
    covered: np.ndarray,
    target_mask: np.ndarray,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """用相邻有效坐标快速填补小空洞，不进行额外精确反算。"""

    if cv2 is None:
        return covered
    if not np.any(target_mask & ~covered):
        return covered
    filled = covered.copy()
    kernel = np.ones((3, 3), dtype=np.float32)
    max_iterations = 32
    if progress_callback is not None:
        progress_callback(0, max_iterations)
    for index in range(max_iterations):
        missing = target_mask & ~filled
        if not np.any(missing):
            break
        count = cv2.filter2D(filled.astype(np.float32), cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
        sum_x = cv2.filter2D(np.where(filled, map_x, 0.0).astype(np.float32), cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
        sum_y = cv2.filter2D(np.where(filled, map_y, 0.0).astype(np.float32), cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
        fillable = missing & (count > 0.0)
        if not np.any(fillable):
            break
        map_x[fillable] = sum_x[fillable] / count[fillable]
        map_y[fillable] = sum_y[fillable] / count[fillable]
        filled[fillable] = True
        if progress_callback is not None:
            progress_callback(index + 1, max_iterations)
    return filled
