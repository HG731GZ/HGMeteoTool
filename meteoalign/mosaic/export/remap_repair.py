"""正向累积 remap 的空洞检测、快速修补与收尾。"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 是 remap 修补依赖。
    cv2 = None


MOSAIC_FORWARD_REMAP_LOW_WEIGHT_THRESHOLD = 0.25
MOSAIC_FORWARD_REMAP_PROPAGATION_MAX_RADIUS_PX = 32
MOSAIC_FORWARD_REMAP_GUARD_MAX_SOURCE_DISTANCE_PX = 64.0
MOSAIC_FORWARD_REMAP_LABEL_BLOCK_ROWS = 512


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
    """将正向累计坐标原位转换为 OpenCV 可用的逆向 map。"""

    # 累积数组在收尾后不再使用，直接复用可少保留两张全尺寸 float32 画布。
    map_x = _reusable_float32_map(accum_x)
    map_y = _reusable_float32_map(accum_y)
    covered = weights > 1e-6
    if np.any(covered):
        np.divide(map_x, weights, out=map_x, where=covered)
        np.divide(map_y, weights, out=map_y, where=covered)
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
                tile_size=tile_size,
                work_buffer=weights,
                progress_callback=fast_progress_callback,
            )
    map_x[~covered] = -1.0
    map_y[~covered] = -1.0
    return np.ascontiguousarray(map_x, dtype=np.float32), np.ascontiguousarray(map_y, dtype=np.float32)


def _reusable_float32_map(values: np.ndarray) -> np.ndarray:
    """优先复用可写的连续 float32 数组，否则建立安全副本。"""

    array = np.asarray(values)
    if array.dtype == np.float32 and array.flags.c_contiguous and array.flags.writeable:
        return array
    return np.array(array, dtype=np.float32, order="C", copy=True)


def forward_inverse_map_target_mask(covered: np.ndarray, tile_size: int) -> np.ndarray:
    """从正向覆盖区域估计源图在目标图上的内部投影范围。"""

    if cv2 is None or not np.any(covered):
        return covered
    radius = _forward_remap_propagation_radius(tile_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def _forward_remap_propagation_radius(tile_size: int) -> int:
    """按正向采样网格估算内部空洞的最大传播半径。"""

    return max(
        1,
        min(
            MOSAIC_FORWARD_REMAP_PROPAGATION_MAX_RADIUS_PX,
            int(tile_size) * 2,
        ),
    )


def fill_forward_inverse_map_holes_fast(
    map_x: np.ndarray,
    map_y: np.ndarray,
    covered: np.ndarray,
    target_mask: np.ndarray,
    *,
    tile_size: int = 16,
    work_buffer: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """单次传播有效坐标，并用局部均值与大范围均值共同保护投影边界。"""

    if cv2 is None:
        return covered
    missing = target_mask & ~covered
    if not np.any(missing):
        return covered

    maximum = 8
    if progress_callback is not None:
        progress_callback(0, maximum)

    radius = _forward_remap_propagation_radius(tile_size)
    guard_kernel_size = radius * 2 + 1
    if (
        work_buffer is not None
        and work_buffer.shape == map_x.shape
        and work_buffer.dtype == np.float32
        and work_buffer.flags.c_contiguous
        and work_buffer.flags.writeable
    ):
        scratch = work_buffer
    else:
        scratch = np.empty_like(map_x, dtype=np.float32)
    scratch[...] = covered
    guard_count = cv2.boxFilter(
        scratch,
        cv2.CV_32F,
        (guard_kernel_size, guard_kernel_size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    guard_fillable = missing & (guard_count > 0.0)
    if progress_callback is not None:
        progress_callback(1, maximum)

    scratch.fill(0.0)
    np.copyto(scratch, map_x, where=covered)
    guard_x = cv2.boxFilter(
        scratch,
        cv2.CV_32F,
        (guard_kernel_size, guard_kernel_size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    np.divide(guard_x, guard_count, out=guard_x, where=guard_fillable)
    map_x[guard_fillable] = guard_x[guard_fillable]
    del guard_x
    if progress_callback is not None:
        progress_callback(2, maximum)

    scratch.fill(0.0)
    np.copyto(scratch, map_y, where=covered)
    guard_y = cv2.boxFilter(
        scratch,
        cv2.CV_32F,
        (guard_kernel_size, guard_kernel_size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    np.divide(guard_y, guard_count, out=guard_y, where=guard_fillable)
    map_y[guard_fillable] = guard_y[guard_fillable]
    del guard_count, guard_y
    if progress_callback is not None:
        progress_callback(3, maximum)

    scratch[...] = covered
    local_count = cv2.boxFilter(
        scratch,
        cv2.CV_32F,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    local_fillable = missing & (local_count > 0.0)
    scratch.fill(0.0)
    np.copyto(scratch, map_x, where=covered)
    local_x = cv2.boxFilter(
        scratch,
        cv2.CV_32F,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    np.divide(local_x, local_count, out=local_x, where=local_fillable)
    if progress_callback is not None:
        progress_callback(4, maximum)

    scratch.fill(0.0)
    np.copyto(scratch, map_y, where=covered)
    local_y = cv2.boxFilter(
        scratch,
        cv2.CV_32F,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    np.divide(local_y, local_count, out=local_y, where=local_fillable)
    del local_count, scratch

    guard_distance = min(
        MOSAIC_FORWARD_REMAP_GUARD_MAX_SOURCE_DISTANCE_PX,
        max(4.0, float(tile_size) * 2.0),
    )
    _apply_guarded_candidate_maps(
        map_x,
        map_y,
        local_x,
        local_y,
        local_fillable,
        guard_distance,
    )
    del local_x, local_y
    if progress_callback is not None:
        progress_callback(5, maximum)

    remaining = missing & ~local_fillable
    if progress_callback is not None:
        progress_callback(6, maximum)
    if np.any(remaining):
        _fill_remaining_from_nearest_covered(
            map_x,
            map_y,
            covered,
            remaining,
            guard_distance,
        )
    if progress_callback is not None:
        progress_callback(7, maximum)

    filled = covered | guard_fillable
    map_x[~filled] = -1.0
    map_y[~filled] = -1.0
    if progress_callback is not None:
        progress_callback(maximum, maximum)
    return filled


def _apply_guarded_candidate_maps(
    map_x: np.ndarray,
    map_y: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    candidate_mask: np.ndarray,
    maximum_source_distance_px: float,
) -> None:
    """候选坐标与大范围保护值相近时才覆盖，避免跨投影边界传播。"""

    safe_rows = max(1, int(MOSAIC_FORWARD_REMAP_LABEL_BLOCK_ROWS))
    threshold_squared = float(maximum_source_distance_px) ** 2
    for row_start in range(0, map_x.shape[0], safe_rows):
        row_end = min(map_x.shape[0], row_start + safe_rows)
        mask_block = candidate_mask[row_start:row_end]
        if not np.any(mask_block):
            continue
        delta_x = candidate_x[row_start:row_end] - map_x[row_start:row_end]
        delta_y = candidate_y[row_start:row_end] - map_y[row_start:row_end]
        accepted = mask_block & (delta_x * delta_x + delta_y * delta_y <= threshold_squared)
        map_x_block = map_x[row_start:row_end]
        map_y_block = map_y[row_start:row_end]
        candidate_x_block = candidate_x[row_start:row_end]
        candidate_y_block = candidate_y[row_start:row_end]
        map_x_block[accepted] = candidate_x_block[accepted]
        map_y_block[accepted] = candidate_y_block[accepted]


def _fill_remaining_from_nearest_covered(
    map_x: np.ndarray,
    map_y: np.ndarray,
    covered: np.ndarray,
    remaining: np.ndarray,
    maximum_source_distance_px: float,
) -> None:
    """用距离变换一次定位最近有效坐标，并继续应用投影边界保护。"""

    distance_input = (~covered).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    del distance, distance_input
    maximum_label = int(labels.max())
    nearest_x = np.full(maximum_label + 1, -1.0, dtype=np.float32)
    nearest_y = np.full(maximum_label + 1, -1.0, dtype=np.float32)
    safe_rows = max(1, int(MOSAIC_FORWARD_REMAP_LABEL_BLOCK_ROWS))
    for row_start in range(0, map_x.shape[0], safe_rows):
        row_end = min(map_x.shape[0], row_start + safe_rows)
        covered_block = covered[row_start:row_end]
        if not np.any(covered_block):
            continue
        label_block = labels[row_start:row_end]
        nearest_x[label_block[covered_block]] = map_x[row_start:row_end][covered_block]
        nearest_y[label_block[covered_block]] = map_y[row_start:row_end][covered_block]

    threshold_squared = float(maximum_source_distance_px) ** 2
    for row_start in range(0, map_x.shape[0], safe_rows):
        row_end = min(map_x.shape[0], row_start + safe_rows)
        remaining_block = remaining[row_start:row_end]
        if not np.any(remaining_block):
            continue
        label_block = labels[row_start:row_end]
        candidate_x = nearest_x[label_block]
        candidate_y = nearest_y[label_block]
        map_x_block = map_x[row_start:row_end]
        map_y_block = map_y[row_start:row_end]
        delta_x = candidate_x - map_x_block
        delta_y = candidate_y - map_y_block
        accepted = remaining_block & (delta_x * delta_x + delta_y * delta_y <= threshold_squared)
        map_x_block[accepted] = candidate_x[accepted]
        map_y_block[accepted] = candidate_y[accepted]
