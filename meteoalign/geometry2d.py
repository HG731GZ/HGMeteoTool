from __future__ import annotations

import numpy as np


def expand_polygon_radially(points: np.ndarray, pixels: float) -> np.ndarray:
    """把二维多边形按质心方向外扩，用于遮住逐格绘制产生的细缝。"""

    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("points 必须是形状为 (N, 2) 的二维坐标数组。")
    if polygon.shape[0] == 0 or pixels <= 0.0:
        return polygon.copy()

    center = np.mean(polygon, axis=0)
    vectors = polygon - center
    lengths = np.linalg.norm(vectors, axis=1)
    expanded = polygon.copy()
    valid = lengths > 1e-6
    if np.any(valid):
        expanded[valid] = center + vectors[valid] * ((lengths[valid] + float(pixels)) / lengths[valid])[:, None]
    return expanded


def cell_crosses_angle_break(
    angle_grid_rad: np.ndarray,
    row: int,
    column: int,
    *,
    break_threshold_rad: float = np.pi,
) -> bool:
    """判断网格单元是否跨过圆柱投影经度断点。"""

    grid = np.asarray(angle_grid_rad, dtype=np.float64)
    cell_angles = np.asarray(
        [
            grid[row, column],
            grid[row, column + 1],
            grid[row + 1, column + 1],
            grid[row + 1, column],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(cell_angles)):
        return True
    closed = np.concatenate((cell_angles, cell_angles[:1]))
    return bool(np.any(np.abs(np.diff(closed)) > float(break_threshold_rad)))


def grid_cell_quad(x_grid: np.ndarray, y_grid: np.ndarray, row: int, column: int) -> np.ndarray:
    """从同形状的 x/y 网格中取出一个四边形单元。"""

    return np.asarray(
        [
            [x_grid[row, column], y_grid[row, column]],
            [x_grid[row, column + 1], y_grid[row, column + 1]],
            [x_grid[row + 1, column + 1], y_grid[row + 1, column + 1]],
            [x_grid[row + 1, column], y_grid[row + 1, column]],
        ],
        dtype=np.float64,
    )
