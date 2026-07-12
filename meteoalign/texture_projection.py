from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QImage

from .geometry2d import cell_crosses_angle_break, expand_polygon_radially, grid_cell_quad

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 是项目依赖；兜底方便环境诊断。
    cv2 = None


def texture_projection_available() -> bool:
    """返回当前环境是否可以执行逐网格纹理投影。"""

    return cv2 is not None


def qimage_to_rgb_array(image: QImage) -> np.ndarray:
    """把 QImage 转成独立持有内存的 RGB uint8 数组。"""

    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    pointer = rgb_image.bits()
    pointer.setsize(rgb_image.byteCount())
    raw = np.frombuffer(pointer, dtype=np.uint8).reshape((height, bytes_per_line))
    return raw[:, : width * 3].reshape((height, width, 3)).copy()


def rgba_array_to_qimage(rgba: np.ndarray) -> QImage:
    """把 RGBA uint8 数组转成独立持有内存的 QImage。"""

    image_array = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width, _channels = image_array.shape
    qimage = QImage(image_array.data, width, height, width * 4, QImage.Format_RGBA8888)
    return qimage.copy()


def expand_quad_for_seamless_fill(quad: np.ndarray, padding_px: float) -> np.ndarray:
    """把目标四边形轻微外扩，盖住逐格贴片留下的接缝。"""

    return expand_polygon_radially(quad, padding_px).astype(np.float32)


def _cell_crosses_longitude_break(screen_longitudes_rad: np.ndarray, row: int, column: int) -> bool:
    return cell_crosses_angle_break(screen_longitudes_rad, row, column)


def warp_grid_texture_to_rgba(
    rgba: np.ndarray,
    *,
    source_rgb: np.ndarray,
    source_grid_x_px: np.ndarray,
    source_grid_y_px: np.ndarray,
    source_scale_x: float,
    source_scale_y: float,
    screen_x_px: np.ndarray,
    screen_y_px: np.ndarray,
    valid_points: np.ndarray,
    projection_valid_points: np.ndarray | None = None,
    target_valid_mask: np.ndarray | None = None,
    opacity: float = 1.0,
    screen_longitudes_rad: np.ndarray | None = None,
    seam_padding_px: float = 0.75,
    max_cell_bbox_area_fraction: float = 0.45,
) -> bool:
    """把源图网格纹理逐单元透视贴到目标 RGBA 缓冲区。

    valid_points 表示源网格点本身是否有效。投影边界外的点应通过
    projection_valid_points 标记，而不应让整个网格单元被跳过；最终由
    target_valid_mask 把单元在目标视场边缘裁掉。
    """

    if cv2 is None:
        return False
    height, width, _channels = rgba.shape
    source_height, source_width, _source_channels = source_rgb.shape
    max_cell_bbox_area = max(64.0, width * height * float(max_cell_bbox_area_fraction))
    opacity_alpha = int(round(255.0 * max(0.0, min(1.0, float(opacity)))))
    if opacity_alpha <= 0:
        return True

    rows, columns = valid_points.shape
    projection_valid = (
        np.asarray(projection_valid_points, dtype=bool)
        if projection_valid_points is not None
        else np.asarray(valid_points, dtype=bool)
    )
    target_mask = None if target_valid_mask is None else np.asarray(target_valid_mask, dtype=bool)
    if projection_valid.shape != valid_points.shape:
        raise ValueError("projection_valid_points 的形状必须与 valid_points 一致。")
    if target_mask is not None and target_mask.shape != (height, width):
        raise ValueError("target_valid_mask 的形状必须与目标 RGBA 缓冲区一致。")
    for row in range(rows - 1):
        for column in range(columns - 1):
            source_cell_valid = bool(
                valid_points[row, column]
                and valid_points[row, column + 1]
                and valid_points[row + 1, column + 1]
                and valid_points[row + 1, column]
            )
            if not source_cell_valid:
                continue
            projection_cell_valid = np.asarray(
                [
                    projection_valid[row, column],
                    projection_valid[row, column + 1],
                    projection_valid[row + 1, column + 1],
                    projection_valid[row + 1, column],
                ],
                dtype=bool,
            )
            if not bool(np.any(projection_cell_valid)):
                continue
            if screen_longitudes_rad is not None and _cell_crosses_longitude_break(
                screen_longitudes_rad,
                row,
                column,
            ):
                continue

            dst_quad = expand_quad_for_seamless_fill(
                grid_cell_quad(screen_x_px, screen_y_px, row, column).astype(np.float32),
                seam_padding_px,
            )
            if not np.all(np.isfinite(dst_quad)):
                continue
            if abs(float(cv2.contourArea(dst_quad))) < 0.25:
                continue

            dst_min = np.floor(np.min(dst_quad, axis=0) - 1.0).astype(np.int64)
            dst_max = np.ceil(np.max(dst_quad, axis=0) + 1.0).astype(np.int64)
            bbox_left = max(0, int(dst_min[0]))
            bbox_top = max(0, int(dst_min[1]))
            bbox_right = min(width - 1, int(dst_max[0]))
            bbox_bottom = min(height - 1, int(dst_max[1]))
            bbox_width = bbox_right - bbox_left + 1
            bbox_height = bbox_bottom - bbox_top + 1
            if bbox_width <= 1 or bbox_height <= 1:
                continue
            is_partially_outside_view = bool(np.any(projection_cell_valid) and not np.all(projection_cell_valid))
            if bbox_width * bbox_height > max_cell_bbox_area and not (
                target_mask is not None and is_partially_outside_view
            ):
                continue

            src_quad = np.asarray(
                [
                    [
                        source_grid_x_px[row, column] * source_scale_x,
                        source_grid_y_px[row, column] * source_scale_y,
                    ],
                    [
                        source_grid_x_px[row, column + 1] * source_scale_x,
                        source_grid_y_px[row, column + 1] * source_scale_y,
                    ],
                    [
                        source_grid_x_px[row + 1, column + 1] * source_scale_x,
                        source_grid_y_px[row + 1, column + 1] * source_scale_y,
                    ],
                    [
                        source_grid_x_px[row + 1, column] * source_scale_x,
                        source_grid_y_px[row + 1, column] * source_scale_y,
                    ],
                ],
                dtype=np.float32,
            )
            src_min = np.floor(np.min(src_quad, axis=0) - 1.0).astype(np.int64)
            src_max = np.ceil(np.max(src_quad, axis=0) + 1.0).astype(np.int64)
            src_left = max(0, int(src_min[0]))
            src_top = max(0, int(src_min[1]))
            src_right = min(source_width - 1, int(src_max[0]))
            src_bottom = min(source_height - 1, int(src_max[1]))
            if src_right <= src_left or src_bottom <= src_top:
                continue

            src_crop = source_rgb[src_top : src_bottom + 1, src_left : src_right + 1]
            src_relative = src_quad - np.asarray([src_left, src_top], dtype=np.float32)
            dst_relative = dst_quad - np.asarray([bbox_left, bbox_top], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_relative, dst_relative)
            warped = cv2.warpPerspective(
                src_crop,
                matrix,
                (bbox_width, bbox_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            mask = np.zeros((bbox_height, bbox_width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.rint(dst_relative).astype(np.int32), 255, lineType=cv2.LINE_8)
            target = rgba[bbox_top : bbox_bottom + 1, bbox_left : bbox_right + 1]
            visible = mask > 0
            if target_mask is not None:
                visible &= target_mask[bbox_top : bbox_bottom + 1, bbox_left : bbox_right + 1]
            if not np.any(visible):
                continue
            target[visible, :3] = warped[visible]
            target[visible, 3] = np.minimum(mask[visible], opacity_alpha)
    return True
