from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QImage

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

    if padding_px <= 0.0:
        return quad.astype(np.float32, copy=True)
    center = np.mean(quad.astype(np.float64), axis=0).astype(np.float32)
    vectors = quad.astype(np.float32) - center
    lengths = np.linalg.norm(vectors, axis=1).astype(np.float32)
    expanded = quad.astype(np.float32, copy=True)
    valid = lengths > 1e-6
    if np.any(valid):
        scale = ((lengths[valid] + float(padding_px)) / lengths[valid]).astype(np.float32)
        expanded[valid] = center + vectors[valid] * scale[:, None]
    return expanded


def _cell_crosses_longitude_break(screen_longitudes_rad: np.ndarray, row: int, column: int) -> bool:
    cell_longitudes = np.asarray(
        [
            screen_longitudes_rad[row, column],
            screen_longitudes_rad[row, column + 1],
            screen_longitudes_rad[row + 1, column + 1],
            screen_longitudes_rad[row + 1, column],
        ],
        dtype=np.float64,
    )
    return (
        not np.all(np.isfinite(cell_longitudes))
        or np.any(np.abs(np.diff(np.concatenate((cell_longitudes, cell_longitudes[:1])))) > np.pi)
    )


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
    opacity: float = 1.0,
    screen_longitudes_rad: np.ndarray | None = None,
    seam_padding_px: float = 0.75,
    max_cell_bbox_area_fraction: float = 0.45,
) -> bool:
    """把源图网格纹理逐单元透视贴到目标 RGBA 缓冲区。"""

    if cv2 is None:
        return False
    height, width, _channels = rgba.shape
    source_height, source_width, _source_channels = source_rgb.shape
    max_cell_bbox_area = max(64.0, width * height * float(max_cell_bbox_area_fraction))
    opacity_alpha = int(round(255.0 * max(0.0, min(1.0, float(opacity)))))
    if opacity_alpha <= 0:
        return True

    rows, columns = valid_points.shape
    for row in range(rows - 1):
        for column in range(columns - 1):
            if not bool(
                valid_points[row, column]
                and valid_points[row, column + 1]
                and valid_points[row + 1, column + 1]
                and valid_points[row + 1, column]
            ):
                continue
            if screen_longitudes_rad is not None and _cell_crosses_longitude_break(
                screen_longitudes_rad,
                row,
                column,
            ):
                continue

            dst_quad = expand_quad_for_seamless_fill(
                np.asarray(
                    [
                        [screen_x_px[row, column], screen_y_px[row, column]],
                        [screen_x_px[row, column + 1], screen_y_px[row, column + 1]],
                        [screen_x_px[row + 1, column + 1], screen_y_px[row + 1, column + 1]],
                        [screen_x_px[row + 1, column], screen_y_px[row + 1, column]],
                    ],
                    dtype=np.float32,
                ),
                seam_padding_px,
            )
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
            if bbox_width * bbox_height > max_cell_bbox_area:
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
            target[visible, :3] = warped[visible]
            target[visible, 3] = np.minimum(mask[visible], opacity_alpha)
    return True
