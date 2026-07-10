from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPolygonF

from ..geometry2d import cell_crosses_angle_break, grid_cell_quad
from ..image_preview import load_image_preview
from ..projection.grid import project_altaz_grid_to_screen, radec_grid_to_altaz
from ..simulator import CameraSettings, ObserverSettings, ViewSettings
from ..texture_projection import qimage_to_rgb_array
from .model_io import (
    MosaicCoverageCache,
    MosaicSourceModel,
    MosaicSourceTextureCache,
    _expanded_polygon_points,
)

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 是纹理投影依赖；兜底方便环境诊断。
    cv2 = None


# 源图纹理默认长边像素上限；运行时优先使用 UI 配置传入的值。
MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX = 1920


def coverage_altaz(
    cache: MosaicCoverageCache,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将覆盖缓存的 RA/Dec 网格转换为当前观测者的地平坐标。"""
    return radec_grid_to_altaz(cache.ra_deg, cache.dec_deg, cache.valid, observer)


def paint_coverage_overlay(
    image: QImage,
    cache: MosaicCoverageCache,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    opacity: float,
) -> None:
    """在已渲染的星空预览图上绘制源图覆盖范围（半透明填充+边界线）。

    参数
    ----
    image : QImage
        已渲染星空背景的 ARGB32 图像，覆盖范围直接绘制其上。
    cache : MosaicCoverageCache
        源图像素网格到天球的映射缓存。
    camera : CameraSettings
        当前输出投影的相机参数。
    view : ViewSettings
        当前取景方向。
    observer : ObserverSettings
        观测者位置与时间。
    opacity : float
        覆盖填充不透明度，0.0-1.0。
    """
    if cache is None:
        return

    cache_alt, cache_az, cache_valid = coverage_altaz(cache, observer)
    screen_grid = project_altaz_grid_to_screen(
        cache_alt, cache_az, camera=camera, view=view, valid=cache_valid,
    )
    screen_x = screen_grid.x_px
    screen_y = screen_grid.y_px
    valid = screen_grid.valid
    screen_longitudes = screen_grid.screen_longitudes_rad
    fill_color = QColor(205, 205, 205, int(round(255.0 * opacity)))
    edge_color = QColor(245, 245, 245, int(round(220.0 * min(1.0, opacity + 0.25))))
    max_cell_bbox_area = max(256.0, image.width() * image.height() * 0.35)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(fill_color)

    for row in range(cache.grid_rows - 1):
        for column in range(cache.grid_columns - 1):
            if not bool(
                valid[row, column]
                and valid[row, column + 1]
                and valid[row + 1, column + 1]
                and valid[row + 1, column]
            ):
                continue
            quad = grid_cell_quad(screen_x, screen_y, row, column)
            xs = quad[:, 0]
            ys = quad[:, 1]
            if screen_longitudes is not None and cell_crosses_angle_break(screen_longitudes, row, column):
                continue
            bbox_area = max(float(np.max(xs) - np.min(xs)), 0.0) * max(float(np.max(ys) - np.min(ys)), 0.0)
            if bbox_area <= 0.05 or bbox_area > max_cell_bbox_area:
                continue
            polygon = QPolygonF(_expanded_polygon_points(xs, ys, 0.75))
            painter.drawPolygon(polygon)

    painter.setPen(edge_color)
    painter.setBrush(Qt.NoBrush)
    _paint_coverage_boundary(painter, screen_x, screen_y, valid)
    painter.end()


def _paint_coverage_boundary(
    painter: QPainter,
    screen_x: np.ndarray,
    screen_y: np.ndarray,
    valid: np.ndarray,
) -> None:
    """沿覆盖网格的四条外边绘制边界折线，跳过无效点和跨越过大的断点。"""
    edge_indices = [
        [(0, column) for column in range(screen_x.shape[1])],
        [(row, screen_x.shape[1] - 1) for row in range(screen_x.shape[0])],
        [(screen_x.shape[0] - 1, column) for column in range(screen_x.shape[1] - 1, -1, -1)],
        [(row, 0) for row in range(screen_x.shape[0] - 1, -1, -1)],
    ]
    for edge in edge_indices:
        current = QPolygonF()
        previous_x: float | None = None
        previous_y: float | None = None
        for row, column in edge:
            if not bool(valid[row, column]):
                if len(current) >= 2:
                    painter.drawPolyline(current)
                current = QPolygonF()
                previous_x = None
                previous_y = None
                continue
            x_value = float(screen_x[row, column])
            y_value = float(screen_y[row, column])
            if previous_x is not None and math.hypot(x_value - previous_x, y_value - previous_y) > 240.0:
                if len(current) >= 2:
                    painter.drawPolyline(current)
                current = QPolygonF()
            current.append(QPointF(x_value, y_value))
            previous_x = x_value
            previous_y = y_value
        if len(current) >= 2:
            painter.drawPolyline(current)


def load_source_texture(
    source_model: MosaicSourceModel,
    existing_cache: MosaicSourceTextureCache | None = None,
    *,
    max_long_side_px: int | None = None,
) -> MosaicSourceTextureCache | None:
    """加载或复用源图纹理缓存。

    参数
    ----
    source_model : MosaicSourceModel
        包含源图路径和尺寸信息的模型。
    existing_cache : MosaicSourceTextureCache | None
        已有的缓存，若路径匹配则直接返回。
    max_long_side_px : int | None
        贴图长边像素上限；None 时使用默认值。

    返回
    ----
    MosaicSourceTextureCache | None
        纹理缓存，加载失败时返回 None。
    """
    image_path = source_model.source_image_path
    if image_path is None:
        return None
    try:
        resolved_path = image_path.expanduser().resolve()
    except OSError:
        resolved_path = image_path.expanduser()
    texture_long_side = _safe_texture_long_side_px(max_long_side_px)
    if existing_cache is not None and existing_cache.source_image_path == resolved_path:
        if existing_cache.texture_max_long_side_px == texture_long_side:
            return existing_cache
        scaled_cache = _scaled_texture_cache(existing_cache, texture_long_side)
        if scaled_cache is not None:
            return scaled_cache
    if not resolved_path.exists():
        return None
    try:
        preview = load_image_preview(resolved_path, max_long_side_px=texture_long_side)
        source_rgb = qimage_to_rgb_array(preview.image)
    except Exception:
        return None
    return MosaicSourceTextureCache(
        source_image_path=resolved_path,
        source_rgb=source_rgb.astype(np.uint8),
        source_scale_x=source_rgb.shape[1] / max(float(source_model.image_width_px), 1.0),
        source_scale_y=source_rgb.shape[0] / max(float(source_model.image_height_px), 1.0),
        source_width_px=int(source_model.image_width_px),
        source_height_px=int(source_model.image_height_px),
        texture_max_long_side_px=texture_long_side,
    )


def _safe_texture_long_side_px(max_long_side_px: int | None) -> int:
    """规整贴图长边上限，避免配置异常导致空图或超大贴图。"""

    if max_long_side_px is None:
        return MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX
    try:
        value = int(round(float(max_long_side_px)))
    except (TypeError, ValueError):
        value = MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX
    return max(1, value)


def _scaled_texture_cache(
    existing_cache: MosaicSourceTextureCache,
    max_long_side_px: int,
) -> MosaicSourceTextureCache | None:
    """从已有较高清缓存降采样生成低清贴图缓存。"""

    if existing_cache.texture_max_long_side_px < max_long_side_px:
        return None
    source_rgb = existing_cache.source_rgb
    current_height, current_width = source_rgb.shape[:2]
    current_long_side = max(int(current_width), int(current_height))
    if current_long_side <= 0:
        return None
    if current_long_side <= max_long_side_px:
        scaled_rgb = source_rgb
    else:
        if cv2 is None:
            return None
        scale = max_long_side_px / float(current_long_side)
        target_width = max(1, int(round(current_width * scale)))
        target_height = max(1, int(round(current_height * scale)))
        scaled_rgb = cv2.resize(source_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return MosaicSourceTextureCache(
        source_image_path=existing_cache.source_image_path,
        source_rgb=np.ascontiguousarray(scaled_rgb, dtype=np.uint8),
        source_scale_x=scaled_rgb.shape[1] / max(float(existing_cache.source_width_px), 1.0),
        source_scale_y=scaled_rgb.shape[0] / max(float(existing_cache.source_height_px), 1.0),
        source_width_px=existing_cache.source_width_px,
        source_height_px=existing_cache.source_height_px,
        texture_max_long_side_px=max_long_side_px,
    )


def paint_source_image_overlay(
    image: QImage,
    texture_renderer,
    cache: MosaicCoverageCache,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    texture: MosaicSourceTextureCache,
    opacity: float,
) -> None:
    """将源图纹理按天球网格重投影并叠加到输出图像上。"""
    cache_alt, cache_az, cache_valid = coverage_altaz(cache, observer)
    texture_renderer.paint_on_qimage(
        image,
        camera=camera,
        view=view,
        source_rgb=texture.source_rgb,
        source_grid_x_px=cache.grid_x_px,
        source_grid_y_px=cache.grid_y_px,
        source_scale_x=texture.source_scale_x,
        source_scale_y=texture.source_scale_y,
        alt_deg=cache_alt,
        az_deg=cache_az,
        valid_points=cache_valid,
        opacity=opacity,
    )


__all__ = [
    "coverage_altaz",
    "load_source_texture",
    "paint_coverage_overlay",
    "paint_source_image_overlay",
]
