from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PyQt5.QtCore import QEvent, QObject, QPoint, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QDialog, QGraphicsScene, QMessageBox, QProgressDialog

from .app_constants import TOUCHPAD_ZOOM_MAX_FACTOR, TOUCHPAD_ZOOM_MIN_FACTOR, TOUCHPAD_ZOOM_SENSITIVITY
from .app_graphics_items import GraphicsImageItem
from .config import StarMapUiConfig
from .fixed_camera_model import FixedCameraModel
from .renderer import StarMapRenderer
from .simulator import (
    CameraSettings,
    FISHEYE_LENS_MODELS,
    HorizontalMilkyWayCatalog,
    HorizontalSolarSystemCatalog,
    HorizontalStarCatalog,
    ObserverSettings,
    ViewSettings,
    _project_altaz_points,
    camera_basis_from_view,
    horizontal_fov_deg,
    project_horizontal_catalog,
    vertical_fov_deg,
)
from .ui.ui_mapping_validation_dialog import Ui_MappingValidationDialog

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 是项目依赖；兜底方便环境诊断。
    cv2 = None


VALIDATION_PIXEL_TO_SKY_CHUNK_SIZE = 24_576
VALIDATION_MIN_RENDER_SIZE_PX = 96
VALIDATION_ZOOM_FACTOR = 1.18
VALIDATION_DEFAULT_GRID_CAP_PIXELS = 60_000_000
VALIDATION_SOURCE_PREVIEW_LONG_SIDE_PX = 4096
VALIDATION_GRID_MIN_PRECISION = 12
VALIDATION_GRID_MAX_PRECISION = 180


@dataclass(frozen=True)
class ImageSkyProjectionCache:
    source_width_px: int
    source_height_px: int
    source_scale_x: float
    source_scale_y: float
    grid_precision: int
    grid_rows: int
    grid_columns: int
    grid_point_count: int
    grid_x_px: np.ndarray
    grid_y_px: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray
    valid: np.ndarray
    source_rgb: np.ndarray


def _qimage_to_rgb_array(image: QImage) -> np.ndarray:
    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    pointer = rgb_image.bits()
    pointer.setsize(rgb_image.byteCount())
    raw = np.frombuffer(pointer, dtype=np.uint8).reshape((height, bytes_per_line))
    return raw[:, : width * 3].reshape((height, width, 3)).copy()


def _rgba_array_to_qimage(rgba: np.ndarray) -> QImage:
    image_array = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width, _channels = image_array.shape
    qimage = QImage(image_array.data, width, height, width * 4, QImage.Format_RGBA8888)
    return qimage.copy()


class PixelToSkyValidationWorker(QObject):
    """后台计算真实图像网格点到固定相机本地地平坐标的验证缓存。"""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        model: FixedCameraModel,
        source_image: QImage,
        grid_precision: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.source_image = source_image.copy()
        self.grid_precision = max(
            VALIDATION_GRID_MIN_PRECISION,
            min(VALIDATION_GRID_MAX_PRECISION, int(grid_precision)),
        )

    def run(self) -> None:
        try:
            cache = self._build_cache()
            self.finished.emit(cache)
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把模型和星历错误传回界面。
            self.failed.emit(str(exc))

    def _grid_shape(self, width: int, height: int) -> tuple[int, int]:
        if width >= height:
            columns = self.grid_precision
            rows = max(2, int(round(self.grid_precision * height / max(float(width), 1.0))))
        else:
            rows = self.grid_precision
            columns = max(2, int(round(self.grid_precision * width / max(float(height), 1.0))))
        return rows, columns

    def _source_preview_rgb(self, width: int, height: int) -> tuple[np.ndarray, float, float]:
        scale = min(1.0, VALIDATION_SOURCE_PREVIEW_LONG_SIDE_PX / max(float(width), float(height), 1.0))
        if scale < 1.0:
            preview_width = max(1, int(round(width * scale)))
            preview_height = max(1, int(round(height * scale)))
            preview = self.source_image.scaled(preview_width, preview_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            preview = self.source_image
        rgb = _qimage_to_rgb_array(preview)
        return rgb, rgb.shape[1] / max(float(width), 1.0), rgb.shape[0] / max(float(height), 1.0)

    def _build_cache(self) -> ImageSkyProjectionCache:
        width = self.source_image.width()
        height = self.source_image.height()
        rows, columns = self._grid_shape(width, height)
        x_values = np.linspace(0.0, max(width - 1, 0), columns, dtype=np.float64)
        y_values = np.linspace(0.0, max(height - 1, 0), rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        point_count = int(grid_points.shape[0])

        self.progress.emit(
            2,
            "正在准备真实图像网格到天球方向的验证缓存...\n"
            f"图像 {width} x {height} px，网格 {columns} x {rows}。",
        )

        alt_deg = np.full(point_count, np.nan, dtype=np.float64)
        az_deg = np.full(point_count, np.nan, dtype=np.float64)
        valid = np.zeros(point_count, dtype=bool)
        for start in range(0, point_count, VALIDATION_PIXEL_TO_SKY_CHUNK_SIZE):
            end = min(start + VALIDATION_PIXEL_TO_SKY_CHUNK_SIZE, point_count)
            alt_values, az_values, valid_values = self.model.pixel_to_altaz_points(grid_points[start:end])
            alt_deg[start:end] = alt_values
            az_deg[start:end] = az_values
            valid[start:end] = valid_values
            progress = 2 + int(62 * end / max(point_count, 1))
            self.progress.emit(progress, f"正在按固定相机模型反解网格点 pixel_to_altaz：{end} / {point_count}。")

        self.progress.emit(94, "正在生成显示用真实图像缓存...")
        source_rgb, source_scale_x, source_scale_y = self._source_preview_rgb(width, height)
        valid = (valid & np.isfinite(alt_deg) & np.isfinite(az_deg)).reshape((rows, columns))
        self.progress.emit(100, "映射验证缓存已就绪。")
        return ImageSkyProjectionCache(
            source_width_px=width,
            source_height_px=height,
            source_scale_x=float(source_scale_x),
            source_scale_y=float(source_scale_y),
            grid_precision=self.grid_precision,
            grid_rows=rows,
            grid_columns=columns,
            grid_point_count=point_count,
            grid_x_px=grid_x.astype(np.float64),
            grid_y_px=grid_y.astype(np.float64),
            alt_deg=alt_deg.reshape((rows, columns)).astype(np.float64),
            az_deg=az_deg.reshape((rows, columns)).astype(np.float64),
            valid=valid,
            source_rgb=source_rgb.astype(np.uint8),
        )


class MappingValidationDialog(QDialog):
    """源图映射验证窗口。"""

    def __init__(
        self,
        *,
        parent,
        renderer: StarMapRenderer,
        model: FixedCameraModel,
        source_image: QImage,
        observer: ObserverSettings,
        base_camera: CameraSettings,
        initial_view: ViewSettings,
        visible_mag_limit: float,
        horizontal_catalog: HorizontalStarCatalog,
        horizontal_milky_way: HorizontalMilkyWayCatalog,
        horizontal_solar_system: HorizontalSolarSystemCatalog,
        ui_config: StarMapUiConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.ui = Ui_MappingValidationDialog()
        self.ui.setupUi(self)
        self.renderer = renderer
        self.model = model
        self.source_image = source_image.copy()
        self.observer = observer
        self.base_camera = base_camera
        self.visible_mag_limit = float(visible_mag_limit)
        self.horizontal_catalog = horizontal_catalog
        self.horizontal_milky_way = horizontal_milky_way
        self.horizontal_solar_system = horizontal_solar_system
        self.ui_config = ui_config or StarMapUiConfig()
        self.center_az_deg = float(initial_view.center_az_deg) % 360.0
        self.center_alt_deg = max(-90.0, min(90.0, float(initial_view.center_alt_deg)))
        self.fixed_roll_deg = float(initial_view.roll_deg)
        self.focal_length_mm = max(float(base_camera.focal_length_mm), 1e-6)
        self.fisheye_fov_deg = max(1.0, min(300.0, float(base_camera.fisheye_fov_deg)))
        self.cache: ImageSkyProjectionCache | None = None
        self.last_drag_pos: QPoint | None = None
        self.cache_thread: QThread | None = None
        self.cache_worker: PixelToSkyValidationWorker | None = None
        self.cache_progress: QProgressDialog | None = None
        self._set_default_grid_precision()

        self.scene = QGraphicsScene(self)
        self.sky_item = GraphicsImageItem()
        self.image_overlay_item = GraphicsImageItem()
        self.image_overlay_item.setZValue(10.0)
        self.scene.addItem(self.sky_item)
        self.scene.addItem(self.image_overlay_item)
        self.ui.graphicsViewValidation.setScene(self.scene)
        self.ui.graphicsViewValidation.installEventFilter(self)
        self.ui.graphicsViewValidation.viewport().installEventFilter(self)
        self.ui.graphicsViewValidation.viewport().setMouseTracking(True)

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render_now)
        self.ui.doubleSpinBoxImageOpacity.valueChanged.connect(self._update_overlay_opacity)
        self.ui.checkBoxOverlayEnabled.toggled.connect(self._handle_overlay_toggled)
        self.ui.spinBoxGridPrecision.valueChanged.connect(self._update_grid_precision_tooltip)
        self.ui.pushButtonResolveGrid.clicked.connect(self.resolve_grid_again)
        self._update_overlay_opacity()
        self._handle_overlay_toggled(self.ui.checkBoxOverlayEnabled.isChecked())
        self._start_cache_worker()
        QTimer.singleShot(0, self.render_now)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.cache_thread is not None:
            QMessageBox.information(self, "正在准备验证", "映射验证缓存仍在计算，请等待完成后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched in (self.ui.graphicsViewValidation, self.ui.graphicsViewValidation.viewport()):
            if event.type() == QEvent.NativeGesture and self._handle_native_fov_zoom(event):
                return True
        if watched is self.ui.graphicsViewValidation.viewport():
            if event.type() == QEvent.Resize:
                self.schedule_render()
                return False
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.last_drag_pos = event.pos()
                self.ui.graphicsViewValidation.viewport().setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self.last_drag_pos is not None:
                dx = event.pos().x() - self.last_drag_pos.x()
                dy = event.pos().y() - self.last_drag_pos.y()
                self.last_drag_pos = event.pos()
                if event.modifiers() & Qt.ControlModifier:
                    self._apply_roll_drag_delta(dx)
                else:
                    self._apply_drag_delta(dx, dy)
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.last_drag_pos = None
                self.ui.graphicsViewValidation.viewport().unsetCursor()
                self.render_now()
                return True
            if event.type() == QEvent.Wheel:
                if not self._wheel_zoom_enabled():
                    return False
                self._apply_fov_wheel(event.angleDelta().y())
                return True
        return super().eventFilter(watched, event)

    def schedule_render(self, delay_ms: int = 30) -> None:
        self.render_timer.start(delay_ms)

    def _set_default_grid_precision(self) -> None:
        pixel_count = max(1, self.source_image.width() * self.source_image.height())
        capped_pixels = min(pixel_count, VALIDATION_DEFAULT_GRID_CAP_PIXELS)
        default_precision = int(round(36.0 + 7.0 * math.sqrt(capped_pixels / 1_000_000.0)))
        default_precision = max(
            VALIDATION_GRID_MIN_PRECISION,
            min(VALIDATION_GRID_MAX_PRECISION, default_precision),
        )
        was_blocked = self.ui.spinBoxGridPrecision.blockSignals(True)
        self.ui.spinBoxGridPrecision.setValue(default_precision)
        self.ui.spinBoxGridPrecision.blockSignals(was_blocked)
        self._update_grid_precision_tooltip()

    def _grid_precision_value(self) -> int:
        return max(
            VALIDATION_GRID_MIN_PRECISION,
            min(VALIDATION_GRID_MAX_PRECISION, int(self.ui.spinBoxGridPrecision.value())),
        )

    def _update_grid_precision_tooltip(self) -> None:
        precision = self._grid_precision_value()
        width = max(1, self.source_image.width())
        height = max(1, self.source_image.height())
        if width >= height:
            columns = precision
            rows = max(2, int(round(precision * height / float(width))))
        else:
            rows = precision
            columns = max(2, int(round(precision * width / float(height))))
        text = f"下一次重新求解将使用 {columns} x {rows} 个精确网格点。"
        self.ui.spinBoxGridPrecision.setToolTip(text)
        self.ui.pushButtonResolveGrid.setToolTip(text)

    def _set_grid_controls_enabled(self, enabled: bool) -> None:
        self.ui.spinBoxGridPrecision.setEnabled(enabled)
        self.ui.pushButtonResolveGrid.setEnabled(enabled)

    def _render_size(self) -> tuple[int, int]:
        viewport_size = self.ui.graphicsViewValidation.viewport().size()
        width = max(VALIDATION_MIN_RENDER_SIZE_PX, int(viewport_size.width()))
        height = max(VALIDATION_MIN_RENDER_SIZE_PX, int(viewport_size.height()))
        return width, height

    def _camera_for_render(self, width: int, height: int) -> CameraSettings:
        return replace(
            self.base_camera,
            image_width_px=width,
            image_height_px=height,
            focal_length_mm=self.focal_length_mm,
            fisheye_fov_deg=self.fisheye_fov_deg,
        )

    def _view_for_render(self) -> ViewSettings:
        return ViewSettings(
            center_az_deg=self.center_az_deg,
            center_alt_deg=self.center_alt_deg,
            roll_deg=self.fixed_roll_deg,
        )

    def _apply_drag_delta(self, dx: int, dy: int) -> None:
        width, height = self._render_size()
        camera = self._camera_for_render(width, height)
        az_degrees_per_pixel = max(horizontal_fov_deg(camera) / max(width, 1), 0.005)
        alt_degrees_per_pixel = max(vertical_fov_deg(camera) / max(height, 1), 0.005)
        self.center_az_deg = (self.center_az_deg - dx * az_degrees_per_pixel) % 360.0
        self.center_alt_deg = max(-90.0, min(90.0, self.center_alt_deg + dy * alt_degrees_per_pixel))
        self.schedule_render(delay_ms=10)

    def _apply_roll_drag_delta(self, dx: int) -> None:
        self.fixed_roll_deg -= dx * 0.25
        while self.fixed_roll_deg > 180.0:
            self.fixed_roll_deg -= 360.0
        while self.fixed_roll_deg < -180.0:
            self.fixed_roll_deg += 360.0
        self.schedule_render(delay_ms=10)

    def _apply_fov_wheel(self, wheel_delta: int) -> None:
        if wheel_delta == 0:
            return
        zoom_factor = VALIDATION_ZOOM_FACTOR if wheel_delta > 0 else 1.0 / VALIDATION_ZOOM_FACTOR
        self._apply_fov_zoom_factor(zoom_factor)

    def _apply_fov_zoom_factor(self, zoom_factor: float) -> None:
        if not np.isfinite(zoom_factor) or zoom_factor <= 0.0 or abs(zoom_factor - 1.0) <= 1e-4:
            return
        if self.base_camera.lens_model in FISHEYE_LENS_MODELS:
            self.fisheye_fov_deg = max(1.0, min(300.0, self.fisheye_fov_deg / zoom_factor))
        else:
            self.focal_length_mm = max(0.2, min(5000.0, self.focal_length_mm * zoom_factor))
        self.render_now()

    def _wheel_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "wheel_zoom_enabled", True))

    def _touchpad_pinch_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "touchpad_pinch_zoom_enabled", True))

    def _native_gesture_zoom_value(self, event) -> float:  # type: ignore[no-untyped-def]
        if not self._touchpad_pinch_zoom_enabled():
            return 0.0
        zoom_gesture = getattr(Qt, "ZoomNativeGesture", None)
        if zoom_gesture is None or event.gestureType() != zoom_gesture:
            return 0.0
        try:
            value = float(event.value())
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value) or abs(value) <= 1e-6:
            return 0.0
        return value

    def _handle_native_fov_zoom(self, event) -> bool:  # type: ignore[no-untyped-def]
        value = self._native_gesture_zoom_value(event)
        if value == 0.0:
            return False
        zoom_factor = math.exp(value * TOUCHPAD_ZOOM_SENSITIVITY)
        zoom_factor = max(TOUCHPAD_ZOOM_MIN_FACTOR, min(TOUCHPAD_ZOOM_MAX_FACTOR, zoom_factor))
        self._apply_fov_zoom_factor(zoom_factor)
        return True

    def _update_overlay_opacity(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if not self.ui.checkBoxOverlayEnabled.isChecked():
            opacity = 0.0
        else:
            opacity = max(0.0, min(1.0, self.ui.doubleSpinBoxImageOpacity.value() / 100.0))
        self.image_overlay_item.setOpacity(opacity)
        self.ui.doubleSpinBoxImageOpacity.setToolTip(f"真实图像叠加透明度：{opacity * 100.0:.1f}%")

    def _handle_overlay_toggled(self, checked: bool) -> None:
        self.image_overlay_item.setVisible(bool(checked))
        self.ui.labelImageOpacity.setEnabled(bool(checked))
        self.ui.doubleSpinBoxImageOpacity.setEnabled(bool(checked))
        self._update_overlay_opacity()
        if checked and self.cache is not None:
            self.render_now()

    def resolve_grid_again(self) -> None:
        self._update_grid_precision_tooltip()
        if self.cache_thread is not None:
            QMessageBox.information(self, "正在求解网格", "当前映射验证网格仍在计算，请等待完成后再重新求解。")
            return
        self._start_cache_worker()

    def render_now(self) -> None:
        width, height = self._render_size()
        camera = self._camera_for_render(width, height)
        view = self._view_for_render()
        star_map = project_horizontal_catalog(
            horizontal_catalog=self.horizontal_catalog,
            camera=camera,
            view=view,
            visible_mag_limit=self.visible_mag_limit,
            horizontal_milky_way=self.horizontal_milky_way,
            horizontal_solar_system=self.horizontal_solar_system,
        )
        star_map = replace(
            star_map,
            alpha=np.full_like(star_map.alpha, 255, dtype=np.uint8),
            grid_lines=(),
            direction_labels=(),
            horizon_shadow_rects=(),
            solar_system_objects=tuple(replace(item, alpha=255) for item in star_map.solar_system_objects),
        )
        sky_image = self.renderer.render(
            star_map,
            reference_stars=(),
            element_scale=1.0,
            draw_common_names=False,
            number_reference_stars=False,
            draw_background=True,
            draw_horizon_shadow=False,
            draw_grid=False,
            draw_solar_system_labels=True,
            draw_direction_labels=False,
        )
        self.sky_item.set_image(sky_image)
        if self.ui.checkBoxOverlayEnabled.isChecked():
            self.image_overlay_item.set_image(self._render_overlay_image(camera, view))
        self.scene.setSceneRect(0.0, 0.0, float(width), float(height))
        self.ui.graphicsViewValidation.resetTransform()

    def _render_overlay_image(self, camera: CameraSettings, view: ViewSettings) -> QImage:
        width = int(camera.image_width_px)
        height = int(camera.image_height_px)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        cache = self.cache
        if cache is None or cache.alt_deg.size == 0 or cv2 is None:
            return _rgba_array_to_qimage(rgba)

        basis = camera_basis_from_view(view)
        x_px, y_px, valid_projection = _project_altaz_points(
            cache.alt_deg.ravel(),
            cache.az_deg.ravel(),
            camera=camera,
            basis=basis,
        )
        screen_x = x_px.reshape((cache.grid_rows, cache.grid_columns))
        screen_y = y_px.reshape((cache.grid_rows, cache.grid_columns))
        valid_screen = (
            valid_projection.reshape((cache.grid_rows, cache.grid_columns))
            & cache.valid
            & np.isfinite(screen_x)
            & np.isfinite(screen_y)
        )
        self._warp_grid_cells_to_overlay(rgba, cache, screen_x, screen_y, valid_screen)
        return _rgba_array_to_qimage(rgba)

    def _warp_grid_cells_to_overlay(
        self,
        rgba: np.ndarray,
        cache: ImageSkyProjectionCache,
        screen_x: np.ndarray,
        screen_y: np.ndarray,
        valid_screen: np.ndarray,
    ) -> None:
        height, width, _channels = rgba.shape
        source_rgb = cache.source_rgb
        source_height, source_width, _source_channels = source_rgb.shape
        max_cell_bbox_area = max(64.0, width * height * 0.45)

        for row in range(cache.grid_rows - 1):
            for column in range(cache.grid_columns - 1):
                if not bool(
                    valid_screen[row, column]
                    and valid_screen[row, column + 1]
                    and valid_screen[row + 1, column + 1]
                    and valid_screen[row + 1, column]
                ):
                    continue

                dst_quad = np.asarray(
                    [
                        [screen_x[row, column], screen_y[row, column]],
                        [screen_x[row, column + 1], screen_y[row, column + 1]],
                        [screen_x[row + 1, column + 1], screen_y[row + 1, column + 1]],
                        [screen_x[row + 1, column], screen_y[row + 1, column]],
                    ],
                    dtype=np.float32,
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
                            cache.grid_x_px[row, column] * cache.source_scale_x,
                            cache.grid_y_px[row, column] * cache.source_scale_y,
                        ],
                        [
                            cache.grid_x_px[row, column + 1] * cache.source_scale_x,
                            cache.grid_y_px[row, column + 1] * cache.source_scale_y,
                        ],
                        [
                            cache.grid_x_px[row + 1, column + 1] * cache.source_scale_x,
                            cache.grid_y_px[row + 1, column + 1] * cache.source_scale_y,
                        ],
                        [
                            cache.grid_x_px[row + 1, column] * cache.source_scale_x,
                            cache.grid_y_px[row + 1, column] * cache.source_scale_y,
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
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                mask = np.zeros((bbox_height, bbox_width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.rint(dst_relative).astype(np.int32), 255, lineType=cv2.LINE_AA)
                target = rgba[bbox_top : bbox_bottom + 1, bbox_left : bbox_right + 1]
                visible = mask > 0
                target[visible, :3] = warped[visible]
                target[visible, 3] = mask[visible]

    def _start_cache_worker(self) -> None:
        self._set_grid_controls_enabled(False)
        self.cache_progress = QProgressDialog(self)
        self.cache_progress.setWindowTitle("正在准备映射验证")
        self.cache_progress.setLabelText("正在计算真实图像网格到天球方向的验证缓存...")
        self.cache_progress.setRange(0, 100)
        self.cache_progress.setValue(0)
        self.cache_progress.setCancelButton(None)
        self.cache_progress.setWindowModality(Qt.WindowModal)
        self.cache_progress.setMinimumDuration(0)
        self.cache_progress.setAutoClose(False)
        self.cache_progress.setAutoReset(False)
        self.cache_progress.show()

        thread = QThread(self)
        worker = PixelToSkyValidationWorker(
            self.model,
            self.source_image,
            grid_precision=self._grid_precision_value(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_cache_progress)
        worker.finished.connect(self._handle_cache_finished)
        worker.failed.connect(self._handle_cache_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_cache_worker)
        self.cache_thread = thread
        self.cache_worker = worker
        thread.start()

    def _handle_cache_progress(self, value: int, label_text: str) -> None:
        if self.cache_progress is None:
            return
        self.cache_progress.setLabelText(label_text)
        self.cache_progress.setValue(max(0, min(100, int(value))))

    def _handle_cache_finished(self, cache: object) -> None:
        self.cache = cache  # type: ignore[assignment]
        if self.cache_progress is not None:
            self.cache_progress.setValue(100)
            self.cache_progress.close()
        if isinstance(cache, ImageSkyProjectionCache):
            self.ui.spinBoxGridPrecision.setToolTip(
                f"当前缓存：{cache.grid_columns} x {cache.grid_rows} 个精确网格点。"
            )
        self.render_now()

    def _handle_cache_failed(self, error_message: str) -> None:
        if self.cache_progress is not None:
            self.cache_progress.close()
        QMessageBox.critical(self, "映射验证失败", error_message)

    def _cleanup_cache_worker(self) -> None:
        self.cache_thread = None
        self.cache_worker = None
        self.cache_progress = None
        self._set_grid_controls_enabled(True)
