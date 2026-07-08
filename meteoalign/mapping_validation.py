from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PyQt5.QtCore import QEvent, QObject, QPoint, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QDialog, QGraphicsScene, QMessageBox, QProgressDialog

from .app_graphics_items import GraphicsImageItem
from .config import StarMapUiConfig
from .fixed_camera_model import FixedCameraModel
from .frame_astrometry import FrameAstrometricModel
from .projected_texture_renderer import ProjectedTextureRenderer
from .projection_grid import build_pixel_grid, grid_shape_for_long_side
from .projection_view_state import ProjectionViewState
from .qt_tasks import WorkerTaskHandle, create_progress_dialog, start_qt_worker_task
from .renderer import StarMapRenderer
from .sky_scene_service import SkyPreviewRenderService, SkyPreviewStyle, SkySceneData
from .simulator import (
    CameraSettings,
    FISHEYE_LENS_MODELS,
    HorizontalMilkyWayCatalog,
    HorizontalSolarSystemCatalog,
    HorizontalStarCatalog,
    ObserverSettings,
    ViewSettings,
    compute_altaz_from_radec,
    horizontal_fov_deg,
    vertical_fov_deg,
)
from .texture_projection import qimage_to_rgb_array
from .ui.ui_mapping_validation_dialog import Ui_MappingValidationDialog
from .view_gestures import (
    ViewZoomPolicy,
    roll_after_drag,
    sky_center_after_drag,
    native_gesture_zoom_factor,
    wheel_zoom_factor,
)


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


class PixelToSkyValidationWorker(QObject):
    """后台计算真实图像网格点到当前 Scene Observer 地平坐标的验证缓存。"""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        model: FrameAstrometricModel | FixedCameraModel,
        source_image: QImage,
        observer: ObserverSettings,
        grid_precision: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.source_image = source_image.copy()
        self.observer = observer
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
        return grid_shape_for_long_side(width, height, self.grid_precision, min_minor_cells=2)

    def _source_preview_rgb(self, width: int, height: int) -> tuple[np.ndarray, float, float]:
        scale = min(1.0, VALIDATION_SOURCE_PREVIEW_LONG_SIDE_PX / max(float(width), float(height), 1.0))
        if scale < 1.0:
            preview_width = max(1, int(round(width * scale)))
            preview_height = max(1, int(round(height * scale)))
            preview = self.source_image.scaled(preview_width, preview_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            preview = self.source_image
        rgb = qimage_to_rgb_array(preview)
        return rgb, rgb.shape[1] / max(float(width), 1.0), rgb.shape[0] / max(float(height), 1.0)

    def _build_cache(self) -> ImageSkyProjectionCache:
        width = self.source_image.width()
        height = self.source_image.height()
        rows, columns = self._grid_shape(width, height)
        pixel_grid = build_pixel_grid(width, height, rows, columns)
        grid_points = pixel_grid.points
        point_count = pixel_grid.point_count

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
            alt_values, az_values, valid_values = self._pixel_to_altaz_points(grid_points[start:end])
            alt_deg[start:end] = alt_values
            az_deg[start:end] = az_values
            valid[start:end] = valid_values
            progress = 2 + int(62 * end / max(point_count, 1))
            self.progress.emit(progress, f"正在按源图模型反解网格点 pixel_to_sky：{end} / {point_count}。")

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
            grid_x_px=pixel_grid.x_px,
            grid_y_px=pixel_grid.y_px,
            alt_deg=alt_deg.reshape((rows, columns)).astype(np.float64),
            az_deg=az_deg.reshape((rows, columns)).astype(np.float64),
            valid=valid,
            source_rgb=source_rgb.astype(np.uint8),
        )

    def _pixel_to_altaz_points(self, pixel_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(self.model, FrameAstrometricModel):
            radec = self.model.pixel_to_sky_points(pixel_points)
            valid_input = np.all(np.isfinite(radec), axis=1)
            alt_deg = np.full(radec.shape[0], np.nan, dtype=np.float64)
            az_deg = np.full(radec.shape[0], np.nan, dtype=np.float64)
            if np.any(valid_input):
                alt_values, az_values = compute_altaz_from_radec(
                    radec[valid_input, 0],
                    radec[valid_input, 1],
                    self.observer,
                )
                alt_deg[valid_input] = alt_values
                az_deg[valid_input] = az_values
            valid = valid_input & np.isfinite(alt_deg) & np.isfinite(az_deg)
            return alt_deg.astype(np.float64), az_deg.astype(np.float64), valid.astype(bool)
        return self.model.pixel_to_altaz_points(pixel_points)


class MappingValidationDialog(QDialog):
    """源图映射验证窗口。"""

    def __init__(
        self,
        *,
        parent,
        renderer: StarMapRenderer,
        model: FrameAstrometricModel | FixedCameraModel,
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
        self.sky_preview_renderer = SkyPreviewRenderService(renderer)
        self.texture_renderer = ProjectedTextureRenderer()
        self.center_az_deg = float(initial_view.center_az_deg) % 360.0
        self.center_alt_deg = max(-90.0, min(90.0, float(initial_view.center_alt_deg)))
        self.fixed_roll_deg = float(initial_view.roll_deg)
        self.focal_length_mm = max(float(base_camera.focal_length_mm), 1e-6)
        self.fisheye_fov_deg = max(1.0, min(300.0, float(base_camera.fisheye_fov_deg)))
        self.cache: ImageSkyProjectionCache | None = None
        self.last_drag_pos: QPoint | None = None
        self.cache_task: WorkerTaskHandle | None = None
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
        if self.cache_task is not None:
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
        rows, columns = grid_shape_for_long_side(width, height, precision, min_minor_cells=2)
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

    def _view_for_render(self, camera: CameraSettings) -> ViewSettings:
        state = ProjectionViewState.from_camera_and_view(
            camera,
            ViewSettings(
                center_az_deg=self.center_az_deg,
                center_alt_deg=self.center_alt_deg,
                roll_deg=self.fixed_roll_deg,
            ),
        )
        return state.to_view_settings()

    def _apply_drag_delta(self, dx: int, dy: int) -> None:
        width, height = self._render_size()
        camera = self._camera_for_render(width, height)
        self.center_az_deg, self.center_alt_deg = sky_center_after_drag(
            center_az_deg=self.center_az_deg,
            center_alt_deg=self.center_alt_deg,
            dx_px=dx,
            dy_px=dy,
            horizontal_fov_deg=horizontal_fov_deg(camera),
            vertical_fov_deg=vertical_fov_deg(camera),
            viewport_width_px=width,
            viewport_height_px=height,
            min_degrees_per_pixel=0.005,
        )
        self.schedule_render(delay_ms=10)

    def _apply_roll_drag_delta(self, dx: int) -> None:
        self.fixed_roll_deg = roll_after_drag(self.fixed_roll_deg, dx, drag_sign=-1.0)
        self.schedule_render(delay_ms=10)

    def _apply_fov_wheel(self, wheel_delta: int) -> None:
        zoom_factor = wheel_zoom_factor(wheel_delta, VALIDATION_ZOOM_FACTOR)
        if zoom_factor is None:
            return
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

    def _zoom_policy(self) -> ViewZoomPolicy:
        return ViewZoomPolicy(
            wheel_enabled=self._wheel_zoom_enabled(),
            pinch_enabled=self._touchpad_pinch_zoom_enabled(),
            min_fov=1.0,
            max_fov=300.0,
        )

    def _handle_native_fov_zoom(self, event) -> bool:  # type: ignore[no-untyped-def]
        zoom_factor = native_gesture_zoom_factor(event, self._zoom_policy())
        if zoom_factor is None:
            return False
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
        if self.cache_task is not None:
            QMessageBox.information(self, "正在求解网格", "当前映射验证网格仍在计算，请等待完成后再重新求解。")
            return
        self._start_cache_worker()

    def render_now(self) -> None:
        width, height = self._render_size()
        camera = self._camera_for_render(width, height)
        view = self._view_for_render(camera)
        sky_image = self.sky_preview_renderer.render(
            scene=SkySceneData(
                horizontal_catalog=self.horizontal_catalog,
                horizontal_milky_way=self.horizontal_milky_way,
                horizontal_solar_system=self.horizontal_solar_system,
            ),
            camera=camera,
            view=view,
            visible_mag_limit=self.visible_mag_limit,
            style=SkyPreviewStyle(
                draw_background=True,
                draw_horizon_shadow=False,
                draw_grid=False,
                draw_solar_system_labels=True,
                draw_direction_labels=False,
                force_opaque=True,
                clear_grid_lines=True,
                clear_direction_labels=True,
                clear_horizon_shadow=True,
            ),
        )
        self.sky_item.set_image(sky_image)
        if self.ui.checkBoxOverlayEnabled.isChecked():
            self.image_overlay_item.set_image(self._render_overlay_image(camera, view))
        self.scene.setSceneRect(0.0, 0.0, float(width), float(height))
        self.ui.graphicsViewValidation.resetTransform()

    def _render_overlay_image(self, camera: CameraSettings, view: ViewSettings) -> QImage:
        width = int(camera.image_width_px)
        height = int(camera.image_height_px)
        cache = self.cache
        if cache is None or cache.alt_deg.size == 0:
            return self.texture_renderer.render_qimage(
                width=width,
                height=height,
                camera=camera,
                view=view,
                source_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                source_grid_x_px=np.zeros((2, 2), dtype=np.float64),
                source_grid_y_px=np.zeros((2, 2), dtype=np.float64),
                source_scale_x=1.0,
                source_scale_y=1.0,
                alt_deg=np.full((2, 2), np.nan, dtype=np.float64),
                az_deg=np.full((2, 2), np.nan, dtype=np.float64),
                valid_points=np.zeros((2, 2), dtype=bool),
            )
        return self.texture_renderer.render_qimage(
            width=width,
            height=height,
            camera=camera,
            view=view,
            source_rgb=cache.source_rgb,
            source_grid_x_px=cache.grid_x_px,
            source_grid_y_px=cache.grid_y_px,
            source_scale_x=cache.source_scale_x,
            source_scale_y=cache.source_scale_y,
            alt_deg=cache.alt_deg,
            az_deg=cache.az_deg,
            valid_points=cache.valid,
        )

    def _start_cache_worker(self) -> None:
        self._set_grid_controls_enabled(False)
        self.cache_progress = create_progress_dialog(
            self,
            title="正在准备映射验证",
            label_text="正在计算真实图像网格到天球方向的验证缓存...",
        )
        worker = PixelToSkyValidationWorker(
            self.model,
            self.source_image,
            self.observer,
            grid_precision=self._grid_precision_value(),
        )
        self.cache_worker = worker
        self.cache_task = start_qt_worker_task(
            parent=self,
            worker=worker,
            progress_signal=worker.progress,
            finished_signal=worker.finished,
            failed_signal=worker.failed,
            on_progress=self._handle_cache_progress,
            on_finished=self._handle_cache_finished,
            on_failed=self._handle_cache_failed,
            on_cleanup=self._cleanup_cache_worker,
            progress_dialog=self.cache_progress,
        )

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
        self.cache_task = None
        self.cache_worker = None
        self.cache_progress = None
        self._set_grid_controls_enabled(True)
