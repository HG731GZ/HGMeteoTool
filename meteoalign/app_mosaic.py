from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, QEvent, QTimer, Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QMessageBox

from .alignment.constants import (
    SKY_KNOWN_PROJECTION_DISPLAY_NAMES,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .app_constants import SOURCE_MODEL_JSON_FILTER
from .app_graphics_items import GraphicsImageItem
from .catalog import project_root
from .mosaic_common import (
    MOSAIC_COVERAGE_GRID_LONG_SIDE,
    MOSAIC_GRID_MAX_PRECISION,
    MOSAIC_GRID_MIN_PRECISION,
    MOSAIC_OVERLAY_MODE_COVERAGE,
    MOSAIC_OVERLAY_MODE_SOURCE_IMAGE,
    MOSAIC_OVERLAY_MODES,
    MOSAIC_PROJECTION_MODELS,
    MOSAIC_RENDER_MIN_SIZE_PX,
    MOSAIC_ZOOM_FACTOR,
)
from .mosaic_grid_service import (
    build_coverage_cache,
    compute_center_from_model,
    suggest_fov_from_coverage,
)
from .mosaic_model_io import (
    MosaicCoverageCache,
    MosaicSourceModel,
    MosaicSourceTextureCache,
    _load_mosaic_source_model,
)
from .mosaic_overlay_renderer import (
    load_source_texture,
    paint_coverage_overlay,
    paint_source_image_overlay,
)
from .projected_texture_renderer import ProjectedTextureRenderer
from .projection_grid import grid_shape_for_long_side
from .projection_interaction_controller import ProjectionInteractionController
from .projection_view_state import ProjectionViewState
from .simulator import (
    CameraSettings,
    ObserverSettings,
    ViewSettings,
    horizontal_fov_deg,
    vertical_fov_deg,
)
from .sky_scene_service import SkyPreviewRenderService, SkyPreviewStyle, SkySceneData
from .view_gestures import (
    ViewZoomPolicy,
    clamp_fov,
    roll_after_drag,
    sky_center_after_drag,
)


class MosaicProjectionMixin:
    """自由投影拼图预览 Mixin。"""

    ui: object
    renderer: object
    catalog: object
    milky_way_catalog: object
    ui_config: object

    def _init_mosaic_projection_page(self) -> None:
        if not hasattr(self.ui, "mosaicProjectionView"):
            return
        self.mosaic_scene = QGraphicsScene(self)
        self.mosaic_image_item = GraphicsImageItem()
        self.mosaic_scene.addItem(self.mosaic_image_item)
        self.ui.mosaicProjectionView.setScene(self.mosaic_scene)

        # 使用统一的投影视图交互控制器处理拖拽、缩放、触控板手势
        self._mosaic_interaction_controller = ProjectionInteractionController(
            view=self.ui.mosaicProjectionView,
            on_pan=self._handle_mosaic_pan_delta,
            on_roll=self._handle_mosaic_roll_delta,
            on_zoom=self._handle_mosaic_zoom_factor,
            on_resize=self._handle_mosaic_resize_event,
            on_interaction_end=self._handle_mosaic_interaction_end_event,
            zoom_step_factor=MOSAIC_ZOOM_FACTOR,
            zoom_policy=self._mosaic_zoom_policy(),
            parent=self,
        )

        # 保留 MainWindow 的事件过滤器用于标签省略号显示
        self.ui.mosaicProjectionView.installEventFilter(self)
        self.ui.mosaicProjectionView.viewport().installEventFilter(self)

        self.mosaic_render_timer = QTimer(self)
        self.mosaic_render_timer.setSingleShot(True)
        self.mosaic_render_timer.timeout.connect(self.render_mosaic_projection_now)
        self._mosaic_sky_preview_renderer = SkyPreviewRenderService(self.renderer)
        self._mosaic_texture_renderer = ProjectedTextureRenderer()
        self._mosaic_source_model: MosaicSourceModel | None = None
        self._mosaic_coverage_cache: MosaicCoverageCache | None = None
        self._mosaic_source_texture_cache: MosaicSourceTextureCache | None = None
        self._mosaic_center_az_deg = 0.0
        self._mosaic_center_alt_deg = 20.0
        self._mosaic_roll_deg = 0.0
        self._init_mosaic_projection_defaults()

    def _init_mosaic_projection_defaults(self) -> None:
        if not hasattr(self.ui, "comboBoxMosaicProjection"):
            return
        self.ui.comboBoxMosaicProjection.setCurrentIndex(0)
        self.ui.doubleSpinBoxMosaicFov.setValue(120.0)
        if hasattr(self.ui, "doubleSpinBoxMosaicAz"):
            self.ui.doubleSpinBoxMosaicAz.setValue(self._mosaic_center_az_deg)
        if hasattr(self.ui, "doubleSpinBoxMosaicAlt"):
            self.ui.doubleSpinBoxMosaicAlt.setValue(self._mosaic_center_alt_deg)
        if hasattr(self.ui, "doubleSpinBoxMosaicRoll"):
            self.ui.doubleSpinBoxMosaicRoll.setValue(self._mosaic_roll_deg)
        self.ui.doubleSpinBoxMosaicMagLimit.setValue(6.5)
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            self.ui.comboBoxMosaicOverlayMode.setCurrentIndex(0)
        if hasattr(self.ui, "checkBoxMosaicSkyOnly"):
            self.ui.checkBoxMosaicSkyOnly.setChecked(False)
        if hasattr(self.ui, "doubleSpinBoxMosaicOverlayOpacity"):
            self.ui.doubleSpinBoxMosaicOverlayOpacity.setValue(100.0)
        elif hasattr(self.ui, "doubleSpinBoxMosaicCoverageOpacity"):
            self.ui.doubleSpinBoxMosaicCoverageOpacity.setValue(100.0)
        if hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            self.ui.spinBoxMosaicGridPrecision.setValue(MOSAIC_COVERAGE_GRID_LONG_SIDE)
        self._init_mosaic_observer_controls()
        self._update_mosaic_projection_controls()
        self._update_mosaic_grid_precision_tooltip()
        self._set_mosaic_grid_controls_enabled(False)
        self._update_mosaic_model_labels()
        self._update_mosaic_view_label()

    def _init_mosaic_observer_controls(self) -> None:
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return
        self.ui.dateTimeEditMosaicObservation.setDateTime(QDateTime.currentDateTime())
        utc_offset = datetime.now().astimezone().utcoffset()
        if utc_offset is None:
            utc_offset = timedelta(hours=8)
        self.ui.doubleSpinBoxMosaicUtcOffset.setValue(utc_offset.total_seconds() / 3600.0)
        self.ui.doubleSpinBoxMosaicLatitude.setValue(0.0)
        self.ui.doubleSpinBoxMosaicLongitude.setValue(0.0)
        self.ui.doubleSpinBoxMosaicElevation.setValue(0.0)
        self._set_mosaic_observer_controls_enabled(False)

    def _connect_mosaic_projection_inputs(self) -> None:
        if not hasattr(self.ui, "pushButtonImportMosaicModel"):
            return
        self.ui.pushButtonImportMosaicModel.clicked.connect(self.import_mosaic_model_json)
        self.ui.comboBoxMosaicProjection.currentIndexChanged.connect(self._handle_mosaic_projection_changed)
        self.ui.doubleSpinBoxMosaicFov.valueChanged.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "doubleSpinBoxMosaicAz"):
            self.ui.doubleSpinBoxMosaicAz.valueChanged.connect(self._handle_mosaic_view_controls_changed)
        if hasattr(self.ui, "doubleSpinBoxMosaicAlt"):
            self.ui.doubleSpinBoxMosaicAlt.valueChanged.connect(self._handle_mosaic_view_controls_changed)
        if hasattr(self.ui, "doubleSpinBoxMosaicRoll"):
            self.ui.doubleSpinBoxMosaicRoll.valueChanged.connect(self._handle_mosaic_view_controls_changed)
        self.ui.doubleSpinBoxMosaicMagLimit.valueChanged.connect(self.schedule_mosaic_render)
        self.ui.checkBoxMosaicShowGrid.toggled.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "checkBoxMosaicShowCoverage"):
            self.ui.checkBoxMosaicShowCoverage.toggled.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            self.ui.comboBoxMosaicOverlayMode.currentIndexChanged.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "checkBoxMosaicSkyOnly"):
            self.ui.checkBoxMosaicSkyOnly.toggled.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "doubleSpinBoxMosaicOverlayOpacity"):
            self.ui.doubleSpinBoxMosaicOverlayOpacity.valueChanged.connect(self.schedule_mosaic_render)
        elif hasattr(self.ui, "doubleSpinBoxMosaicCoverageOpacity"):
            self.ui.doubleSpinBoxMosaicCoverageOpacity.valueChanged.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            self.ui.spinBoxMosaicGridPrecision.valueChanged.connect(self._update_mosaic_grid_precision_tooltip)
        if hasattr(self.ui, "pushButtonResolveMosaicGrid"):
            self.ui.pushButtonResolveMosaicGrid.clicked.connect(self.resolve_mosaic_grid_again)
        self.ui.pushButtonResetMosaicView.clicked.connect(self.reset_mosaic_projection_view)
        if hasattr(self.ui, "dateTimeEditMosaicObservation"):
            self.ui.dateTimeEditMosaicObservation.dateTimeChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicUtcOffset.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicLatitude.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicLongitude.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicElevation.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.pushButtonResetMosaicObserver.clicked.connect(self.reset_mosaic_observer_from_json)
        self.ui.labelMosaicModelPath.installEventFilter(self)
        self.ui.labelMosaicSourceImage.installEventFilter(self)
        self.ui.labelMosaicModelInfo.installEventFilter(self)
        self.ui.labelMosaicViewInfo.installEventFilter(self)

    def _handle_mosaic_projection_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_mosaic_projection_controls()
        self.schedule_mosaic_render()

    def _handle_mosaic_view_controls_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self.ui, "doubleSpinBoxMosaicAz"):
            self._mosaic_center_az_deg = float(self.ui.doubleSpinBoxMosaicAz.value()) % 360.0
        if hasattr(self.ui, "doubleSpinBoxMosaicAlt"):
            self._mosaic_center_alt_deg = max(-90.0, min(90.0, float(self.ui.doubleSpinBoxMosaicAlt.value())))
        if hasattr(self.ui, "doubleSpinBoxMosaicRoll"):
            self._mosaic_roll_deg = float(self.ui.doubleSpinBoxMosaicRoll.value())
            while self._mosaic_roll_deg > 180.0:
                self._mosaic_roll_deg -= 360.0
            while self._mosaic_roll_deg < -180.0:
                self._mosaic_roll_deg += 360.0
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _handle_mosaic_observer_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self.schedule_mosaic_render(delay_ms=120)

    def _update_mosaic_projection_controls(self) -> None:
        if not hasattr(self.ui, "doubleSpinBoxMosaicFov"):
            return
        projection_model = self._mosaic_projection_model()
        if projection_model == SKY_MATCHING_MODEL_RECTILINEAR:
            maximum_fov = 160.0
        elif projection_model in (SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID):
            maximum_fov = 300.0
        else:
            maximum_fov = 360.0
        was_blocked = self.ui.doubleSpinBoxMosaicFov.blockSignals(True)
        self.ui.doubleSpinBoxMosaicFov.setMaximum(maximum_fov)
        if self.ui.doubleSpinBoxMosaicFov.value() > maximum_fov:
            self.ui.doubleSpinBoxMosaicFov.setValue(maximum_fov)
        self.ui.doubleSpinBoxMosaicFov.blockSignals(was_blocked)

    def _mosaic_projection_model(self) -> str:
        if not hasattr(self.ui, "comboBoxMosaicProjection"):
            return SKY_MATCHING_MODEL_RECTILINEAR
        index = self.ui.comboBoxMosaicProjection.currentIndex()
        if index < 0 or index >= len(MOSAIC_PROJECTION_MODELS):
            return SKY_MATCHING_MODEL_RECTILINEAR
        return MOSAIC_PROJECTION_MODELS[index]

    def import_mosaic_model_json(self) -> None:
        default_dir = project_root() / "outputs"
        if self._mosaic_source_model is not None:
            default_dir = self._mosaic_source_model.json_path.parent
        elif not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入源图模型 JSON",
            str(default_dir),
            SOURCE_MODEL_JSON_FILTER,
        )
        if not file_path:
            return
        self.load_mosaic_model_json(file_path)

    def load_mosaic_model_json(self, file_path: str | Path, *, quiet: bool = False) -> bool:
        json_path = Path(file_path).expanduser()
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            source_model = _load_mosaic_source_model(json_path)
            coverage_cache = self._build_mosaic_coverage_cache(source_model.model)
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON 和模型错误直接反馈到界面。
            if not quiet:
                QMessageBox.critical(self, "导入源图模型失败", str(exc))
            self.ui.statusbar.showMessage(f"导入源图模型失败: {exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

        self._mosaic_source_model = source_model
        self._mosaic_coverage_cache = coverage_cache
        self._mosaic_source_texture_cache = None
        self._set_mosaic_projection_from_source_model(source_model)
        self._set_mosaic_overlay_defaults_for_model(source_model)
        self._set_mosaic_observer_controls_from_source_model(source_model)
        self._set_mosaic_observer_controls_enabled(True)
        self._set_mosaic_grid_controls_enabled(True)
        self._update_mosaic_grid_precision_tooltip()
        self._reset_mosaic_center_from_model()
        self._update_mosaic_model_labels()
        self.schedule_mosaic_render(delay_ms=0)
        if not quiet:
            self.ui.statusbar.showMessage(f"已导入源图模型: {source_model.json_path}")
        return True

    def _set_mosaic_projection_from_source_model(self, source_model: MosaicSourceModel) -> bool:
        if not hasattr(self.ui, "comboBoxMosaicProjection"):
            return False
        projection_model = str(source_model.model.camera_calibration_profile.base_projection_type)
        if projection_model not in MOSAIC_PROJECTION_MODELS:
            return False
        index = MOSAIC_PROJECTION_MODELS.index(projection_model)
        was_blocked = self.ui.comboBoxMosaicProjection.blockSignals(True)
        self.ui.comboBoxMosaicProjection.setCurrentIndex(index)
        self.ui.comboBoxMosaicProjection.blockSignals(was_blocked)
        self._update_mosaic_projection_controls()
        return True

    def _set_mosaic_overlay_defaults_for_model(self, source_model: MosaicSourceModel) -> None:
        if hasattr(self.ui, "checkBoxMosaicSkyOnly"):
            was_blocked = self.ui.checkBoxMosaicSkyOnly.blockSignals(True)
            self.ui.checkBoxMosaicSkyOnly.setChecked(False)
            self.ui.checkBoxMosaicSkyOnly.blockSignals(was_blocked)
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            overlay_mode = (
                MOSAIC_OVERLAY_MODE_SOURCE_IMAGE
                if source_model.source_image_path is not None
                else MOSAIC_OVERLAY_MODE_COVERAGE
            )
            index = MOSAIC_OVERLAY_MODES.index(overlay_mode)
            was_blocked = self.ui.comboBoxMosaicOverlayMode.blockSignals(True)
            self.ui.comboBoxMosaicOverlayMode.setCurrentIndex(index)
            self.ui.comboBoxMosaicOverlayMode.blockSignals(was_blocked)
        opacity_control = None
        if hasattr(self.ui, "doubleSpinBoxMosaicOverlayOpacity"):
            opacity_control = self.ui.doubleSpinBoxMosaicOverlayOpacity
        elif hasattr(self.ui, "doubleSpinBoxMosaicCoverageOpacity"):
            opacity_control = self.ui.doubleSpinBoxMosaicCoverageOpacity
        if opacity_control is not None:
            was_blocked = opacity_control.blockSignals(True)
            opacity_control.setValue(50.0)
            opacity_control.blockSignals(was_blocked)

    def _bounded_mosaic_utc_offset(self, offset_hours: float) -> float:
        if not hasattr(self.ui, "doubleSpinBoxMosaicUtcOffset"):
            return float(offset_hours)
        return max(
            float(self.ui.doubleSpinBoxMosaicUtcOffset.minimum()),
            min(float(self.ui.doubleSpinBoxMosaicUtcOffset.maximum()), float(offset_hours)),
        )

    def _mosaic_qdatetime_for_observer(self, observer: ObserverSettings, utc_offset_hours: float) -> QDateTime:
        local_timezone = timezone(timedelta(hours=self._bounded_mosaic_utc_offset(utc_offset_hours)))
        local_time = observer.observation_time_utc.astimezone(local_timezone)
        qt_time = QDateTime.fromString(local_time.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd HH:mm:ss")
        if qt_time.isValid():
            return qt_time
        return QDateTime.currentDateTime()

    def _set_mosaic_observer_controls_enabled(self, enabled: bool) -> None:
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return
        for control_name in (
            "dateTimeEditMosaicObservation",
            "doubleSpinBoxMosaicUtcOffset",
            "doubleSpinBoxMosaicLatitude",
            "doubleSpinBoxMosaicLongitude",
            "doubleSpinBoxMosaicElevation",
            "pushButtonResetMosaicObserver",
        ):
            getattr(self.ui, control_name).setEnabled(enabled)

    def _set_mosaic_observer_controls_from_source_model(self, source_model: MosaicSourceModel) -> None:
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return
        controls = (
            self.ui.dateTimeEditMosaicObservation,
            self.ui.doubleSpinBoxMosaicUtcOffset,
            self.ui.doubleSpinBoxMosaicLatitude,
            self.ui.doubleSpinBoxMosaicLongitude,
            self.ui.doubleSpinBoxMosaicElevation,
        )
        previous_blocks = [control.blockSignals(True) for control in controls]
        try:
            utc_offset_hours = self._bounded_mosaic_utc_offset(source_model.utc_offset_hours)
            self.ui.dateTimeEditMosaicObservation.setDateTime(
                self._mosaic_qdatetime_for_observer(source_model.observer, utc_offset_hours)
            )
            self.ui.doubleSpinBoxMosaicUtcOffset.setValue(utc_offset_hours)
            self.ui.doubleSpinBoxMosaicLatitude.setValue(float(source_model.observer.latitude_deg))
            self.ui.doubleSpinBoxMosaicLongitude.setValue(float(source_model.observer.longitude_deg))
            self.ui.doubleSpinBoxMosaicElevation.setValue(float(source_model.observer.elevation_m))
        finally:
            for control, was_blocked in zip(controls, previous_blocks, strict=True):
                control.blockSignals(was_blocked)

    def reset_mosaic_observer_from_json(self) -> None:
        source_model = self._mosaic_source_model
        if source_model is None:
            return
        self._set_mosaic_observer_controls_from_source_model(source_model)
        self.schedule_mosaic_render(delay_ms=0)

    def _mosaic_observer_from_controls(self) -> ObserverSettings | None:
        if self._mosaic_source_model is None:
            return None
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return self._mosaic_source_model.observer
        local_dt = self.ui.dateTimeEditMosaicObservation.dateTime().toPyDateTime()
        offset = timezone(timedelta(hours=float(self.ui.doubleSpinBoxMosaicUtcOffset.value())))
        aware_dt = local_dt.replace(tzinfo=offset)
        return ObserverSettings(
            observation_time_utc=aware_dt.astimezone(timezone.utc),
            latitude_deg=float(self.ui.doubleSpinBoxMosaicLatitude.value()),
            longitude_deg=float(self.ui.doubleSpinBoxMosaicLongitude.value()),
            elevation_m=float(self.ui.doubleSpinBoxMosaicElevation.value()),
        )

    def _mosaic_grid_precision_value(self) -> int:
        if not hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            return MOSAIC_COVERAGE_GRID_LONG_SIDE
        return max(
            MOSAIC_GRID_MIN_PRECISION,
            min(MOSAIC_GRID_MAX_PRECISION, int(self.ui.spinBoxMosaicGridPrecision.value())),
        )

    def _mosaic_grid_shape_for_size(self, width: int, height: int) -> tuple[int, int]:
        precision = self._mosaic_grid_precision_value()
        return grid_shape_for_long_side(width, height, precision, min_minor_cells=3)

    def _update_mosaic_grid_precision_tooltip(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            return
        source_model = self._mosaic_source_model
        if source_model is None:
            precision = self._mosaic_grid_precision_value()
            text = f"下一次导入模型将使用长边 {precision} 点的贴图网格。"
        else:
            rows, columns = self._mosaic_grid_shape_for_size(
                source_model.image_width_px,
                source_model.image_height_px,
            )
            text = f"下一次重新求解将使用 {columns} x {rows} 个源图网格点。"
        self.ui.spinBoxMosaicGridPrecision.setToolTip(text)
        if hasattr(self.ui, "pushButtonResolveMosaicGrid"):
            self.ui.pushButtonResolveMosaicGrid.setToolTip(text)

    def _set_mosaic_grid_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            self.ui.spinBoxMosaicGridPrecision.setEnabled(True)
        if hasattr(self.ui, "pushButtonResolveMosaicGrid"):
            self.ui.pushButtonResolveMosaicGrid.setEnabled(bool(enabled))

    def _build_mosaic_coverage_cache(self, model: FrameAstrometricModel) -> MosaicCoverageCache:
        """委托 mosaic_grid_service 构建覆盖网格缓存。"""
        return build_coverage_cache(
            model,
            grid_precision=self._mosaic_grid_precision_value(),
            min_minor_cells=3,
        )

    def resolve_mosaic_grid_again(self) -> None:
        source_model = self._mosaic_source_model
        if source_model is None:
            self.ui.statusbar.showMessage("尚未导入源图模型，无法重新求解自由投影网格。")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._mosaic_coverage_cache = self._build_mosaic_coverage_cache(source_model.model)
        except Exception as exc:  # noqa: BLE001 - 手动重算入口需要把模型错误直接反馈到界面。
            QMessageBox.critical(self, "重新求解自由投影网格失败", str(exc))
            self.ui.statusbar.showMessage(f"重新求解自由投影网格失败: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._update_mosaic_grid_precision_tooltip()
        self.schedule_mosaic_render(delay_ms=0)
        cache = self._mosaic_coverage_cache
        if cache is not None:
            self.ui.statusbar.showMessage(f"已重新求解自由投影网格: {cache.grid_columns} x {cache.grid_rows}")

    def _mosaic_coverage_altaz(
        self,
        cache: MosaicCoverageCache,
        observer: ObserverSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """委托 mosaic_grid_service 转换覆盖网格到地平坐标。"""
        from .mosaic_grid_service import coverage_altaz as _coverage_altaz
        return _coverage_altaz(cache, observer)

    def _reset_mosaic_center_from_model(self) -> None:
        """委托 mosaic_grid_service 从源图模型计算初始视图中心与 FOV。"""
        source_model = self._mosaic_source_model
        effective_model, cache, observer = self._effective_mosaic_model_and_coverage()
        if source_model is None or effective_model is None or cache is None or observer is None:
            self._mosaic_center_az_deg = 0.0
            self._mosaic_center_alt_deg = 20.0
            self._mosaic_roll_deg = 0.0
            self._update_mosaic_view_label()
            return

        self._mosaic_center_az_deg, self._mosaic_center_alt_deg = compute_center_from_model(
            effective_model, cache, observer,
            source_model.image_width_px, source_model.image_height_px,
        )
        self._mosaic_roll_deg = 0.0
        self._set_mosaic_fov_from_coverage(cache)
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()

    def _set_mosaic_fov_from_coverage(self, cache: MosaicCoverageCache | None = None) -> None:
        """委托 mosaic_grid_service 从覆盖网格推算合适的初始 FOV。"""
        if cache is None:
            cache = self._mosaic_coverage_cache
        if cache is None:
            return
        observer = self._mosaic_observer_from_controls()
        if observer is None:
            source_model = self._mosaic_source_model
            observer = None if source_model is None else source_model.observer
        if observer is None:
            return
        suggested = suggest_fov_from_coverage(
            cache, observer,
            self._mosaic_center_az_deg, self._mosaic_center_alt_deg,
            min_fov_deg=25.0,
            max_fov_deg=self.ui.doubleSpinBoxMosaicFov.maximum(),
        )
        was_blocked = self.ui.doubleSpinBoxMosaicFov.blockSignals(True)
        self.ui.doubleSpinBoxMosaicFov.setValue(suggested)
        self.ui.doubleSpinBoxMosaicFov.blockSignals(was_blocked)

    def reset_mosaic_projection_view(self) -> None:
        self._reset_mosaic_center_from_model()
        self.schedule_mosaic_render(delay_ms=0)

    def schedule_mosaic_render(self, *unused, delay_ms: int = 40) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self, "mosaic_render_timer"):
            return
        self.mosaic_render_timer.start(delay_ms)

    def _mosaic_render_size(self) -> tuple[int, int]:
        size = self.ui.mosaicProjectionView.viewport().size()
        return (
            max(MOSAIC_RENDER_MIN_SIZE_PX, int(size.width())),
            max(MOSAIC_RENDER_MIN_SIZE_PX, int(size.height())),
        )

    def _mosaic_camera_for_render(self, width: int, height: int) -> CameraSettings:
        projection_model = self._mosaic_projection_model()
        fov_deg = max(1.0, float(self.ui.doubleSpinBoxMosaicFov.value()))
        sensor_width_mm = 36.0
        sensor_height_mm = sensor_width_mm * height / max(float(width), 1.0)
        focal_length_mm = 24.0
        if projection_model == SKY_MATCHING_MODEL_RECTILINEAR:
            fov_deg = min(fov_deg, 160.0)
            focal_length_mm = sensor_width_mm / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        return CameraSettings(
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            image_width_px=width,
            image_height_px=height,
            focal_length_mm=max(focal_length_mm, 0.05),
            lens_model=projection_model,
            fisheye_fov_deg=fov_deg,
        )

    def _mosaic_view_settings(self, camera: CameraSettings | None = None) -> ViewSettings:
        if camera is None:
            width, height = self._mosaic_render_size()
            camera = self._mosaic_camera_for_render(width, height)
        state = ProjectionViewState.from_camera_and_view(
            camera,
            ViewSettings(
                center_az_deg=self._mosaic_center_az_deg,
                center_alt_deg=self._mosaic_center_alt_deg,
                roll_deg=self._mosaic_roll_deg,
            ),
        )
        return state.to_view_settings()

    def _effective_mosaic_model_and_coverage(
        self,
    ) -> tuple[FrameAstrometricModel | None, MosaicCoverageCache | None, ObserverSettings | None]:
        source_model = self._mosaic_source_model
        if source_model is None:
            return None, None, None
        observer = self._mosaic_observer_from_controls() or source_model.observer
        return source_model.model, self._mosaic_coverage_cache, observer

    def _mosaic_overlay_mode(self) -> str:
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            index = self.ui.comboBoxMosaicOverlayMode.currentIndex()
            if 0 <= index < len(MOSAIC_OVERLAY_MODES):
                return MOSAIC_OVERLAY_MODES[index]
        return MOSAIC_OVERLAY_MODE_COVERAGE

    def _mosaic_overlay_opacity(self) -> float:
        if hasattr(self.ui, "doubleSpinBoxMosaicOverlayOpacity"):
            value = self.ui.doubleSpinBoxMosaicOverlayOpacity.value()
        elif hasattr(self.ui, "doubleSpinBoxMosaicCoverageOpacity"):
            value = self.ui.doubleSpinBoxMosaicCoverageOpacity.value()
        else:
            value = 100.0
        return max(0.0, min(1.0, float(value) / 100.0))

    def _mosaic_overlay_enabled(self) -> bool:
        if hasattr(self.ui, "checkBoxMosaicSkyOnly") and self.ui.checkBoxMosaicSkyOnly.isChecked():
            return False
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            return True
        if hasattr(self.ui, "checkBoxMosaicShowCoverage"):
            return bool(self.ui.checkBoxMosaicShowCoverage.isChecked())
        return True

    def _set_mosaic_view_controls_from_state(self) -> None:
        control_values = (
            ("doubleSpinBoxMosaicAz", self._mosaic_center_az_deg % 360.0),
            ("doubleSpinBoxMosaicAlt", self._mosaic_center_alt_deg),
            ("doubleSpinBoxMosaicRoll", self._mosaic_roll_deg),
        )
        for control_name, value in control_values:
            if not hasattr(self.ui, control_name):
                continue
            control = getattr(self.ui, control_name)
            was_blocked = control.blockSignals(True)
            control.setValue(float(value))
            control.blockSignals(was_blocked)

    def render_mosaic_projection_now(self) -> None:
        if not hasattr(self.ui, "mosaicProjectionView"):
            return
        width, height = self._mosaic_render_size()
        source_model = self._mosaic_source_model
        effective_model, coverage_cache, observer = self._effective_mosaic_model_and_coverage()
        if source_model is None or effective_model is None or coverage_cache is None or observer is None:
            image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
            image.fill(QColor(0, 0, 0))
            self.mosaic_image_item.set_image(image)
            self.mosaic_scene.setSceneRect(0.0, 0.0, float(width), float(height))
            self.ui.mosaicProjectionView.resetTransform()
            return

        try:
            camera = self._mosaic_camera_for_render(width, height)
            view = self._mosaic_view_settings(camera)
            mag_limit = float(self.ui.doubleSpinBoxMosaicMagLimit.value())
            horizontal_catalog = self._get_horizontal_catalog(observer, mag_limit)
            horizontal_milky_way = self._get_horizontal_milky_way(observer)
            horizontal_solar_system = self._get_horizontal_solar_system(observer)
            image = self._mosaic_sky_preview_renderer.render(
                scene=SkySceneData(
                    horizontal_catalog=horizontal_catalog,
                    horizontal_milky_way=horizontal_milky_way,
                    horizontal_solar_system=horizontal_solar_system,
                ),
                camera=camera,
                view=view,
                visible_mag_limit=mag_limit,
                style=SkyPreviewStyle(
                    draw_common_names=False,
                    number_reference_stars=False,
                    draw_background=True,
                    draw_horizon_shadow=True,
                    draw_grid=self.ui.checkBoxMosaicShowGrid.isChecked(),
                    draw_solar_system_labels=True,
                    draw_direction_labels=self.ui.checkBoxMosaicShowGrid.isChecked(),
                ),
            )
            if self._mosaic_overlay_enabled():
                if self._mosaic_overlay_mode() == MOSAIC_OVERLAY_MODE_SOURCE_IMAGE:
                    self._paint_mosaic_source_image(image, camera, view, coverage_cache, observer, source_model)
                else:
                    self._paint_mosaic_coverage(image, camera, view, coverage_cache, observer)
            self.mosaic_image_item.set_image(image)
            self.mosaic_scene.setSceneRect(0.0, 0.0, float(width), float(height))
            self.ui.mosaicProjectionView.resetTransform()
            self._update_mosaic_view_label()
        except Exception as exc:  # noqa: BLE001 - 预览渲染要把模型和投影错误反馈给用户。
            self.ui.statusbar.showMessage(f"自由投影预览渲染失败: {exc}")

    def _paint_mosaic_coverage(
        self,
        image: QImage,
        camera: CameraSettings,
        view: ViewSettings,
        cache: MosaicCoverageCache | None,
        observer: ObserverSettings,
    ) -> None:
        """委托 mosaic_overlay_renderer 绘制覆盖范围叠加。"""
        if cache is None:
            return
        paint_coverage_overlay(image, cache, camera, view, observer, self._mosaic_overlay_opacity())

    def _mosaic_source_texture(self, source_model: MosaicSourceModel) -> MosaicSourceTextureCache | None:
        """委托 mosaic_overlay_renderer 加载源图纹理。"""
        texture = load_source_texture(source_model, self._mosaic_source_texture_cache)
        if texture is None and source_model.source_image_path is not None:
            resolved = source_model.source_image_path
            try:
                resolved = resolved.expanduser().resolve()
            except OSError:
                pass
            if not resolved.exists():
                self.ui.statusbar.showMessage(f"源图不存在，无法显示原图: {resolved}")
            else:
                self.ui.statusbar.showMessage("源图模型 JSON 未记录真实图像路径，无法显示原图。")
            return None
        if texture is not None:
            self._mosaic_source_texture_cache = texture
        return texture

    def _paint_mosaic_source_image(
        self,
        image: QImage,
        camera: CameraSettings,
        view: ViewSettings,
        cache: MosaicCoverageCache | None,
        observer: ObserverSettings,
        source_model: MosaicSourceModel,
    ) -> None:
        """委托 mosaic_overlay_renderer 将源图纹理重投影叠加到输出图像。"""
        if cache is None:
            return
        texture = self._mosaic_source_texture(source_model)
        if texture is None:
            return
        paint_source_image_overlay(
            image, self._mosaic_texture_renderer,
            cache, camera, view, observer, texture,
            self._mosaic_overlay_opacity(),
        )

    def _handle_mosaic_event_filter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        """处理 mosaic 页面特有的标签省略号事件。

        视图交互（拖拽、缩放、触控板）已交由 ProjectionInteractionController 处理。
        """
        if not hasattr(self.ui, "mosaicProjectionView"):
            return False
        # 标签省略号更新
        if watched in (
            getattr(self.ui, "labelMosaicModelPath", None),
            getattr(self.ui, "labelMosaicSourceImage", None),
            getattr(self.ui, "labelMosaicModelInfo", None),
            getattr(self.ui, "labelMosaicViewInfo", None),
        ):
            if event.type() in (QEvent.Resize, QEvent.Show):
                QTimer.singleShot(0, lambda label=watched: self._refresh_elided_label(label))
            return False
        return False

    # ------------------------------------------------------------------
    # 交互回调（由 ProjectionInteractionController 驱动）
    # ------------------------------------------------------------------

    def _handle_mosaic_pan_delta(self, dx: int, dy: int) -> None:
        """处理平移拖拽。"""
        width, height = self._mosaic_render_size()
        camera = self._mosaic_camera_for_render(width, height)
        self._mosaic_center_az_deg, self._mosaic_center_alt_deg = sky_center_after_drag(
            center_az_deg=self._mosaic_center_az_deg,
            center_alt_deg=self._mosaic_center_alt_deg,
            dx_px=dx,
            dy_px=dy,
            horizontal_fov_deg=horizontal_fov_deg(camera),
            vertical_fov_deg=vertical_fov_deg(camera),
            viewport_width_px=width,
            viewport_height_px=height,
            min_degrees_per_pixel=0.005,
        )
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _handle_mosaic_roll_delta(self, dx: int) -> None:
        """处理滚转拖拽。"""
        self._mosaic_roll_deg = roll_after_drag(self._mosaic_roll_deg, dx, drag_sign=-1.0)
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _handle_mosaic_zoom_factor(self, zoom_factor: float) -> None:
        """处理缩放（滚轮或触控板）。"""
        self._apply_mosaic_fov_zoom_factor(zoom_factor)
        self.render_mosaic_projection_now()

    def _handle_mosaic_resize_event(self) -> None:
        """处理视图尺寸变化。"""
        self.schedule_mosaic_render()

    def _handle_mosaic_interaction_end_event(self) -> None:
        """交互结束（鼠标释放）时触发最终高质量渲染。"""
        self.render_mosaic_projection_now()

    def _apply_mosaic_fov_zoom_factor(self, zoom_factor: float) -> None:
        if not np.isfinite(zoom_factor) or zoom_factor <= 0.0 or abs(zoom_factor - 1.0) <= 1e-4:
            return
        current = float(self.ui.doubleSpinBoxMosaicFov.value())
        target = clamp_fov(
            current / zoom_factor,
            self.ui.doubleSpinBoxMosaicFov.minimum(),
            self.ui.doubleSpinBoxMosaicFov.maximum(),
        )
        self.ui.doubleSpinBoxMosaicFov.setValue(target)

    def _mosaic_wheel_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "wheel_zoom_enabled", True))

    def _mosaic_touchpad_pinch_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "touchpad_pinch_zoom_enabled", True))

    def _mosaic_zoom_policy(self) -> ViewZoomPolicy:
        maximum = self.ui.doubleSpinBoxMosaicFov.maximum() if hasattr(self.ui, "doubleSpinBoxMosaicFov") else 360.0
        minimum = self.ui.doubleSpinBoxMosaicFov.minimum() if hasattr(self.ui, "doubleSpinBoxMosaicFov") else 1.0
        return ViewZoomPolicy(
            wheel_enabled=self._mosaic_wheel_zoom_enabled(),
            pinch_enabled=self._mosaic_touchpad_pinch_zoom_enabled(),
            min_fov=float(minimum),
            max_fov=float(maximum),
        )

    def _update_mosaic_model_labels(self) -> None:
        if not hasattr(self.ui, "labelMosaicModelPath"):
            return
        source_model = self._mosaic_source_model
        if source_model is None:
            self._set_elided_label_text(self.ui.labelMosaicModelPath, "未导入")
            self._set_elided_label_text(self.ui.labelMosaicSourceImage, "未导入")
            self._set_elided_label_text(self.ui.labelMosaicModelInfo, "-")
            return
        self._set_elided_label_text(
            self.ui.labelMosaicModelPath,
            source_model.json_path.name,
            str(source_model.json_path),
        )
        source_tooltip = str(source_model.source_image_path) if source_model.source_image_path is not None else ""
        self._set_elided_label_text(self.ui.labelMosaicSourceImage, source_model.source_image_text, source_tooltip)
        rms_text = f"{source_model.rms_px:.2f}px" if np.isfinite(source_model.rms_px) else "-"
        info_text = (
            f"{source_model.image_width_px} x {source_model.image_height_px} px  "
            f"星点 {source_model.pair_count}  RMS {rms_text}"
        )
        self._set_elided_label_text(self.ui.labelMosaicModelInfo, info_text)

    def _update_mosaic_view_label(self) -> None:
        if not hasattr(self.ui, "labelMosaicViewInfo"):
            return
        projection_name = SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(self._mosaic_projection_model(), self._mosaic_projection_model())
        text = (
            f"{projection_name}  Az {self._mosaic_center_az_deg:.2f} deg  "
            f"Alt {self._mosaic_center_alt_deg:.2f} deg  Roll {self._mosaic_roll_deg:.2f} deg"
        )
        self._set_elided_label_text(self.ui.labelMosaicViewInfo, text)

__all__ = [name for name in globals() if not name.startswith("__")]
