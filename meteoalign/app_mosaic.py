from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, QElapsedTimer, QEvent, QRectF, QTimer, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QMessageBox, QProgressDialog

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
    MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX,
    MOSAIC_ZOOM_FACTOR,
)
from .mosaic_grid_service import (
    build_coverage_cache,
    compute_center_from_model,
    suggest_fov_from_coverage,
)
from .mosaic_export import (
    MOSAIC_EXPORT_TIFF_FILTER,
    build_mosaic_reprojection_map_payload,
    load_mosaic_export_source_image,
    mosaic_export_available,
    mosaic_export_block_rows,
    mosaic_export_cropped_geometry,
    mosaic_reprojection_map_payload_matches,
    write_mosaic_reprojection_tiff,
)
from .mosaic_framing import (
    MOSAIC_FRAMING_SCHEMA,
    MOSAIC_FRAMING_VERSION,
    MOSAIC_RESOLUTION_METHOD,
    MosaicResolutionEstimate,
    estimate_mosaic_optimal_resolution,
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


MOSAIC_FRAMING_JSON_FILTER = "自由投影取景 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)"


class MosaicProjectionMixin:
    """自由投影拼图预览 Mixin。"""

    ui: object
    renderer: object
    catalog: object
    milky_way_catalog: object
    ui_config: object

    MOSAIC_RENDER_FPS_LIMIT = 60
    MOSAIC_INTERACTION_TEXTURE_SCALE = 0.5

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
            on_interaction_start=self._handle_mosaic_interaction_start_event,
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
        self._mosaic_render_clock = QElapsedTimer()
        self._mosaic_render_clock.start()
        self._mosaic_last_render_msecs = -10_000
        self._mosaic_min_render_interval_ms = max(1, int(round(1000.0 / self._mosaic_render_fps_limit())))
        self._mosaic_sky_preview_renderer = SkyPreviewRenderService(self.renderer)
        self._mosaic_texture_renderer = ProjectedTextureRenderer()
        self._mosaic_source_model: MosaicSourceModel | None = None
        self._mosaic_coverage_cache: MosaicCoverageCache | None = None
        self._mosaic_source_texture_cache: MosaicSourceTextureCache | None = None
        self._mosaic_interaction_source_texture_cache: MosaicSourceTextureCache | None = None
        self._mosaic_interaction_coverage_cache: MosaicCoverageCache | None = None
        self._mosaic_interaction_coverage_source_id: int | None = None
        self._mosaic_interaction_active = False
        self._mosaic_center_az_deg = 0.0
        self._mosaic_center_alt_deg = 20.0
        self._mosaic_roll_deg = 0.0
        self._mosaic_output_boundary_width_px = 0
        self._mosaic_output_boundary_height_px = 0
        self._mosaic_resolution_estimate: MosaicResolutionEstimate | None = None
        self._mosaic_framing_observer: ObserverSettings | None = None
        self._mosaic_framing_utc_offset_hours = 0.0
        self._mosaic_reprojection_map_payload: dict[str, object] | None = None
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
            self.ui.spinBoxMosaicGridPrecision.setValue(self._mosaic_default_grid_precision())
        self._init_mosaic_observer_controls()
        self._update_mosaic_projection_controls()
        self._update_mosaic_grid_precision_tooltip()
        self._set_mosaic_grid_controls_enabled(False)
        self._update_mosaic_output_labels()
        self._update_mosaic_crop_control_limits()
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
        self.ui.doubleSpinBoxMosaicFov.valueChanged.connect(self._handle_mosaic_fov_changed)
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
        if hasattr(self.ui, "pushButtonCalculateMosaicResolution"):
            self.ui.pushButtonCalculateMosaicResolution.clicked.connect(self.calculate_mosaic_optimal_resolution)
        if hasattr(self.ui, "pushButtonExportMosaicFraming"):
            self.ui.pushButtonExportMosaicFraming.clicked.connect(self.export_mosaic_framing_json)
        if hasattr(self.ui, "pushButtonImportMosaicFraming"):
            self.ui.pushButtonImportMosaicFraming.clicked.connect(self.import_mosaic_framing_json)
        if hasattr(self.ui, "pushButtonExportMosaicProjectedImage"):
            self.ui.pushButtonExportMosaicProjectedImage.clicked.connect(self.export_mosaic_projected_image)
        for control_name in (
            "doubleSpinBoxMosaicCropTop",
            "doubleSpinBoxMosaicCropBottom",
            "doubleSpinBoxMosaicCropLeft",
            "doubleSpinBoxMosaicCropRight",
        ):
            if hasattr(self.ui, control_name):
                getattr(self.ui, control_name).valueChanged.connect(self._handle_mosaic_crop_changed)
        self.ui.labelMosaicModelPath.installEventFilter(self)
        self.ui.labelMosaicSourceImage.installEventFilter(self)
        self.ui.labelMosaicModelInfo.installEventFilter(self)
        self.ui.labelMosaicViewInfo.installEventFilter(self)

    def _handle_mosaic_projection_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._invalidate_mosaic_reprojection_map()
        self._update_mosaic_projection_controls()
        self.schedule_mosaic_render()

    def _handle_mosaic_fov_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._invalidate_mosaic_reprojection_map()
        self.schedule_mosaic_render()

    def _handle_mosaic_view_controls_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._invalidate_mosaic_reprojection_map()
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
        self._invalidate_mosaic_reprojection_map()
        observer = self._mosaic_observer_from_control_values()
        if observer is not None:
            self._mosaic_framing_observer = observer
            self._mosaic_framing_utc_offset_hours = self._mosaic_utc_offset_from_controls()
        self.schedule_mosaic_render(delay_ms=120)

    def _handle_mosaic_crop_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._invalidate_mosaic_reprojection_map()
        self.schedule_mosaic_render(delay_ms=0)

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

    def _mosaic_output_size(self) -> tuple[int, int]:
        return (
            max(0, int(getattr(self, "_mosaic_output_boundary_width_px", 0))),
            max(0, int(getattr(self, "_mosaic_output_boundary_height_px", 0))),
        )

    def _invalidate_mosaic_reprojection_map(self) -> None:
        self._mosaic_reprojection_map_payload = None

    def _mosaic_export_block_rows(self) -> int:
        return mosaic_export_block_rows(getattr(self, "ui_config", object()))

    def _set_mosaic_output_resolution(
        self,
        width_px: int,
        height_px: int,
        estimate: MosaicResolutionEstimate | None = None,
    ) -> None:
        self._mosaic_output_boundary_width_px = max(0, int(round(width_px)))
        self._mosaic_output_boundary_height_px = max(0, int(round(height_px)))
        self._mosaic_resolution_estimate = estimate
        self._invalidate_mosaic_reprojection_map()
        self._update_mosaic_output_labels()
        self._update_mosaic_crop_control_limits()

    def _update_mosaic_output_labels(self) -> None:
        if not hasattr(self.ui, "labelMosaicOptimalWidth"):
            return
        width_px, height_px = self._mosaic_output_size()
        self.ui.labelMosaicOptimalWidth.setText(f"{width_px} px" if width_px > 0 else "-")
        self.ui.labelMosaicOptimalHeight.setText(f"{height_px} px" if height_px > 0 else "-")
        if hasattr(self.ui, "labelMosaicResolutionInfo"):
            estimate = getattr(self, "_mosaic_resolution_estimate", None)
            if estimate is None:
                info = "尚未计算" if width_px <= 0 or height_px <= 0 else "导入取景"
            else:
                arcsec_per_px = estimate.center_angular_resolution_rad_per_px * 180.0 * 3600.0 / math.pi
                info = f"中心 {arcsec_per_px:.2f} arcsec/px"
            self.ui.labelMosaicResolutionInfo.setText(info)

    def _update_mosaic_crop_control_limits(self) -> None:
        width_px, height_px = self._mosaic_output_size()
        limits = {
            "doubleSpinBoxMosaicCropLeft": width_px,
            "doubleSpinBoxMosaicCropRight": width_px,
            "doubleSpinBoxMosaicCropTop": height_px,
            "doubleSpinBoxMosaicCropBottom": height_px,
        }
        for control_name, output_limit in limits.items():
            if not hasattr(self.ui, control_name):
                continue
            control = getattr(self.ui, control_name)
            maximum = float(output_limit) if output_limit > 0 else 1_000_000_000.0
            was_blocked = control.blockSignals(True)
            control.setMaximum(max(1.0, maximum))
            if control.value() > control.maximum():
                control.setValue(control.maximum())
            control.blockSignals(was_blocked)

    def _mosaic_crop_margins(self) -> dict[str, float]:
        margins: dict[str, float] = {}
        controls = {
            "top_px": "doubleSpinBoxMosaicCropTop",
            "bottom_px": "doubleSpinBoxMosaicCropBottom",
            "left_px": "doubleSpinBoxMosaicCropLeft",
            "right_px": "doubleSpinBoxMosaicCropRight",
        }
        for key, control_name in controls.items():
            value = 0.0
            if hasattr(self.ui, control_name):
                raw_value = float(getattr(self.ui, control_name).value())
                value = raw_value if np.isfinite(raw_value) else 0.0
            margins[key] = max(0.0, float(value))
        return margins

    def _set_mosaic_crop_controls(self, margins: dict[str, object]) -> None:
        controls = {
            "top_px": "doubleSpinBoxMosaicCropTop",
            "bottom_px": "doubleSpinBoxMosaicCropBottom",
            "left_px": "doubleSpinBoxMosaicCropLeft",
            "right_px": "doubleSpinBoxMosaicCropRight",
        }
        for key, control_name in controls.items():
            if not hasattr(self.ui, control_name):
                continue
            try:
                value = float(margins.get(key, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            control = getattr(self.ui, control_name)
            was_blocked = control.blockSignals(True)
            control.setValue(max(0.0, min(float(control.maximum()), value if np.isfinite(value) else 0.0)))
            control.blockSignals(was_blocked)

    def _mosaic_crop_rect_payload(self) -> dict[str, object]:
        width_px, height_px = self._mosaic_output_size()
        margins = self._mosaic_crop_margins()
        left_px = min(margins["left_px"], float(width_px))
        right_px = min(margins["right_px"], max(0.0, float(width_px) - left_px))
        top_px = min(margins["top_px"], float(height_px))
        bottom_px = min(margins["bottom_px"], max(0.0, float(height_px) - top_px))
        crop_width = max(0.0, float(width_px) - left_px - right_px)
        crop_height = max(0.0, float(height_px) - top_px - bottom_px)
        return {
            "top_px": float(top_px),
            "bottom_px": float(bottom_px),
            "left_px": float(left_px),
            "right_px": float(right_px),
            "x_px": float(left_px),
            "y_px": float(top_px),
            "width_px": float(crop_width),
            "height_px": float(crop_height),
        }

    def _mosaic_estimate_optimal_resolution(self) -> MosaicResolutionEstimate:
        source_model = self._mosaic_source_model
        if source_model is None:
            raise ValueError("需要先导入源图模型，才能按源图中心角分辨率计算最优输出尺寸。")
        width, height = self._mosaic_render_size()
        return estimate_mosaic_optimal_resolution(
            source_model.model,
            source_image_width_px=source_model.image_width_px,
            source_image_height_px=source_model.image_height_px,
            projection_model=self._mosaic_projection_model(),
            fov_deg=float(self.ui.doubleSpinBoxMosaicFov.value()),
            viewport_width_px=width,
            viewport_height_px=height,
        )

    def calculate_mosaic_optimal_resolution(self) -> bool:
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            estimate = self._mosaic_estimate_optimal_resolution()
        except Exception as exc:  # noqa: BLE001 - 手动计算入口需要把模型与投影错误直接反馈到界面。
            QMessageBox.warning(self, "计算最优分辨率失败", str(exc))
            self.ui.statusbar.showMessage(f"计算最优分辨率失败: {exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

        self._set_mosaic_output_resolution(
            estimate.boundary_width_px,
            estimate.boundary_height_px,
            estimate,
        )
        self.schedule_mosaic_render(delay_ms=0)
        self.ui.statusbar.showMessage(
            f"最优输出边界: {estimate.boundary_width_px} x {estimate.boundary_height_px} px"
        )
        return True

    def _mosaic_observer_payload(self, observer: ObserverSettings) -> dict[str, object]:
        return {
            "observation_time_utc": observer.observation_time_utc.astimezone(timezone.utc).isoformat(),
            "utc_offset_hours": float(self._mosaic_utc_offset_from_controls()),
            "latitude_deg": float(observer.latitude_deg),
            "longitude_deg": float(observer.longitude_deg),
            "elevation_m": float(observer.elevation_m),
        }

    def _mosaic_resolution_payload(self) -> dict[str, object]:
        estimate = getattr(self, "_mosaic_resolution_estimate", None)
        if estimate is None:
            return {
                "method": MOSAIC_RESOLUTION_METHOD,
                "status": "imported_or_manual",
            }
        return {
            "method": MOSAIC_RESOLUTION_METHOD,
            "source_center_x_px": float(estimate.source_center_x_px),
            "source_center_y_px": float(estimate.source_center_y_px),
            "source_center_angular_resolution_rad_per_px": float(
                estimate.center_angular_resolution_rad_per_px
            ),
            "target_center_px_per_rad": float(estimate.target_center_px_per_rad),
            "source_jacobian_rad_per_px": [
                [float(value) for value in row]
                for row in estimate.source_jacobian_rad_per_px
            ],
        }

    def _mosaic_framing_payload(self) -> dict[str, object]:
        width_px, height_px = self._mosaic_output_size()
        if width_px <= 0 or height_px <= 0:
            raise ValueError("请先计算最优分辨率，或导入包含输出边界的取景 JSON。")
        observer = self._mosaic_observer_from_controls()
        if observer is None:
            raise ValueError("取景 JSON 需要有效的拍摄时间和地点。")
        render_width, render_height = self._mosaic_render_size()
        projection_model = self._mosaic_projection_model()
        crop_payload = self._mosaic_crop_rect_payload()
        source_model = self._mosaic_source_model
        payload: dict[str, object] = {
            "schema": MOSAIC_FRAMING_SCHEMA,
            "version": MOSAIC_FRAMING_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "projection": {
                "model": projection_model,
                "display_name": SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(projection_model, projection_model),
                "fov_deg": float(self.ui.doubleSpinBoxMosaicFov.value()),
            },
            "view": {
                "center_az_deg": float(self._mosaic_center_az_deg) % 360.0,
                "center_alt_deg": float(self._mosaic_center_alt_deg),
                "roll_deg": float(self._mosaic_roll_deg),
            },
            "observer": self._mosaic_observer_payload(observer),
            "output": {
                "boundary_width_px": int(width_px),
                "boundary_height_px": int(height_px),
                "pixel_convention": "0-based_pixel_center",
                "crop": crop_payload,
                "cropped_width_px": float(crop_payload["width_px"]),
                "cropped_height_px": float(crop_payload["height_px"]),
            },
            "viewport_reference": {
                "width_px": int(render_width),
                "height_px": int(render_height),
                "height_over_width": float(render_height / max(float(render_width), 1.0)),
            },
            "resolution_method": self._mosaic_resolution_payload(),
        }
        if source_model is not None:
            payload["source_model"] = {
                "path": str(source_model.json_path),
                "image_width_px": int(source_model.image_width_px),
                "image_height_px": int(source_model.image_height_px),
                "rms_px": float(source_model.rms_px),
            }
        return payload

    def _mosaic_geometry_from_framing_payload(self, payload: dict[str, object]):
        output_payload = payload.get("output")
        if not isinstance(output_payload, dict):
            raise ValueError("当前取景缺少输出边界。")
        return mosaic_export_cropped_geometry(
            boundary_width_px=int(output_payload["boundary_width_px"]),
            boundary_height_px=int(output_payload["boundary_height_px"]),
            crop=output_payload.get("crop") if isinstance(output_payload.get("crop"), dict) else {},
        )

    def _mosaic_current_map_matches(self, geometry) -> bool:  # type: ignore[no-untyped-def]
        source_model = self._mosaic_source_model
        observer = self._mosaic_observer_from_controls()
        if source_model is None or observer is None:
            return False
        camera = self._mosaic_camera_for_render(geometry.boundary_width_px, geometry.boundary_height_px)
        view = self._mosaic_view_settings(camera)
        return mosaic_reprojection_map_payload_matches(
            getattr(self, "_mosaic_reprojection_map_payload", None),
            geometry=geometry,
            source_model_path=str(source_model.json_path),
            camera=camera,
            view=view,
            observer=observer,
        )

    def _build_mosaic_reprojection_map_for_geometry(
        self,
        geometry,
        *,
        title: str,
    ) -> dict[str, object]:  # type: ignore[no-untyped-def]
        source_model = self._mosaic_source_model
        observer = self._mosaic_observer_from_controls()
        if source_model is None or observer is None:
            raise ValueError("生成 map_x/map_y 需要先导入源图模型，并设置有效拍摄信息。")
        if not mosaic_export_available():
            raise ValueError("当前环境缺少 OpenCV 或 tifffile，无法生成 map_x/map_y。")

        progress = QProgressDialog("正在生成 map_x/map_y...", "取消", 0, int(geometry.output_height_px), self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def update_progress(rows_done: int) -> None:
            progress.setValue(int(rows_done))
            QApplication.processEvents()
            if progress.wasCanceled():
                raise InterruptedError("用户取消了 map_x/map_y 生成。")

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            camera = self._mosaic_camera_for_render(geometry.boundary_width_px, geometry.boundary_height_px)
            view = self._mosaic_view_settings(camera)
            map_payload = build_mosaic_reprojection_map_payload(
                source_model=source_model.model,
                camera=camera,
                view=view,
                observer=observer,
                geometry=geometry,
                source_model_path=str(source_model.json_path),
                source_image_width_px=source_model.image_width_px,
                source_image_height_px=source_model.image_height_px,
                block_rows=self._mosaic_export_block_rows(),
                progress_callback=update_progress,
            )
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()
        self._mosaic_reprojection_map_payload = map_payload
        return map_payload

    def export_mosaic_framing_json(self) -> None:
        if self._mosaic_source_model is not None:
            try:
                estimate = self._mosaic_estimate_optimal_resolution()
                self._set_mosaic_output_resolution(
                    estimate.boundary_width_px,
                    estimate.boundary_height_px,
                    estimate,
                )
            except Exception as exc:  # noqa: BLE001 - 导出前自动刷新尺寸失败时直接反馈用户。
                QMessageBox.warning(self, "导出取景失败", f"无法刷新最优分辨率：{exc}")
                self.ui.statusbar.showMessage(f"导出取景失败: {exc}")
                return
        try:
            payload = self._mosaic_framing_payload()
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把缺失参数直接反馈到界面。
            QMessageBox.warning(self, "导出取景失败", str(exc))
            self.ui.statusbar.showMessage(f"导出取景失败: {exc}")
            return

        default_dir = project_root() / "outputs"
        if self._mosaic_source_model is not None:
            default_dir = self._mosaic_source_model.json_path.parent
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出自由投影取景 JSON",
            str(default_dir / "mosaic_framing.json"),
            MOSAIC_FRAMING_JSON_FILTER,
        )
        if not file_path:
            return
        json_path = Path(file_path).expanduser()
        try:
            geometry = self._mosaic_geometry_from_framing_payload(payload)
            if self._mosaic_source_model is not None and geometry.output_width_px > 0 and geometry.output_height_px > 0:
                if self._mosaic_current_map_matches(geometry):
                    payload["reprojection_map"] = self._mosaic_reprojection_map_payload
                else:
                    payload["reprojection_map"] = self._build_mosaic_reprojection_map_for_geometry(
                        geometry,
                        title="导出自由投影取景",
                    )
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except InterruptedError as exc:
            self.ui.statusbar.showMessage(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 文件写入错误需要直接反馈。
            QMessageBox.critical(self, "导出取景失败", str(exc))
            self.ui.statusbar.showMessage(f"导出取景失败: {exc}")
            return
        self.ui.statusbar.showMessage(f"已导出自由投影取景: {json_path}")

    def export_mosaic_projected_image(self) -> None:
        source_model = self._mosaic_source_model
        if source_model is None:
            QMessageBox.warning(self, "导出重投影图失败", "请先导入源图模型 JSON。")
            return
        if source_model.source_image_path is None:
            QMessageBox.warning(self, "导出重投影图失败", "源图模型 JSON 未记录原图路径。")
            return
        if not mosaic_export_available():
            QMessageBox.critical(self, "导出重投影图失败", "当前环境缺少 OpenCV 或 tifffile，无法写入 16-bit TIFF。")
            return

        if self._mosaic_output_size()[0] <= 0 or self._mosaic_output_size()[1] <= 0:
            if not self.calculate_mosaic_optimal_resolution():
                return
        try:
            payload = self._mosaic_framing_payload()
            geometry = self._mosaic_geometry_from_framing_payload(payload)
            if geometry.output_width_px <= 0 or geometry.output_height_px <= 0:
                raise ValueError("裁剪后的导出尺寸无效，请减小四边裁剪量。")
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把缺失参数直接反馈到界面。
            QMessageBox.warning(self, "导出重投影图失败", str(exc))
            self.ui.statusbar.showMessage(f"导出重投影图失败: {exc}")
            return

        default_dir = source_model.json_path.parent if source_model.json_path.parent.exists() else project_root()
        default_name = f"{source_model.json_path.stem}_mosaic_{geometry.output_width_px}x{geometry.output_height_px}.tif"
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出自由投影重投影图",
            str(default_dir / default_name),
            MOSAIC_EXPORT_TIFF_FILTER,
        )
        if not file_path:
            return
        output_path = Path(file_path).expanduser()
        if output_path.suffix.lower() not in (".tif", ".tiff"):
            output_path = output_path.with_suffix(".tif")
        if QMessageBox.question(
            self,
            "确认导出重投影图",
            (
                f"导出文件：\n{output_path}\n\n"
                f"尺寸：{geometry.output_width_px} x {geometry.output_height_px} px\n"
                "格式：无压缩 16-bit TIFF"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return

        map_payload = getattr(self, "_mosaic_reprojection_map_payload", None)
        try:
            if not self._mosaic_current_map_matches(geometry):
                map_payload = self._build_mosaic_reprojection_map_for_geometry(
                    geometry,
                    title="准备导出重投影图",
                )
            payload["reprojection_map"] = map_payload
        except InterruptedError as exc:
            self.ui.statusbar.showMessage(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - map 生成失败需要阻止导出错图。
            QMessageBox.critical(self, "导出重投影图失败", str(exc))
            self.ui.statusbar.showMessage(f"导出重投影图失败: {exc}")
            return

        self._write_mosaic_projected_image(output_path, payload, geometry, map_payload)

    def _write_mosaic_projected_image(
        self,
        output_path: Path,
        framing_payload: dict[str, object],
        geometry,
        map_payload: dict[str, object] | None,
    ) -> None:  # type: ignore[no-untyped-def]
        source_model = self._mosaic_source_model
        observer = self._mosaic_observer_from_controls()
        if source_model is None or observer is None:
            return
        temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix or '.tif'}")
        progress = QProgressDialog("正在导出重投影 TIFF...", "取消", 0, int(geometry.output_height_px), self)
        progress.setWindowTitle("导出自由投影重投影图")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def update_progress(rows_done: int) -> None:
            progress.setValue(int(rows_done))
            QApplication.processEvents()
            if progress.wasCanceled():
                raise InterruptedError("用户取消了导出。")

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            source_image = load_mosaic_export_source_image(source_model.source_image_path)
            if source_image.width_px != source_model.image_width_px or source_image.height_px != source_model.image_height_px:
                raise ValueError(
                    "原图尺寸与源图模型不一致："
                    f"原图 {source_image.width_px} x {source_image.height_px} px，"
                    f"模型 {source_model.image_width_px} x {source_model.image_height_px} px。"
                )
            camera = self._mosaic_camera_for_render(geometry.boundary_width_px, geometry.boundary_height_px)
            view = self._mosaic_view_settings(camera)
            write_mosaic_reprojection_tiff(
                output_path=temp_path,
                source_model=source_model.model,
                source_image=source_image,
                camera=camera,
                view=view,
                observer=observer,
                geometry=geometry,
                framing_payload=framing_payload,
                block_rows=self._mosaic_export_block_rows(),
                progress_callback=update_progress,
                map_payload=map_payload,
            )
            temp_path.replace(output_path)
        except InterruptedError as exc:
            self.ui.statusbar.showMessage(str(exc))
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            return
        except Exception as exc:  # noqa: BLE001 - 长流程导出需要把文件和几何错误直接反馈。
            QMessageBox.critical(self, "导出重投影图失败", str(exc))
            self.ui.statusbar.showMessage(f"导出重投影图失败: {exc}")
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()
        self.ui.statusbar.showMessage(f"已导出自由投影重投影图: {output_path}")

    def import_mosaic_framing_json(self) -> None:
        default_dir = project_root() / "outputs"
        if self._mosaic_source_model is not None:
            default_dir = self._mosaic_source_model.json_path.parent
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入自由投影取景 JSON",
            str(default_dir),
            MOSAIC_FRAMING_JSON_FILTER,
        )
        if not file_path:
            return
        self.load_mosaic_framing_json(file_path)

    def load_mosaic_framing_json(self, file_path: str | Path) -> bool:
        json_path = Path(file_path).expanduser()
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("取景 JSON 根对象必须是对象。")
            self._apply_mosaic_framing_payload(payload)
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON 错误直接反馈到界面。
            QMessageBox.critical(self, "导入取景失败", str(exc))
            self.ui.statusbar.showMessage(f"导入取景失败: {exc}")
            return False
        self.ui.statusbar.showMessage(f"已导入自由投影取景: {json_path}")
        return True

    def _mosaic_payload_float(self, payload: dict[str, object], key: str, default: float) -> float:
        try:
            value = float(payload.get(key, default))
        except (TypeError, ValueError):
            value = float(default)
        return float(value) if np.isfinite(value) else float(default)

    def _observer_from_mosaic_framing_payload(self, payload: dict[str, object]) -> tuple[ObserverSettings, float]:
        time_text = str(payload.get("observation_time_utc") or "").strip()
        if not time_text:
            raise ValueError("取景 JSON 缺少 observer.observation_time_utc。")
        parsed_time = datetime.fromisoformat(time_text.replace("Z", "+00:00"))
        if parsed_time.tzinfo is None:
            parsed_time = parsed_time.replace(tzinfo=timezone.utc)
        observer = ObserverSettings(
            observation_time_utc=parsed_time.astimezone(timezone.utc),
            latitude_deg=self._mosaic_payload_float(payload, "latitude_deg", 0.0),
            longitude_deg=self._mosaic_payload_float(payload, "longitude_deg", 0.0),
            elevation_m=self._mosaic_payload_float(payload, "elevation_m", 0.0),
        )
        utc_offset_hours = self._mosaic_payload_float(payload, "utc_offset_hours", 0.0)
        return observer, utc_offset_hours

    def _apply_mosaic_framing_payload(self, payload: dict[str, object]) -> None:
        schema = str(payload.get("schema") or "")
        if schema != MOSAIC_FRAMING_SCHEMA:
            raise ValueError("这不是 MeteoAlign 自由投影取景 JSON。")

        projection_payload = payload.get("projection")
        if not isinstance(projection_payload, dict):
            raise ValueError("取景 JSON 缺少 projection 对象。")
        projection_model = str(projection_payload.get("model") or "")
        if projection_model not in MOSAIC_PROJECTION_MODELS:
            raise ValueError(f"取景 JSON 包含不支持的目标投影：{projection_model}")
        projection_index = MOSAIC_PROJECTION_MODELS.index(projection_model)
        was_blocked = self.ui.comboBoxMosaicProjection.blockSignals(True)
        self.ui.comboBoxMosaicProjection.setCurrentIndex(projection_index)
        self.ui.comboBoxMosaicProjection.blockSignals(was_blocked)
        self._update_mosaic_projection_controls()

        fov_value = self._mosaic_payload_float(projection_payload, "fov_deg", float(self.ui.doubleSpinBoxMosaicFov.value()))
        was_blocked = self.ui.doubleSpinBoxMosaicFov.blockSignals(True)
        self.ui.doubleSpinBoxMosaicFov.setValue(
            max(
                float(self.ui.doubleSpinBoxMosaicFov.minimum()),
                min(float(self.ui.doubleSpinBoxMosaicFov.maximum()), fov_value),
            )
        )
        self.ui.doubleSpinBoxMosaicFov.blockSignals(was_blocked)

        view_payload = payload.get("view")
        if not isinstance(view_payload, dict):
            raise ValueError("取景 JSON 缺少 view 对象。")
        self._mosaic_center_az_deg = self._mosaic_payload_float(view_payload, "center_az_deg", self._mosaic_center_az_deg) % 360.0
        self._mosaic_center_alt_deg = max(
            -90.0,
            min(90.0, self._mosaic_payload_float(view_payload, "center_alt_deg", self._mosaic_center_alt_deg)),
        )
        self._mosaic_roll_deg = self._mosaic_payload_float(view_payload, "roll_deg", self._mosaic_roll_deg)
        while self._mosaic_roll_deg > 180.0:
            self._mosaic_roll_deg -= 360.0
        while self._mosaic_roll_deg < -180.0:
            self._mosaic_roll_deg += 360.0
        self._set_mosaic_view_controls_from_state()

        observer_payload = payload.get("observer")
        if not isinstance(observer_payload, dict):
            raise ValueError("取景 JSON 缺少 observer 对象。")
        observer, utc_offset_hours = self._observer_from_mosaic_framing_payload(observer_payload)
        self._mosaic_framing_observer = observer
        self._mosaic_framing_utc_offset_hours = utc_offset_hours
        self._set_mosaic_observer_controls_from_values(observer, utc_offset_hours)
        self._set_mosaic_observer_controls_enabled(True)

        output_payload = payload.get("output")
        if not isinstance(output_payload, dict):
            raise ValueError("取景 JSON 缺少 output 对象。")
        width_px = int(round(self._mosaic_payload_float(output_payload, "boundary_width_px", 0.0)))
        height_px = int(round(self._mosaic_payload_float(output_payload, "boundary_height_px", 0.0)))
        if width_px <= 0 or height_px <= 0:
            raise ValueError("取景 JSON 的输出边界尺寸无效。")
        self._set_mosaic_output_resolution(width_px, height_px, None)
        crop_payload = output_payload.get("crop")
        self._set_mosaic_crop_controls(crop_payload if isinstance(crop_payload, dict) else {})
        map_payload = payload.get("reprojection_map")
        self._mosaic_reprojection_map_payload = map_payload if isinstance(map_payload, dict) else None
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=0)

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
        self._mosaic_interaction_source_texture_cache = None
        self._mosaic_interaction_coverage_cache = None
        self._mosaic_interaction_coverage_source_id = None
        self._invalidate_mosaic_reprojection_map()
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
        self._mosaic_framing_observer = source_model.observer
        self._mosaic_framing_utc_offset_hours = source_model.utc_offset_hours
        self._set_mosaic_observer_controls_from_values(source_model.observer, source_model.utc_offset_hours)

    def _set_mosaic_observer_controls_from_values(
        self,
        observer: ObserverSettings,
        utc_offset_hours: float,
    ) -> None:
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
            utc_offset_hours = self._bounded_mosaic_utc_offset(utc_offset_hours)
            self.ui.dateTimeEditMosaicObservation.setDateTime(
                self._mosaic_qdatetime_for_observer(observer, utc_offset_hours)
            )
            self.ui.doubleSpinBoxMosaicUtcOffset.setValue(utc_offset_hours)
            self.ui.doubleSpinBoxMosaicLatitude.setValue(float(observer.latitude_deg))
            self.ui.doubleSpinBoxMosaicLongitude.setValue(float(observer.longitude_deg))
            self.ui.doubleSpinBoxMosaicElevation.setValue(float(observer.elevation_m))
        finally:
            for control, was_blocked in zip(controls, previous_blocks, strict=True):
                control.blockSignals(was_blocked)

    def reset_mosaic_observer_from_json(self) -> None:
        self._invalidate_mosaic_reprojection_map()
        source_model = self._mosaic_source_model
        if source_model is None:
            observer = getattr(self, "_mosaic_framing_observer", None)
            if observer is None:
                return
            self._set_mosaic_observer_controls_from_values(
                observer,
                getattr(self, "_mosaic_framing_utc_offset_hours", 0.0),
            )
            self.schedule_mosaic_render(delay_ms=0)
            return
        self._set_mosaic_observer_controls_from_source_model(source_model)
        self.schedule_mosaic_render(delay_ms=0)

    def _mosaic_utc_offset_from_controls(self) -> float:
        if hasattr(self.ui, "doubleSpinBoxMosaicUtcOffset"):
            return float(self.ui.doubleSpinBoxMosaicUtcOffset.value())
        source_model = self._mosaic_source_model
        if source_model is not None:
            return float(source_model.utc_offset_hours)
        return float(getattr(self, "_mosaic_framing_utc_offset_hours", 0.0))

    def _mosaic_observer_from_control_values(self) -> ObserverSettings | None:
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return None
        try:
            local_dt = self.ui.dateTimeEditMosaicObservation.dateTime().toPyDateTime()
            offset = timezone(timedelta(hours=self._mosaic_utc_offset_from_controls()))
            aware_dt = local_dt.replace(tzinfo=offset)
            return ObserverSettings(
                observation_time_utc=aware_dt.astimezone(timezone.utc),
                latitude_deg=float(self.ui.doubleSpinBoxMosaicLatitude.value()),
                longitude_deg=float(self.ui.doubleSpinBoxMosaicLongitude.value()),
                elevation_m=float(self.ui.doubleSpinBoxMosaicElevation.value()),
            )
        except Exception:  # noqa: BLE001 - 控件读值失败时使用已有取景或源图观察者兜底。
            return None

    def _mosaic_observer_from_controls(self) -> ObserverSettings | None:
        fallback_observer = None
        if self._mosaic_source_model is not None:
            fallback_observer = self._mosaic_source_model.observer
        elif getattr(self, "_mosaic_framing_observer", None) is not None:
            fallback_observer = self._mosaic_framing_observer
        if fallback_observer is None:
            return None
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return fallback_observer
        return self._mosaic_observer_from_control_values() or fallback_observer

    def _mosaic_grid_precision_value(self) -> int:
        if not hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            return self._mosaic_default_grid_precision()
        return max(
            MOSAIC_GRID_MIN_PRECISION,
            min(MOSAIC_GRID_MAX_PRECISION, int(self.ui.spinBoxMosaicGridPrecision.value())),
        )

    def _mosaic_default_grid_precision(self) -> int:
        """从配置读取默认贴图网格精度。"""

        configured = getattr(self.ui_config, "mosaic_grid_precision_default", MOSAIC_COVERAGE_GRID_LONG_SIDE)
        try:
            value = int(configured)
        except (TypeError, ValueError):
            value = MOSAIC_COVERAGE_GRID_LONG_SIDE
        return max(MOSAIC_GRID_MIN_PRECISION, min(MOSAIC_GRID_MAX_PRECISION, value))

    def _mosaic_render_fps_limit(self) -> int:
        """从配置读取自由拼图预览的最高刷新率。"""

        configured = getattr(self.ui_config, "mosaic_render_fps_limit", self.MOSAIC_RENDER_FPS_LIMIT)
        try:
            value = int(configured)
        except (TypeError, ValueError):
            value = self.MOSAIC_RENDER_FPS_LIMIT
        return max(1, min(240, value))

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
            self._mosaic_interaction_coverage_cache = None
            self._mosaic_interaction_coverage_source_id = None
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
        self._invalidate_mosaic_reprojection_map()
        self._reset_mosaic_center_from_model()
        self.schedule_mosaic_render(delay_ms=0)

    def schedule_mosaic_render(self, *unused, delay_ms: int = 40) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self, "mosaic_render_timer"):
            return
        requested_delay = max(0, int(delay_ms))
        if hasattr(self, "_mosaic_render_clock") and self._mosaic_render_clock.isValid():
            elapsed_msecs = int(self._mosaic_render_clock.elapsed())
            since_last_render = elapsed_msecs - int(getattr(self, "_mosaic_last_render_msecs", -10_000))
            rate_limit_delay = max(0, int(getattr(self, "_mosaic_min_render_interval_ms", 33)) - since_last_render)
            requested_delay = max(requested_delay, rate_limit_delay)
        remaining_delay = self.mosaic_render_timer.remainingTime()
        if remaining_delay >= 0 and remaining_delay <= requested_delay:
            return
        self.mosaic_render_timer.start(requested_delay)

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
            return None, None, self._mosaic_observer_from_controls()
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

    def _mosaic_render_coverage_cache(self, cache: MosaicCoverageCache) -> MosaicCoverageCache:
        """交互拖动时使用降采样网格，静止时使用完整网格。"""

        if not bool(getattr(self, "_mosaic_interaction_active", False)):
            return cache
        source_id = id(cache)
        interaction_cache = getattr(self, "_mosaic_interaction_coverage_cache", None)
        if interaction_cache is None or getattr(self, "_mosaic_interaction_coverage_source_id", None) != source_id:
            interaction_cache = self._reduced_mosaic_coverage_cache(cache)
            self._mosaic_interaction_coverage_cache = interaction_cache
            self._mosaic_interaction_coverage_source_id = source_id
        return interaction_cache

    @staticmethod
    def _reduced_mosaic_coverage_cache(cache: MosaicCoverageCache) -> MosaicCoverageCache:
        """从完整覆盖网格抽取约一半行列，保留首尾边界点。"""

        row_indices = MosaicProjectionMixin._mosaic_interaction_grid_indices(cache.grid_rows)
        column_indices = MosaicProjectionMixin._mosaic_interaction_grid_indices(cache.grid_columns)
        if len(row_indices) == cache.grid_rows and len(column_indices) == cache.grid_columns:
            return cache
        grid_index = np.ix_(row_indices, column_indices)
        return MosaicCoverageCache(
            grid_rows=int(len(row_indices)),
            grid_columns=int(len(column_indices)),
            grid_x_px=np.asarray(cache.grid_x_px[grid_index], dtype=np.float64),
            grid_y_px=np.asarray(cache.grid_y_px[grid_index], dtype=np.float64),
            ra_deg=np.asarray(cache.ra_deg[grid_index], dtype=np.float64),
            dec_deg=np.asarray(cache.dec_deg[grid_index], dtype=np.float64),
            valid=np.asarray(cache.valid[grid_index], dtype=bool),
        )

    @staticmethod
    def _mosaic_interaction_grid_indices(length: int) -> np.ndarray:
        """生成交互渲染用的隔点索引。"""

        safe_length = max(1, int(length))
        if safe_length <= 3:
            return np.arange(safe_length, dtype=np.int64)
        indices = np.arange(0, safe_length, 2, dtype=np.int64)
        if int(indices[-1]) != safe_length - 1:
            indices = np.append(indices, safe_length - 1)
        return np.unique(indices).astype(np.int64)

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
        if hasattr(self, "mosaic_render_timer") and self.mosaic_render_timer.isActive():
            self.mosaic_render_timer.stop()
        if hasattr(self, "_mosaic_render_clock") and self._mosaic_render_clock.isValid():
            self._mosaic_last_render_msecs = int(self._mosaic_render_clock.elapsed())
        width, height = self._mosaic_render_size()
        source_model = self._mosaic_source_model
        effective_model, coverage_cache, observer = self._effective_mosaic_model_and_coverage()
        if observer is None:
            image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
            image.fill(QColor(0, 0, 0))
            self._paint_mosaic_crop_rect(image)
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
            if (
                source_model is not None
                and effective_model is not None
                and coverage_cache is not None
                and self._mosaic_overlay_enabled()
            ):
                render_cache = self._mosaic_render_coverage_cache(coverage_cache)
                if self._mosaic_overlay_mode() == MOSAIC_OVERLAY_MODE_SOURCE_IMAGE:
                    self._paint_mosaic_source_image(image, camera, view, render_cache, observer, source_model)
                else:
                    self._paint_mosaic_coverage(image, camera, view, render_cache, observer)
            self._paint_mosaic_crop_rect(image)
            self.mosaic_image_item.set_image(image)
            self.mosaic_scene.setSceneRect(0.0, 0.0, float(width), float(height))
            self.ui.mosaicProjectionView.resetTransform()
            self._update_mosaic_view_label()
        except Exception as exc:  # noqa: BLE001 - 预览渲染要把模型和投影错误反馈给用户。
            self.ui.statusbar.showMessage(f"自由投影预览渲染失败: {exc}")

    def _mosaic_crop_rect_for_preview(self, image_width_px: int, image_height_px: int) -> QRectF | None:
        output_width_px, output_height_px = self._mosaic_output_size()
        if output_width_px <= 0 or output_height_px <= 0 or image_width_px <= 0 or image_height_px <= 0:
            return None
        crop = self._mosaic_crop_rect_payload()
        if float(crop["width_px"]) <= 0.0 or float(crop["height_px"]) <= 0.0:
            return None
        scale_x = float(image_width_px) / float(output_width_px)
        scale_y = float(image_height_px) / float(output_height_px)
        return QRectF(
            float(crop["x_px"]) * scale_x,
            float(crop["y_px"]) * scale_y,
            float(crop["width_px"]) * scale_x,
            float(crop["height_px"]) * scale_y,
        )

    def _paint_mosaic_crop_rect(self, image: QImage) -> None:
        """在预览图上按输出边界比例绘制红色裁剪框。"""

        rect = self._mosaic_crop_rect_for_preview(image.width(), image.height())
        if rect is None:
            return
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(255, 32, 32, 230))
        pen.setWidthF(max(1.5, min(float(image.width()), float(image.height())) / 320.0))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        inset = pen.widthF() * 0.5
        painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
        painter.end()

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
        interaction = bool(getattr(self, "_mosaic_interaction_active", False))
        texture_long_side = self._mosaic_source_texture_long_side_px(source_model, interaction=interaction)
        existing_cache = self._mosaic_existing_texture_cache_for_render(interaction=interaction)
        texture = load_source_texture(
            source_model,
            existing_cache,
            max_long_side_px=texture_long_side,
        )
        if texture is None:
            self._show_mosaic_source_texture_error(source_model)
            return None
        if texture is not None:
            if interaction:
                self._mosaic_interaction_source_texture_cache = texture
            else:
                self._mosaic_source_texture_cache = texture
        return texture

    def _mosaic_existing_texture_cache_for_render(self, *, interaction: bool) -> MosaicSourceTextureCache | None:
        """按当前渲染质量选择可复用的贴图缓存。"""

        if interaction:
            return self._mosaic_interaction_source_texture_cache or self._mosaic_source_texture_cache
        return self._mosaic_source_texture_cache

    def _mosaic_source_texture_long_side_px(
        self,
        source_model: MosaicSourceModel,
        *,
        interaction: bool,
    ) -> int:
        """根据配置计算当前渲染应使用的贴图长边像素。"""

        original_long_side = max(1, int(max(source_model.image_width_px, source_model.image_height_px)))
        scale_percent = getattr(self.ui_config, "mosaic_texture_scale_percent", 25.0)
        max_long_side = getattr(self.ui_config, "mosaic_texture_max_long_side_px", MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX)
        try:
            scaled_long_side = int(round(original_long_side * float(scale_percent) / 100.0))
        except (TypeError, ValueError):
            scaled_long_side = int(round(original_long_side * 0.25))
        try:
            configured_max_long_side = int(max_long_side)
        except (TypeError, ValueError):
            configured_max_long_side = MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX
        texture_long_side = max(1, min(original_long_side, scaled_long_side, configured_max_long_side))
        if interaction:
            texture_long_side = max(1, int(round(texture_long_side * self.MOSAIC_INTERACTION_TEXTURE_SCALE)))
        return texture_long_side

    def _show_mosaic_source_texture_error(self, source_model: MosaicSourceModel) -> None:
        """把源图贴图加载失败原因反馈到状态栏。"""

        if source_model.source_image_path is None:
            self.ui.statusbar.showMessage("源图模型 JSON 未记录真实图像路径，无法显示原图。")
            return
        resolved = source_model.source_image_path
        try:
            resolved = resolved.expanduser().resolve()
        except OSError:
            pass
        if not resolved.exists():
            self.ui.statusbar.showMessage(f"源图不存在，无法显示原图: {resolved}")
        else:
            self.ui.statusbar.showMessage(f"源图读取失败，无法显示原图: {resolved}")

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
        self._invalidate_mosaic_reprojection_map()
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
        self._invalidate_mosaic_reprojection_map()
        self._mosaic_roll_deg = roll_after_drag(self._mosaic_roll_deg, dx, drag_sign=-1.0)
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _handle_mosaic_zoom_factor(self, zoom_factor: float) -> None:
        """处理缩放（滚轮或触控板）。"""
        self._invalidate_mosaic_reprojection_map()
        self._apply_mosaic_fov_zoom_factor(zoom_factor)
        self.schedule_mosaic_render(delay_ms=0)

    def _handle_mosaic_resize_event(self) -> None:
        """处理视图尺寸变化。"""
        self.schedule_mosaic_render()

    def _handle_mosaic_interaction_start_event(self) -> None:
        """交互开始时切换到低清贴图和低密度网格。"""

        if not bool(getattr(self, "_mosaic_interaction_active", False)):
            self._mosaic_interaction_active = True
            self.schedule_mosaic_render(delay_ms=0)

    def _handle_mosaic_interaction_end_event(self) -> None:
        """交互结束（鼠标释放）时触发最终高质量渲染。"""
        self._mosaic_interaction_active = False
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
