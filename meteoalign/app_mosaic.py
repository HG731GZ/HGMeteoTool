from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, QEvent, QPoint, QPointF, QTimer, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPolygonF
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QMessageBox

from .alignment import (
    SKY_KNOWN_PROJECTION_DISPLAY_NAMES,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .app_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT,
    SOURCE_MODEL_JSON_FILTER,
    TOUCHPAD_ZOOM_MAX_FACTOR,
    TOUCHPAD_ZOOM_MIN_FACTOR,
    TOUCHPAD_ZOOM_SENSITIVITY,
)
from .app_graphics_items import GraphicsImageItem
from .catalog import project_root
from .frame_astrometry import FrameAstrometricModel
from .image_preview import load_image_preview
from .texture_projection import (
    qimage_to_rgb_array,
    rgba_array_to_qimage,
    texture_projection_available,
    warp_grid_texture_to_rgba,
)
from .simulator import (
    CYLINDRICAL_LENS_MODELS,
    CameraSettings,
    ObserverSettings,
    ViewSettings,
    camera_basis_from_view,
    compute_altaz_from_radec,
    horizontal_fov_deg,
    local_vectors_from_altaz,
    project_horizontal_catalog,
    vertical_fov_deg,
    _camera_longitudes_from_altaz,
    _project_altaz_points,
)


MOSAIC_PROJECTION_MODELS = (
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
)
MOSAIC_RENDER_MIN_SIZE_PX = 128
MOSAIC_COVERAGE_GRID_LONG_SIDE = 54
MOSAIC_GRID_MIN_PRECISION = 12
MOSAIC_GRID_MAX_PRECISION = 180
MOSAIC_ZOOM_FACTOR = 1.18
MOSAIC_MODEL_REFIT_MIN_PAIRS = 4
MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX = 4096
MOSAIC_OVERLAY_MODE_COVERAGE = "coverage"
MOSAIC_OVERLAY_MODE_SOURCE_IMAGE = "source_image"
MOSAIC_OVERLAY_MODES = (
    MOSAIC_OVERLAY_MODE_COVERAGE,
    MOSAIC_OVERLAY_MODE_SOURCE_IMAGE,
)


@dataclass(frozen=True)
class MosaicModelFitData:
    ra_dec_points: np.ndarray
    pixel_points: np.ndarray
    point_weights: np.ndarray
    residual_anchor_mask: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.ra_dec_points.shape[0])


@dataclass(frozen=True)
class MosaicSourceModel:
    json_path: Path
    source_image_path: Path | None
    source_image_text: str
    model: FrameAstrometricModel
    observer: ObserverSettings
    utc_offset_hours: float
    fit_data: MosaicModelFitData | None
    image_width_px: int
    image_height_px: int
    pair_count: int
    rms_px: float


@dataclass(frozen=True)
class MosaicCoverageCache:
    grid_rows: int
    grid_columns: int
    grid_x_px: np.ndarray
    grid_y_px: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class MosaicSourceTextureCache:
    source_image_path: Path
    source_rgb: np.ndarray
    source_scale_x: float
    source_scale_y: float
    source_width_px: int
    source_height_px: int


def _parse_datetime_utc(value: object, field_name: str) -> datetime:
    if value is None:
        raise ValueError(f"模型 JSON 缺少时间字段：{field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"模型 JSON 时间字段为空：{field_name}")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"模型 JSON 字段 {field_name} 必须是对象。")
    return value


def _payload_float(value: object, field_name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"模型 JSON 缺少字段：{field_name}")
        return float(default)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"模型 JSON 字段 {field_name} 不是有效数值。")
    return result


def _payload_optional_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(result) if np.isfinite(result) else float(default)


def _payload_observer(payload: dict[str, object]) -> ObserverSettings:
    dynamic: dict[str, object] | None = None
    fit_metadata = payload.get("fit_metadata")
    if isinstance(fit_metadata, dict) and isinstance(fit_metadata.get("scene_observer_hint"), dict):
        dynamic = fit_metadata.get("scene_observer_hint")
    elif isinstance(payload.get("dynamic_sky_conversion"), dict):
        dynamic = payload.get("dynamic_sky_conversion")
    elif isinstance(payload.get("reference_payload"), dict):
        reference_payload = payload.get("reference_payload")
        observer_payload = reference_payload.get("observer") if isinstance(reference_payload, dict) else None
        if isinstance(observer_payload, dict):
            dynamic = observer_payload
    if dynamic is None:
        dynamic = {}

    time_value = None
    time_field = ""
    for candidate in (
        "frame_effective_time_utc",
        "observation_time_utc",
        "frame_nominal_time_utc",
        "capture_time_utc",
        "first_frame_nominal_time_utc",
    ):
        if dynamic.get(candidate):
            time_value = dynamic.get(candidate)
            time_field = f"dynamic_sky_conversion.{candidate}"
            break
    if time_value is None:
        source_image = payload.get("source_image")
        if isinstance(source_image, dict) and source_image.get("capture_time_utc"):
            time_value = source_image.get("capture_time_utc")
            time_field = "source_image.capture_time_utc"
    if time_value is None:
        time_value = datetime.now(timezone.utc).isoformat()
        time_field = "generated_default_scene_observer_time"
    return ObserverSettings(
        observation_time_utc=_parse_datetime_utc(time_value, time_field or "observation_time_utc"),
        latitude_deg=_payload_float(dynamic.get("latitude_deg"), "scene_observer.latitude_deg", default=0.0),
        longitude_deg=_payload_float(dynamic.get("longitude_deg"), "scene_observer.longitude_deg", default=0.0),
        elevation_m=_payload_float(dynamic.get("elevation_m"), "scene_observer.elevation_m", default=0.0),
    )


def _payload_utc_offset_hours(payload: dict[str, object]) -> float:
    fit_metadata = payload.get("fit_metadata")
    if isinstance(fit_metadata, dict) and isinstance(fit_metadata.get("scene_observer_hint"), dict):
        return _payload_optional_float(fit_metadata["scene_observer_hint"].get("utc_offset_hours"), 0.0)
    dynamic = payload.get("dynamic_sky_conversion")
    if isinstance(dynamic, dict):
        return _payload_optional_float(dynamic.get("utc_offset_hours"), 0.0)
    reference_payload = payload.get("reference_payload")
    if isinstance(reference_payload, dict):
        observer_payload = reference_payload.get("observer")
        if isinstance(observer_payload, dict):
            return _payload_optional_float(observer_payload.get("utc_offset_hours"), 0.0)
    return 0.0


def _expanded_polygon_points(xs: np.ndarray, ys: np.ndarray, padding_px: float) -> list[QPointF]:
    """让覆盖面片相互轻微重叠，避免逐格填充时出现内部接缝。"""

    points = np.column_stack((xs, ys)).astype(np.float64)
    if padding_px > 0.0:
        center = np.mean(points, axis=0)
        vectors = points - center
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-6
        if np.any(valid):
            points[valid] = center + vectors[valid] * ((lengths[valid] + padding_px) / lengths[valid])[:, None]
    return [QPointF(float(x_value), float(y_value)) for x_value, y_value in points]


def _load_mosaic_fit_data(payload: dict[str, object]) -> MosaicModelFitData | None:
    pair_payload = payload.get("fit_pairs")
    if not isinstance(pair_payload, list):
        pair_payload = payload.get("pairs")
    if not isinstance(pair_payload, list):
        return None

    ra_dec_points: list[tuple[float, float]] = []
    pixel_points: list[tuple[float, float]] = []
    point_weights: list[float] = []
    residual_anchor_mask: list[bool] = []
    for record in pair_payload:
        if not isinstance(record, dict):
            continue
        ra_deg = _payload_optional_float(record.get("ra_deg"), float("nan"))
        dec_deg = _payload_optional_float(record.get("dec_deg"), float("nan"))
        image_x_px = _payload_optional_float(record.get("image_x_px"), float("nan"))
        image_y_px = _payload_optional_float(record.get("image_y_px"), float("nan"))
        if not all(np.isfinite(value) for value in (ra_deg, dec_deg, image_x_px, image_y_px)):
            continue
        fit_weight = _payload_optional_float(record.get("fit_weight"), 1.0)
        constraint_mode = str(record.get("fit_constraint_mode") or "").strip()
        ra_dec_points.append((float(ra_deg), float(dec_deg)))
        pixel_points.append((float(image_x_px), float(image_y_px)))
        point_weights.append(float(fit_weight))
        residual_anchor_mask.append(constraint_mode != AUTO_MATCH_CONSTRAINT_SOFT)

    if len(ra_dec_points) < MOSAIC_MODEL_REFIT_MIN_PAIRS:
        return None
    return MosaicModelFitData(
        ra_dec_points=np.asarray(ra_dec_points, dtype=np.float64),
        pixel_points=np.asarray(pixel_points, dtype=np.float64),
        point_weights=np.asarray(point_weights, dtype=np.float64),
        residual_anchor_mask=np.asarray(residual_anchor_mask, dtype=bool),
    )


def _resolve_source_image_path(payload: dict[str, object], json_path: Path) -> tuple[Path | None, str]:
    source_image = payload.get("source_image")
    if not isinstance(source_image, dict):
        return None, "未记录源图路径"
    raw_path = str(source_image.get("path") or "").strip()
    if raw_path:
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = (json_path.parent / image_path).resolve()
        return image_path, image_path.name
    relative_path = str(source_image.get("relative_path") or "").strip()
    if relative_path:
        image_path = (json_path.parent / relative_path).resolve()
        return image_path, image_path.name
    file_name = str(source_image.get("file_name") or "").strip()
    if file_name:
        return None, file_name
    return None, "未记录源图路径"


def _load_mosaic_source_model(json_path: Path) -> MosaicSourceModel:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("模型 JSON 根对象必须是对象。")
    model = FrameAstrometricModel.from_json_payload(payload)
    observer = _payload_observer(payload)
    utc_offset_hours = _payload_utc_offset_hours(payload)
    fit_data = _load_mosaic_fit_data(payload)
    source_image_path, source_image_text = _resolve_source_image_path(payload, json_path)
    diagnostics_mapping = model.diagnostics
    fit_metadata = model.fit_metadata
    diagnostics_pair_count = int(
        diagnostics_mapping.get("pair_count", 0)
        or fit_metadata.get("control_point_count", 0)
        or 0
    )
    return MosaicSourceModel(
        json_path=json_path,
        source_image_path=source_image_path,
        source_image_text=source_image_text,
        model=model,
        observer=observer,
        utc_offset_hours=utc_offset_hours,
        fit_data=fit_data,
        image_width_px=int(model.image_width_px),
        image_height_px=int(model.image_height_px),
        pair_count=diagnostics_pair_count or (0 if fit_data is None else fit_data.pair_count),
        rms_px=float(diagnostics_mapping.get("rms_px", float("nan"))),
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
        self.ui.mosaicProjectionView.installEventFilter(self)
        self.ui.mosaicProjectionView.viewport().installEventFilter(self)
        self.ui.mosaicProjectionView.viewport().setMouseTracking(True)

        self.mosaic_render_timer = QTimer(self)
        self.mosaic_render_timer.setSingleShot(True)
        self.mosaic_render_timer.timeout.connect(self.render_mosaic_projection_now)
        self._mosaic_source_model: MosaicSourceModel | None = None
        self._mosaic_coverage_cache: MosaicCoverageCache | None = None
        self._mosaic_source_texture_cache: MosaicSourceTextureCache | None = None
        self._mosaic_last_drag_pos: QPoint | None = None
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

    def load_mosaic_model_json(self, file_path: str | Path) -> None:
        json_path = Path(file_path).expanduser()
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            source_model = _load_mosaic_source_model(json_path)
            coverage_cache = self._build_mosaic_coverage_cache(source_model.model)
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON 和模型错误直接反馈到界面。
            QMessageBox.critical(self, "导入源图模型失败", str(exc))
            self.ui.statusbar.showMessage(f"导入源图模型失败: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._mosaic_source_model = source_model
        self._mosaic_coverage_cache = coverage_cache
        self._mosaic_source_texture_cache = None
        self._set_mosaic_observer_controls_from_source_model(source_model)
        self._set_mosaic_observer_controls_enabled(True)
        self._set_mosaic_grid_controls_enabled(True)
        self._update_mosaic_grid_precision_tooltip()
        self._reset_mosaic_center_from_model()
        self._update_mosaic_model_labels()
        self.schedule_mosaic_render(delay_ms=0)
        self.ui.statusbar.showMessage(f"已导入源图模型: {source_model.json_path}")

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
        if width >= height:
            columns = precision
            rows = max(3, int(round(precision * height / float(max(width, 1)))))
        else:
            rows = precision
            columns = max(3, int(round(precision * width / float(max(height, 1)))))
        return rows, columns

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
        width = max(1, int(model.image_width_px))
        height = max(1, int(model.image_height_px))
        rows, columns = self._mosaic_grid_shape_for_size(width, height)
        x_values = np.linspace(0.0, max(width - 1, 0), columns, dtype=np.float64)
        y_values = np.linspace(0.0, max(height - 1, 0), rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        radec = model.pixel_to_sky_points(points)
        valid = np.all(np.isfinite(radec), axis=1)
        return MosaicCoverageCache(
            grid_rows=rows,
            grid_columns=columns,
            grid_x_px=grid_x.astype(np.float64),
            grid_y_px=grid_y.astype(np.float64),
            ra_deg=radec[:, 0].reshape((rows, columns)).astype(np.float64),
            dec_deg=radec[:, 1].reshape((rows, columns)).astype(np.float64),
            valid=valid.reshape((rows, columns)).astype(bool),
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
        flat_ra = cache.ra_deg.ravel()
        flat_dec = cache.dec_deg.ravel()
        valid_input = cache.valid.ravel() & np.isfinite(flat_ra) & np.isfinite(flat_dec)
        alt_deg = np.full(flat_ra.shape, np.nan, dtype=np.float64)
        az_deg = np.full(flat_ra.shape, np.nan, dtype=np.float64)
        if np.any(valid_input):
            alt_values, az_values = compute_altaz_from_radec(flat_ra[valid_input], flat_dec[valid_input], observer)
            alt_deg[valid_input] = alt_values
            az_deg[valid_input] = az_values
        valid = valid_input & np.isfinite(alt_deg) & np.isfinite(az_deg)
        return (
            alt_deg.reshape((cache.grid_rows, cache.grid_columns)).astype(np.float64),
            az_deg.reshape((cache.grid_rows, cache.grid_columns)).astype(np.float64),
            valid.reshape((cache.grid_rows, cache.grid_columns)).astype(bool),
        )

    def _reset_mosaic_center_from_model(self) -> None:
        source_model = self._mosaic_source_model
        effective_model, cache, _observer = self._effective_mosaic_model_and_coverage()
        if source_model is None or effective_model is None or cache is None:
            self._mosaic_center_az_deg = 0.0
            self._mosaic_center_alt_deg = 20.0
            self._mosaic_roll_deg = 0.0
            self._update_mosaic_view_label()
            return

        center_point = np.asarray(
            [[source_model.image_width_px * 0.5, source_model.image_height_px * 0.5]],
            dtype=np.float64,
        )
        center_radec = effective_model.pixel_to_sky_points(center_point)
        if np.all(np.isfinite(center_radec)):
            alt_deg, az_deg = compute_altaz_from_radec(center_radec[:, 0], center_radec[:, 1], _observer)
            self._mosaic_center_alt_deg = float(alt_deg[0])
            self._mosaic_center_az_deg = float(az_deg[0]) % 360.0
        else:
            cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, _observer)
            valid_cache = cache_valid & np.isfinite(cache_alt) & np.isfinite(cache_az)
            if np.any(valid_cache):
                vectors = local_vectors_from_altaz(cache_alt[valid_cache], cache_az[valid_cache])
                mean_vector = np.mean(vectors, axis=0)
                norm = float(np.linalg.norm(mean_vector))
                if norm > 1e-12:
                    mean_vector = mean_vector / norm
                    self._mosaic_center_alt_deg = float(np.rad2deg(np.arcsin(np.clip(mean_vector[2], -1.0, 1.0))))
                    self._mosaic_center_az_deg = float(np.rad2deg(np.arctan2(mean_vector[0], mean_vector[1]))) % 360.0
        self._mosaic_roll_deg = 0.0
        self._set_mosaic_fov_from_coverage(cache)
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()

    def _set_mosaic_fov_from_coverage(self, cache: MosaicCoverageCache | None = None) -> None:
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
        cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, observer)
        valid = cache_valid & np.isfinite(cache_alt) & np.isfinite(cache_az)
        if not np.any(valid):
            return
        center_vector = local_vectors_from_altaz(
            np.asarray([self._mosaic_center_alt_deg], dtype=np.float64),
            np.asarray([self._mosaic_center_az_deg], dtype=np.float64),
        )[0]
        vectors = local_vectors_from_altaz(cache_alt[valid], cache_az[valid])
        dots = np.sum(vectors * center_vector[None, :], axis=1)
        angles_deg = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
        if angles_deg.size == 0:
            return
        suggested_fov = float(np.percentile(angles_deg, 98.0) * 2.35)
        suggested_fov = max(25.0, min(self.ui.doubleSpinBoxMosaicFov.maximum(), suggested_fov))
        was_blocked = self.ui.doubleSpinBoxMosaicFov.blockSignals(True)
        self.ui.doubleSpinBoxMosaicFov.setValue(suggested_fov)
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

    def _mosaic_view_settings(self) -> ViewSettings:
        return ViewSettings(
            center_az_deg=self._mosaic_center_az_deg,
            center_alt_deg=self._mosaic_center_alt_deg,
            roll_deg=self._mosaic_roll_deg,
        )

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
            view = self._mosaic_view_settings()
            mag_limit = float(self.ui.doubleSpinBoxMosaicMagLimit.value())
            horizontal_catalog = self._get_horizontal_catalog(observer, mag_limit)
            horizontal_milky_way = self._get_horizontal_milky_way(observer)
            horizontal_solar_system = self._get_horizontal_solar_system(observer)
            star_map = project_horizontal_catalog(
                horizontal_catalog=horizontal_catalog,
                camera=camera,
                view=view,
                visible_mag_limit=mag_limit,
                horizontal_milky_way=horizontal_milky_way,
                horizontal_solar_system=horizontal_solar_system,
            )
            image = self.renderer.render(
                star_map,
                reference_stars=(),
                element_scale=1.0,
                draw_common_names=False,
                number_reference_stars=False,
                draw_background=True,
                draw_horizon_shadow=True,
                draw_grid=self.ui.checkBoxMosaicShowGrid.isChecked(),
                draw_solar_system_labels=True,
                draw_direction_labels=self.ui.checkBoxMosaicShowGrid.isChecked(),
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
        if cache is None:
            return
        cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, observer)
        basis = camera_basis_from_view(view)
        x_px, y_px, valid_projection = _project_altaz_points(
            cache_alt.ravel(),
            cache_az.ravel(),
            camera=camera,
            basis=basis,
        )
        screen_x = x_px.reshape((cache.grid_rows, cache.grid_columns))
        screen_y = y_px.reshape((cache.grid_rows, cache.grid_columns))
        valid = (
            valid_projection.reshape((cache.grid_rows, cache.grid_columns))
            & cache_valid
            & np.isfinite(screen_x)
            & np.isfinite(screen_y)
        )
        screen_longitudes = None
        if camera.lens_model in CYLINDRICAL_LENS_MODELS:
            screen_longitudes = _camera_longitudes_from_altaz(
                cache_alt.ravel(),
                cache_az.ravel(),
                basis,
            ).reshape((cache.grid_rows, cache.grid_columns))
        opacity = self._mosaic_overlay_opacity()
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
                xs = np.asarray(
                    [
                        screen_x[row, column],
                        screen_x[row, column + 1],
                        screen_x[row + 1, column + 1],
                        screen_x[row + 1, column],
                    ],
                    dtype=np.float64,
                )
                ys = np.asarray(
                    [
                        screen_y[row, column],
                        screen_y[row, column + 1],
                        screen_y[row + 1, column + 1],
                        screen_y[row + 1, column],
                    ],
                    dtype=np.float64,
                )
                if screen_longitudes is not None:
                    cell_longitudes = np.asarray(
                        [
                            screen_longitudes[row, column],
                            screen_longitudes[row, column + 1],
                            screen_longitudes[row + 1, column + 1],
                            screen_longitudes[row + 1, column],
                        ],
                        dtype=np.float64,
                    )
                    # 覆盖面片和银河面片一样，跨圆柱投影断点时不能直接闭合填充。
                    if (
                        not np.all(np.isfinite(cell_longitudes))
                        or np.any(np.abs(np.diff(np.concatenate((cell_longitudes, cell_longitudes[:1])))) > np.pi)
                    ):
                        continue
                bbox_area = max(float(np.max(xs) - np.min(xs)), 0.0) * max(float(np.max(ys) - np.min(ys)), 0.0)
                if bbox_area <= 0.05 or bbox_area > max_cell_bbox_area:
                    continue
                polygon = QPolygonF(_expanded_polygon_points(xs, ys, 0.75))
                painter.drawPolygon(polygon)

        painter.setPen(edge_color)
        painter.setBrush(Qt.NoBrush)
        self._paint_mosaic_coverage_boundary(painter, screen_x, screen_y, valid)
        painter.end()

    def _mosaic_source_texture(self, source_model: MosaicSourceModel) -> MosaicSourceTextureCache | None:
        image_path = source_model.source_image_path
        if image_path is None:
            self.ui.statusbar.showMessage("源图模型 JSON 未记录真实图像路径，无法显示原图。")
            return None
        try:
            resolved_path = image_path.expanduser().resolve()
        except OSError:
            resolved_path = image_path.expanduser()
        cache = self._mosaic_source_texture_cache
        if cache is not None and cache.source_image_path == resolved_path:
            return cache
        if not resolved_path.exists():
            self.ui.statusbar.showMessage(f"源图不存在，无法显示原图: {resolved_path}")
            return None
        try:
            preview = load_image_preview(resolved_path, max_long_side_px=MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX)
            source_rgb = qimage_to_rgb_array(preview.image)
        except Exception as exc:  # noqa: BLE001 - 预览渲染入口只在状态栏提示缺图或解码错误。
            self.ui.statusbar.showMessage(f"读取源图失败，无法显示原图: {exc}")
            return None
        texture_cache = MosaicSourceTextureCache(
            source_image_path=resolved_path,
            source_rgb=source_rgb.astype(np.uint8),
            source_scale_x=source_rgb.shape[1] / max(float(source_model.image_width_px), 1.0),
            source_scale_y=source_rgb.shape[0] / max(float(source_model.image_height_px), 1.0),
            source_width_px=int(source_model.image_width_px),
            source_height_px=int(source_model.image_height_px),
        )
        self._mosaic_source_texture_cache = texture_cache
        return texture_cache

    def _paint_mosaic_source_image(
        self,
        image: QImage,
        camera: CameraSettings,
        view: ViewSettings,
        cache: MosaicCoverageCache | None,
        observer: ObserverSettings,
        source_model: MosaicSourceModel,
    ) -> None:
        if cache is None or not texture_projection_available():
            return
        texture = self._mosaic_source_texture(source_model)
        if texture is None:
            return
        cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, observer)
        basis = camera_basis_from_view(view)
        x_px, y_px, valid_projection = _project_altaz_points(
            cache_alt.ravel(),
            cache_az.ravel(),
            camera=camera,
            basis=basis,
        )
        screen_x = x_px.reshape((cache.grid_rows, cache.grid_columns))
        screen_y = y_px.reshape((cache.grid_rows, cache.grid_columns))
        valid = (
            valid_projection.reshape((cache.grid_rows, cache.grid_columns))
            & cache_valid
            & np.isfinite(screen_x)
            & np.isfinite(screen_y)
        )
        screen_longitudes = None
        if camera.lens_model in CYLINDRICAL_LENS_MODELS:
            screen_longitudes = _camera_longitudes_from_altaz(
                cache_alt.ravel(),
                cache_az.ravel(),
                basis,
            ).reshape((cache.grid_rows, cache.grid_columns))
        rgba = np.zeros((image.height(), image.width(), 4), dtype=np.uint8)
        warp_grid_texture_to_rgba(
            rgba,
            source_rgb=texture.source_rgb,
            source_grid_x_px=cache.grid_x_px,
            source_grid_y_px=cache.grid_y_px,
            source_scale_x=texture.source_scale_x,
            source_scale_y=texture.source_scale_y,
            screen_x_px=screen_x,
            screen_y_px=screen_y,
            valid_points=valid,
            opacity=self._mosaic_overlay_opacity(),
            screen_longitudes_rad=screen_longitudes,
        )
        overlay = rgba_array_to_qimage(rgba)
        painter = QPainter(image)
        painter.drawImage(0, 0, overlay)
        painter.end()

    def _paint_mosaic_coverage_boundary(
        self,
        painter: QPainter,
        screen_x: np.ndarray,
        screen_y: np.ndarray,
        valid: np.ndarray,
    ) -> None:
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

    def _handle_mosaic_event_filter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if not hasattr(self.ui, "mosaicProjectionView"):
            return False
        if watched in (
            getattr(self.ui, "labelMosaicModelPath", None),
            getattr(self.ui, "labelMosaicSourceImage", None),
            getattr(self.ui, "labelMosaicModelInfo", None),
            getattr(self.ui, "labelMosaicViewInfo", None),
        ):
            if event.type() in (QEvent.Resize, QEvent.Show):
                QTimer.singleShot(0, lambda label=watched: self._refresh_elided_label(label))
            return False
        if watched in (self.ui.mosaicProjectionView, self.ui.mosaicProjectionView.viewport()):
            if event.type() == QEvent.NativeGesture and self._handle_mosaic_native_fov_zoom(event):
                return True
        if watched is not self.ui.mosaicProjectionView.viewport():
            return False
        if event.type() == QEvent.Resize:
            self.schedule_mosaic_render()
            return False
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._mosaic_last_drag_pos = event.pos()
            self.ui.mosaicProjectionView.viewport().setCursor(Qt.ClosedHandCursor)
            return True
        if event.type() == QEvent.MouseMove and self._mosaic_last_drag_pos is not None:
            dx = event.pos().x() - self._mosaic_last_drag_pos.x()
            dy = event.pos().y() - self._mosaic_last_drag_pos.y()
            self._mosaic_last_drag_pos = event.pos()
            if event.modifiers() & Qt.ControlModifier:
                self._apply_mosaic_roll_drag_delta(dx)
            else:
                self._apply_mosaic_center_drag_delta(dx, dy)
            return True
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self._mosaic_last_drag_pos = None
            self.ui.mosaicProjectionView.viewport().unsetCursor()
            self.render_mosaic_projection_now()
            return True
        if event.type() == QEvent.Wheel:
            if not self._mosaic_wheel_zoom_enabled():
                return False
            self._apply_mosaic_fov_wheel(event.angleDelta().y())
            return True
        return False

    def _apply_mosaic_center_drag_delta(self, dx: int, dy: int) -> None:
        width, height = self._mosaic_render_size()
        camera = self._mosaic_camera_for_render(width, height)
        az_degrees_per_pixel = max(horizontal_fov_deg(camera) / max(width, 1), 0.005)
        alt_degrees_per_pixel = max(vertical_fov_deg(camera) / max(height, 1), 0.005)
        self._mosaic_center_az_deg = (self._mosaic_center_az_deg - dx * az_degrees_per_pixel) % 360.0
        self._mosaic_center_alt_deg = max(-90.0, min(90.0, self._mosaic_center_alt_deg + dy * alt_degrees_per_pixel))
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _apply_mosaic_roll_drag_delta(self, dx: int) -> None:
        self._mosaic_roll_deg -= dx * 0.25
        while self._mosaic_roll_deg > 180.0:
            self._mosaic_roll_deg -= 360.0
        while self._mosaic_roll_deg < -180.0:
            self._mosaic_roll_deg += 360.0
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _apply_mosaic_fov_wheel(self, wheel_delta: int) -> None:
        if wheel_delta == 0:
            return
        zoom_factor = MOSAIC_ZOOM_FACTOR if wheel_delta > 0 else 1.0 / MOSAIC_ZOOM_FACTOR
        self._apply_mosaic_fov_zoom_factor(zoom_factor)
        self.render_mosaic_projection_now()

    def _apply_mosaic_fov_zoom_factor(self, zoom_factor: float) -> None:
        if not np.isfinite(zoom_factor) or zoom_factor <= 0.0 or abs(zoom_factor - 1.0) <= 1e-4:
            return
        current = float(self.ui.doubleSpinBoxMosaicFov.value())
        target = current / zoom_factor
        target = max(self.ui.doubleSpinBoxMosaicFov.minimum(), min(self.ui.doubleSpinBoxMosaicFov.maximum(), target))
        self.ui.doubleSpinBoxMosaicFov.setValue(target)

    def _mosaic_wheel_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "wheel_zoom_enabled", True))

    def _mosaic_touchpad_pinch_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "touchpad_pinch_zoom_enabled", True))

    def _mosaic_native_zoom_value(self, event) -> float:  # type: ignore[no-untyped-def]
        if not self._mosaic_touchpad_pinch_zoom_enabled():
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

    def _handle_mosaic_native_fov_zoom(self, event) -> bool:  # type: ignore[no-untyped-def]
        value = self._mosaic_native_zoom_value(event)
        if value == 0.0:
            return False
        zoom_factor = math.exp(value * TOUCHPAD_ZOOM_SENSITIVITY)
        zoom_factor = max(TOUCHPAD_ZOOM_MIN_FACTOR, min(TOUCHPAD_ZOOM_MAX_FACTOR, zoom_factor))
        self._apply_mosaic_fov_zoom_factor(zoom_factor)
        self.render_mosaic_projection_now()
        return True

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
