from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, QTimer, Qt
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
from .app_constants import SOURCE_MODEL_JSON_FILTER
from .app_graphics_items import GraphicsImageItem
from .catalog import project_root
from .fixed_camera_model import FixedCameraModel
from .simulator import (
    CYLINDRICAL_LENS_MODELS,
    CameraSettings,
    ObserverSettings,
    ViewSettings,
    camera_basis_from_view,
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
MOSAIC_ZOOM_FACTOR = 1.18


@dataclass(frozen=True)
class MosaicSourceModel:
    json_path: Path
    source_image_path: Path | None
    source_image_text: str
    model: FixedCameraModel
    observer: ObserverSettings
    image_width_px: int
    image_height_px: int
    pair_count: int
    rms_px: float


@dataclass(frozen=True)
class MosaicCoverageCache:
    grid_rows: int
    grid_columns: int
    alt_deg: np.ndarray
    az_deg: np.ndarray
    valid: np.ndarray


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


def _payload_observer(payload: dict[str, object]) -> ObserverSettings:
    dynamic = _payload_mapping(payload.get("dynamic_sky_conversion"), "dynamic_sky_conversion")
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
    return ObserverSettings(
        observation_time_utc=_parse_datetime_utc(time_value, time_field or "observation_time_utc"),
        latitude_deg=_payload_float(dynamic.get("latitude_deg"), "dynamic_sky_conversion.latitude_deg"),
        longitude_deg=_payload_float(dynamic.get("longitude_deg"), "dynamic_sky_conversion.longitude_deg"),
        elevation_m=_payload_float(dynamic.get("elevation_m"), "dynamic_sky_conversion.elevation_m", default=0.0),
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
    fixed_payload = payload.get("fixed_camera_model")
    if fixed_payload is None and payload.get("kind") == "fixed_camera_enu_model":
        fixed_payload = payload
    if not isinstance(fixed_payload, dict):
        raise ValueError("当前仅支持包含 fixed_camera_model 的固定相机模型 JSON，请重新导出当前版本的映射 JSON。")

    model = FixedCameraModel.from_json_payload(fixed_payload)
    observer = _payload_observer(payload)
    source_image_path, source_image_text = _resolve_source_image_path(payload, json_path)
    diagnostics = fixed_payload.get("diagnostics")
    diagnostics_mapping = diagnostics if isinstance(diagnostics, dict) else {}
    return MosaicSourceModel(
        json_path=json_path,
        source_image_path=source_image_path,
        source_image_text=source_image_text,
        model=model,
        observer=observer,
        image_width_px=int(model.image_width_px),
        image_height_px=int(model.image_height_px),
        pair_count=int(diagnostics_mapping.get("fit_pair_count", 0) or 0),
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
        self.ui.mosaicProjectionView.viewport().installEventFilter(self)
        self.ui.mosaicProjectionView.viewport().setMouseTracking(True)

        self.mosaic_render_timer = QTimer(self)
        self.mosaic_render_timer.setSingleShot(True)
        self.mosaic_render_timer.timeout.connect(self.render_mosaic_projection_now)
        self._mosaic_source_model: MosaicSourceModel | None = None
        self._mosaic_coverage_cache: MosaicCoverageCache | None = None
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
        self.ui.doubleSpinBoxMosaicMagLimit.setValue(6.5)
        self.ui.doubleSpinBoxMosaicCoverageOpacity.setValue(45.0)
        self._update_mosaic_projection_controls()
        self._update_mosaic_model_labels()
        self._update_mosaic_view_label()

    def _connect_mosaic_projection_inputs(self) -> None:
        if not hasattr(self.ui, "pushButtonImportMosaicModel"):
            return
        self.ui.pushButtonImportMosaicModel.clicked.connect(self.import_mosaic_model_json)
        self.ui.comboBoxMosaicProjection.currentIndexChanged.connect(self._handle_mosaic_projection_changed)
        self.ui.doubleSpinBoxMosaicFov.valueChanged.connect(self.schedule_mosaic_render)
        self.ui.doubleSpinBoxMosaicMagLimit.valueChanged.connect(self.schedule_mosaic_render)
        self.ui.checkBoxMosaicShowGrid.toggled.connect(self.schedule_mosaic_render)
        self.ui.checkBoxMosaicShowCoverage.toggled.connect(self.schedule_mosaic_render)
        self.ui.doubleSpinBoxMosaicCoverageOpacity.valueChanged.connect(self.schedule_mosaic_render)
        self.ui.pushButtonResetMosaicView.clicked.connect(self.reset_mosaic_projection_view)
        self.ui.labelMosaicModelPath.installEventFilter(self)
        self.ui.labelMosaicSourceImage.installEventFilter(self)
        self.ui.labelMosaicModelInfo.installEventFilter(self)
        self.ui.labelMosaicViewInfo.installEventFilter(self)

    def _handle_mosaic_projection_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_mosaic_projection_controls()
        self.schedule_mosaic_render()

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
        self._reset_mosaic_center_from_model()
        self._update_mosaic_model_labels()
        self.schedule_mosaic_render(delay_ms=0)
        self.ui.statusbar.showMessage(f"已导入源图模型: {source_model.json_path}")

    def _build_mosaic_coverage_cache(self, model: FixedCameraModel) -> MosaicCoverageCache:
        width = max(1, int(model.image_width_px))
        height = max(1, int(model.image_height_px))
        if width >= height:
            columns = MOSAIC_COVERAGE_GRID_LONG_SIDE
            rows = max(3, int(round(MOSAIC_COVERAGE_GRID_LONG_SIDE * height / float(width))))
        else:
            rows = MOSAIC_COVERAGE_GRID_LONG_SIDE
            columns = max(3, int(round(MOSAIC_COVERAGE_GRID_LONG_SIDE * width / float(height))))
        x_values = np.linspace(0.0, max(width - 1, 0), columns, dtype=np.float64)
        y_values = np.linspace(0.0, max(height - 1, 0), rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        alt_deg, az_deg, valid = model.pixel_to_altaz_points(points)
        valid = valid & np.isfinite(alt_deg) & np.isfinite(az_deg)
        return MosaicCoverageCache(
            grid_rows=rows,
            grid_columns=columns,
            alt_deg=alt_deg.reshape((rows, columns)).astype(np.float64),
            az_deg=az_deg.reshape((rows, columns)).astype(np.float64),
            valid=valid.reshape((rows, columns)).astype(bool),
        )

    def _reset_mosaic_center_from_model(self) -> None:
        source_model = self._mosaic_source_model
        cache = self._mosaic_coverage_cache
        if source_model is None or cache is None:
            self._mosaic_center_az_deg = 0.0
            self._mosaic_center_alt_deg = 20.0
            self._mosaic_roll_deg = 0.0
            self._update_mosaic_view_label()
            return

        center_point = np.asarray(
            [[source_model.image_width_px * 0.5, source_model.image_height_px * 0.5]],
            dtype=np.float64,
        )
        alt_deg, az_deg, valid = source_model.model.pixel_to_altaz_points(center_point)
        if bool(valid[0]) and np.isfinite(alt_deg[0]) and np.isfinite(az_deg[0]):
            self._mosaic_center_alt_deg = float(alt_deg[0])
            self._mosaic_center_az_deg = float(az_deg[0]) % 360.0
        else:
            valid_cache = cache.valid & np.isfinite(cache.alt_deg) & np.isfinite(cache.az_deg)
            if np.any(valid_cache):
                vectors = local_vectors_from_altaz(cache.alt_deg[valid_cache], cache.az_deg[valid_cache])
                mean_vector = np.mean(vectors, axis=0)
                norm = float(np.linalg.norm(mean_vector))
                if norm > 1e-12:
                    mean_vector = mean_vector / norm
                    self._mosaic_center_alt_deg = float(np.rad2deg(np.arcsin(np.clip(mean_vector[2], -1.0, 1.0))))
                    self._mosaic_center_az_deg = float(np.rad2deg(np.arctan2(mean_vector[0], mean_vector[1]))) % 360.0
        self._mosaic_roll_deg = 0.0
        self._set_mosaic_fov_from_coverage()
        self._update_mosaic_view_label()

    def _set_mosaic_fov_from_coverage(self) -> None:
        cache = self._mosaic_coverage_cache
        if cache is None:
            return
        valid = cache.valid & np.isfinite(cache.alt_deg) & np.isfinite(cache.az_deg)
        if not np.any(valid):
            return
        center_vector = local_vectors_from_altaz(
            np.asarray([self._mosaic_center_alt_deg], dtype=np.float64),
            np.asarray([self._mosaic_center_az_deg], dtype=np.float64),
        )[0]
        vectors = local_vectors_from_altaz(cache.alt_deg[valid], cache.az_deg[valid])
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

    def render_mosaic_projection_now(self) -> None:
        if not hasattr(self.ui, "mosaicProjectionView"):
            return
        width, height = self._mosaic_render_size()
        source_model = self._mosaic_source_model
        if source_model is None:
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
            horizontal_catalog = self._get_horizontal_catalog(source_model.observer, mag_limit)
            horizontal_milky_way = self._get_horizontal_milky_way(source_model.observer)
            horizontal_solar_system = self._get_horizontal_solar_system(source_model.observer)
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
            if self.ui.checkBoxMosaicShowCoverage.isChecked():
                self._paint_mosaic_coverage(image, camera, view)
            self.mosaic_image_item.set_image(image)
            self.mosaic_scene.setSceneRect(0.0, 0.0, float(width), float(height))
            self.ui.mosaicProjectionView.resetTransform()
            self._update_mosaic_view_label()
        except Exception as exc:  # noqa: BLE001 - 预览渲染要把模型和投影错误反馈给用户。
            self.ui.statusbar.showMessage(f"自由投影预览渲染失败: {exc}")

    def _paint_mosaic_coverage(self, image: QImage, camera: CameraSettings, view: ViewSettings) -> None:
        cache = self._mosaic_coverage_cache
        if cache is None:
            return
        basis = camera_basis_from_view(view)
        x_px, y_px, valid_projection = _project_altaz_points(
            cache.alt_deg.ravel(),
            cache.az_deg.ravel(),
            camera=camera,
            basis=basis,
        )
        screen_x = x_px.reshape((cache.grid_rows, cache.grid_columns))
        screen_y = y_px.reshape((cache.grid_rows, cache.grid_columns))
        valid = (
            valid_projection.reshape((cache.grid_rows, cache.grid_columns))
            & cache.valid
            & np.isfinite(screen_x)
            & np.isfinite(screen_y)
        )
        screen_longitudes = None
        if camera.lens_model in CYLINDRICAL_LENS_MODELS:
            screen_longitudes = _camera_longitudes_from_altaz(
                cache.alt_deg.ravel(),
                cache.az_deg.ravel(),
                basis,
            ).reshape((cache.grid_rows, cache.grid_columns))
        opacity = max(0.0, min(1.0, float(self.ui.doubleSpinBoxMosaicCoverageOpacity.value()) / 100.0))
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
                polygon = QPolygonF()
                for x_value, y_value in zip(xs, ys):
                    polygon.append(QPointF(float(x_value), float(y_value)))
                painter.drawPolygon(polygon)

        painter.setPen(edge_color)
        painter.setBrush(Qt.NoBrush)
        self._paint_mosaic_coverage_boundary(painter, screen_x, screen_y, valid)
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
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _apply_mosaic_roll_drag_delta(self, dx: int) -> None:
        self._mosaic_roll_deg += dx * 0.25
        while self._mosaic_roll_deg > 180.0:
            self._mosaic_roll_deg -= 360.0
        while self._mosaic_roll_deg < -180.0:
            self._mosaic_roll_deg += 360.0
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _apply_mosaic_fov_wheel(self, wheel_delta: int) -> None:
        if wheel_delta == 0:
            return
        current = float(self.ui.doubleSpinBoxMosaicFov.value())
        factor = 1.0 / MOSAIC_ZOOM_FACTOR if wheel_delta > 0 else MOSAIC_ZOOM_FACTOR
        target = max(self.ui.doubleSpinBoxMosaicFov.minimum(), min(self.ui.doubleSpinBoxMosaicFov.maximum(), current * factor))
        self.ui.doubleSpinBoxMosaicFov.setValue(target)
        self.render_mosaic_projection_now()

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
