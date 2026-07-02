from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PyQt5.QtCore import QDateTime, QEvent, QPoint, QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QMainWindow, QMessageBox

from .catalog_download import ensure_catalogs_ready_or_handle
from .catalog import load_default_catalog, project_root
from .reference import build_reference_payload, save_reference_outputs
from .renderer import StarMapRenderer
from .simulator import (
    CameraSettings,
    FISHEYE_EQUIDISTANT,
    FISHEYE_EQUISOLID,
    FISHEYE_ORTHOGRAPHIC,
    FISHEYE_STEREOGRAPHIC,
    HorizontalStarCatalog,
    ObserverSettings,
    ProjectedStarMap,
    RECTILINEAR_LENS_MODEL,
    ViewSettings,
    compute_horizontal_catalog,
    horizontal_fov_deg,
    project_horizontal_catalog,
    select_reference_stars,
    vertical_fov_deg,
)
from .ui.ui_main_window import Ui_MainWindow


LENS_MODELS = (
    RECTILINEAR_LENS_MODEL,
    FISHEYE_EQUIDISTANT,
    FISHEYE_EQUISOLID,
    FISHEYE_STEREOGRAPHIC,
    FISHEYE_ORTHOGRAPHIC,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.catalog = load_default_catalog(mag_limit=None)
        self.renderer = StarMapRenderer()
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.ui.starMapView.setScene(self.scene)
        self.ui.starMapView.viewport().installEventFilter(self)

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render_now)
        self.drag_start: QPoint | None = None
        self.last_drag_pos: QPoint | None = None
        self._horizontal_cache_key: tuple[object, ...] | None = None
        self._horizontal_cache: HorizontalStarCatalog | None = None
        self._last_render_size: tuple[int, int] | None = None

        self._init_defaults()
        self._connect_inputs()
        self.schedule_render(delay_ms=0)

    def _init_defaults(self) -> None:
        self.ui.dateTimeEditObservation.setDateTime(QDateTime.currentDateTime())
        utc_offset = datetime.now().astimezone().utcoffset()
        if utc_offset is None:
            utc_offset = timedelta(hours=8)
        self.ui.doubleSpinBoxUtcOffset.setValue(utc_offset.total_seconds() / 3600.0)
        self.ui.doubleSpinBoxLatitude.setValue(40.0)
        self.ui.doubleSpinBoxLongitude.setValue(116.0)
        self.ui.doubleSpinBoxElevation.setValue(50.0)
        self.ui.doubleSpinBoxSensorWidth.setValue(36.0)
        self.ui.doubleSpinBoxSensorHeight.setValue(24.0)
        self.ui.spinBoxImageWidth.setValue(1920)
        self.ui.spinBoxImageHeight.setValue(1280)
        self.ui.doubleSpinBoxFocalLength.setValue(24.0)
        self.ui.comboBoxLensModel.setCurrentIndex(0)
        self.ui.doubleSpinBoxFisheyeFov.setValue(180.0)
        self.ui.doubleSpinBoxMagLimit.setValue(6.5)
        self.ui.horizontalSliderCommonNameLimit.setValue(10)
        self._update_common_name_limit_label()
        self.ui.doubleSpinBoxAz.setValue(0.0)
        self.ui.doubleSpinBoxAlt.setValue(20.0)
        self.ui.doubleSpinBoxRoll.setValue(0.0)
        self.ui.spinBoxReferenceStarCount.setValue(12)
        self._update_lens_model_controls()

    def _connect_inputs(self) -> None:
        widgets = (
            self.ui.dateTimeEditObservation,
            self.ui.doubleSpinBoxUtcOffset,
            self.ui.doubleSpinBoxLatitude,
            self.ui.doubleSpinBoxLongitude,
            self.ui.doubleSpinBoxElevation,
            self.ui.doubleSpinBoxSensorWidth,
            self.ui.doubleSpinBoxSensorHeight,
            self.ui.spinBoxImageWidth,
            self.ui.spinBoxImageHeight,
            self.ui.doubleSpinBoxFocalLength,
            self.ui.doubleSpinBoxFisheyeFov,
            self.ui.doubleSpinBoxMagLimit,
            self.ui.horizontalSliderCommonNameLimit,
            self.ui.doubleSpinBoxAz,
            self.ui.doubleSpinBoxAlt,
            self.ui.doubleSpinBoxRoll,
        )
        for widget in widgets:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.schedule_render)
            elif hasattr(widget, "dateTimeChanged"):
                widget.dateTimeChanged.connect(self.schedule_render)
        self.ui.comboBoxLensModel.currentIndexChanged.connect(self._handle_lens_model_changed)
        self.ui.horizontalSliderCommonNameLimit.valueChanged.connect(self._update_common_name_limit_label)
        self.ui.pushButtonRender.clicked.connect(lambda: self.render_now())
        self.ui.pushButtonExportReference.clicked.connect(self.export_reference_map)

    def schedule_render(self, *unused, delay_ms: int = 120) -> None:  # type: ignore[no-untyped-def]
        self.render_timer.start(delay_ms)

    def _common_name_mag_limit(self) -> float:
        return self.ui.horizontalSliderCommonNameLimit.value() / 10.0

    def _update_common_name_limit_label(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self.ui.labelCommonNameLimitValue.setText(f"{self._common_name_mag_limit():.1f} mag")

    def _lens_model(self) -> str:
        index = self.ui.comboBoxLensModel.currentIndex()
        if index < 0 or index >= len(LENS_MODELS):
            return RECTILINEAR_LENS_MODEL
        return LENS_MODELS[index]

    def _update_lens_model_controls(self) -> None:
        lens_model = self._lens_model()
        is_fisheye = lens_model != RECTILINEAR_LENS_MODEL
        if lens_model == FISHEYE_STEREOGRAPHIC:
            max_fov = 179.0
        elif lens_model == FISHEYE_ORTHOGRAPHIC:
            max_fov = 180.0
        else:
            max_fov = 300.0
        self.ui.doubleSpinBoxFisheyeFov.setMaximum(max_fov)
        if self.ui.doubleSpinBoxFisheyeFov.value() > max_fov:
            self.ui.doubleSpinBoxFisheyeFov.setValue(max_fov)
        self.ui.labelFisheyeFov.setEnabled(is_fisheye)
        self.ui.doubleSpinBoxFisheyeFov.setEnabled(is_fisheye)

    def _handle_lens_model_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_lens_model_controls()
        self.schedule_render()

    def _observer_settings(self) -> ObserverSettings:
        local_dt = self.ui.dateTimeEditObservation.dateTime().toPyDateTime()
        offset = timezone(timedelta(hours=self.ui.doubleSpinBoxUtcOffset.value()))
        aware_dt = local_dt.replace(tzinfo=offset)
        return ObserverSettings(
            observation_time_utc=aware_dt.astimezone(timezone.utc),
            latitude_deg=self.ui.doubleSpinBoxLatitude.value(),
            longitude_deg=self.ui.doubleSpinBoxLongitude.value(),
            elevation_m=self.ui.doubleSpinBoxElevation.value(),
        )

    def _camera_settings(self) -> CameraSettings:
        return CameraSettings(
            sensor_width_mm=self.ui.doubleSpinBoxSensorWidth.value(),
            sensor_height_mm=self.ui.doubleSpinBoxSensorHeight.value(),
            image_width_px=self.ui.spinBoxImageWidth.value(),
            image_height_px=self.ui.spinBoxImageHeight.value(),
            focal_length_mm=self.ui.doubleSpinBoxFocalLength.value(),
            lens_model=self._lens_model(),
            fisheye_fov_deg=self.ui.doubleSpinBoxFisheyeFov.value(),
        )

    def _view_settings(self) -> ViewSettings:
        return ViewSettings(
            center_az_deg=self.ui.doubleSpinBoxAz.value(),
            center_alt_deg=self.ui.doubleSpinBoxAlt.value(),
            roll_deg=self.ui.doubleSpinBoxRoll.value(),
        )

    def _get_horizontal_catalog(self, observer: ObserverSettings, mag_limit: float) -> HorizontalStarCatalog:
        cache_key = (
            int(observer.observation_time_utc.timestamp()),
            round(observer.latitude_deg, 8),
            round(observer.longitude_deg, 8),
            round(observer.elevation_m, 3),
            round(mag_limit, 3),
        )
        if self._horizontal_cache_key != cache_key or self._horizontal_cache is None:
            self._horizontal_cache = compute_horizontal_catalog(
                catalog=self.catalog,
                observer=observer,
                visible_mag_limit=mag_limit,
            )
            self._horizontal_cache_key = cache_key
        return self._horizontal_cache

    def _build_projected_star_map(self) -> tuple[ObserverSettings, CameraSettings, ViewSettings, float, ProjectedStarMap]:
        observer = self._observer_settings()
        camera = self._camera_settings()
        view = self._view_settings()
        mag_limit = self.ui.doubleSpinBoxMagLimit.value()
        horizontal_catalog = self._get_horizontal_catalog(observer, mag_limit)
        star_map = project_horizontal_catalog(
            horizontal_catalog=horizontal_catalog,
            camera=camera,
            view=view,
            visible_mag_limit=mag_limit,
        )
        return observer, camera, view, mag_limit, star_map

    def _display_star_map_image(self, star_map: ProjectedStarMap, image) -> None:  # type: ignore[no-untyped-def]
        render_size = (star_map.width, star_map.height)
        should_fit = self._last_render_size != render_size or self.pixmap_item.pixmap().isNull()
        self.pixmap_item.setPos(0, 0)
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.scene.setSceneRect(0, 0, star_map.width, star_map.height)
        self._last_render_size = render_size
        if should_fit:
            self.fit_star_map()

    def render_now(self) -> None:
        try:
            _observer, _camera, view, _mag_limit, star_map = self._build_projected_star_map()
            common_name_mag_limit = self._common_name_mag_limit()
            image = self.renderer.render(star_map, common_name_mag_limit=common_name_mag_limit)
            self._display_star_map_image(star_map, image)
            self.ui.statusbar.showMessage(
                "星表: {catalog_count}  视野内: {visible_count}  地平线上: {above_count}  "
                "俗名: <= {name_limit:.1f} mag  镜头: {lens_name}  Az: {az:.2f} deg  Alt: {alt:.2f} deg".format(
                    catalog_count=star_map.catalog_count,
                    visible_count=len(star_map),
                    above_count=star_map.above_horizon_count,
                    name_limit=common_name_mag_limit,
                    lens_name=self.ui.comboBoxLensModel.currentText(),
                    az=view.center_az_deg,
                    alt=view.center_alt_deg,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 界面层需要把可恢复输入错误显示出来。
            self.ui.statusbar.showMessage(f"渲染失败: {exc}")

    def _next_reference_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return project_root() / "outputs" / f"reference_{timestamp}"

    def export_reference_map(self) -> None:
        try:
            observer, camera, view, mag_limit, star_map = self._build_projected_star_map()
            reference_stars = select_reference_stars(
                star_map=star_map,
                max_count=self.ui.spinBoxReferenceStarCount.value(),
            )
            if not reference_stars:
                QMessageBox.warning(self, "无法生成参考图", "当前视野内没有可用的地平线上参考星。")
                return

            common_name_mag_limit = self._common_name_mag_limit()
            image = self.renderer.render(
                star_map,
                common_name_mag_limit=common_name_mag_limit,
                reference_stars=reference_stars,
            )
            payload = build_reference_payload(
                star_map=star_map,
                reference_stars=reference_stars,
                observer=observer,
                camera=camera,
                view=view,
                visible_mag_limit=mag_limit,
                common_name_mag_limit=common_name_mag_limit,
            )
            image_path, json_path = save_reference_outputs(image, payload, self._next_reference_output_dir())
            self._display_star_map_image(star_map, image)
            self.ui.statusbar.showMessage(
                f"已导出参考图: {image_path}  参考星表: {json_path}  标注星数: {len(reference_stars)}"
            )
            QMessageBox.information(self, "参考图已导出", f"PNG：{image_path}\nJSON：{json_path}")
        except Exception as exc:  # noqa: BLE001 - 界面层需要把可恢复输入错误显示出来。
            self.ui.statusbar.showMessage(f"导出参考图失败: {exc}")
            QMessageBox.critical(self, "导出参考图失败", str(exc))

    def fit_star_map(self) -> None:
        if not self.pixmap_item.pixmap().isNull():
            self.ui.starMapView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        QTimer.singleShot(0, self.fit_star_map)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is self.ui.starMapView.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.drag_start = event.pos()
                self.last_drag_pos = event.pos()
                self.render_timer.stop()
                self.ui.starMapView.viewport().setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self.last_drag_pos is not None:
                dx = event.pos().x() - self.last_drag_pos.x()
                dy = event.pos().y() - self.last_drag_pos.y()
                self.last_drag_pos = event.pos()
                if event.modifiers() & Qt.ShiftModifier:
                    self._apply_roll_drag_delta(dx)
                else:
                    self._apply_drag_delta(dx, dy)
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.drag_start = None
                self.last_drag_pos = None
                self.ui.starMapView.viewport().unsetCursor()
                self.render_now()
                return True
        return super().eventFilter(watched, event)

    def _apply_drag_delta(self, dx: int, dy: int) -> None:
        camera = self._camera_settings()
        az_degrees_per_pixel = max(horizontal_fov_deg(camera) / max(self.ui.starMapView.viewport().width(), 1), 0.01)
        alt_degrees_per_pixel = max(vertical_fov_deg(camera) / max(self.ui.starMapView.viewport().height(), 1), 0.01)
        az = (self.ui.doubleSpinBoxAz.value() - dx * az_degrees_per_pixel) % 360.0
        alt = max(-90.0, min(90.0, self.ui.doubleSpinBoxAlt.value() + dy * alt_degrees_per_pixel))
        self.ui.doubleSpinBoxAz.blockSignals(True)
        self.ui.doubleSpinBoxAlt.blockSignals(True)
        self.ui.doubleSpinBoxAz.setValue(az)
        self.ui.doubleSpinBoxAlt.setValue(alt)
        self.ui.doubleSpinBoxAz.blockSignals(False)
        self.ui.doubleSpinBoxAlt.blockSignals(False)
        self.render_now()

    def _apply_roll_drag_delta(self, dx: int) -> None:
        roll = self.ui.doubleSpinBoxRoll.value() + dx * 0.25
        while roll > 180.0:
            roll -= 360.0
        while roll < -180.0:
            roll += 360.0
        self.ui.doubleSpinBoxRoll.blockSignals(True)
        self.ui.doubleSpinBoxRoll.setValue(roll)
        self.ui.doubleSpinBoxRoll.blockSignals(False)
        self.render_now()


def main(argv: list[str] | None = None) -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv or sys.argv)
    if not ensure_catalogs_ready_or_handle():
        return 0

    window = MainWindow()
    window.show()
    return int(app.exec_())
