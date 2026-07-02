from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from PyQt5.QtCore import QDateTime, QEvent, QPoint, QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QMainWindow

from .catalog_download import ensure_catalogs_ready_or_handle
from .catalog import load_default_catalog
from .renderer import StarMapRenderer
from .simulator import (
    CameraSettings,
    HorizontalStarCatalog,
    ObserverSettings,
    ViewSettings,
    compute_horizontal_catalog,
    horizontal_fov_deg,
    project_horizontal_catalog,
    vertical_fov_deg,
)
from .ui.ui_main_window import Ui_MainWindow


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
        self.ui.doubleSpinBoxMagLimit.setValue(6.5)
        self.ui.doubleSpinBoxAz.setValue(0.0)
        self.ui.doubleSpinBoxAlt.setValue(20.0)
        self.ui.doubleSpinBoxRoll.setValue(0.0)

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
            self.ui.doubleSpinBoxMagLimit,
            self.ui.doubleSpinBoxAz,
            self.ui.doubleSpinBoxAlt,
            self.ui.doubleSpinBoxRoll,
        )
        for widget in widgets:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.schedule_render)
            elif hasattr(widget, "dateTimeChanged"):
                widget.dateTimeChanged.connect(self.schedule_render)
        self.ui.pushButtonRender.clicked.connect(lambda: self.render_now())

    def schedule_render(self, *unused, delay_ms: int = 120) -> None:  # type: ignore[no-untyped-def]
        self.render_timer.start(delay_ms)

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

    def render_now(self) -> None:
        try:
            camera = self._camera_settings()
            mag_limit = self.ui.doubleSpinBoxMagLimit.value()
            horizontal_catalog = self._get_horizontal_catalog(self._observer_settings(), mag_limit)
            star_map = project_horizontal_catalog(
                horizontal_catalog=horizontal_catalog,
                camera=camera,
                view=self._view_settings(),
                visible_mag_limit=mag_limit,
            )
            image = self.renderer.render(star_map)
            render_size = (star_map.width, star_map.height)
            should_fit = self._last_render_size != render_size or self.pixmap_item.pixmap().isNull()
            self.pixmap_item.setPos(0, 0)
            self.pixmap_item.setPixmap(QPixmap.fromImage(image))
            self.scene.setSceneRect(0, 0, star_map.width, star_map.height)
            self._last_render_size = render_size
            if should_fit:
                self.fit_star_map()
            self.ui.statusbar.showMessage(
                "星表: {catalog_count}  视野内: {visible_count}  地平线上: {above_count}  "
                "Az: {az:.2f} deg  Alt: {alt:.2f} deg".format(
                    catalog_count=star_map.catalog_count,
                    visible_count=len(star_map),
                    above_count=star_map.above_horizon_count,
                    az=self.ui.doubleSpinBoxAz.value(),
                    alt=self.ui.doubleSpinBoxAlt.value(),
                )
            )
        except Exception as exc:  # noqa: BLE001 - UI should report recoverable input errors.
            self.ui.statusbar.showMessage(f"渲染失败: {exc}")

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


def main(argv: list[str] | None = None) -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv or sys.argv)
    if not ensure_catalogs_ready_or_handle():
        return 0

    window = MainWindow()
    window.show()
    return int(app.exec_())
