from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PyQt5.QtCore import QDateTime, QEvent, QObject, QPoint, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QCursor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QTableWidgetItem,
)

from .catalog_download import ensure_catalogs_ready_or_handle
from .catalog import load_default_catalog, project_root
from .config import StarMapUiConfig, load_star_map_ui_config
from .image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from .milky_way import MilkyWayCatalog, load_milky_way
from .reference import build_reference_payload, save_reference_outputs
from .renderer import StarMapRenderer
from .simulator import (
    CameraSettings,
    FISHEYE_EQUIDISTANT,
    FISHEYE_EQUISOLID,
    HorizontalMilkyWayCatalog,
    HorizontalSolarSystemCatalog,
    HorizontalStarCatalog,
    ObserverSettings,
    ProjectedStarMap,
    ReferenceStar,
    RECTILINEAR_LENS_MODEL,
    ViewSettings,
    compute_horizontal_catalog,
    compute_horizontal_milky_way,
    compute_horizontal_solar_system,
    horizontal_fov_deg,
    project_horizontal_catalog,
    select_reference_stars,
    vertical_fov_deg,
)
from .star_fitting import FittedStarPosition, fit_star_position
from .ui.ui_main_window import Ui_MainWindow


LENS_MODELS = (
    RECTILINEAR_LENS_MODEL,
    FISHEYE_EQUIDISTANT,
    FISHEYE_EQUISOLID,
)
REFERENCE_LABEL_MODE_FIXED_COUNT = "fixed_count"
REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT = "fixed_mag_limit"
REFERENCE_LABEL_MODES = (
    REFERENCE_LABEL_MODE_FIXED_COUNT,
    REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT,
)
PREVIEW_LONG_SIDE_PX = 1920
REAL_IMAGE_MAX_ZOOM_SCALE = 2.0
IMAGE_VIEW_ZOOM_IN_FACTOR = 1.25
IMAGE_VIEW_ZOOM_OUT_FACTOR = 0.8
STAR_PICK_CIRCLE_STEP_PX = 10
MIN_PSF_RADIUS_PX = 4


class GraphicsImageItem(QGraphicsItem):
    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()

    def set_image(self, image: QImage) -> None:
        self.prepareGeometryChange()
        self.image = image

    def isNull(self) -> bool:
        return self.image.isNull()

    def pixmap(self) -> QPixmap:
        return QPixmap.fromImage(self.image)

    def boundingRect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        return QRectF(0.0, 0.0, float(self.image.width()), float(self.image.height()))

    def paint(self, painter, option, widget=None) -> None:  # type: ignore[no-untyped-def]
        if self.image.isNull():
            return
        painter.drawImage(0, 0, self.image)


class ImagePreviewLoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str | Path, max_long_side_px: int | None) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.max_long_side_px = max_long_side_px

    def run(self) -> None:
        try:
            preview = load_image_preview(self.file_path, max_long_side_px=self.max_long_side_px)
            self.finished.emit(preview)
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有读取错误传回界面层。
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui_config = load_star_map_ui_config()
        self._apply_ui_font_config(self.ui_config)

        self.catalog = load_default_catalog(mag_limit=None)
        self.milky_way_catalog: MilkyWayCatalog = load_milky_way()
        self.renderer = StarMapRenderer(self.ui_config)
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.ui.starMapView.setScene(self.scene)
        self.ui.starMapView.viewport().installEventFilter(self)

        self.reference_scene = QGraphicsScene(self)
        self.reference_pixmap_item = QGraphicsPixmapItem()
        self.reference_scene.addItem(self.reference_pixmap_item)
        self.ui.referenceImageView.setScene(self.reference_scene)
        self.ui.referenceImageView.viewport().installEventFilter(self)

        self.real_image_scene = QGraphicsScene(self)
        self.real_image_item = GraphicsImageItem()
        self.real_image_pixmap_item = self.real_image_item
        self.real_image_scene.addItem(self.real_image_item)
        self.ui.realImageView.setScene(self.real_image_scene)
        self.ui.realImageView.viewport().installEventFilter(self)

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render_now)
        self.drag_start: QPoint | None = None
        self.last_drag_pos: QPoint | None = None
        self._horizontal_cache_key: tuple[object, ...] | None = None
        self._horizontal_cache: HorizontalStarCatalog | None = None
        self._milky_way_cache_key: tuple[object, ...] | None = None
        self._milky_way_cache: HorizontalMilkyWayCatalog | None = None
        self._solar_system_cache_key: tuple[object, ...] | None = None
        self._solar_system_cache: HorizontalSolarSystemCatalog | None = None
        self._last_render_size: tuple[int, int] | None = None
        self._last_reference_render_size: tuple[int, int] | None = None
        self.current_image_preview: ImagePreview | None = None
        self._image_import_thread: QThread | None = None
        self._image_import_worker: ImagePreviewLoadWorker | None = None
        self._image_import_progress: QProgressDialog | None = None
        self._real_image_zoom_max_scale = REAL_IMAGE_MAX_ZOOM_SCALE
        self._syncing_camera_dimensions = False
        self._active_star_pair_row: int | None = None
        self._star_pick_cursor: QCursor | None = None
        self._star_pick_circle_diameter_px = self.ui_config.star_pick_circle_default_diameter_px
        self._star_pick_previous_drag_mode = self.ui.realImageView.dragMode()
        self._star_pair_annotations: dict[str, tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}

        self._init_defaults()
        self._connect_inputs()
        self.schedule_render(delay_ms=0)

    def _apply_ui_font_config(self, ui_config: StarMapUiConfig) -> None:
        controls_font = QFont(self.font())
        controls_font.setPointSize(ui_config.controls_font_size_pt)
        self.setFont(controls_font)
        self.ui.centralwidget.setFont(controls_font)

        status_font = QFont(self.ui.statusbar.font())
        status_font.setPointSize(ui_config.status_bar_font_size_pt)
        self.ui.statusbar.setFont(status_font)

    def _init_defaults(self) -> None:
        self.ui.dateTimeEditObservation.setDateTime(QDateTime.currentDateTime())
        utc_offset = datetime.now().astimezone().utcoffset()
        if utc_offset is None:
            utc_offset = timedelta(hours=8)
        self.ui.doubleSpinBoxUtcOffset.setValue(utc_offset.total_seconds() / 3600.0)
        self.ui.doubleSpinBoxLatitude.setValue(self.ui_config.default_latitude_deg)
        self.ui.doubleSpinBoxLongitude.setValue(self.ui_config.default_longitude_deg)
        self.ui.doubleSpinBoxElevation.setValue(self.ui_config.default_elevation_m)
        self.ui.doubleSpinBoxSensorWidth.setValue(36.0)
        self.ui.doubleSpinBoxSensorHeight.setValue(24.0)
        self.ui.spinBoxImageWidth.setValue(1920)
        self.ui.spinBoxImageHeight.setValue(1280)
        self.ui.doubleSpinBoxFocalLength.setValue(24.0)
        self.ui.comboBoxLensModel.setCurrentIndex(0)
        self.ui.doubleSpinBoxFisheyeFov.setValue(180.0)
        self.ui.doubleSpinBoxMagLimit.setValue(6.5)
        self.ui.doubleSpinBoxAz.setValue(0.0)
        self.ui.doubleSpinBoxAlt.setValue(20.0)
        self.ui.doubleSpinBoxRoll.setValue(0.0)
        self.ui.comboBoxReferenceLabelMode.setCurrentIndex(0)
        self.ui.spinBoxReferenceStarCount.setValue(12)
        self.ui.doubleSpinBoxReferenceMagLimit.setValue(3.0)
        self._reset_imported_image_labels()
        self._update_reference_label_controls()
        self._update_lens_model_controls()

    def _connect_inputs(self) -> None:
        widgets = (
            self.ui.dateTimeEditObservation,
            self.ui.doubleSpinBoxUtcOffset,
            self.ui.doubleSpinBoxLatitude,
            self.ui.doubleSpinBoxLongitude,
            self.ui.doubleSpinBoxElevation,
            self.ui.doubleSpinBoxFocalLength,
            self.ui.doubleSpinBoxFisheyeFov,
            self.ui.doubleSpinBoxMagLimit,
            self.ui.doubleSpinBoxAz,
            self.ui.doubleSpinBoxAlt,
            self.ui.doubleSpinBoxRoll,
            self.ui.doubleSpinBoxReferenceMagLimit,
        )
        for widget in widgets:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.schedule_render)
            elif hasattr(widget, "dateTimeChanged"):
                widget.dateTimeChanged.connect(self.schedule_render)
        self.ui.doubleSpinBoxSensorWidth.valueChanged.connect(self._handle_sensor_size_changed)
        self.ui.doubleSpinBoxSensorHeight.valueChanged.connect(self._handle_sensor_size_changed)
        self.ui.spinBoxImageWidth.valueChanged.connect(self._handle_image_width_changed)
        self.ui.spinBoxImageHeight.valueChanged.connect(self._handle_image_height_changed)
        self.ui.comboBoxLensModel.currentIndexChanged.connect(self._handle_lens_model_changed)
        self.ui.comboBoxReferenceLabelMode.currentIndexChanged.connect(self._handle_reference_label_mode_changed)
        self.ui.spinBoxReferenceStarCount.valueChanged.connect(self.schedule_render)
        self.ui.pushButtonSwapOrientation.clicked.connect(self._swap_camera_orientation)
        self.ui.pushButtonRender.clicked.connect(lambda: self.render_now())
        self.ui.pushButtonExportReference.clicked.connect(self.export_reference_map)
        self.ui.pushButtonImportSingleImage.clicked.connect(self.import_single_image)
        self.ui.pushButtonImportImageSequence.clicked.connect(self.show_sequence_import_placeholder)
        self.ui.actionImportSingleImage.triggered.connect(self.import_single_image)
        self.ui.actionImportImageSequence.triggered.connect(self.show_sequence_import_placeholder)
        self.ui.tabWidgetMain.currentChanged.connect(self._handle_tab_changed)
        self.ui.tableWidgetStarPairs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.tableWidgetStarPairs.customContextMenuRequested.connect(self._show_star_pair_context_menu)
        self.ui.tableWidgetStarPairs.itemChanged.connect(self._handle_star_pair_item_changed)
        self.ui.tableWidgetStarPairs.installEventFilter(self)
        self.ui.pushButtonImportReferenceJson.clicked.connect(self.import_reference_json)

    def _reset_imported_image_labels(self) -> None:
        self.ui.labelImportedImagePath.setText("未导入")
        self.ui.labelImportedImageSize.setText("-")

    def _update_imported_image_labels(self, preview: ImagePreview) -> None:
        self.ui.labelImportedImagePath.setText(str(preview.path))
        self.ui.labelImportedImageSize.setText(f"{preview.original_width} x {preview.original_height} px")

    def _collect_star_pair_positions(self) -> dict[str, str]:
        positions: dict[str, str] = {}
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            position_item = table.item(row, 2)
            if name_item is None or position_item is None:
                continue
            star_id = str(name_item.data(Qt.UserRole) or "")
            position_text = position_item.text().strip()
            if star_id and position_text:
                positions[star_id] = position_text
        return positions

    def _read_only_table_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _update_star_pair_table(self, reference_stars: tuple[ReferenceStar, ...]) -> None:
        table = self.ui.tableWidgetStarPairs
        saved_positions = self._collect_star_pair_positions()
        table.blockSignals(True)
        table.setRowCount(len(reference_stars))
        for row, star in enumerate(reference_stars):
            star_id = star.star_id.strip()
            star_name = star.common_name.strip() or star_id

            index_item = self._read_only_table_item(str(star.index))
            name_item = self._read_only_table_item(star_name)
            name_item.setData(Qt.UserRole, star_id)
            position_item = QTableWidgetItem(saved_positions.get(star_id, ""))
            position_item.setData(Qt.UserRole, star_id)

            table.setItem(row, 0, index_item)
            table.setItem(row, 1, name_item)
            table.setItem(row, 2, position_item)
        table.blockSignals(False)
        table.resizeColumnToContents(0)
        table.resizeColumnToContents(1)
        self._sync_star_pair_annotations_to_table()
        self._refresh_star_pair_table_styles()
        self._restore_star_pair_annotations_from_table()

    def _handle_star_pair_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 2:
            return
        star_id = self._star_pair_star_id(item.row())
        if star_id and not item.text().strip():
            self._remove_star_pair_annotation(star_id)
        self._refresh_star_pair_row_style(item.row())

    def _star_pair_star_id(self, row: int) -> str:
        name_item = self.ui.tableWidgetStarPairs.item(row, 1)
        if name_item is None:
            return ""
        return str(name_item.data(Qt.UserRole) or "")

    def _star_pair_position_text(self, row: int) -> str:
        position_item = self.ui.tableWidgetStarPairs.item(row, 2)
        if position_item is None:
            return ""
        return position_item.text().strip()

    def _parse_star_pair_position_text(self, row: int) -> tuple[float, float] | None:
        position_text = self._star_pair_position_text(row)
        if not position_text:
            return None
        normalized_text = position_text.replace("，", ",")
        parts = [part.strip() for part in normalized_text.split(",")]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def _star_pair_label(self, row: int) -> str:
        index_item = self.ui.tableWidgetStarPairs.item(row, 0)
        index_text = index_item.text() if index_item is not None else str(row + 1)
        star_name = self._star_pair_name(row)
        return f"{index_text}. {star_name}" if star_name else index_text

    def _refresh_star_pair_table_styles(self) -> None:
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            self._refresh_star_pair_row_style(row)

    def _refresh_star_pair_row_style(self, row: int) -> None:
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount():
            return
        table = self.ui.tableWidgetStarPairs
        if self._active_star_pair_row == row:
            background = QColor(255, 242, 153)
        elif self._star_pair_position_text(row):
            background = QColor(210, 244, 214)
        else:
            background = QColor(255, 255, 255)

        signals_were_blocked = table.blockSignals(True)
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is not None:
                item.setBackground(QBrush(background))
        table.blockSignals(signals_were_blocked)

    def _create_star_pick_cursor(self) -> QCursor:
        if self._star_pick_cursor is not None:
            return self._star_pick_cursor

        diameter = self._star_pick_circle_diameter_px + 1
        radius = self._star_pick_circle_diameter_px // 2
        pixmap = QPixmap(diameter, diameter)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(255, 220, 80), 2))
        painter.drawEllipse(1, 1, diameter - 3, diameter - 3)
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawPoint(radius, radius)
        painter.end()

        self._star_pick_cursor = QCursor(pixmap, radius, radius)
        return self._star_pick_cursor

    def _set_star_pick_circle_diameter(self, diameter_px: int, show_status: bool = True) -> None:
        minimum = self.ui_config.star_pick_circle_min_diameter_px
        maximum = self.ui_config.star_pick_circle_max_diameter_px
        new_diameter = min(max(int(diameter_px), minimum), maximum)
        if new_diameter == self._star_pick_circle_diameter_px:
            if show_status and self._active_star_pair_row is not None:
                self.ui.statusbar.showMessage(
                    f"选星圈直径已到边界：{new_diameter} px。左键确认，右键取消。"
                )
            return

        self._star_pick_circle_diameter_px = new_diameter
        self._star_pick_cursor = None
        if self._active_star_pair_row is not None:
            self.ui.realImageView.viewport().setCursor(self._create_star_pick_cursor())
            if show_status:
                self.ui.statusbar.showMessage(
                    f"选星圈直径：{new_diameter} px。左键确认，右键取消，Ctrl+滚轮 / Ctrl+加减继续缩放。"
                )

    def _adjust_star_pick_circle_diameter(self, step_count: int) -> None:
        if step_count == 0:
            return
        self._set_star_pick_circle_diameter(
            self._star_pick_circle_diameter_px + step_count * STAR_PICK_CIRCLE_STEP_PX
        )

    def _star_pick_circle_image_radius_px(self, viewport_pos: QPoint) -> int:
        scene_center = self.ui.realImageView.mapToScene(viewport_pos)
        screen_radius = max(1, self._star_pick_circle_diameter_px // 2)
        scene_edge = self.ui.realImageView.mapToScene(viewport_pos + QPoint(screen_radius, 0))
        image_radius = ((scene_edge.x() - scene_center.x()) ** 2 + (scene_edge.y() - scene_center.y()) ** 2) ** 0.5
        return max(MIN_PSF_RADIUS_PX, int(round(image_radius)))

    def _star_pick_psf_radius_px(self, viewport_pos: QPoint) -> int:
        circle_radius = self._star_pick_circle_image_radius_px(viewport_pos)
        psf_radius = circle_radius * self.ui_config.star_pick_psf_radius_scale
        bounded_radius = min(psf_radius, float(self.ui_config.star_pick_psf_max_radius_px))
        return max(MIN_PSF_RADIUS_PX, int(round(bounded_radius)))

    def _show_star_pick_status_hint(self, row: int) -> None:
        self.ui.statusbar.showMessage(
            "正在点选 {label}；左键确认，右键取消，Ctrl+滚轮 / Ctrl+加减缩放选星圈。"
            "当前选星圈直径：{diameter} px，PSF半径比例：{scale:.2f}，上限：{max_radius} px。".format(
                label=self._star_pair_label(row),
                diameter=self._star_pick_circle_diameter_px,
                scale=self.ui_config.star_pick_psf_radius_scale,
                max_radius=self.ui_config.star_pick_psf_max_radius_px,
            )
        )

    def _clear_star_pair_annotations(self) -> None:
        for ellipse_item, label_item in self._star_pair_annotations.values():
            self.real_image_scene.removeItem(ellipse_item)
            self.real_image_scene.removeItem(label_item)
        self._star_pair_annotations.clear()

    def _remove_star_pair_annotation(self, star_id: str) -> None:
        items = self._star_pair_annotations.pop(star_id, None)
        if items is None:
            return
        ellipse_item, label_item = items
        self.real_image_scene.removeItem(ellipse_item)
        self.real_image_scene.removeItem(label_item)

    def _sync_star_pair_annotations_to_table(self) -> None:
        valid_star_ids: set[str] = set()
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            star_id = self._star_pair_star_id(row)
            if not star_id:
                continue
            valid_star_ids.add(star_id)
            items = self._star_pair_annotations.get(star_id)
            if items is not None:
                _ellipse_item, label_item = items
                label_item.setText(self._star_pair_label(row))

        for star_id in tuple(self._star_pair_annotations):
            if star_id not in valid_star_ids:
                self._remove_star_pair_annotation(star_id)

    def _restore_star_pair_annotations_from_table(self) -> None:
        if self.current_image_preview is None:
            return
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            position = self._parse_star_pair_position_text(row)
            if position is None:
                continue
            image_x, image_y = position
            if not (0.0 <= image_x < self.current_image_preview.image.width()):
                continue
            if not (0.0 <= image_y < self.current_image_preview.image.height()):
                continue
            fitted_position = FittedStarPosition(
                x=image_x,
                y=image_y,
                amplitude=0.0,
                background=0.0,
                sigma_x=0.0,
                sigma_y=0.0,
            )
            self._add_or_update_star_pair_annotation(
                row,
                fitted_position,
                image_radius_px=self.ui_config.star_pick_psf_max_radius_px,
            )

    def _add_or_update_star_pair_annotation(
        self,
        row: int,
        fitted_position: FittedStarPosition,
        image_radius_px: int,
    ) -> None:
        star_id = self._star_pair_star_id(row)
        if not star_id:
            return

        self._remove_star_pair_annotation(star_id)
        radius = max(float(image_radius_px), 1.0)
        ellipse_item = QGraphicsEllipseItem(
            fitted_position.x - radius,
            fitted_position.y - radius,
            radius * 2.0,
            radius * 2.0,
        )
        marker_pen = QPen(QColor(255, 220, 80), 2)
        marker_pen.setCosmetic(True)
        ellipse_item.setPen(marker_pen)
        ellipse_item.setBrush(QBrush(Qt.NoBrush))
        ellipse_item.setZValue(20.0)

        label_item = QGraphicsSimpleTextItem(self._star_pair_label(row))
        label_font = QFont(self.font())
        label_font.setPointSize(self.ui_config.star_name_font_size_pt)
        label_item.setFont(label_font)
        label_item.setBrush(QBrush(QColor(255, 220, 80)))
        label_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label_item.setPos(fitted_position.x + radius, fitted_position.y - radius)
        label_item.setZValue(21.0)

        self.real_image_scene.addItem(ellipse_item)
        self.real_image_scene.addItem(label_item)
        self._star_pair_annotations[star_id] = (ellipse_item, label_item)

    def _show_star_pair_context_menu(self, point: QPoint) -> None:
        table = self.ui.tableWidgetStarPairs
        row = table.rowAt(point.y())
        if row < 0:
            return

        table.selectRow(row)
        menu = QMenu(self)
        pick_action = menu.addAction("点选位置")
        pick_action.setEnabled(self.current_image_preview is not None)
        clear_action = None
        if self._star_pair_position_text(row):
            clear_action = menu.addAction("清除配对")
        selected_action = menu.exec_(table.viewport().mapToGlobal(point))
        if selected_action is pick_action:
            self._enter_star_pick_mode(row)
        elif clear_action is not None and selected_action is clear_action:
            self._clear_star_pair_position(row)

    def _enter_star_pick_mode(self, row: int) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再点选星点位置。")
            return
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount():
            return

        self._active_star_pair_row = row
        self._star_pick_previous_drag_mode = self.ui.realImageView.dragMode()
        self.ui.realImageView.setDragMode(QGraphicsView.NoDrag)
        self.ui.realImageView.viewport().setFocusPolicy(Qt.StrongFocus)
        self.ui.realImageView.viewport().setFocus()
        self.ui.realImageView.viewport().setMouseTracking(True)
        self.ui.realImageView.viewport().setCursor(self._create_star_pick_cursor())
        self._refresh_star_pair_table_styles()
        self._show_star_pick_status_hint(row)

    def _leave_star_pick_mode(self) -> None:
        self._active_star_pair_row = None
        self.ui.realImageView.setDragMode(self._star_pick_previous_drag_mode)
        self.ui.realImageView.viewport().unsetCursor()
        self._refresh_star_pair_table_styles()

    def _star_pair_name(self, row: int) -> str:
        item = self.ui.tableWidgetStarPairs.item(row, 1)
        if item is None:
            return ""
        return item.text()

    def _set_star_pair_position(self, row: int, fitted_position: FittedStarPosition) -> None:
        table = self.ui.tableWidgetStarPairs
        if row < 0 or row >= table.rowCount():
            return

        position_item = table.item(row, 2)
        if position_item is None:
            position_item = QTableWidgetItem()
            table.setItem(row, 2, position_item)
        name_item = table.item(row, 1)
        if name_item is not None:
            position_item.setData(Qt.UserRole, name_item.data(Qt.UserRole))
        position_item.setText(f"{fitted_position.x:.2f}, {fitted_position.y:.2f}")
        table.selectRow(row)
        self._refresh_star_pair_row_style(row)

    def _clear_star_pair_positions(self) -> None:
        table = self.ui.tableWidgetStarPairs
        table.blockSignals(True)
        for row in range(table.rowCount()):
            position_item = table.item(row, 2)
            if position_item is not None:
                position_item.setText("")
        table.blockSignals(False)
        self._clear_star_pair_annotations()
        self._refresh_star_pair_table_styles()

    def _clear_star_pair_position(self, row: int) -> None:
        table = self.ui.tableWidgetStarPairs
        if row < 0 or row >= table.rowCount():
            return
        star_label = self._star_pair_label(row)
        star_id = self._star_pair_star_id(row)
        position_item = table.item(row, 2)
        if position_item is not None:
            signals_were_blocked = table.blockSignals(True)
            position_item.setText("")
            table.blockSignals(signals_were_blocked)
        if star_id:
            self._remove_star_pair_annotation(star_id)
        self._refresh_star_pair_row_style(row)
        self.ui.statusbar.showMessage(f"已清除 {star_label} 的真实图像配对。右键该行可重新点选位置。")

    def _clear_selected_star_pair_positions(self) -> bool:
        rows = sorted({index.row() for index in self.ui.tableWidgetStarPairs.selectedIndexes()})
        if not rows:
            return False
        for row in rows:
            self._clear_star_pair_position(row)
        return True

    def _handle_real_image_pick_click(self, viewport_pos: QPoint) -> None:
        if self._active_star_pair_row is None or self.current_image_preview is None:
            return

        image = self.current_image_preview.image
        scene_pos = self.ui.realImageView.mapToScene(viewport_pos)
        image_x = float(scene_pos.x())
        image_y = float(scene_pos.y())
        if not (0.0 <= image_x < image.width() and 0.0 <= image_y < image.height()):
            self.ui.statusbar.showMessage("点击位置不在真实图像范围内，请重新点选。")
            return

        image_radius_px = self._star_pick_psf_radius_px(viewport_pos)
        try:
            fitted_position = fit_star_position(
                image,
                click_x=image_x,
                click_y=image_y,
                radius_px=image_radius_px,
            )
        except Exception as exc:  # noqa: BLE001 - 交互式点选需要把拟合失败原因直接反馈给用户。
            self.ui.statusbar.showMessage(f"PSF 拟合失败: {exc}")
            QMessageBox.warning(self, "PSF 拟合失败", str(exc))
            return

        row = self._active_star_pair_row
        star_name = self._star_pair_name(row)
        self._set_star_pair_position(row, fitted_position)
        self._add_or_update_star_pair_annotation(row, fitted_position, image_radius_px)
        self._leave_star_pick_mode()
        self.ui.statusbar.showMessage(
            "已记录 {name} 的图像坐标: x={x:.2f}, y={y:.2f}；拟合窗口半径 {radius} px，"
            "PSF sigma=({sigma_x:.2f}, {sigma_y:.2f}) px。右键配对表行可继续点选。".format(
                name=star_name,
                x=fitted_position.x,
                y=fitted_position.y,
                radius=image_radius_px,
                sigma_x=fitted_position.sigma_x,
                sigma_y=fitted_position.sigma_y,
            )
        )

    def import_single_image(self) -> None:
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return
        default_dir = project_root() / "testimages"
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入单张图像",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        self.start_single_image_import(file_path)

    def start_single_image_import(self, file_path: str | Path) -> None:
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return

        image_path = Path(file_path)
        self._set_image_import_controls_enabled(False)
        self.ui.statusbar.showMessage(f"正在读取整张图像并量化为 8-bit: {image_path}")

        progress = QProgressDialog(self)
        progress.setWindowTitle("正在导入图像")
        progress.setLabelText(f"正在读取整张图像并量化为 8-bit 显示图...\n{image_path}")
        progress.setRange(0, 0)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        thread = QThread(self)
        worker = ImagePreviewLoadWorker(image_path, None)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_single_image_import_finished)
        worker.failed.connect(self._handle_single_image_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_single_image_import)

        self._image_import_thread = thread
        self._image_import_worker = worker
        self._image_import_progress = progress
        thread.start()

    def load_single_image(self, file_path: str | Path) -> None:
        try:
            preview = load_image_preview(file_path, max_long_side_px=None)
            self._apply_loaded_image_preview(preview)
        except Exception as exc:  # noqa: BLE001 - 文件导入错误需要以对话框形式提示用户。
            self.ui.statusbar.showMessage(f"导入图像失败: {exc}")
            QMessageBox.critical(self, "导入图像失败", str(exc))

    def _set_image_import_controls_enabled(self, enabled: bool) -> None:
        self.ui.pushButtonImportSingleImage.setEnabled(enabled)
        self.ui.pushButtonImportImageSequence.setEnabled(enabled)
        self.ui.actionImportSingleImage.setEnabled(enabled)
        self.ui.actionImportImageSequence.setEnabled(enabled)

    def _apply_loaded_image_preview(self, preview: ImagePreview) -> None:
        self.current_image_preview = preview
        self._update_imported_image_labels(preview)
        self._display_real_image_preview(preview)
        self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        self.ui.statusbar.showMessage(
            "已导入图像: {path}  原始: {width} x {height} px。右键配对表行选择“点选位置”。".format(
                path=preview.path,
                width=preview.original_width,
                height=preview.original_height,
            )
        )

    def _handle_single_image_import_finished(self, preview: object) -> None:
        if self._image_import_progress is not None:
            self._image_import_progress.close()
        self._apply_loaded_image_preview(preview)  # type: ignore[arg-type]

    def _handle_single_image_import_failed(self, error_message: str) -> None:
        if self._image_import_progress is not None:
            self._image_import_progress.close()
        self.ui.statusbar.showMessage(f"导入图像失败: {error_message}")
        QMessageBox.critical(self, "导入图像失败", error_message)

    def _cleanup_single_image_import(self) -> None:
        self._image_import_thread = None
        self._image_import_worker = None
        self._image_import_progress = None
        self._set_image_import_controls_enabled(True)

    def show_sequence_import_placeholder(self) -> None:
        QMessageBox.information(self, "序列图像导入", "序列图像导入入口已预留，将在后续阶段实现。")

    def _handle_tab_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        QTimer.singleShot(0, self.fit_all_graphics_views)

    def schedule_render(self, *unused, delay_ms: int = 120) -> None:  # type: ignore[no-untyped-def]
        self.render_timer.start(delay_ms)

    def _reference_label_mode(self) -> str:
        index = self.ui.comboBoxReferenceLabelMode.currentIndex()
        if index < 0 or index >= len(REFERENCE_LABEL_MODES):
            return REFERENCE_LABEL_MODE_FIXED_COUNT
        return REFERENCE_LABEL_MODES[index]

    def _update_reference_label_controls(self) -> None:
        is_fixed_count = self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_COUNT
        self.ui.labelReferenceStarCount.setEnabled(is_fixed_count)
        self.ui.spinBoxReferenceStarCount.setEnabled(is_fixed_count)
        self.ui.labelReferenceMagLimit.setEnabled(not is_fixed_count)
        self.ui.doubleSpinBoxReferenceMagLimit.setEnabled(not is_fixed_count)

    def _handle_reference_label_mode_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_reference_label_controls()
        self.schedule_render()

    def _select_current_reference_stars(self, star_map: ProjectedStarMap) -> tuple[ReferenceStar, ...]:
        if self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT:
            return select_reference_stars(
                star_map=star_map,
                max_count=None,
                mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            )
        return select_reference_stars(
            star_map=star_map,
            max_count=self.ui.spinBoxReferenceStarCount.value(),
        )

    def _sensor_aspect_ratio(self) -> float:
        sensor_height = max(self.ui.doubleSpinBoxSensorHeight.value(), 1e-6)
        return max(self.ui.doubleSpinBoxSensorWidth.value() / sensor_height, 1e-6)

    def _bounded_image_width(self, value: float) -> int:
        return min(max(int(round(value)), self.ui.spinBoxImageWidth.minimum()), self.ui.spinBoxImageWidth.maximum())

    def _bounded_image_height(self, value: float) -> int:
        return min(max(int(round(value)), self.ui.spinBoxImageHeight.minimum()), self.ui.spinBoxImageHeight.maximum())

    def _set_image_dimensions(self, width_px: int, height_px: int) -> None:
        self._syncing_camera_dimensions = True
        self.ui.spinBoxImageWidth.blockSignals(True)
        self.ui.spinBoxImageHeight.blockSignals(True)
        self.ui.spinBoxImageWidth.setValue(self._bounded_image_width(width_px))
        self.ui.spinBoxImageHeight.setValue(self._bounded_image_height(height_px))
        self.ui.spinBoxImageWidth.blockSignals(False)
        self.ui.spinBoxImageHeight.blockSignals(False)
        self._syncing_camera_dimensions = False

    def _sync_image_size_to_sensor_long_side(self) -> None:
        aspect_ratio = self._sensor_aspect_ratio()
        long_side = max(self.ui.spinBoxImageWidth.value(), self.ui.spinBoxImageHeight.value())
        if aspect_ratio >= 1.0:
            width_px = self._bounded_image_width(long_side)
            height_px = self._bounded_image_height(width_px / aspect_ratio)
        else:
            height_px = self._bounded_image_height(long_side)
            width_px = self._bounded_image_width(height_px * aspect_ratio)
        self._set_image_dimensions(width_px, height_px)

    def _handle_sensor_size_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self._syncing_camera_dimensions:
            return
        self._sync_image_size_to_sensor_long_side()
        self.schedule_render()

    def _handle_image_width_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self._syncing_camera_dimensions:
            return
        width_px = self.ui.spinBoxImageWidth.value()
        height_px = self._bounded_image_height(width_px / self._sensor_aspect_ratio())
        self._set_image_dimensions(width_px, height_px)

    def _handle_image_height_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self._syncing_camera_dimensions:
            return
        height_px = self.ui.spinBoxImageHeight.value()
        width_px = self._bounded_image_width(height_px * self._sensor_aspect_ratio())
        self._set_image_dimensions(width_px, height_px)

    def _swap_camera_orientation(self) -> None:
        sensor_width = self.ui.doubleSpinBoxSensorWidth.value()
        sensor_height = self.ui.doubleSpinBoxSensorHeight.value()
        self._syncing_camera_dimensions = True
        self.ui.doubleSpinBoxSensorWidth.blockSignals(True)
        self.ui.doubleSpinBoxSensorHeight.blockSignals(True)
        self.ui.doubleSpinBoxSensorWidth.setValue(sensor_height)
        self.ui.doubleSpinBoxSensorHeight.setValue(sensor_width)
        self.ui.doubleSpinBoxSensorWidth.blockSignals(False)
        self.ui.doubleSpinBoxSensorHeight.blockSignals(False)
        self._syncing_camera_dimensions = False
        self._sync_image_size_to_sensor_long_side()
        self.schedule_render()

    def _lens_model(self) -> str:
        index = self.ui.comboBoxLensModel.currentIndex()
        if index < 0 or index >= len(LENS_MODELS):
            return RECTILINEAR_LENS_MODEL
        return LENS_MODELS[index]

    def _update_lens_model_controls(self) -> None:
        lens_model = self._lens_model()
        is_fisheye = lens_model != RECTILINEAR_LENS_MODEL
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

    def _camera_settings_for_image_size(self, image_width_px: int, image_height_px: int) -> CameraSettings:
        return CameraSettings(
            sensor_width_mm=self.ui.doubleSpinBoxSensorWidth.value(),
            sensor_height_mm=self.ui.doubleSpinBoxSensorHeight.value(),
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            focal_length_mm=self.ui.doubleSpinBoxFocalLength.value(),
            lens_model=self._lens_model(),
            fisheye_fov_deg=self.ui.doubleSpinBoxFisheyeFov.value(),
        )

    def _output_camera_settings(self) -> CameraSettings:
        return self._camera_settings_for_image_size(
            image_width_px=self.ui.spinBoxImageWidth.value(),
            image_height_px=self.ui.spinBoxImageHeight.value(),
        )

    def _preview_image_size(self) -> tuple[int, int]:
        aspect_ratio = self._sensor_aspect_ratio()
        if aspect_ratio >= 1.0:
            width_px = PREVIEW_LONG_SIDE_PX
            height_px = max(128, int(round(PREVIEW_LONG_SIDE_PX / aspect_ratio)))
        else:
            height_px = PREVIEW_LONG_SIDE_PX
            width_px = max(128, int(round(PREVIEW_LONG_SIDE_PX * aspect_ratio)))
        return width_px, height_px

    def _preview_camera_settings(self) -> CameraSettings:
        image_width_px, image_height_px = self._preview_image_size()
        return self._camera_settings_for_image_size(image_width_px=image_width_px, image_height_px=image_height_px)

    def _render_element_scale(self, camera: CameraSettings) -> float:
        return max(camera.image_width_px, camera.image_height_px) / float(PREVIEW_LONG_SIDE_PX)

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

    def _get_horizontal_milky_way(self, observer: ObserverSettings) -> HorizontalMilkyWayCatalog:
        cache_key = (
            int(observer.observation_time_utc.timestamp()),
            round(observer.latitude_deg, 8),
            round(observer.longitude_deg, 8),
            round(observer.elevation_m, 3),
        )
        if self._milky_way_cache_key != cache_key or self._milky_way_cache is None:
            self._milky_way_cache = compute_horizontal_milky_way(
                milky_way=self.milky_way_catalog,
                observer=observer,
            )
            self._milky_way_cache_key = cache_key
        return self._milky_way_cache

    def _get_horizontal_solar_system(self, observer: ObserverSettings) -> HorizontalSolarSystemCatalog:
        cache_key = (
            int(observer.observation_time_utc.timestamp()),
            round(observer.latitude_deg, 8),
            round(observer.longitude_deg, 8),
            round(observer.elevation_m, 3),
        )
        if self._solar_system_cache_key != cache_key or self._solar_system_cache is None:
            self._solar_system_cache = compute_horizontal_solar_system(observer)
            self._solar_system_cache_key = cache_key
        return self._solar_system_cache

    def _build_projected_star_map(
        self,
        camera: CameraSettings | None = None,
    ) -> tuple[ObserverSettings, CameraSettings, ViewSettings, float, ProjectedStarMap]:
        observer = self._observer_settings()
        camera = camera or self._preview_camera_settings()
        view = self._view_settings()
        mag_limit = self.ui.doubleSpinBoxMagLimit.value()
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

    def _display_reference_map_image(self, star_map: ProjectedStarMap, image) -> None:  # type: ignore[no-untyped-def]
        render_size = (star_map.width, star_map.height)
        should_fit = self._last_reference_render_size != render_size or self.reference_pixmap_item.pixmap().isNull()
        self.reference_pixmap_item.setPos(0, 0)
        self.reference_pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.reference_scene.setSceneRect(0, 0, star_map.width, star_map.height)
        self._last_reference_render_size = render_size
        if should_fit:
            self.fit_reference_map()

    def _display_real_image_preview(self, preview: ImagePreview) -> None:
        self._clear_star_pair_annotations()
        self.real_image_item.setPos(0, 0)
        self.real_image_item.set_image(preview.image)
        self.real_image_scene.setSceneRect(0, 0, preview.image.width(), preview.image.height())
        self._restore_star_pair_annotations_from_table()
        self.fit_real_image()

    def render_now(self) -> None:
        try:
            _observer, _camera, view, _mag_limit, star_map = self._build_projected_star_map()
            reference_stars = self._select_current_reference_stars(star_map)
            image = self.renderer.render(
                star_map,
                reference_stars=reference_stars,
                draw_common_names=False,
                number_reference_stars=False,
            )
            self._display_star_map_image(star_map, image)
            self._update_star_pair_table(reference_stars)
            reference_image = self.renderer.render(
                star_map,
                reference_stars=reference_stars,
                draw_common_names=False,
            )
            self._display_reference_map_image(star_map, reference_image)
            if self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT:
                reference_mode_text = f"标注星等 <= {self.ui.doubleSpinBoxReferenceMagLimit.value():.1f} mag"
            else:
                reference_mode_text = f"标注星数 {self.ui.spinBoxReferenceStarCount.value()} 颗"
            self.ui.statusbar.showMessage(
                "星表: {catalog_count}  视野内: {visible_count}  地平线上: {above_count}  "
                "银河面: {mw_count}  太阳系: {solar_count}  参考星: {reference_count} ({reference_mode})  "
                "镜头: {lens_name}  Az: {az:.2f} deg  Alt: {alt:.2f} deg".format(
                    catalog_count=star_map.catalog_count,
                    visible_count=len(star_map),
                    above_count=star_map.above_horizon_count,
                    mw_count=len(star_map.milky_way_polygons),
                    solar_count=len(star_map.solar_system_objects),
                    reference_count=len(reference_stars),
                    reference_mode=reference_mode_text,
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

    def import_reference_json(self) -> None:
        default_dir = project_root() / "outputs"
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入预览 JSON",
            str(default_dir),
            "MeteoAlign 参考图 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)",
        )
        if not file_path:
            return
        self.load_reference_json(file_path)

    def load_reference_json(self, file_path: str | Path) -> None:
        json_path = Path(file_path)
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self._apply_reference_payload(payload, json_path)
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON/字段错误以对话框提示。
            self.ui.statusbar.showMessage(f"导入预览 JSON 失败: {exc}")
            QMessageBox.critical(self, "导入预览 JSON 失败", str(exc))

    def _payload_section(self, payload: dict[str, object], section_name: str) -> dict[str, object]:
        section = payload.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"JSON 缺少 {section_name} 字段。")
        return section

    def _payload_float(self, section: dict[str, object], key: str) -> float:
        try:
            return float(section[key])
        except KeyError as exc:
            raise ValueError(f"JSON 缺少 {key} 字段。") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效数字。") from exc

    def _payload_int(self, section: dict[str, object], key: str) -> int:
        try:
            return int(section[key])
        except KeyError as exc:
            raise ValueError(f"JSON 缺少 {key} 字段。") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效整数。") from exc

    def _payload_optional_float(self, section: dict[str, object], key: str, default_value: float) -> float:
        if key not in section or section.get(key) is None:
            return default_value
        try:
            return float(section[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效数字。") from exc

    def _payload_datetime_utc(self, section: dict[str, object], key: str) -> datetime:
        raw_value = section.get(key)
        if not isinstance(raw_value, str):
            raise ValueError(f"JSON 缺少 {key} 字段。")
        try:
            parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效 ISO 时间。") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _apply_reference_payload(self, payload: object, source_path: Path) -> None:
        if not isinstance(payload, dict):
            raise ValueError("JSON 根对象必须是字典。")
        if payload.get("format") != "meteoalign_phase1_reference":
            raise ValueError("当前只支持 MeteoAlign 导出的参考图 JSON。")

        observer = self._payload_section(payload, "observer")
        camera = self._payload_section(payload, "camera")
        view = self._payload_section(payload, "view")
        render = self._payload_section(payload, "render")

        observation_time_utc = self._payload_datetime_utc(observer, "observation_time_utc")
        utc_offset_hours = self._payload_optional_float(observer, "utc_offset_hours", 0.0)
        utc_offset_hours = min(
            max(utc_offset_hours, self.ui.doubleSpinBoxUtcOffset.minimum()),
            self.ui.doubleSpinBoxUtcOffset.maximum(),
        )
        local_observation_time = observation_time_utc.astimezone(timezone(timedelta(hours=utc_offset_hours)))
        local_datetime_text = local_observation_time.strftime("%Y-%m-%d %H:%M:%S")
        qt_observation_time = QDateTime.fromString(local_datetime_text, "yyyy-MM-dd HH:mm:ss")
        if not qt_observation_time.isValid():
            raise ValueError("JSON 中的观测时间无法转换为界面时间。")

        widgets_to_block = (
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
            self.ui.comboBoxLensModel,
            self.ui.doubleSpinBoxFisheyeFov,
            self.ui.doubleSpinBoxMagLimit,
            self.ui.doubleSpinBoxAz,
            self.ui.doubleSpinBoxAlt,
            self.ui.doubleSpinBoxRoll,
            self.ui.comboBoxReferenceLabelMode,
            self.ui.spinBoxReferenceStarCount,
            self.ui.doubleSpinBoxReferenceMagLimit,
        )
        previous_signal_states = [widget.blockSignals(True) for widget in widgets_to_block]
        previous_syncing = self._syncing_camera_dimensions
        self._syncing_camera_dimensions = True
        try:
            self.ui.dateTimeEditObservation.setDateTime(qt_observation_time)
            self.ui.doubleSpinBoxUtcOffset.setValue(utc_offset_hours)
            self.ui.doubleSpinBoxLatitude.setValue(self._payload_float(observer, "latitude_deg"))
            self.ui.doubleSpinBoxLongitude.setValue(self._payload_float(observer, "longitude_deg"))
            self.ui.doubleSpinBoxElevation.setValue(self._payload_float(observer, "elevation_m"))

            self.ui.doubleSpinBoxSensorWidth.setValue(self._payload_float(camera, "sensor_width_mm"))
            self.ui.doubleSpinBoxSensorHeight.setValue(self._payload_float(camera, "sensor_height_mm"))
            self.ui.spinBoxImageWidth.setValue(self._payload_int(camera, "image_width_px"))
            self.ui.spinBoxImageHeight.setValue(self._payload_int(camera, "image_height_px"))
            self.ui.doubleSpinBoxFocalLength.setValue(self._payload_float(camera, "focal_length_mm"))
            lens_model = str(camera.get("lens_model", RECTILINEAR_LENS_MODEL))
            lens_index = LENS_MODELS.index(lens_model) if lens_model in LENS_MODELS else 0
            self.ui.comboBoxLensModel.setCurrentIndex(lens_index)
            self.ui.doubleSpinBoxFisheyeFov.setValue(self._payload_float(camera, "fisheye_fov_deg"))

            self.ui.doubleSpinBoxAz.setValue(self._payload_float(view, "center_az_deg"))
            self.ui.doubleSpinBoxAlt.setValue(self._payload_float(view, "center_alt_deg"))
            self.ui.doubleSpinBoxRoll.setValue(self._payload_float(view, "roll_deg"))

            self.ui.doubleSpinBoxMagLimit.setValue(self._payload_float(render, "visible_mag_limit"))
            reference_label_mode = str(render.get("reference_label_mode", REFERENCE_LABEL_MODE_FIXED_COUNT))
            if reference_label_mode not in REFERENCE_LABEL_MODES:
                reference_label_mode = REFERENCE_LABEL_MODE_FIXED_COUNT
            self.ui.comboBoxReferenceLabelMode.setCurrentIndex(REFERENCE_LABEL_MODES.index(reference_label_mode))
            self.ui.spinBoxReferenceStarCount.setValue(self._payload_int(render, "reference_star_count"))
            self.ui.doubleSpinBoxReferenceMagLimit.setValue(
                self._payload_optional_float(render, "reference_mag_limit", self.ui.doubleSpinBoxReferenceMagLimit.value())
            )
        finally:
            self._syncing_camera_dimensions = previous_syncing
            for widget, was_blocked in zip(widgets_to_block, previous_signal_states):
                widget.blockSignals(was_blocked)

        self._update_reference_label_controls()
        self._update_lens_model_controls()
        self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabSimulator)
        self.render_now()
        self.ui.statusbar.showMessage(f"已导入预览 JSON 并恢复星空模拟参数: {source_path}")

    def export_reference_map(self) -> None:
        try:
            output_camera = self._output_camera_settings()
            observer, camera, view, mag_limit, star_map = self._build_projected_star_map(camera=output_camera)
            reference_stars = self._select_current_reference_stars(star_map)
            if not reference_stars:
                QMessageBox.warning(self, "无法生成参考图", "当前视野内没有可用的地平线上参考星。")
                return

            image = self.renderer.render(
                star_map,
                reference_stars=reference_stars,
                element_scale=self._render_element_scale(camera),
                draw_common_names=False,
            )
            payload = build_reference_payload(
                star_map=star_map,
                reference_stars=reference_stars,
                observer=observer,
                camera=camera,
                view=view,
                visible_mag_limit=mag_limit,
                utc_offset_hours=self.ui.doubleSpinBoxUtcOffset.value(),
                reference_label_mode=self._reference_label_mode(),
                reference_mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            )
            image_path, json_path = save_reference_outputs(image, payload, self._next_reference_output_dir())
            self.render_now()
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

    def fit_reference_map(self) -> None:
        if not self.reference_pixmap_item.pixmap().isNull():
            self.ui.referenceImageView.fitInView(self.reference_scene.sceneRect(), Qt.KeepAspectRatio)

    def fit_real_image(self) -> None:
        if not self.real_image_item.isNull():
            self.ui.realImageView.fitInView(self.real_image_scene.sceneRect(), Qt.KeepAspectRatio)
            self._cap_graphics_view_to_max_scale(self.ui.realImageView)

    def fit_all_graphics_views(self) -> None:
        self.fit_star_map()
        self.fit_reference_map()
        self.fit_real_image()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "图像预览仍在生成，请等待导入完成后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        QTimer.singleShot(0, self.fit_all_graphics_views)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is self.ui.tableWidgetStarPairs and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                return self._clear_selected_star_pair_positions()
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
        if watched is self.ui.referenceImageView.viewport() and event.type() == QEvent.Wheel:
            self._apply_graphics_view_zoom(self.ui.referenceImageView, event.angleDelta().y())
            return True
        if watched is self.ui.realImageView.viewport():
            if self._active_star_pair_row is not None:
                if event.type() in (QEvent.Enter, QEvent.MouseMove):
                    self.ui.realImageView.viewport().setCursor(self._create_star_pick_cursor())
                    self._show_star_pick_status_hint(self._active_star_pair_row)
                if event.type() == QEvent.Leave:
                    self.ui.realImageView.viewport().unsetCursor()
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._handle_real_image_pick_click(event.pos())
                    return True
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                    self._leave_star_pick_mode()
                    self.ui.statusbar.showMessage("已取消当前星点位置点选。")
                    return True
                if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
                    wheel_delta = event.angleDelta().y()
                    if wheel_delta == 0:
                        return True
                    wheel_steps = int(wheel_delta / 120)
                    if wheel_steps == 0:
                        wheel_steps = 1 if wheel_delta > 0 else -1
                    self._adjust_star_pick_circle_diameter(wheel_steps)
                    return True
                if event.type() == QEvent.KeyPress and self._handle_star_pick_key_press(event):
                    return True
            if event.type() == QEvent.Wheel:
                self._apply_graphics_view_zoom(self.ui.realImageView, event.angleDelta().y())
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._handle_star_pick_key_press(event):
            return
        super().keyPressEvent(event)

    def _handle_star_pick_key_press(self, event) -> bool:  # type: ignore[no-untyped-def]
        if self._active_star_pair_row is None or not (event.modifiers() & Qt.ControlModifier):
            return False
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._adjust_star_pick_circle_diameter(1)
            return True
        if key in (Qt.Key_Minus, Qt.Key_Underscore):
            self._adjust_star_pick_circle_diameter(-1)
            return True
        return False

    def _apply_graphics_view_zoom(self, view: QGraphicsView, wheel_delta: int) -> None:
        if wheel_delta == 0:
            return
        factor = IMAGE_VIEW_ZOOM_IN_FACTOR if wheel_delta > 0 else IMAGE_VIEW_ZOOM_OUT_FACTOR
        if wheel_delta < 0:
            min_scale = self._graphics_view_fit_scale(view)
            current_scale = self._graphics_view_current_scale(view)
            if current_scale * factor <= min_scale:
                self._fit_graphics_view_to_scene(view)
                return
        if wheel_delta > 0:
            max_scale = self._graphics_view_max_scale(view)
            current_scale = self._graphics_view_current_scale(view)
            if max_scale is not None and current_scale * factor >= max_scale:
                self._set_graphics_view_scale(view, max_scale)
                return
        view.scale(factor, factor)

    def _graphics_view_current_scale(self, view: QGraphicsView) -> float:
        transform = view.transform()
        return min(abs(transform.m11()), abs(transform.m22()))

    def _graphics_view_fit_scale(self, view: QGraphicsView) -> float:
        scene = view.scene()
        if scene is None:
            return 1.0
        scene_rect = scene.sceneRect()
        if scene_rect.width() <= 0.0 or scene_rect.height() <= 0.0:
            return 1.0
        viewport_size = view.viewport().size()
        width_scale = viewport_size.width() / scene_rect.width()
        height_scale = viewport_size.height() / scene_rect.height()
        fit_scale = max(min(width_scale, height_scale), 1e-6)
        max_scale = self._graphics_view_max_scale(view)
        if max_scale is not None:
            return min(fit_scale, max_scale)
        return fit_scale

    def _fit_graphics_view_to_scene(self, view: QGraphicsView) -> None:
        scene = view.scene()
        if scene is None or scene.sceneRect().isEmpty():
            return
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        self._cap_graphics_view_to_max_scale(view)

    def _graphics_view_max_scale(self, view: QGraphicsView) -> float | None:
        if view is self.ui.realImageView:
            return self._real_image_zoom_max_scale
        return None

    def _cap_graphics_view_to_max_scale(self, view: QGraphicsView) -> None:
        max_scale = self._graphics_view_max_scale(view)
        if max_scale is None:
            return
        if self._graphics_view_current_scale(view) > max_scale:
            self._set_graphics_view_scale(view, max_scale)

    def _set_graphics_view_scale(self, view: QGraphicsView, target_scale: float) -> None:
        center = view.mapToScene(view.viewport().rect().center())
        view.resetTransform()
        view.scale(target_scale, target_scale)
        view.centerOn(center)

    def _apply_drag_delta(self, dx: int, dy: int) -> None:
        camera = self._preview_camera_settings()
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
