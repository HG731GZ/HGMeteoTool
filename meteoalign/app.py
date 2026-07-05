from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, QEvent, QObject, QPoint, QPointF, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QCursor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QTableWidgetItem,
)

from .alignment import (
    MIN_ALIGNMENT_PAIRS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_POLYNOMIAL,
    SKY_MATCHING_MODEL_RECTILINEAR,
    SkyAlignmentTransform,
    fit_sky_alignment,
)
from .catalog_download import ensure_catalogs_ready_or_handle
from .catalog import load_default_catalog, project_root
from .config import StarMapUiConfig, load_star_map_ui_config
from .coordinates import radec_to_unit_vectors
from .image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from .milky_way import MilkyWayCatalog, load_milky_way
from .reference import build_reference_payload, save_reference_outputs
from .renderer import StarMapRenderer
from .source_model import SourceAstrometricModel, fit_source_astrometric_model
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
    camera_basis_from_view,
    compute_horizontal_catalog,
    compute_horizontal_milky_way,
    compute_horizontal_solar_system,
    horizontal_fov_deg,
    local_vectors_from_altaz,
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
SKY_ALIGNMENT_MODELS = (
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
)
SKY_ALIGNMENT_MODEL_ALIASES = {
    SKY_MATCHING_MODEL_POLYNOMIAL: SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
}
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
TOUCHPAD_ZOOM_SENSITIVITY = 1.6
TOUCHPAD_ZOOM_MIN_FACTOR = 0.55
TOUCHPAD_ZOOM_MAX_FACTOR = 1.8
STAR_PICK_TOUCHPAD_STEPS_PER_ZOOM_UNIT = 12.0
STAR_PICK_CIRCLE_STEP_PX = 10
MIN_PSF_RADIUS_PX = 4
AUTO_PAIR_MAX_SEARCH_RADIUS_PX = 120
AUTO_PAIR_RMS_RADIUS_SCALE = 3.0
REFERENCE_STAR_PICK_SCREEN_RADIUS_PX = 32
STAR_PAIR_INDEX_COLUMN = 0
STAR_PAIR_NAME_COLUMN = 1
STAR_PAIR_POSITION_COLUMN = 2
STAR_PAIR_RESIDUAL_COLUMN = 3
STAR_PAIR_ROW_TYPE_ROLE = Qt.UserRole + 1
STAR_PAIR_FIT_ROLE = Qt.UserRole + 2
STAR_PAIR_CONSTRAINT_MODE_ROLE = Qt.UserRole + 3
STAR_PAIR_FIT_WEIGHT_ROLE = Qt.UserRole + 4
STAR_PAIR_ROW_TYPE_MANUAL = "manual"
STAR_PAIR_ROW_TYPE_AUTO_GROUP = "auto_match_group"
STAR_PAIR_ROW_TYPE_AUTO_MATCH = "auto_match"
STAR_PAIR_SESSION_FORMAT = "meteoalign_star_pair_session"
STAR_PAIR_SESSION_VERSION = 1
STAR_PAIR_SESSION_JSON_FILTER = "MeteoAlign 星点配对 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)"
SOURCE_MODEL_JSON_FILTER = "MeteoAlign 源图映射 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)"
ALIGNMENT_STATUS_MAX_CHARS = 68
RESIDUAL_WARNING_MIN_PX = 25.0
RESIDUAL_SEVERE_MIN_PX = 50.0
RESIDUAL_SEVERE_RMS_SCALE = 2.0
STAR_RADIUS_ZOOM_EXPONENT = 0.32
STAR_RADIUS_MIN_ZOOM_SCALE = 0.48
AUTO_MATCH_SEARCH_MAG_LIMIT = 8.0
AUTO_MATCH_CONSTRAINT_ANCHOR = "anchor"
AUTO_MATCH_CONSTRAINT_SOFT = "soft"
AUTO_MATCH_CONSTRAINT_MODES = (
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_SOFT,
)
AUTO_MATCH_DEFAULT_SOFT_WEIGHT = 0.3
AUTO_MATCH_MIN_AMPLITUDE = 2.0
AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX = 4.0
AUTO_MATCH_ANNOTATION_LIMIT = 250
STAR_ANNOTATION_PSF_SIGMA_SCALE = 3.0
STAR_ANNOTATION_MIN_RADIUS_PX = 5.0
STAR_ANNOTATION_FALLBACK_RADIUS_PX = 8.0
STAR_ANNOTATION_MAX_RADIUS_PX = 80.0
STAR_PAIR_FOCUS_MIN_MATCHED_COUNT = 6
STAR_PAIR_FOCUS_ZOOM_FIT_SCALE = 8.0
STAR_PAIR_FOCUS_MARKER_RADIUS_PX = 24.0


def _session_image_candidate(path_value: object, source_path: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    image_path = Path(path_value).expanduser()
    if not image_path.is_absolute():
        image_path = source_path.parent / image_path
    return image_path.resolve()


def _resolve_star_pair_session_real_image_path(payload: object, source_path: Path) -> Path:
    if not isinstance(payload, dict):
        raise ValueError("JSON 根对象必须是字典。")
    if payload.get("format") != STAR_PAIR_SESSION_FORMAT:
        raise ValueError("当前只支持 MeteoAlign 星点配对 JSON。")
    real_image = payload.get("real_image")
    if not isinstance(real_image, dict):
        raise ValueError("JSON 缺少 real_image 字段。")

    searched_paths: list[Path] = []
    for key in ("relative_path", "path"):
        image_path = _session_image_candidate(real_image.get(key), source_path)
        if image_path is None:
            continue
        searched_paths.append(image_path)
        if image_path.exists():
            return image_path

    if not searched_paths:
        raise ValueError("JSON 缺少真实图像相对路径与完整路径。")
    searched_text = "\n".join(str(path) for path in searched_paths)
    raise FileNotFoundError(f"真实图像不存在，已按相对路径和完整路径查找：\n{searched_text}")


def _relative_image_path_for_session(image_path: Path, json_path: Path) -> str:
    json_dir = json_path.expanduser().resolve().parent
    try:
        return os.path.relpath(str(image_path), start=str(json_dir))
    except ValueError:
        # Windows 不同盘符之间没有有效相对路径，此时保留文件名并继续依赖完整路径兜底。
        return image_path.name


def _qimage_to_binary_mask(image: QImage) -> np.ndarray:
    if image.isNull():
        raise ValueError("蒙版图像为空。")

    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    buffer_size = rgb_image.sizeInBytes() if hasattr(rgb_image, "sizeInBytes") else rgb_image.byteCount()
    image_bits = rgb_image.bits()
    image_bits.setsize(buffer_size)

    raw = np.frombuffer(image_bits, dtype=np.uint8)
    rows = raw.reshape((height, bytes_per_line))
    rgb = rows[:, : width * 3].reshape((height, width, 3))
    return np.any(rgb != 0, axis=2)


def _image_with_binary_mask(image: QImage, mask: np.ndarray) -> QImage:
    if image.isNull():
        return QImage()

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != (image.height(), image.width()):
        raise ValueError("蒙版尺寸必须与图像尺寸一致。")

    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    buffer_size = rgb_image.sizeInBytes() if hasattr(rgb_image, "sizeInBytes") else rgb_image.byteCount()
    image_bits = rgb_image.bits()
    image_bits.setsize(buffer_size)

    raw = np.frombuffer(image_bits, dtype=np.uint8)
    rows = np.array(raw.reshape((height, bytes_per_line)), copy=True)
    pixels = rows[:, : width * 3].reshape((height, width, 3))
    pixels[~mask_array] = 0
    return QImage(rows.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


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


class LiveStarMapGraphicsItem(QGraphicsItem):
    def __init__(
        self,
        renderer: StarMapRenderer,
        draw_background: bool = True,
        draw_horizon_shadow: bool = True,
    ) -> None:
        super().__init__()
        self.renderer = renderer
        self.draw_background = draw_background
        self.draw_horizon_shadow = draw_horizon_shadow
        self.star_map: ProjectedStarMap | None = None
        self.reference_stars: tuple[ReferenceStar, ...] = ()
        self.sky_transform: SkyAlignmentTransform | None = None
        self.target_size: tuple[int, int] | None = None
        self.element_scale = 1.0
        self.draw_common_names = True
        self.number_reference_stars = True
        self.view_zoom_scale = 1.0
        self._bounding_rect = QRectF()

    def set_star_map(
        self,
        star_map: ProjectedStarMap | None,
        reference_stars: tuple[ReferenceStar, ...] = (),
        sky_transform: SkyAlignmentTransform | None = None,
        target_size: tuple[int, int] | None = None,
        element_scale: float = 1.0,
        draw_common_names: bool = True,
        number_reference_stars: bool = True,
    ) -> None:
        self.prepareGeometryChange()
        self.star_map = star_map
        self.reference_stars = tuple(reference_stars)
        self.sky_transform = sky_transform
        self.target_size = target_size
        self.element_scale = max(float(element_scale), 0.05)
        self.draw_common_names = draw_common_names
        self.number_reference_stars = number_reference_stars
        self._bounding_rect = self._build_bounding_rect()
        self.update()

    def clear(self) -> None:
        self.set_star_map(None)

    def set_view_zoom_scale(self, zoom_scale: float) -> None:
        new_zoom_scale = max(float(zoom_scale), 1.0)
        if abs(new_zoom_scale - self.view_zoom_scale) <= 1e-3:
            return
        self.view_zoom_scale = new_zoom_scale
        self.update()

    def star_radius_zoom_scale(self) -> float:
        if self.view_zoom_scale <= 1.0:
            return 1.0
        return max(STAR_RADIUS_MIN_ZOOM_SCALE, self.view_zoom_scale ** (-STAR_RADIUS_ZOOM_EXPONENT))

    def boundingRect(self) -> QRectF:
        return QRectF(self._bounding_rect)

    def paint(self, painter, option, widget=None) -> None:  # type: ignore[no-untyped-def]
        star_map = self.star_map
        if star_map is None:
            return

        if self.sky_transform is None:
            self.renderer.paint(
                painter,
                star_map,
                reference_stars=self.reference_stars,
                element_scale=self.element_scale,
                draw_common_names=self.draw_common_names,
                number_reference_stars=self.number_reference_stars,
                draw_background=self.draw_background,
                draw_horizon_shadow=self.draw_horizon_shadow,
                star_radius_scale=self.star_radius_zoom_scale(),
            )
            return

        self._paint_sky_aligned_map(painter, star_map)

    def _build_bounding_rect(self) -> QRectF:
        star_map = self.star_map
        if star_map is None:
            return QRectF()
        if self.sky_transform is not None and self.target_size is not None:
            return QRectF(0.0, 0.0, float(self.target_size[0]), float(self.target_size[1]))
        return QRectF(0.0, 0.0, float(star_map.width), float(star_map.height))

    def _paint_sky_aligned_map(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        transform = self.sky_transform
        if transform is None:
            return

        rect = self.boundingRect()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.draw_background:
            painter.fillRect(rect, QColor(0, 0, 0))

        if len(star_map) > 0:
            points = transform.transform_radec_points(np.column_stack((star_map.ra_deg, star_map.dec_deg)))
            inside = self._inside_rect_mask(points, rect)
            order = star_map.radius_px.argsort()
            star_radius_scale = self.star_radius_zoom_scale()
            painter.setPen(Qt.NoPen)
            for index in order:
                if not bool(inside[index]):
                    continue
                red, green, blue = (int(value) for value in star_map.star_rgb[index])
                alpha = int(star_map.alpha[index])
                painter.setBrush(QColor(red, green, blue, alpha))
                radius = max(0.8, float(star_map.radius_px[index]) * self.element_scale * star_radius_scale)
                center = QPointF(float(points[index, 0]), float(points[index, 1]))
                painter.drawEllipse(QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0))

        self._draw_sky_aligned_solar_system_objects(painter, star_map, rect)
        if self.reference_stars and self.number_reference_stars:
            self._draw_sky_aligned_reference_stars(painter, rect)

    def _inside_rect_mask(self, points: np.ndarray, rect: QRectF) -> np.ndarray:
        return (
            np.all(np.isfinite(points), axis=1)
            & (points[:, 0] >= rect.left())
            & (points[:, 0] <= rect.right())
            & (points[:, 1] >= rect.top())
            & (points[:, 1] <= rect.bottom())
        )

    def _draw_sky_aligned_solar_system_objects(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        rect: QRectF,
    ) -> None:
        transform = self.sky_transform
        if transform is None or not star_map.solar_system_objects:
            return

        ra_dec = np.asarray(
            [(solar_object.ra_deg, solar_object.dec_deg) for solar_object in star_map.solar_system_objects],
            dtype=np.float64,
        )
        points = transform.transform_radec_points(ra_dec)
        inside = self._inside_rect_mask(points, rect)
        painter.setPen(Qt.NoPen)
        for index, solar_object in enumerate(star_map.solar_system_objects):
            if not bool(inside[index]):
                continue
            red, green, blue = solar_object.color_rgb
            radius = max(1.0, solar_object.radius_px * self.element_scale * self.star_radius_zoom_scale())
            alpha = int(solar_object.alpha)
            center = QPointF(float(points[index, 0]), float(points[index, 1]))
            painter.setBrush(QColor(red, green, blue, alpha))
            painter.drawEllipse(QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0))

    def _draw_sky_aligned_reference_stars(self, painter: QPainter, rect: QRectF) -> None:
        transform = self.sky_transform
        if transform is None:
            return

        label_scale = max(self.element_scale, 0.75)
        font = QFont()
        font.setPointSizeF(self.renderer.ui_config.reference_label_font_size_pt * label_scale)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        edge_padding_px = 4.0 * label_scale
        label_padding_x_px = 14.0 * label_scale
        label_padding_y_px = 8.0 * label_scale
        marker_radius = 17.0 * label_scale

        for reference_star in self.reference_stars:
            point_x, point_y = transform.transform_radec(reference_star.ra_deg, reference_star.dec_deg)
            if not rect.adjusted(-marker_radius, -marker_radius, marker_radius, marker_radius).contains(
                QPointF(point_x, point_y)
            ):
                continue

            marker_rect = QRectF(
                point_x - marker_radius,
                point_y - marker_radius,
                marker_radius * 2.0,
                marker_radius * 2.0,
            )
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 230), 5.0 * label_scale))
            painter.drawEllipse(marker_rect)
            painter.setPen(QPen(QColor(255, 230, 80, 255), 2.4 * label_scale))
            painter.drawEllipse(marker_rect)

            label_text = f"{reference_star.index}. {reference_star.name}"
            max_label_width = max(40.0 * label_scale, rect.width() - edge_padding_px * 2.0)
            visible_label_text = metrics.elidedText(label_text, Qt.ElideRight, int(max_label_width - label_padding_x_px))
            label_width = min(metrics.horizontalAdvance(visible_label_text) + label_padding_x_px, max_label_width)
            label_height = metrics.height() + label_padding_y_px
            preferred_x = point_x + marker_radius + 8.0 * label_scale
            if preferred_x + label_width > rect.right() - edge_padding_px:
                preferred_x = point_x - marker_radius - label_width - 8.0 * label_scale
            label_x = min(
                max(preferred_x, rect.left() + edge_padding_px),
                max(rect.left() + edge_padding_px, rect.right() - label_width - edge_padding_px),
            )
            label_y = min(
                max(point_y - label_height / 2.0, rect.top() + edge_padding_px),
                max(rect.top() + edge_padding_px, rect.bottom() - label_height - edge_padding_px),
            )
            label_rect = QRectF(label_x, label_y, label_width, label_height)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(label_rect, 4.0 * label_scale, 4.0 * label_scale)
            painter.setPen(QPen(QColor(255, 240, 130, 255), 1.0 * label_scale))
            painter.drawText(label_rect, Qt.AlignCenter, visible_label_text)


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


class SkyMaskLoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        file_path: str | Path,
        expected_size: tuple[int, int],
        source_image: QImage,
        source_path: Path,
    ) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.expected_size = expected_size
        self.source_image = source_image
        self.source_path = source_path

    def run(self) -> None:
        try:
            preview = load_image_preview(self.file_path, max_long_side_px=None)
            expected_width, expected_height = self.expected_size
            if preview.image.width() != expected_width or preview.image.height() != expected_height:
                raise ValueError(
                    "蒙版尺寸必须与真实图像一致：真实图像 {image_width} x {image_height} px，"
                    "蒙版 {mask_width} x {mask_height} px。".format(
                        image_width=expected_width,
                        image_height=expected_height,
                        mask_width=preview.image.width(),
                        mask_height=preview.image.height(),
                    )
                )

            mask = _qimage_to_binary_mask(preview.image)
            if not np.any(mask):
                raise ValueError("蒙版中没有任何非零像素，无法参与星点匹配。")

            masked_image = _image_with_binary_mask(self.source_image, mask)
            self.finished.emit((preview.path, self.source_path, mask, masked_image))
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有蒙版读取错误传回界面层。
            self.failed.emit(str(exc))


class ReferenceJsonImportWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self.file_path = Path(file_path)

    def run(self) -> None:
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            self.finished.emit((self.file_path, payload))
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有 JSON 读取错误传回界面层。
            self.failed.emit(str(exc))


class StarPairSessionImportWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self.file_path = Path(file_path)

    def run(self) -> None:
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            image_path = self._real_image_path(payload)
            preview = load_image_preview(image_path, max_long_side_px=None)
            self.finished.emit((self.file_path, payload, preview))
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有 JSON/图像读取错误传回界面层。
            self.failed.emit(str(exc))

    def _real_image_path(self, payload: object) -> Path:
        return _resolve_star_pair_session_real_image_path(payload, self.file_path)


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
        self.star_map_item = LiveStarMapGraphicsItem(self.renderer)
        self.scene.addItem(self.star_map_item)
        self.ui.starMapView.setScene(self.scene)
        self.ui.starMapView.viewport().installEventFilter(self)

        self.reference_scene = QGraphicsScene(self)
        self.reference_star_map_item = LiveStarMapGraphicsItem(self.renderer)
        self.reference_scene.addItem(self.reference_star_map_item)
        self.ui.referenceImageView.setScene(self.reference_scene)
        self.ui.referenceImageView.installEventFilter(self)
        self.ui.referenceImageView.viewport().installEventFilter(self)
        self.ui.referenceImageView.viewport().setFocusPolicy(Qt.StrongFocus)
        self.ui.referenceImageView.viewport().setMouseTracking(True)

        self.real_image_scene = QGraphicsScene(self)
        self.real_image_item = GraphicsImageItem()
        self.real_image_scene.addItem(self.real_image_item)
        self.real_reference_overlay_item = LiveStarMapGraphicsItem(
            self.renderer,
            draw_background=False,
            draw_horizon_shadow=False,
        )
        self.real_reference_overlay_item.setZValue(5.0)
        self.real_reference_overlay_item.setVisible(False)
        self.real_image_scene.addItem(self.real_reference_overlay_item)
        self.ui.realImageView.setScene(self.real_image_scene)
        self.ui.realImageView.installEventFilter(self)
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
        self._json_import_thread: QThread | None = None
        self._json_import_worker: QObject | None = None
        self._json_import_progress: QProgressDialog | None = None
        self._mask_import_thread: QThread | None = None
        self._mask_import_worker: QObject | None = None
        self._mask_import_progress: QProgressDialog | None = None
        self._real_image_zoom_max_scale = REAL_IMAGE_MAX_ZOOM_SCALE
        self._syncing_camera_dimensions = False
        self._active_star_pair_row: int | None = None
        self._reference_pick_press_pos: QPoint | None = None
        self._star_pick_cursor: QCursor | None = None
        self._star_pick_circle_diameter_px = self.ui_config.star_pick_circle_default_diameter_px
        self._star_pick_previous_drag_mode = self.ui.realImageView.dragMode()
        self._star_pair_annotations: dict[str, tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        self._focused_star_annotations: list[QGraphicsItem] = []
        self._current_star_map: ProjectedStarMap | None = None
        self._current_reference_stars: tuple[ReferenceStar, ...] = ()
        self._sky_alignment_transform: SkyAlignmentTransform | None = None
        self._source_astrometric_model: SourceAstrometricModel | None = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""
        self._syncing_reference_real_views = False
        self._syncing_reference_preview_splitter = False
        self._suspend_alignment_updates = False
        self._manual_reference_star_ids: list[str] = []
        self._auto_match_reference_star_ids: list[str] = []
        self._auto_match_constraint_by_star_id: dict[str, tuple[str, float]] = {}
        self._auto_match_group_expanded = False
        self._excluded_reference_star_ids: list[str] = []
        self._star_pick_native_zoom_remainder = 0.0
        self.current_sky_mask_path: Path | None = None
        self.current_sky_mask: np.ndarray | None = None
        self.current_sky_masked_image: QImage | None = None

        self._init_defaults()
        self._connect_inputs()
        self._configure_reference_preview_splitter()
        self.schedule_render(delay_ms=0)

    def _apply_ui_font_config(self, ui_config: StarMapUiConfig) -> None:
        controls_font = QFont(self.font())
        controls_font.setPointSize(ui_config.controls_font_size_pt)
        self.setFont(controls_font)
        self.ui.centralwidget.setFont(controls_font)

        status_font = QFont(self.ui.statusbar.font())
        status_font.setPointSize(ui_config.status_bar_font_size_pt)
        self.ui.statusbar.setFont(status_font)

    def _set_plain_label_text(self, label: QLabel, text: str, tooltip: str | None = None) -> None:
        display_text = text.strip()
        label.setText(display_text)
        label.setToolTip((tooltip or display_text).strip())

    def _refresh_elided_label(self, label: QLabel) -> None:
        full_text = str(label.property("fullText") or "")
        if not full_text:
            return
        available_width = max(12, label.contentsRect().width() - 2)
        label.setText(label.fontMetrics().elidedText(full_text, Qt.ElideRight, available_width))

    def _set_elided_label_text(self, label: QLabel, text: str, tooltip: str | None = None) -> None:
        full_text = text.strip()
        label.setProperty("fullText", full_text)
        label.setToolTip((tooltip or full_text).strip())
        self._refresh_elided_label(label)
        QTimer.singleShot(0, lambda label=label: self._refresh_elided_label(label))

    def _refresh_all_elided_labels(self) -> None:
        self._refresh_elided_label(self.ui.labelImportedImagePath)
        self._refresh_elided_label(self.ui.labelSkyMaskStatus)
        self._refresh_elided_label(self.ui.labelAlignmentTransformStatus)

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
        self.ui.comboBoxSkyAlignmentModel.setCurrentIndex(0)
        self.ui.spinBoxAutoMatchCount.setValue(self.ui_config.auto_match_default_new_count)
        constraint_index = (
            AUTO_MATCH_CONSTRAINT_MODES.index(self.ui_config.auto_match_default_constraint_mode)
            if self.ui_config.auto_match_default_constraint_mode in AUTO_MATCH_CONSTRAINT_MODES
            else AUTO_MATCH_CONSTRAINT_MODES.index(AUTO_MATCH_CONSTRAINT_SOFT)
        )
        self.ui.comboBoxAutoMatchConstraintMode.setCurrentIndex(constraint_index)
        self.ui.doubleSpinBoxAutoMatchSoftWeight.setValue(self.ui_config.auto_match_default_soft_weight)
        self.ui.spinBoxAutoMatchRadius.setValue(30)
        self._reset_imported_image_labels()
        self._reset_sky_mask_status()
        self._update_reference_label_controls()
        self._update_auto_match_controls()
        self._update_lens_model_controls()
        self._update_reference_overlay_opacity_label()
        self._update_reference_alignment_controls()

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
        self.ui.spinBoxReferenceStarCount.valueChanged.connect(self._handle_reference_label_options_changed)
        self.ui.doubleSpinBoxReferenceMagLimit.valueChanged.connect(self._handle_reference_label_options_changed)
        self.ui.pushButtonSwapOrientation.clicked.connect(self._swap_camera_orientation)
        self.ui.pushButtonExportReference.clicked.connect(self.export_reference_map)
        self.ui.pushButtonImportSingleImage.clicked.connect(self.import_single_image)
        self.ui.pushButtonImportImageSequence.clicked.connect(self.show_sequence_import_placeholder)
        self.ui.pushButtonExportStarPairs.clicked.connect(self.export_star_pair_session)
        self.ui.pushButtonImportStarPairs.clicked.connect(self.import_star_pair_session)
        self.ui.pushButtonClearStarPairs.clicked.connect(self.clear_all_star_pair_positions)
        self.ui.pushButtonImportSkyMask.clicked.connect(self.import_sky_mask)
        self.ui.pushButtonClearSkyMask.clicked.connect(self.clear_sky_mask)
        self.ui.checkBoxShowSkyMask.toggled.connect(self._refresh_real_image_display_for_mask)
        self.ui.comboBoxSkyAlignmentModel.currentIndexChanged.connect(self._handle_alignment_model_changed)
        self.ui.comboBoxAutoMatchConstraintMode.currentIndexChanged.connect(self._update_auto_match_controls)
        self.ui.pushButtonAutoMatchFieldStars.clicked.connect(self.auto_match_field_stars)
        self.ui.pushButtonExportSourceModel.clicked.connect(self.export_source_model_json)
        self.ui.actionImportSingleImage.triggered.connect(self.import_single_image)
        self.ui.actionImportImageSequence.triggered.connect(self.show_sequence_import_placeholder)
        self.ui.tabWidgetMain.currentChanged.connect(self._handle_tab_changed)
        self.ui.tableWidgetStarPairs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.tableWidgetStarPairs.customContextMenuRequested.connect(self._show_star_pair_context_menu)
        self.ui.tableWidgetStarPairs.itemChanged.connect(self._handle_star_pair_item_changed)
        self.ui.tableWidgetStarPairs.cellClicked.connect(self._handle_star_pair_cell_clicked)
        self.ui.tableWidgetStarPairs.cellDoubleClicked.connect(self._handle_star_pair_cell_double_clicked)
        self.ui.tableWidgetStarPairs.installEventFilter(self)
        self.ui.labelImportedImagePath.installEventFilter(self)
        self.ui.labelSkyMaskStatus.installEventFilter(self)
        self.ui.labelAlignmentTransformStatus.installEventFilter(self)
        self.ui.pushButtonImportReferenceJson.clicked.connect(self.import_reference_json)
        self.ui.checkBoxOverlayReferenceMap.toggled.connect(self._update_reference_alignment_display)
        self.ui.doubleSpinBoxReferenceOverlayOpacity.valueChanged.connect(self._handle_reference_overlay_opacity_changed)
        self.ui.checkBoxSyncReferenceAndRealView.toggled.connect(self._handle_reference_real_sync_toggled)
        self.ui.checkBoxHideAllAnnotations.toggled.connect(self._handle_hide_all_annotations_toggled)
        self.ui.referenceImageView.horizontalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.referenceImageView)
        )
        self.ui.referenceImageView.verticalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.referenceImageView)
        )
        self.ui.realImageView.horizontalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.realImageView)
        )
        self.ui.realImageView.verticalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.realImageView)
        )

    def _configure_reference_preview_splitter(self) -> None:
        splitter = self.ui.splitterReferenceAndRealImage
        splitter.setChildrenCollapsible(False)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.installEventFilter(self)
        splitter.splitterMoved.connect(lambda _pos, _index: self._set_equal_reference_preview_sizes())
        QTimer.singleShot(0, self._set_equal_reference_preview_sizes)

    def _set_equal_reference_preview_sizes(self) -> None:
        if self._syncing_reference_preview_splitter:
            return
        splitter = self.ui.splitterReferenceAndRealImage
        total_width = sum(splitter.sizes())
        if total_width <= 0:
            total_width = max(0, splitter.width() - splitter.handleWidth())
        if total_width <= 0:
            return

        # 两个预览区需要并排对照星点，始终把分割器恢复成左右等宽。
        left_width = total_width // 2
        right_width = total_width - left_width
        self._syncing_reference_preview_splitter = True
        try:
            splitter.setSizes([left_width, right_width])
        finally:
            self._syncing_reference_preview_splitter = False

    def _reset_imported_image_labels(self) -> None:
        self._set_elided_label_text(self.ui.labelImportedImagePath, "未导入", "")
        self.ui.labelImportedImageSize.setText("-")

    def _update_imported_image_labels(self, preview: ImagePreview) -> None:
        image_path = str(Path(preview.path).expanduser().resolve())
        self._set_elided_label_text(self.ui.labelImportedImagePath, image_path, image_path)
        self.ui.labelImportedImageSize.setText(f"{preview.original_width} x {preview.original_height} px")

    def _reset_sky_mask_status(self) -> None:
        self.current_sky_mask_path = None
        self.current_sky_mask = None
        self.current_sky_masked_image = None
        self._set_elided_label_text(self.ui.labelSkyMaskStatus, "未使用蒙版", "")

    def _update_sky_mask_status(self) -> None:
        if self.current_sky_mask is None:
            self._reset_sky_mask_status()
            return

        valid_fraction = float(np.count_nonzero(self.current_sky_mask)) / max(float(self.current_sky_mask.size), 1.0)
        path_text = str(self.current_sky_mask_path) if self.current_sky_mask_path is not None else "内存蒙版"
        self._set_elided_label_text(
            self.ui.labelSkyMaskStatus,
            f"蒙版有效区域 {valid_fraction * 100.0:.1f}%",
            path_text,
        )

    def _sky_mask_allows_point(self, x_px: float, y_px: float) -> bool:
        if self.current_sky_mask is None:
            return True
        if not (math.isfinite(x_px) and math.isfinite(y_px)):
            return False

        mask_height, mask_width = self.current_sky_mask.shape
        image_x = int(round(x_px))
        image_y = int(round(y_px))
        if image_x < 0 or image_x >= mask_width or image_y < 0 or image_y >= mask_height:
            return False
        return bool(self.current_sky_mask[image_y, image_x])

    def _clear_sky_mask_if_size_mismatch(self, image_width: int, image_height: int) -> None:
        if self.current_sky_mask is None:
            return
        mask_height, mask_width = self.current_sky_mask.shape
        if mask_width == image_width and mask_height == image_height:
            self.current_sky_masked_image = None
            return
        self._reset_sky_mask_status()
        self.ui.statusbar.showMessage("新的真实图像尺寸与已有蒙版不一致，已自动清除蒙版。")

    def _alignment_model(self) -> str:
        index = self.ui.comboBoxSkyAlignmentModel.currentIndex()
        if index < 0 or index >= len(SKY_ALIGNMENT_MODELS):
            return SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION
        return SKY_ALIGNMENT_MODELS[index]

    def _auto_match_constraint_mode(self) -> str:
        index = self.ui.comboBoxAutoMatchConstraintMode.currentIndex()
        if index < 0 or index >= len(AUTO_MATCH_CONSTRAINT_MODES):
            return AUTO_MATCH_CONSTRAINT_SOFT
        return AUTO_MATCH_CONSTRAINT_MODES[index]

    def _auto_match_soft_weight(self) -> float:
        if self._auto_match_constraint_mode() != AUTO_MATCH_CONSTRAINT_SOFT:
            return 1.0
        return max(0.01, min(1.0, float(self.ui.doubleSpinBoxAutoMatchSoftWeight.value())))

    def _set_alignment_model(self, model: object) -> None:
        model_text = str(model or "").strip()
        model_text = SKY_ALIGNMENT_MODEL_ALIASES.get(model_text, model_text)
        if model_text not in SKY_ALIGNMENT_MODELS:
            return
        self.ui.comboBoxSkyAlignmentModel.setCurrentIndex(SKY_ALIGNMENT_MODELS.index(model_text))

    def _handle_alignment_model_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_reference_alignment_transform()

    def _update_auto_match_controls(self, *unused) -> None:  # type: ignore[no-untyped-def]
        soft_mode = self._auto_match_constraint_mode() == AUTO_MATCH_CONSTRAINT_SOFT
        self.ui.labelAutoMatchSoftWeight.setEnabled(soft_mode)
        self.ui.doubleSpinBoxAutoMatchSoftWeight.setEnabled(soft_mode)

    def _real_image_for_current_mask_preview(self) -> QImage:
        if self.current_image_preview is None:
            return QImage()
        image = self.current_image_preview.image
        if self.current_sky_mask is not None and self.ui.checkBoxShowSkyMask.isChecked():
            if self.current_sky_masked_image is None:
                self.current_sky_masked_image = _image_with_binary_mask(image, self.current_sky_mask)
            return self.current_sky_masked_image
        return image

    def _refresh_real_image_display_for_mask(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self.current_image_preview is None:
            return
        self.real_image_item.set_image(self._real_image_for_current_mask_preview())

    def _set_mask_import_controls_enabled(self, enabled: bool) -> None:
        self.ui.pushButtonImportSkyMask.setEnabled(enabled and self.current_image_preview is not None)
        self.ui.pushButtonClearSkyMask.setEnabled(enabled and self.current_sky_mask is not None)
        self.ui.checkBoxShowSkyMask.setEnabled(enabled and self.current_sky_mask is not None)

    def _reference_star_lookup(self) -> dict[str, ReferenceStar]:
        return {star.star_id.strip(): star for star in self._current_reference_stars if star.star_id.strip()}

    def _reference_star_for_row(self, row: int) -> ReferenceStar | None:
        star_id = self._star_pair_star_id(row)
        if not star_id:
            return None
        return self._reference_star_lookup().get(star_id)

    def _matched_sky_alignment_points(self) -> tuple[np.ndarray, np.ndarray]:
        sky_points, target_points, _weights, _anchor_mask = self._matched_sky_alignment_data()
        return sky_points, target_points

    def _matched_sky_alignment_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        star_lookup = self._reference_star_lookup()
        sky_points: list[tuple[float, float]] = []
        target_points: list[tuple[float, float]] = []
        point_weights: list[float] = []
        anchor_flags: list[bool] = []
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            star_id = self._star_pair_star_id(row)
            reference_star = star_lookup.get(star_id)
            target_position = self._parse_star_pair_position_text(row)
            if reference_star is None or target_position is None:
                continue
            sky_points.append((reference_star.ra_deg, reference_star.dec_deg))
            target_points.append(target_position)
            mode, fit_weight = self._star_pair_fit_constraint(row)
            point_weights.append(fit_weight)
            anchor_flags.append(mode != AUTO_MATCH_CONSTRAINT_SOFT)
        return (
            np.asarray(sky_points, dtype=np.float64),
            np.asarray(target_points, dtype=np.float64),
            np.asarray(point_weights, dtype=np.float64),
            np.asarray(anchor_flags, dtype=bool),
        )

    def _initial_projection_rotation_matrix(self) -> np.ndarray | None:
        reference_stars = [
            star
            for star in self._current_reference_stars
            if all(
                math.isfinite(value)
                for value in (star.ra_deg, star.dec_deg, star.alt_deg, star.az_deg)
            )
        ]
        if len(reference_stars) < 3:
            return None

        world_vectors = radec_to_unit_vectors(
            np.asarray([star.ra_deg for star in reference_stars], dtype=np.float64),
            np.asarray([star.dec_deg for star in reference_stars], dtype=np.float64),
        )
        local_vectors = local_vectors_from_altaz(
            np.asarray([star.alt_deg for star in reference_stars], dtype=np.float64),
            np.asarray([star.az_deg for star in reference_stars], dtype=np.float64),
        )
        finite = np.all(np.isfinite(world_vectors), axis=1) & np.all(np.isfinite(local_vectors), axis=1)
        if np.count_nonzero(finite) < 3:
            return None

        try:
            local_from_world_transposed, _residuals, _rank, _singular_values = np.linalg.lstsq(
                world_vectors[finite],
                local_vectors[finite],
                rcond=None,
            )
            local_from_world = local_from_world_transposed.T
            u_matrix, _values, vt_matrix = np.linalg.svd(local_from_world)
        except np.linalg.LinAlgError:
            return None
        local_from_world = u_matrix @ vt_matrix
        if np.linalg.det(local_from_world) < 0.0:
            u_matrix[:, -1] *= -1.0
            local_from_world = u_matrix @ vt_matrix

        camera_from_local = np.vstack(camera_basis_from_view(self._view_settings())).astype(np.float64)
        rotation_matrix = camera_from_local @ local_from_world
        if rotation_matrix.shape != (3, 3) or not np.all(np.isfinite(rotation_matrix)):
            return None
        return rotation_matrix.astype(np.float64)

    def _star_pair_alignment_residual(self, row: int) -> tuple[float, float, float] | None:
        transform = self._sky_alignment_transform
        if transform is None:
            return None

        reference_star = self._reference_star_for_row(row)
        target_position = self._parse_star_pair_position_text(row)
        if reference_star is None or target_position is None:
            return None

        predicted_x, predicted_y = transform.transform_radec(reference_star.ra_deg, reference_star.dec_deg)
        if not all(math.isfinite(value) for value in (predicted_x, predicted_y)):
            return None
        dx = predicted_x - target_position[0]
        dy = predicted_y - target_position[1]
        distance = float(np.hypot(dx, dy))
        return float(dx), float(dy), distance

    def _alignment_residual_distances(self) -> np.ndarray:
        distances: list[float] = []
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            residual = self._star_pair_alignment_residual(row)
            if residual is not None:
                distances.append(residual[2])
        return np.asarray(distances, dtype=np.float64)

    def _residual_warning_thresholds(self) -> tuple[float, float]:
        transform = self._sky_alignment_transform
        if transform is None:
            return RESIDUAL_WARNING_MIN_PX, RESIDUAL_SEVERE_MIN_PX
        warning = max(RESIDUAL_WARNING_MIN_PX, float(transform.rms_px))
        severe = max(RESIDUAL_SEVERE_MIN_PX, float(transform.rms_px) * RESIDUAL_SEVERE_RMS_SCALE)
        return warning, severe

    def _ensure_star_pair_residual_item(self, row: int, column: int) -> QTableWidgetItem:
        table = self.ui.tableWidgetStarPairs
        item = table.item(row, column)
        if item is None:
            item = self._read_only_table_item("")
            table.setItem(row, column, item)
        else:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _update_star_pair_residual_columns(self) -> None:
        table = self.ui.tableWidgetStarPairs
        signals_were_blocked = table.blockSignals(True)
        for row in range(table.rowCount()):
            residual_item = self._ensure_star_pair_residual_item(row, STAR_PAIR_RESIDUAL_COLUMN)

            residual = self._star_pair_alignment_residual(row)
            if residual is None:
                residual_item.setText("")
                residual_item.setData(Qt.UserRole, None)
                residual_item.setToolTip("")
                continue

            _dx, _dy, distance = residual
            residual_item.setText(f"{distance:.2f}")
            residual_item.setData(Qt.UserRole, distance)
            residual_item.setToolTip("残差为天球 RA/Dec 模型预测位置与真实图像记录位置之间的像素距离。")
        table.blockSignals(signals_were_blocked)

        table.resizeColumnToContents(STAR_PAIR_RESIDUAL_COLUMN)
        self._update_auto_match_group_row_text()
        self._refresh_star_pair_table_styles()

    def _update_reference_alignment_transform(self) -> None:
        self._sky_alignment_transform = None
        self._source_astrometric_model = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""
        if self._suspend_alignment_updates:
            return
        if self.current_image_preview is None:
            self._reference_alignment_error_message = "导入真实图像后可计算实时星空叠加。"
            self._sky_alignment_error_message = "导入真实图像后可计算天球残差。"
            self._source_model_error_message = "导入真实图像后可生成 xy→RA/Dec 映射。"
            self._update_star_pair_residual_columns()
            self._update_reference_alignment_display()
            return

        sky_points, sky_target_points, fit_weights, anchor_mask = self._matched_sky_alignment_data()
        if sky_points.shape[0] < MIN_ALIGNMENT_PAIRS:
            self._reference_alignment_error_message = (
                f"已配对 {sky_points.shape[0]} 颗星；至少 {MIN_ALIGNMENT_PAIRS} 颗后可实时叠加星空。"
            )
            self._sky_alignment_error_message = (
                f"已配对 {sky_points.shape[0]} 颗星；至少 {MIN_ALIGNMENT_PAIRS} 颗后可计算天球残差。"
            )
            self._source_model_error_message = (
                f"已配对 {sky_points.shape[0]} 颗星；至少 {MIN_ALIGNMENT_PAIRS} 颗后可生成 xy→RA/Dec 映射。"
            )
            self._update_star_pair_residual_columns()
            self._update_reference_alignment_display()
            return

        try:
            image = self.current_image_preview.image
            initial_rotation_matrix = self._initial_projection_rotation_matrix()
            self._sky_alignment_transform = fit_sky_alignment(
                ra_dec_points=sky_points,
                target_points=sky_target_points,
                matching_model=self._alignment_model(),
                image_size=(image.width(), image.height()),
                fisheye_fov_deg=self.ui.doubleSpinBoxFisheyeFov.value(),
                initial_rotation_matrix=initial_rotation_matrix,
                point_weights=fit_weights,
                residual_anchor_mask=anchor_mask,
            )
        except Exception as exc:  # noqa: BLE001 - 天球残差失败需要直接反馈给交互界面。
            self._sky_alignment_error_message = str(exc)
        if self._sky_alignment_transform is not None and self.current_image_preview is not None:
            try:
                image = self.current_image_preview.image
                initial_rotation_matrix = self._initial_projection_rotation_matrix()
                self._source_astrometric_model = fit_source_astrometric_model(
                    ra_dec_points=sky_points,
                    pixel_points=sky_target_points,
                    image_size=(image.width(), image.height()),
                    matching_model=self._alignment_model(),
                    fisheye_fov_deg=self.ui.doubleSpinBoxFisheyeFov.value(),
                    initial_rotation_matrix=initial_rotation_matrix,
                    point_weights=fit_weights,
                    residual_anchor_mask=anchor_mask,
                )
            except Exception as exc:  # noqa: BLE001 - 源图模型错误要保留给导出按钮和状态栏。
                self._source_model_error_message = str(exc)
        self._update_star_pair_residual_columns()
        self._update_reference_alignment_display()

    def _update_reference_overlay_opacity_label(self) -> None:
        opacity = self.ui.doubleSpinBoxReferenceOverlayOpacity.value()
        self.ui.doubleSpinBoxReferenceOverlayOpacity.setToolTip(f"实时星空叠加透明度：{opacity:.1f}%")

    def _reference_overlay_opacity(self) -> float:
        return max(0.0, min(1.0, self.ui.doubleSpinBoxReferenceOverlayOpacity.value() / 100.0))

    def _hide_all_annotations(self) -> bool:
        return self.ui.checkBoxHideAllAnnotations.isChecked()

    def _set_alignment_status_text(self, text: str, tooltip: str | None = None) -> None:
        self._set_elided_label_text(
            self.ui.labelAlignmentTransformStatus,
            text.strip(),
            (tooltip or text).strip(),
        )

    def _handle_reference_overlay_opacity_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_reference_overlay_opacity_label()
        self.real_reference_overlay_item.setOpacity(self._reference_overlay_opacity())
        self._update_reference_alignment_controls()

    def _handle_hide_all_annotations_toggled(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._clear_focused_star_annotations()
        self._update_star_pair_annotation_visibility()
        self._update_reference_alignment_display()

    def _handle_reference_real_sync_toggled(self, checked: bool) -> None:
        if checked and self._can_sync_reference_real_views():
            self._sync_reference_real_view_from(self.ui.realImageView, force=True)

    def _reference_alignment_scene_rect(self) -> QRectF:
        if self.current_image_preview is None:
            return QRectF()
        return QRectF(
            0.0,
            0.0,
            float(self.current_image_preview.image.width()),
            float(self.current_image_preview.image.height()),
        )

    def _update_reference_alignment_controls(self) -> None:
        has_alignment = self._sky_alignment_transform is not None and self.current_image_preview is not None
        has_source_model = self._source_astrometric_model is not None and self.current_image_preview is not None
        self.ui.checkBoxOverlayReferenceMap.setEnabled(has_alignment)
        self.ui.labelReferenceOverlayOpacityTitle.setEnabled(True)
        self.ui.doubleSpinBoxReferenceOverlayOpacity.setEnabled(True)
        self.ui.checkBoxSyncReferenceAndRealView.setEnabled(has_alignment)
        mask_controls_enabled = self._mask_import_thread is None
        self.ui.pushButtonImportSkyMask.setEnabled(mask_controls_enabled and self.current_image_preview is not None)
        self.ui.pushButtonClearSkyMask.setEnabled(mask_controls_enabled and self.current_sky_mask is not None)
        self.ui.checkBoxShowSkyMask.setEnabled(mask_controls_enabled and self.current_sky_mask is not None)
        if self.current_sky_mask is None and self.ui.checkBoxShowSkyMask.isChecked():
            was_blocked = self.ui.checkBoxShowSkyMask.blockSignals(True)
            self.ui.checkBoxShowSkyMask.setChecked(False)
            self.ui.checkBoxShowSkyMask.blockSignals(was_blocked)
        self.ui.pushButtonAutoMatchFieldStars.setEnabled(has_alignment)
        self.ui.pushButtonExportSourceModel.setEnabled(has_source_model)
        if not has_alignment and self.ui.checkBoxSyncReferenceAndRealView.isChecked():
            self.ui.checkBoxSyncReferenceAndRealView.blockSignals(True)
            self.ui.checkBoxSyncReferenceAndRealView.setChecked(False)
            self.ui.checkBoxSyncReferenceAndRealView.blockSignals(False)

        sky_transform = self._sky_alignment_transform
        if sky_transform is not None:
            source_model_text = ""
            if self._source_astrometric_model is not None:
                source_model_text = f"，映射可导出"
            elif self._source_model_error_message:
                source_model_text = "，映射未就绪"
            distances = self._alignment_residual_distances()
            compact_summary = ""
            residual_summary = "暂无逐星残差"
            if distances.size > 0:
                median_distance = float(np.median(distances))
                max_distance = float(np.max(distances))
                compact_summary = f"，中位 {median_distance:.2f}，最大 {max_distance:.2f}"
                residual_summary = f"中位 {median_distance:.2f} px，最大 {max_distance:.2f} px"
            projection_rms = getattr(sky_transform, "projection_rms_px", None)
            projection_summary = ""
            projection_tooltip = ""
            if projection_rms is not None and math.isfinite(float(projection_rms)):
                projection_summary = f"，投影 {float(projection_rms):.2f}px"
                projection_tooltip = f"\n已知投影原始 RMS：{float(projection_rms):.2f} px。"
            soft_count = int(getattr(sky_transform, "residual_soft_constraint_count", 0) or 0)
            soft_summary = f"，软约束 {soft_count}" if soft_count > 0 else ""
            soft_tooltip = ""
            if soft_count > 0:
                soft_tooltip = (
                    "\n残差修正包含 {count} 个软约束点，权重范围 {min_weight:.2f}-{max_weight:.2f}。"
                ).format(
                    count=soft_count,
                    min_weight=float(getattr(sky_transform, "residual_soft_weight_min", 1.0)),
                    max_weight=float(getattr(sky_transform, "residual_soft_weight_max", 1.0)),
                )
            display_text = (
                "配准 {count} 对，{model} RMS {rms:.2f}px{summary}{projection}{soft}{source_model}".format(
                    count=sky_transform.pair_count,
                    model=sky_transform.display_name,
                    rms=sky_transform.rms_px,
                    summary=compact_summary,
                    projection=projection_summary,
                    soft=soft_summary,
                    source_model=source_model_text,
                )
            )
            tooltip = (
                "实时星空叠加：已用 {count} 对星建立 RA/Dec {model}，RMS {rms:.2f} px。\n"
                "天球残差：{residual_summary}。{projection_tooltip}{soft_tooltip}\n"
                "源图映射：{source_model_summary}".format(
                    count=sky_transform.pair_count,
                    model=sky_transform.display_name,
                    rms=sky_transform.rms_px,
                    residual_summary=residual_summary,
                    projection_tooltip=projection_tooltip,
                    soft_tooltip=soft_tooltip,
                    source_model_summary=(
                        "已可导出 xy→RA/Dec JSON。"
                        if self._source_astrometric_model is not None
                        else self._source_model_error_message or "尚未就绪。"
                    ),
                )
            )
            self._set_alignment_status_text(display_text, tooltip)
        else:
            status_text = (
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or self._source_model_error_message
                or f"至少配对 {MIN_ALIGNMENT_PAIRS} 颗星后可自动配准。"
            )
            self._set_alignment_status_text(status_text)

    def _update_reference_alignment_display(self, *unused) -> None:  # type: ignore[no-untyped-def]
        star_map = self._current_star_map
        transform = self._sky_alignment_transform
        has_alignment = transform is not None and self.current_image_preview is not None
        self._update_reference_alignment_controls()

        if star_map is None:
            self.reference_star_map_item.clear()
            self.real_reference_overlay_item.clear()
            self.real_reference_overlay_item.setVisible(False)
            self._last_reference_render_size = None
            return

        if has_alignment:
            assert transform is not None
            scene_rect = self._reference_alignment_scene_rect()
            target_size = (int(scene_rect.width()), int(scene_rect.height()))
            display_key: tuple[object, ...] = ("aligned", target_size[0], target_size[1])
            element_scale = self._aligned_star_element_scale(target_size)
            number_reference_stars = not self._hide_all_annotations()
            self.reference_star_map_item.set_star_map(
                star_map,
                reference_stars=self._current_reference_stars,
                sky_transform=transform,
                target_size=target_size,
                element_scale=element_scale,
                draw_common_names=False,
                number_reference_stars=number_reference_stars,
            )
            if not scene_rect.isEmpty():
                self.reference_scene.setSceneRect(scene_rect)
        else:
            target_size = None
            display_key = ("native", star_map.width, star_map.height)
            number_reference_stars = not self._hide_all_annotations()
            self.reference_star_map_item.set_star_map(
                star_map,
                reference_stars=self._current_reference_stars,
                sky_transform=None,
                target_size=None,
                element_scale=1.0,
                draw_common_names=False,
                number_reference_stars=number_reference_stars,
            )
            self.reference_scene.setSceneRect(0.0, 0.0, float(star_map.width), float(star_map.height))
        self._fit_reference_map_if_display_changed(display_key)

        overlay_visible = (
            has_alignment
            and self.current_image_preview is not None
            and self.ui.checkBoxOverlayReferenceMap.isChecked()
        )
        if overlay_visible:
            assert transform is not None
            assert target_size is not None
            self.real_reference_overlay_item.set_star_map(
                star_map,
                reference_stars=self._current_reference_stars,
                sky_transform=transform,
                target_size=target_size,
                element_scale=self._aligned_star_element_scale(target_size),
                draw_common_names=False,
                number_reference_stars=not self._hide_all_annotations(),
            )
            self.real_reference_overlay_item.setOpacity(self._reference_overlay_opacity())
            self.real_reference_overlay_item.setVisible(True)
        else:
            self.real_reference_overlay_item.setVisible(False)

        self._update_live_star_map_zoom_scale(self.ui.referenceImageView)
        self._update_live_star_map_zoom_scale(self.ui.realImageView)
        if self.ui.checkBoxSyncReferenceAndRealView.isChecked() and self._can_sync_reference_real_views():
            self._sync_reference_real_view_from(self.ui.realImageView, force=True)

    def _can_sync_reference_real_views(self) -> bool:
        return (
            self.ui.checkBoxSyncReferenceAndRealView.isChecked()
            and self._sky_alignment_transform is not None
            and self.current_image_preview is not None
        )

    def _sync_reference_real_view_from(self, source_view: QGraphicsView, force: bool = False) -> None:
        if self._syncing_reference_real_views:
            return
        if not force and not self._can_sync_reference_real_views():
            return
        if source_view not in (self.ui.referenceImageView, self.ui.realImageView):
            return

        target_view = self.ui.realImageView if source_view is self.ui.referenceImageView else self.ui.referenceImageView
        self._syncing_reference_real_views = True
        try:
            target_view.setTransform(source_view.transform())
            source_center = source_view.mapToScene(source_view.viewport().rect().center())
            target_view.centerOn(source_center)
            if target_view is self.ui.realImageView:
                self._cap_graphics_view_to_max_scale(target_view)
            self._update_live_star_map_zoom_scale(source_view)
            self._update_live_star_map_zoom_scale(target_view)
        finally:
            self._syncing_reference_real_views = False

    def _star_pair_row_type(self, row: int) -> str:
        item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_INDEX_COLUMN)
        if item is None:
            return STAR_PAIR_ROW_TYPE_MANUAL
        row_type = str(item.data(STAR_PAIR_ROW_TYPE_ROLE) or STAR_PAIR_ROW_TYPE_MANUAL)
        if row_type not in {
            STAR_PAIR_ROW_TYPE_MANUAL,
            STAR_PAIR_ROW_TYPE_AUTO_GROUP,
            STAR_PAIR_ROW_TYPE_AUTO_MATCH,
        }:
            return STAR_PAIR_ROW_TYPE_MANUAL
        return row_type

    def _is_auto_match_group_row(self, row: int) -> bool:
        return self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_AUTO_GROUP

    def _is_auto_match_row(self, row: int) -> bool:
        return self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_AUTO_MATCH

    def _auto_match_constraint_for_star_id(self, star_id: str) -> tuple[str, float]:
        mode, weight = self._auto_match_constraint_by_star_id.get(
            star_id,
            (AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0),
        )
        if mode not in AUTO_MATCH_CONSTRAINT_MODES:
            mode = AUTO_MATCH_CONSTRAINT_ANCHOR
        try:
            fit_weight = float(weight)
        except (TypeError, ValueError):
            fit_weight = 1.0
        if mode == AUTO_MATCH_CONSTRAINT_SOFT:
            fit_weight = max(0.01, min(1.0, fit_weight))
        else:
            fit_weight = 1.0
        return mode, fit_weight

    def _star_pair_fit_constraint(self, row: int) -> tuple[str, float]:
        if not self._is_auto_match_row(row):
            return AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0
        star_id = self._star_pair_star_id(row)
        if star_id:
            return self._auto_match_constraint_for_star_id(star_id)

        item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_INDEX_COLUMN)
        mode = str(item.data(STAR_PAIR_CONSTRAINT_MODE_ROLE) or AUTO_MATCH_CONSTRAINT_ANCHOR) if item is not None else ""
        try:
            weight = float(item.data(STAR_PAIR_FIT_WEIGHT_ROLE)) if item is not None else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        if mode not in AUTO_MATCH_CONSTRAINT_MODES:
            mode = AUTO_MATCH_CONSTRAINT_ANCHOR
        return (mode, max(0.01, min(1.0, weight))) if mode == AUTO_MATCH_CONSTRAINT_SOFT else (mode, 1.0)

    def _set_star_pair_item_row_type(self, item: QTableWidgetItem, row_type: str) -> QTableWidgetItem:
        item.setData(STAR_PAIR_ROW_TYPE_ROLE, row_type)
        return item

    def _star_pair_fit_payload(self, row: int) -> dict[str, float] | None:
        position_item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_POSITION_COLUMN)
        if position_item is None:
            return None
        payload = position_item.data(STAR_PAIR_FIT_ROLE)
        if not isinstance(payload, dict):
            return None

        fit_payload: dict[str, float] = {}
        for key in ("x", "y", "amplitude", "background", "sigma_x", "sigma_y"):
            try:
                value = float(payload[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                fit_payload[key] = value
        return fit_payload or None

    def _fit_payload_from_position(self, fitted_position: FittedStarPosition) -> dict[str, float]:
        return {
            "x": float(fitted_position.x),
            "y": float(fitted_position.y),
            "amplitude": float(fitted_position.amplitude),
            "background": float(fitted_position.background),
            "sigma_x": float(fitted_position.sigma_x),
            "sigma_y": float(fitted_position.sigma_y),
        }

    def _fitted_position_for_row(self, row: int) -> FittedStarPosition | None:
        position = self._parse_star_pair_position_text(row)
        if position is None:
            return None

        fit_payload = self._star_pair_fit_payload(row) or {}
        image_x, image_y = position
        return FittedStarPosition(
            x=image_x,
            y=image_y,
            amplitude=float(fit_payload.get("amplitude", 0.0)),
            background=float(fit_payload.get("background", 0.0)),
            sigma_x=float(fit_payload.get("sigma_x", 0.0)),
            sigma_y=float(fit_payload.get("sigma_y", 0.0)),
        )

    def _collect_star_pair_states(self) -> dict[str, dict[str, object]]:
        states: dict[str, dict[str, object]] = {}
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            name_item = table.item(row, STAR_PAIR_NAME_COLUMN)
            position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
            if name_item is None or position_item is None:
                continue
            star_id = str(name_item.data(Qt.UserRole) or "")
            position_text = position_item.text().strip()
            if star_id and position_text:
                states[star_id] = {
                    "position_text": position_text,
                    "fit_payload": self._star_pair_fit_payload(row),
                }
        return states

    def _collect_star_pair_positions(self) -> dict[str, str]:
        positions: dict[str, str] = {}
        for star_id, state in self._collect_star_pair_states().items():
            position_text = str(state.get("position_text", "")).strip()
            if position_text:
                positions[star_id] = position_text
        return positions

    def _star_pair_position_count(self) -> int:
        return len(self._collect_star_pair_positions())

    def _clear_star_pair_positions_for_new_input(self, input_name: str) -> int:
        pair_count = self._star_pair_position_count()
        if pair_count <= 0:
            return 0
        self._clear_star_pair_positions()
        self.ui.statusbar.showMessage(f"导入{input_name}前已清除 {pair_count} 个已有匹配。")
        return pair_count

    def _show_json_import_progress(
        self,
        title: str,
        label_text: str,
        status_text: str,
    ) -> QProgressDialog:
        dialog = QProgressDialog(self)
        dialog.setWindowTitle(title)
        dialog.setLabelText(label_text)
        dialog.setRange(0, 0)
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.show()
        self.ui.statusbar.showMessage(status_text)
        QApplication.processEvents()
        return dialog

    def _cleanup_json_import(self) -> None:
        if self._json_import_progress is not None:
            self._json_import_progress.close()
        self._json_import_thread = None
        self._json_import_worker = None
        self._json_import_progress = None
        self._set_json_import_controls_enabled(True)

    def _is_catalog_reference_star(self, star: ReferenceStar) -> bool:
        star_id = star.star_id.strip()
        return star.object_type == "star" and bool(star_id) and not star_id.startswith("solar_system:")

    def _build_reference_payload_for_current_settings(self) -> dict[str, object]:
        output_camera = self._output_camera_settings()
        observer, camera, view, mag_limit, star_map = self._build_projected_star_map(camera=output_camera)
        reference_stars = self._select_current_reference_stars(star_map)
        return build_reference_payload(
            star_map=star_map,
            reference_stars=reference_stars,
            observer=observer,
            camera=camera,
            view=view,
            visible_mag_limit=mag_limit,
            utc_offset_hours=self.ui.doubleSpinBoxUtcOffset.value(),
            reference_label_mode=self._reference_label_mode(),
            reference_mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            manual_reference_star_ids=tuple(self._manual_reference_star_ids),
        )

    def _star_pair_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            reference_star = self._reference_star_for_row(row)
            target_position = self._parse_star_pair_position_text(row)
            if reference_star is None or target_position is None:
                continue
            if not self._is_catalog_reference_star(reference_star):
                continue

            image_x, image_y = target_position
            if not all(math.isfinite(value) for value in (image_x, image_y, reference_star.ra_deg, reference_star.dec_deg)):
                continue

            record: dict[str, object] = {
                "reference_index": reference_star.index,
                "star_id": reference_star.star_id,
                "name": reference_star.name,
                "display_name": reference_star.display_name,
                "common_name": reference_star.common_name,
                "ra_deg": reference_star.ra_deg,
                "dec_deg": reference_star.dec_deg,
                "mag_v": reference_star.mag_v,
                "image_x_px": image_x,
                "image_y_px": image_y,
                "sim_x": reference_star.sim_x,
                "sim_y": reference_star.sim_y,
                "object_type": "star",
                "pair_origin": "auto_match" if self._is_auto_match_row(row) else "manual",
            }
            constraint_mode, fit_weight = self._star_pair_fit_constraint(row)
            record["fit_constraint_mode"] = constraint_mode
            record["fit_weight"] = fit_weight
            fit_payload = self._star_pair_fit_payload(row)
            if fit_payload is not None:
                for key in ("amplitude", "background", "sigma_x", "sigma_y"):
                    if key in fit_payload:
                        record[key] = fit_payload[key]
            residual = self._star_pair_alignment_residual(row)
            if residual is not None:
                dx, dy, distance = residual
                record["residual_dx_px"] = dx
                record["residual_dy_px"] = dy
                record["residual_px"] = distance
            records.append(record)
        return records

    def _auto_match_constraints_payload(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for star_id in self._auto_match_reference_star_ids:
            mode, fit_weight = self._auto_match_constraint_for_star_id(star_id)
            payload[star_id] = {
                "fit_constraint_mode": mode,
                "fit_weight": fit_weight,
            }
        return payload

    def _default_star_pair_session_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return project_root() / "outputs" / f"star_pairs_{timestamp}.json"

    def _build_star_pair_session_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像，再导出星点配对 JSON。")

        preview = self.current_image_preview
        image_path = Path(preview.path).expanduser().resolve()
        relative_image_path = _relative_image_path_for_session(image_path, json_path)
        reference_payload = self._build_reference_payload_for_current_settings()
        pair_records = self._star_pair_records()
        generated_time = datetime.now(timezone.utc)
        return {
            "format": STAR_PAIR_SESSION_FORMAT,
            "version": STAR_PAIR_SESSION_VERSION,
            "generated_at_utc": generated_time.isoformat(),
            "reference_payload": reference_payload,
            "real_image": {
                "path": str(image_path),
                "relative_path": relative_image_path,
                "original_width_px": preview.original_width,
                "original_height_px": preview.original_height,
                "display_width_px": preview.image.width(),
                "display_height_px": preview.image.height(),
            },
            "sky_alignment_model": self._alignment_model(),
            "auto_match_star_ids": list(self._auto_match_reference_star_ids),
            "auto_match_constraints": self._auto_match_constraints_payload(),
            "pair_count": len(pair_records),
            "pairs": pair_records,
        }

    def export_star_pair_session(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导出星点配对 JSON。")
            return

        default_path = self._default_star_pair_session_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出星点配对 JSON",
            str(default_path),
            STAR_PAIR_SESSION_JSON_FILTER,
        )
        if not file_path:
            return

        json_path = Path(file_path)
        if not json_path.suffix:
            json_path = json_path.with_suffix(".json")
        try:
            payload = self._build_star_pair_session_payload(json_path)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            pair_count = int(payload.get("pair_count", 0))
            self.ui.statusbar.showMessage(f"已导出星点配对 JSON: {json_path}  配对数: {pair_count}")
            QMessageBox.information(self, "配对 JSON 已导出", f"JSON：{json_path}\n配对数：{pair_count}")
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把文件和字段错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导出星点配对 JSON 失败: {exc}")
            QMessageBox.critical(self, "导出星点配对 JSON 失败", str(exc))

    def _default_source_model_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return project_root() / "outputs" / f"source_model_{timestamp}.json"

    def _source_image_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像。")
        preview = self.current_image_preview
        image_path = Path(preview.path).expanduser().resolve()
        return {
            "path": str(image_path),
            "relative_path": _relative_image_path_for_session(image_path, json_path),
            "original_width_px": preview.original_width,
            "original_height_px": preview.original_height,
            "model_width_px": preview.image.width(),
            "model_height_px": preview.image.height(),
        }

    def _sky_mask_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_sky_mask is None:
            return {"active": False}
        mask_height, mask_width = self.current_sky_mask.shape
        payload: dict[str, object] = {
            "active": True,
            "width_px": int(mask_width),
            "height_px": int(mask_height),
            "valid_fraction": float(np.count_nonzero(self.current_sky_mask)) / max(float(self.current_sky_mask.size), 1.0),
            "zero_pixels_excluded": True,
        }
        if self.current_sky_mask_path is not None:
            mask_path = self.current_sky_mask_path.expanduser().resolve()
            payload["path"] = str(mask_path)
            payload["relative_path"] = _relative_image_path_for_session(mask_path, json_path)
        return payload

    def _auto_match_settings_payload(self) -> dict[str, object]:
        return {
            "sky_alignment_model": self._alignment_model(),
            "fisheye_fov_deg": float(self.ui.doubleSpinBoxFisheyeFov.value()),
            "new_star_count": int(self.ui.spinBoxAutoMatchCount.value()),
            "new_constraint_mode": self._auto_match_constraint_mode(),
            "soft_constraint_weight": float(self.ui.doubleSpinBoxAutoMatchSoftWeight.value()),
            "search_radius_px": int(self.ui.spinBoxAutoMatchRadius.value()),
            "mask_enabled": self.current_sky_mask is not None,
        }

    def _current_source_model(self) -> SourceAstrometricModel:
        if self._source_astrometric_model is None:
            self._update_reference_alignment_transform()
        if self._source_astrometric_model is None:
            raise ValueError(self._source_model_error_message or f"至少需要 {MIN_ALIGNMENT_PAIRS} 对星点才能导出映射。")
        return self._source_astrometric_model

    def _build_source_model_payload(self, json_path: Path) -> dict[str, object]:
        model = self._current_source_model()
        return model.to_json_payload(
            source_image=self._source_image_payload(json_path),
            fit_pairs=self._star_pair_records(),
            mask=self._sky_mask_payload(json_path),
            matching=self._auto_match_settings_payload(),
        )

    def export_source_model_json(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导出 xy→RA/Dec 映射 JSON。")
            return

        default_path = self._default_source_model_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出 xy→RA/Dec 映射 JSON",
            str(default_path),
            SOURCE_MODEL_JSON_FILTER,
        )
        if not file_path:
            return

        json_path = Path(file_path)
        if not json_path.suffix:
            json_path = json_path.with_suffix(".json")
        try:
            payload = self._build_source_model_payload(json_path)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            diagnostics = payload.get("diagnostics", {})
            pair_count = int(diagnostics.get("pair_count", 0)) if isinstance(diagnostics, dict) else 0
            rms_px = float(diagnostics.get("rms_px", float("nan"))) if isinstance(diagnostics, dict) else float("nan")
            self.ui.statusbar.showMessage(
                f"已导出 xy→RA/Dec 映射 JSON: {json_path}  配对数: {pair_count}  RMS: {rms_px:.2f}px"
            )
            QMessageBox.information(
                self,
                "映射 JSON 已导出",
                f"JSON：{json_path}\n配对数：{pair_count}\nRMS：{rms_px:.2f} px",
            )
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把模型生成与文件错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导出 xy→RA/Dec 映射 JSON 失败: {exc}")
            QMessageBox.critical(self, "导出 xy→RA/Dec 映射 JSON 失败", str(exc))

    def import_star_pair_session(self) -> None:
        default_dir = project_root() / "outputs"
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入星点配对 JSON",
            str(default_dir),
            STAR_PAIR_SESSION_JSON_FILTER,
        )
        if not file_path:
            return
        self.load_star_pair_session(file_path)

    def load_star_pair_session(self, file_path: str | Path) -> None:
        if self._json_import_thread is not None:
            QMessageBox.information(self, "正在导入 JSON", "当前已有 JSON 正在导入，请稍候。")
            return
        json_path = Path(file_path)
        self._set_json_import_controls_enabled(False)
        self._json_import_progress = self._show_json_import_progress(
            title="正在导入配对 JSON",
            label_text=f"正在读取配对 JSON 并恢复真实图像...\n{json_path}",
            status_text=f"正在导入星点配对 JSON: {json_path}",
        )

        thread = QThread(self)
        worker = StarPairSessionImportWorker(json_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_star_pair_session_import_finished)
        worker.failed.connect(self._handle_star_pair_session_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_json_import)

        self._json_import_thread = thread
        self._json_import_worker = worker
        thread.start()

    def _session_pair_star_id(self, pair_payload: object) -> str:
        if not isinstance(pair_payload, dict):
            return ""
        object_type = str(pair_payload.get("object_type", "star")).strip()
        if object_type != "star":
            return ""
        star_id = str(pair_payload.get("star_id", "")).strip()
        if not star_id or star_id.startswith("solar_system:"):
            return ""
        return star_id

    def _session_pair_position(self, pair_payload: object) -> tuple[float, float] | None:
        if not isinstance(pair_payload, dict):
            return None
        try:
            image_x = float(pair_payload["image_x_px"])
            image_y = float(pair_payload["image_y_px"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(image_x) or not math.isfinite(image_y):
            return None
        return image_x, image_y

    def _session_pair_fit_payload(self, pair_payload: object) -> dict[str, float] | None:
        if not isinstance(pair_payload, dict):
            return None

        fit_payload: dict[str, float] = {}
        for key in ("amplitude", "background", "sigma_x", "sigma_y"):
            try:
                value = float(pair_payload[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                fit_payload[key] = value
        return fit_payload or None

    def _session_auto_match_star_ids(self, payload: dict[str, object], pair_payloads: list[object]) -> list[str]:
        auto_match_star_ids: list[str] = []
        raw_star_ids = payload.get("auto_match_star_ids")
        if isinstance(raw_star_ids, list):
            for raw_star_id in raw_star_ids:
                star_id = str(raw_star_id).strip()
                if star_id and star_id not in auto_match_star_ids:
                    auto_match_star_ids.append(star_id)

        for pair_payload in pair_payloads:
            if not isinstance(pair_payload, dict):
                continue
            if str(pair_payload.get("pair_origin", "")).strip() != "auto_match":
                continue
            star_id = self._session_pair_star_id(pair_payload)
            if star_id and star_id not in auto_match_star_ids:
                auto_match_star_ids.append(star_id)
        return auto_match_star_ids

    def _normalized_auto_match_constraint(self, raw_mode: object, raw_weight: object) -> tuple[str, float]:
        mode = str(raw_mode or AUTO_MATCH_CONSTRAINT_ANCHOR).strip()
        if mode not in AUTO_MATCH_CONSTRAINT_MODES:
            mode = AUTO_MATCH_CONSTRAINT_ANCHOR
        try:
            fit_weight = float(raw_weight)
        except (TypeError, ValueError):
            fit_weight = 1.0
        if mode == AUTO_MATCH_CONSTRAINT_SOFT:
            fit_weight = max(0.01, min(1.0, fit_weight))
        else:
            fit_weight = 1.0
        return mode, fit_weight

    def _session_auto_match_constraints(
        self,
        payload: dict[str, object],
        pair_payloads: list[object],
    ) -> dict[str, tuple[str, float]]:
        constraints: dict[str, tuple[str, float]] = {}
        raw_constraints = payload.get("auto_match_constraints")
        if isinstance(raw_constraints, dict):
            for raw_star_id, raw_constraint in raw_constraints.items():
                star_id = str(raw_star_id).strip()
                if not star_id or not isinstance(raw_constraint, dict):
                    continue
                constraints[star_id] = self._normalized_auto_match_constraint(
                    raw_constraint.get("fit_constraint_mode"),
                    raw_constraint.get("fit_weight", 1.0),
                )

        for pair_payload in pair_payloads:
            if not isinstance(pair_payload, dict):
                continue
            if str(pair_payload.get("pair_origin", "")).strip() != "auto_match":
                continue
            star_id = self._session_pair_star_id(pair_payload)
            if not star_id:
                continue
            constraints[star_id] = self._normalized_auto_match_constraint(
                pair_payload.get("fit_constraint_mode"),
                pair_payload.get("fit_weight", 1.0),
            )
        return constraints

    def _ensure_pair_record_stars_visible(self, pair_payloads: list[object]) -> None:
        visible_star_ids = {
            self._star_pair_star_id(row)
            for row in range(self.ui.tableWidgetStarPairs.rowCount())
            if self._star_pair_star_id(row)
        }
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        added_any = False
        for pair_payload in pair_payloads:
            star_id = self._session_pair_star_id(pair_payload)
            if (
                not star_id
                or star_id in visible_star_ids
                or star_id in self._manual_reference_star_ids
                or star_id in auto_match_star_ids
            ):
                continue
            if star_id in self._excluded_reference_star_ids:
                self._excluded_reference_star_ids.remove(star_id)
            if isinstance(pair_payload, dict) and str(pair_payload.get("pair_origin", "")).strip() == "auto_match":
                self._auto_match_reference_star_ids.append(star_id)
                auto_match_star_ids.add(star_id)
                self._auto_match_constraint_by_star_id[star_id] = self._normalized_auto_match_constraint(
                    pair_payload.get("fit_constraint_mode"),
                    pair_payload.get("fit_weight", 1.0),
                )
            else:
                self._manual_reference_star_ids.append(star_id)
            visible_star_ids.add(star_id)
            added_any = True
        if added_any:
            self._refresh_reference_stars_from_current_map()

    def _restore_star_pair_records(self, pair_payloads: list[object], update_alignment: bool = True) -> int:
        recorded_positions: dict[str, tuple[float, float]] = {}
        recorded_fit_payloads: dict[str, dict[str, float] | None] = {}
        for pair_payload in pair_payloads:
            star_id = self._session_pair_star_id(pair_payload)
            position = self._session_pair_position(pair_payload)
            if star_id and position is not None:
                recorded_positions[star_id] = position
                fit_payload = self._session_pair_fit_payload(pair_payload)
                if fit_payload is not None:
                    fit_payload["x"] = position[0]
                    fit_payload["y"] = position[1]
                recorded_fit_payloads[star_id] = fit_payload

        table = self.ui.tableWidgetStarPairs
        signals_were_blocked = table.blockSignals(True)
        self._clear_star_pair_annotations()
        restored_count = 0
        for row in range(table.rowCount()):
            star_id = self._star_pair_star_id(row)
            position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
            if position_item is None:
                position_item = QTableWidgetItem()
                table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
            position_item.setData(Qt.UserRole, star_id)
            position = recorded_positions.get(star_id)
            if position is None:
                position_item.setText("")
                position_item.setData(STAR_PAIR_FIT_ROLE, None)
                continue
            image_x, image_y = position
            position_item.setText(f"{image_x:.2f}, {image_y:.2f}")
            position_item.setData(STAR_PAIR_FIT_ROLE, recorded_fit_payloads.get(star_id))
            restored_count += 1
        table.blockSignals(signals_were_blocked)

        self._restore_star_pair_annotations_from_table()
        self._refresh_star_pair_table_styles()
        if update_alignment:
            self._update_reference_alignment_transform()
        QTimer.singleShot(0, table.scrollToBottom)
        return restored_count

    def _session_real_image_path(self, payload: dict[str, object], source_path: Path) -> Path:
        return _resolve_star_pair_session_real_image_path(payload, source_path)

    def _handle_star_pair_session_import_finished(self, result: object) -> None:
        try:
            source_path, payload, preview = result  # type: ignore[misc]
            if not isinstance(source_path, Path):
                source_path = Path(source_path)
            self._clear_star_pair_positions_for_new_input("新的配对 JSON")
            self._apply_star_pair_session_payload(payload, source_path, preview=preview)
        except Exception as exc:  # noqa: BLE001 - 主线程恢复界面时也需要把错误反馈给用户。
            self.ui.statusbar.showMessage(f"导入星点配对 JSON 失败: {exc}")
            QMessageBox.critical(self, "导入星点配对 JSON 失败", str(exc))

    def _handle_star_pair_session_import_failed(self, error_message: str) -> None:
        self.ui.statusbar.showMessage(f"导入星点配对 JSON 失败: {error_message}")
        QMessageBox.critical(self, "导入星点配对 JSON 失败", error_message)

    def _apply_star_pair_session_payload(
        self,
        payload: object,
        source_path: Path,
        preview: ImagePreview | None = None,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("JSON 根对象必须是字典。")
        if payload.get("format") != STAR_PAIR_SESSION_FORMAT:
            raise ValueError("当前只支持 MeteoAlign 星点配对 JSON。")

        reference_payload = payload.get("reference_payload")
        if not isinstance(reference_payload, dict):
            raise ValueError("JSON 缺少 reference_payload 字段。")
        pair_payloads = payload.get("pairs", [])
        if not isinstance(pair_payloads, list):
            raise ValueError("JSON 中 pairs 字段必须是列表。")
        auto_match_star_ids = self._session_auto_match_star_ids(payload, pair_payloads)
        auto_match_constraints = self._session_auto_match_constraints(payload, pair_payloads)

        image_path = self._session_real_image_path(payload, source_path)
        self._active_star_pair_row = None
        previous_suspend_alignment = self._suspend_alignment_updates
        self._suspend_alignment_updates = True
        try:
            self._set_alignment_model(payload.get("sky_alignment_model"))
            self._apply_reference_payload(reference_payload, source_path)
            self._auto_match_reference_star_ids = auto_match_star_ids
            self._auto_match_constraint_by_star_id = {
                star_id: auto_match_constraints.get(star_id, (AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0))
                for star_id in auto_match_star_ids
            }
            if self._auto_match_reference_star_ids:
                self._refresh_reference_stars_from_current_map()
            self._ensure_pair_record_stars_visible(pair_payloads)
            if preview is None:
                preview = load_image_preview(image_path, max_long_side_px=None)
            self._apply_loaded_image_preview(preview, clear_existing_pairs=False)
            restored_count = self._restore_star_pair_records(pair_payloads, update_alignment=False)
        finally:
            self._suspend_alignment_updates = previous_suspend_alignment
        self._update_reference_alignment_transform()
        self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        self.ui.statusbar.showMessage(
            f"已导入星点配对 JSON: {source_path}  真实图像: {image_path}  恢复配对: {restored_count}"
        )

    def _read_only_table_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _auto_match_group_row(self) -> int | None:
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            if self._is_auto_match_group_row(row):
                return row
        return None

    def _auto_match_group_counts(self) -> tuple[int, int, int]:
        total_count = 0
        paired_count = 0
        soft_count = 0
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            if not self._is_auto_match_row(row):
                continue
            total_count += 1
            if self._star_pair_position_text(row):
                paired_count += 1
            mode, _fit_weight = self._star_pair_fit_constraint(row)
            if mode == AUTO_MATCH_CONSTRAINT_SOFT:
                soft_count += 1
        return total_count, paired_count, soft_count

    def _update_auto_match_group_row_text(self) -> None:
        group_row = self._auto_match_group_row()
        if group_row is None:
            return
        table = self.ui.tableWidgetStarPairs
        total_count, paired_count, soft_count = self._auto_match_group_counts()
        arrow_text = "▼" if self._auto_match_group_expanded else "▶"
        group_text = f"自动扩展匹配 ({total_count})"
        if soft_count > 0:
            group_text = f"自动扩展匹配 ({total_count}，软 {soft_count})"
        values = {
            STAR_PAIR_INDEX_COLUMN: arrow_text,
            STAR_PAIR_NAME_COLUMN: group_text,
            STAR_PAIR_POSITION_COLUMN: f"已配对 {paired_count}/{total_count}",
            STAR_PAIR_RESIDUAL_COLUMN: "",
        }
        signals_were_blocked = table.blockSignals(True)
        for column, text in values.items():
            item = table.item(group_row, column)
            if item is not None:
                item.setText(text)
        table.blockSignals(signals_were_blocked)

    def _apply_auto_match_group_visibility(self) -> None:
        table = self.ui.tableWidgetStarPairs
        group_row = self._auto_match_group_row()
        for row in range(table.rowCount()):
            if self._is_auto_match_row(row):
                table.setRowHidden(row, group_row is not None and not self._auto_match_group_expanded)
            else:
                table.setRowHidden(row, False)
        self._update_auto_match_group_row_text()

    def _toggle_auto_match_group(self) -> None:
        if self._auto_match_group_row() is None:
            return
        self._auto_match_group_expanded = not self._auto_match_group_expanded
        self._apply_auto_match_group_visibility()

    def _handle_star_pair_cell_clicked(self, row: int, _column: int) -> None:
        if self._is_auto_match_group_row(row):
            self._toggle_auto_match_group()

    def _handle_star_pair_cell_double_clicked(self, row: int, _column: int) -> None:
        if self._is_auto_match_group_row(row):
            return
        self._focus_star_pair_theoretical_position(row)

    def _make_star_pair_table_item(
        self,
        text: str,
        row_type: str,
        star_id: str = "",
        editable: bool = False,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setData(STAR_PAIR_ROW_TYPE_ROLE, row_type)
        if star_id:
            item.setData(Qt.UserRole, star_id)
        return item

    def _set_star_pair_table_row(
        self,
        row: int,
        star: ReferenceStar,
        index_text: str,
        row_type: str,
        saved_states: dict[str, dict[str, object]],
    ) -> None:
        table = self.ui.tableWidgetStarPairs
        star_id = star.star_id.strip()
        star_name = star.common_name.strip() or star_id
        saved_state = saved_states.get(star_id, {})
        position_text = str(saved_state.get("position_text", "")).strip()

        index_item = self._make_star_pair_table_item(index_text, row_type, star_id)
        name_item = self._make_star_pair_table_item(star_name, row_type, star_id)
        position_item = self._make_star_pair_table_item(position_text, row_type, star_id, editable=True)
        fit_payload = saved_state.get("fit_payload")
        position_item.setData(STAR_PAIR_FIT_ROLE, fit_payload if isinstance(fit_payload, dict) else None)
        residual_item = self._make_star_pair_table_item("", row_type, star_id)
        mode, fit_weight = (
            self._auto_match_constraint_for_star_id(star_id)
            if row_type == STAR_PAIR_ROW_TYPE_AUTO_MATCH
            else (AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0)
        )
        constraint_tip = (
            f"投影拟合：软约束，权重 {fit_weight:.2f}。"
            if mode == AUTO_MATCH_CONSTRAINT_SOFT
            else "投影拟合：硬锚点。"
        )
        for item in (index_item, name_item, position_item, residual_item):
            item.setData(STAR_PAIR_CONSTRAINT_MODE_ROLE, mode)
            item.setData(STAR_PAIR_FIT_WEIGHT_ROLE, fit_weight)
            item.setToolTip(constraint_tip)

        table.setItem(row, STAR_PAIR_INDEX_COLUMN, index_item)
        table.setItem(row, STAR_PAIR_NAME_COLUMN, name_item)
        table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
        table.setItem(row, STAR_PAIR_RESIDUAL_COLUMN, residual_item)

    def _set_auto_match_group_table_row(self, row: int) -> None:
        table = self.ui.tableWidgetStarPairs
        for column, text in (
            (STAR_PAIR_INDEX_COLUMN, "▶"),
            (STAR_PAIR_NAME_COLUMN, "自动扩展匹配"),
            (STAR_PAIR_POSITION_COLUMN, ""),
            (STAR_PAIR_RESIDUAL_COLUMN, ""),
        ):
            item = self._make_star_pair_table_item(text, STAR_PAIR_ROW_TYPE_AUTO_GROUP)
            font = QFont(item.font())
            font.setBold(True)
            item.setFont(font)
            table.setItem(row, column, item)

    def _update_star_pair_table(self, reference_stars: tuple[ReferenceStar, ...]) -> None:
        self._current_reference_stars = tuple(reference_stars)
        table = self.ui.tableWidgetStarPairs
        saved_states = self._collect_star_pair_states()
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        regular_stars = [star for star in reference_stars if star.star_id.strip() not in auto_match_star_ids]
        auto_match_stars = [star for star in reference_stars if star.star_id.strip() in auto_match_star_ids]
        row_count = len(regular_stars) + len(auto_match_stars)
        if auto_match_stars:
            row_count += 1

        signals_were_blocked = table.blockSignals(True)
        table.setRowCount(row_count)
        row = 0
        for display_index, star in enumerate(regular_stars, start=1):
            self._set_star_pair_table_row(
                row,
                star,
                str(display_index),
                STAR_PAIR_ROW_TYPE_MANUAL,
                saved_states,
            )
            row += 1

        if auto_match_stars:
            self._set_auto_match_group_table_row(row)
            row += 1
            for auto_index, star in enumerate(auto_match_stars, start=1):
                self._set_star_pair_table_row(
                    row,
                    star,
                    f"A{auto_index}",
                    STAR_PAIR_ROW_TYPE_AUTO_MATCH,
                    saved_states,
                )
                row += 1
        table.blockSignals(signals_were_blocked)
        table.resizeColumnToContents(STAR_PAIR_INDEX_COLUMN)
        table.resizeColumnToContents(STAR_PAIR_NAME_COLUMN)
        self._apply_auto_match_group_visibility()
        self._sync_star_pair_annotations_to_table()
        self._refresh_star_pair_table_styles()
        self._restore_star_pair_annotations_from_table()
        self._update_reference_alignment_transform()

    def _handle_star_pair_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != STAR_PAIR_POSITION_COLUMN:
            return
        if self._is_auto_match_group_row(item.row()):
            return
        signals_were_blocked = self.ui.tableWidgetStarPairs.blockSignals(True)
        item.setData(STAR_PAIR_FIT_ROLE, None)
        self.ui.tableWidgetStarPairs.blockSignals(signals_were_blocked)
        star_id = self._star_pair_star_id(item.row())
        if star_id and not item.text().strip():
            self._remove_star_pair_annotation(star_id)
        self._refresh_star_pair_row_style(item.row())
        self._update_reference_alignment_transform()

    def _star_pair_star_id(self, row: int) -> str:
        name_item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_NAME_COLUMN)
        if name_item is None:
            return ""
        return str(name_item.data(Qt.UserRole) or "")

    def _star_pair_position_text(self, row: int) -> str:
        position_item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_POSITION_COLUMN)
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
        if self._is_auto_match_group_row(row):
            return "自动扩展匹配"
        index_item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_INDEX_COLUMN)
        index_text = index_item.text() if index_item is not None else str(row + 1)
        star_name = self._star_pair_name(row)
        return f"{index_text}. {star_name}" if star_name else index_text

    def _refresh_star_pair_table_styles(self) -> None:
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            self._refresh_star_pair_row_style(row)

    def _star_pair_residual_background(self, row: int) -> QColor | None:
        residual = self._star_pair_alignment_residual(row)
        if residual is None:
            return None

        _dx, _dy, distance = residual
        warning_threshold, severe_threshold = self._residual_warning_thresholds()
        if distance >= severe_threshold:
            return QColor(255, 210, 210)
        if distance >= warning_threshold:
            return QColor(255, 232, 190)
        return None

    def _refresh_star_pair_row_style(self, row: int) -> None:
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount():
            return
        table = self.ui.tableWidgetStarPairs
        if self._is_auto_match_group_row(row):
            background = QColor(232, 236, 244)
        else:
            residual_background = self._star_pair_residual_background(row)
            if self._active_star_pair_row == row:
                background = QColor(255, 242, 153)
            elif residual_background is not None:
                background = residual_background
            elif self._star_pair_position_text(row):
                background = QColor(210, 244, 214)
            else:
                background = QColor(255, 255, 255)

        if self._active_star_pair_row == row and not self._is_auto_match_group_row(row):
            background = QColor(255, 242, 153)

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
                    f"选星圈直径已到边界：{new_diameter} px。Ctrl+左键确认，右键取消。"
                )
            return

        self._star_pick_circle_diameter_px = new_diameter
        self._star_pick_cursor = None
        if self._active_star_pair_row is not None:
            self._update_real_image_pick_cursor()
            if show_status:
                self.ui.statusbar.showMessage(
                    f"选星圈直径：{new_diameter} px。Ctrl+左键确认，右键取消，Ctrl+滚轮 / Ctrl+加减继续缩放。"
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
            "正在点选 {label}；普通左键拖动预览，Ctrl+左键确认，右键取消，Ctrl+滚轮 / Ctrl+加减缩放选星圈。"
            "当前选星圈直径：{diameter} px，PSF半径比例：{scale:.2f}，上限：{max_radius} px。".format(
                label=self._star_pair_label(row),
                diameter=self._star_pick_circle_diameter_px,
                scale=self.ui_config.star_pick_psf_radius_scale,
                max_radius=self.ui_config.star_pick_psf_max_radius_px,
            )
        )

    def _clear_star_pair_annotations(self) -> None:
        self._clear_focused_star_annotations()
        for ellipse_item, label_item in self._star_pair_annotations.values():
            self.real_image_scene.removeItem(ellipse_item)
            self.real_image_scene.removeItem(label_item)
        self._star_pair_annotations.clear()

    def _clear_focused_star_annotations(self) -> None:
        for item in self._focused_star_annotations:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        self._focused_star_annotations.clear()

    def _update_star_pair_annotation_visibility(self) -> None:
        visible = not self._hide_all_annotations()
        for ellipse_item, label_item in self._star_pair_annotations.values():
            ellipse_item.setVisible(visible)
            label_item.setVisible(visible)

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

    def _renumber_star_pair_rows_from_table(self) -> None:
        table = self.ui.tableWidgetStarPairs
        star_lookup = self._reference_star_lookup()
        renumbered_stars: list[ReferenceStar] = []
        regular_index = 1
        auto_index = 1
        signals_were_blocked = table.blockSignals(True)
        for row in range(table.rowCount()):
            if self._is_auto_match_group_row(row):
                index_item = table.item(row, STAR_PAIR_INDEX_COLUMN)
                if index_item is not None:
                    index_item.setText("▼" if self._auto_match_group_expanded else "▶")
                continue

            star_id = self._star_pair_star_id(row)
            reference_star = star_lookup.get(star_id)
            if reference_star is not None:
                renumbered_stars.append(self._reference_star_with_index(reference_star, len(renumbered_stars) + 1))

            index_item = table.item(row, STAR_PAIR_INDEX_COLUMN)
            if index_item is None:
                index_item = self._read_only_table_item("")
                table.setItem(row, STAR_PAIR_INDEX_COLUMN, index_item)
            if self._is_auto_match_row(row):
                index_item.setText(f"A{auto_index}")
                auto_index += 1
            else:
                index_item.setText(str(regular_index))
                regular_index += 1
        table.blockSignals(signals_were_blocked)

        self._current_reference_stars = tuple(renumbered_stars)
        self._update_auto_match_group_row_text()
        self._sync_star_pair_annotations_to_table()
        self._refresh_star_pair_table_styles()

    def _restore_star_pair_annotations_from_table(self) -> None:
        if self.current_image_preview is None:
            return
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            if self._is_auto_match_group_row(row):
                continue
            fitted_position = self._fitted_position_for_row(row)
            if fitted_position is None:
                continue
            image_x, image_y = fitted_position.x, fitted_position.y
            if not (0.0 <= image_x < self.current_image_preview.image.width()):
                continue
            if not (0.0 <= image_y < self.current_image_preview.image.height()):
                continue
            self._add_or_update_star_pair_annotation(row, fitted_position)

    def _star_pair_annotation_radius_px(self, fitted_position: FittedStarPosition) -> float:
        sigma_radius = max(abs(float(fitted_position.sigma_x)), abs(float(fitted_position.sigma_y)))
        if sigma_radius > 0.0 and math.isfinite(sigma_radius):
            radius = sigma_radius * STAR_ANNOTATION_PSF_SIGMA_SCALE
        else:
            radius = STAR_ANNOTATION_FALLBACK_RADIUS_PX
        return min(
            max(radius, STAR_ANNOTATION_MIN_RADIUS_PX),
            STAR_ANNOTATION_MAX_RADIUS_PX,
        )

    def _add_or_update_star_pair_annotation(
        self,
        row: int,
        fitted_position: FittedStarPosition,
    ) -> None:
        star_id = self._star_pair_star_id(row)
        if not star_id:
            return

        self._remove_star_pair_annotation(star_id)
        radius = self._star_pair_annotation_radius_px(fitted_position)
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
        self._update_star_pair_annotation_visibility()

    def _create_focus_annotation_items(
        self,
        scene: QGraphicsScene,
        point: QPointF,
        label: str,
    ) -> None:
        radius = STAR_PAIR_FOCUS_MARKER_RADIUS_PX
        ellipse_item = QGraphicsEllipseItem(
            point.x() - radius,
            point.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        shadow_pen = QPen(QColor(0, 0, 0, 235), 5)
        shadow_pen.setCosmetic(True)
        marker_pen = QPen(QColor(80, 220, 255), 2)
        marker_pen.setCosmetic(True)
        ellipse_item.setPen(marker_pen)
        ellipse_item.setBrush(QBrush(Qt.NoBrush))
        ellipse_item.setZValue(40.0)

        shadow_item = QGraphicsEllipseItem(
            point.x() - radius,
            point.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        shadow_item.setPen(shadow_pen)
        shadow_item.setBrush(QBrush(Qt.NoBrush))
        shadow_item.setZValue(39.0)

        label_item = QGraphicsSimpleTextItem(label)
        label_font = QFont(self.font())
        label_font.setPointSize(self.ui_config.star_name_font_size_pt)
        label_font.setBold(True)
        label_item.setFont(label_font)
        label_item.setBrush(QBrush(QColor(80, 220, 255)))
        label_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label_item.setPos(point.x() + radius + 6.0, point.y() - radius)
        label_item.setZValue(41.0)

        for item in (shadow_item, ellipse_item, label_item):
            scene.addItem(item)
            self._focused_star_annotations.append(item)

    def _set_graphics_view_scale_centered(
        self,
        view: QGraphicsView,
        target_scale: float,
        center: QPointF,
    ) -> None:
        view.resetTransform()
        view.scale(target_scale, target_scale)
        view.centerOn(center)
        self._cap_graphics_view_to_max_scale(view)
        view.centerOn(center)
        self._update_live_star_map_zoom_scale(view)

    def _focus_reference_real_views_on_point(self, point: QPointF) -> None:
        target_scale = max(
            self._graphics_view_fit_scale(self.ui.realImageView) * STAR_PAIR_FOCUS_ZOOM_FIT_SCALE,
            self._graphics_view_current_scale(self.ui.realImageView),
        )
        max_scale = self._graphics_view_max_scale(self.ui.realImageView)
        if max_scale is not None:
            target_scale = min(target_scale, max_scale)

        self._syncing_reference_real_views = True
        try:
            self._set_graphics_view_scale_centered(self.ui.realImageView, target_scale, point)
            self.ui.referenceImageView.setTransform(self.ui.realImageView.transform())
            self.ui.referenceImageView.centerOn(point)
            self._cap_graphics_view_to_max_scale(self.ui.referenceImageView)
            self.ui.referenceImageView.centerOn(point)
            self._update_live_star_map_zoom_scale(self.ui.referenceImageView)
        finally:
            self._syncing_reference_real_views = False

    def _set_hide_all_annotations_checked(self) -> None:
        if self.ui.checkBoxHideAllAnnotations.isChecked():
            self._clear_focused_star_annotations()
            return
        self.ui.checkBoxHideAllAnnotations.setChecked(True)

    def _set_reference_real_sync_checked(self) -> None:
        self._update_reference_alignment_controls()
        if self.ui.checkBoxSyncReferenceAndRealView.isEnabled() and not self.ui.checkBoxSyncReferenceAndRealView.isChecked():
            self.ui.checkBoxSyncReferenceAndRealView.setChecked(True)

    def _focus_star_pair_theoretical_position(self, row: int) -> None:
        matched_count = self._star_pair_position_count()
        if matched_count < STAR_PAIR_FOCUS_MIN_MATCHED_COUNT:
            self.ui.statusbar.showMessage(
                f"当前已有 {matched_count} 个匹配；至少 {STAR_PAIR_FOCUS_MIN_MATCHED_COUNT} 个后可双击聚焦理论位置。"
            )
            return
        if self.current_image_preview is None:
            self.ui.statusbar.showMessage("请先导入真实图像，再双击聚焦匹配星。")
            return

        transform = self._sky_alignment_transform
        if transform is None:
            self._update_reference_alignment_transform()
            transform = self._sky_alignment_transform
        if transform is None:
            self.ui.statusbar.showMessage(self._sky_alignment_error_message or "当前配准模型尚未就绪，无法聚焦理论位置。")
            return

        reference_star = self._reference_star_for_row(row)
        if reference_star is None:
            self.ui.statusbar.showMessage("当前行没有可聚焦的参考星。")
            return

        predicted_x, predicted_y = transform.transform_radec(reference_star.ra_deg, reference_star.dec_deg)
        if not all(math.isfinite(value) for value in (predicted_x, predicted_y)):
            self.ui.statusbar.showMessage(f"{self._star_pair_label(row)} 的理论位置不是有效坐标。")
            return

        image = self.current_image_preview.image
        if not (0.0 <= predicted_x < image.width() and 0.0 <= predicted_y < image.height()):
            self.ui.statusbar.showMessage(f"{self._star_pair_label(row)} 的理论位置在真实图像外。")
            return

        if self._active_star_pair_row is not None:
            self._leave_star_pick_mode()
        self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        focus_point = QPointF(float(predicted_x), float(predicted_y))
        focus_label = self._star_pair_label(row)

        self._set_reference_real_sync_checked()
        self._set_hide_all_annotations_checked()
        self._update_reference_alignment_display()
        self._focus_reference_real_views_on_point(focus_point)
        self._create_focus_annotation_items(self.reference_scene, focus_point, focus_label)
        self._create_focus_annotation_items(self.real_image_scene, focus_point, focus_label)
        self.ui.tableWidgetStarPairs.selectRow(row)
        self.ui.statusbar.showMessage(
            f"已聚焦 {focus_label} 的理论位置: x={predicted_x:.2f}, y={predicted_y:.2f}。切换“隐藏所有标注”可重置标注显示。"
        )

    def _show_star_pair_context_menu(self, point: QPoint) -> None:
        table = self.ui.tableWidgetStarPairs
        row = table.rowAt(point.y())
        if row < 0:
            return

        table.selectRow(row)
        menu = QMenu(self)
        if self._is_auto_match_group_row(row):
            toggle_action = menu.addAction("折叠列表" if self._auto_match_group_expanded else "展开列表")
            delete_group_action = menu.addAction("删除自动扩展列表")
            selected_action = menu.exec_(table.viewport().mapToGlobal(point))
            if selected_action is toggle_action:
                self._toggle_auto_match_group()
            elif selected_action is delete_group_action:
                self._delete_auto_match_group()
            return

        pick_action = menu.addAction("点选位置")
        pick_action.setEnabled(self.current_image_preview is not None)
        auto_pair_action = None
        if not self._star_pair_position_text(row):
            auto_pair_action = menu.addAction("自动配对")
            auto_pair_action.setEnabled(self._sky_alignment_transform is not None and self.current_image_preview is not None)
        clear_action = None
        if self._star_pair_position_text(row):
            clear_action = menu.addAction("清除配对")
        delete_action = menu.addAction("删除该行")
        selected_action = menu.exec_(table.viewport().mapToGlobal(point))
        if selected_action is pick_action:
            self._enter_star_pick_mode(row)
        elif auto_pair_action is not None and selected_action is auto_pair_action:
            self._auto_pair_star(row)
        elif clear_action is not None and selected_action is clear_action:
            self._clear_star_pair_position(row)
        elif selected_action is delete_action:
            self._delete_star_pair_row(row)

    def _enter_star_pick_mode(self, row: int) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再点选星点位置。")
            return
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount():
            return

        self._active_star_pair_row = row
        self._star_pick_previous_drag_mode = self.ui.realImageView.dragMode()
        self.ui.realImageView.viewport().setFocusPolicy(Qt.StrongFocus)
        self.ui.realImageView.viewport().setFocus()
        self.ui.realImageView.viewport().setMouseTracking(True)
        self._update_real_image_pick_cursor()
        self._refresh_star_pair_table_styles()
        self._show_star_pick_status_hint(row)

    def _leave_star_pick_mode(self) -> None:
        self._active_star_pair_row = None
        self.ui.realImageView.setDragMode(self._star_pick_previous_drag_mode)
        self.ui.realImageView.viewport().unsetCursor()
        self._refresh_star_pair_table_styles()

    def _ctrl_is_pressed(self) -> bool:
        return bool(QApplication.keyboardModifiers() & Qt.ControlModifier)

    def _event_ctrl_pressed(self, event) -> bool:  # type: ignore[no-untyped-def]
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Control:
            return True
        if event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Control:
            return False
        if hasattr(event, "modifiers"):
            return bool(event.modifiers() & Qt.ControlModifier)
        return self._ctrl_is_pressed()

    def _update_real_image_pick_cursor(self, ctrl_pressed: bool | None = None) -> None:
        if self._active_star_pair_row is None:
            self.ui.realImageView.viewport().unsetCursor()
            return
        if ctrl_pressed is None:
            ctrl_pressed = self._ctrl_is_pressed()
        if ctrl_pressed:
            self.ui.realImageView.viewport().setCursor(self._create_star_pick_cursor())
        else:
            self.ui.realImageView.viewport().unsetCursor()

    def _update_reference_map_cursor(self, ctrl_pressed: bool | None = None) -> None:
        if ctrl_pressed is None:
            ctrl_pressed = self._ctrl_is_pressed()
        if ctrl_pressed:
            self.ui.referenceImageView.viewport().setCursor(Qt.ArrowCursor)
        else:
            self.ui.referenceImageView.viewport().unsetCursor()

    def _star_pair_name(self, row: int) -> str:
        item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_NAME_COLUMN)
        if item is None:
            return ""
        return item.text()

    def _set_star_pair_position(
        self,
        row: int,
        fitted_position: FittedStarPosition,
        update_alignment: bool = True,
    ) -> None:
        table = self.ui.tableWidgetStarPairs
        if row < 0 or row >= table.rowCount():
            return

        position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
        if position_item is None:
            position_item = QTableWidgetItem()
            table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
        name_item = table.item(row, STAR_PAIR_NAME_COLUMN)
        if name_item is not None:
            position_item.setData(Qt.UserRole, name_item.data(Qt.UserRole))
        signals_were_blocked = table.blockSignals(True)
        position_item.setData(STAR_PAIR_FIT_ROLE, self._fit_payload_from_position(fitted_position))
        position_item.setText(f"{fitted_position.x:.2f}, {fitted_position.y:.2f}")
        table.blockSignals(signals_were_blocked)
        table.selectRow(row)
        self._refresh_star_pair_row_style(row)
        self._update_auto_match_group_row_text()
        if update_alignment:
            self._update_reference_alignment_transform()

    def _clear_star_pair_position_data(self, row: int) -> str:
        table = self.ui.tableWidgetStarPairs
        if row < 0 or row >= table.rowCount():
            return ""
        star_id = self._star_pair_star_id(row)
        position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
        if position_item is not None:
            signals_were_blocked = table.blockSignals(True)
            position_item.setText("")
            position_item.setData(STAR_PAIR_FIT_ROLE, None)
            table.blockSignals(signals_were_blocked)
        if star_id:
            self._remove_star_pair_annotation(star_id)
        return star_id

    def _clear_star_pair_positions(self) -> int:
        cleared_count = self._star_pair_position_count()
        if self._active_star_pair_row is not None:
            self._leave_star_pick_mode()
        table = self.ui.tableWidgetStarPairs
        table.blockSignals(True)
        for row in range(table.rowCount()):
            position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
            if position_item is not None:
                position_item.setText("")
                position_item.setData(STAR_PAIR_FIT_ROLE, None)
        table.blockSignals(False)
        self._clear_star_pair_annotations()
        self._refresh_star_pair_table_styles()
        self._update_auto_match_group_row_text()
        self._update_reference_alignment_transform()
        return cleared_count

    def clear_all_star_pair_positions(self) -> None:
        cleared_count = self._clear_star_pair_positions()
        if cleared_count <= 0:
            self.ui.statusbar.showMessage("当前没有可清除的星点匹配。")
            return
        self.ui.statusbar.showMessage(f"已清除 {cleared_count} 个星点匹配。")

    def _clear_star_pair_position(self, row: int) -> None:
        table = self.ui.tableWidgetStarPairs
        if row < 0 or row >= table.rowCount():
            return
        star_label = self._star_pair_label(row)
        self._clear_star_pair_position_data(row)
        self._refresh_star_pair_row_style(row)
        self._update_auto_match_group_row_text()
        self._update_reference_alignment_transform()
        self.ui.statusbar.showMessage(f"已清除 {star_label} 的真实图像配对。右键该行可重新点选位置。")

    def _delete_star_pair_row(self, row: int) -> None:
        table = self.ui.tableWidgetStarPairs
        if row < 0 or row >= table.rowCount():
            return
        if self._is_auto_match_group_row(row):
            self._delete_auto_match_group()
            return

        star_label = self._star_pair_label(row)
        star_id = self._star_pair_star_id(row)
        if self._active_star_pair_row is not None:
            if self._active_star_pair_row == row:
                self._leave_star_pick_mode()
            elif self._active_star_pair_row > row:
                self._active_star_pair_row -= 1

        if star_id:
            self._clear_star_pair_position_data(row)
            self._remove_star_pair_annotation(star_id)
            if self._is_auto_match_row(row):
                self._auto_match_reference_star_ids = [
                    auto_star_id for auto_star_id in self._auto_match_reference_star_ids if auto_star_id != star_id
                ]
                self._auto_match_constraint_by_star_id.pop(star_id, None)
            else:
                self._manual_reference_star_ids = [
                    manual_star_id for manual_star_id in self._manual_reference_star_ids if manual_star_id != star_id
                ]
                if star_id not in self._excluded_reference_star_ids:
                    self._excluded_reference_star_ids.append(star_id)

        if self._is_auto_match_row(row):
            self._refresh_reference_stars_from_current_map()
            table = self.ui.tableWidgetStarPairs
            if table.rowCount() > 0:
                table.selectRow(min(row, table.rowCount() - 1))
        else:
            signals_were_blocked = table.blockSignals(True)
            table.removeRow(row)
            table.blockSignals(signals_were_blocked)
            self._renumber_star_pair_rows_from_table()
            if table.rowCount() > 0:
                table.selectRow(min(row, table.rowCount() - 1))
        self._update_reference_alignment_transform()
        self.ui.statusbar.showMessage(f"已删除参考星 {star_label}，后续序号已重新排列。")

    def _delete_auto_match_group(self) -> None:
        auto_star_ids = list(self._auto_match_reference_star_ids)
        if not auto_star_ids:
            self.ui.statusbar.showMessage("当前没有自动扩展匹配列表可删除。")
            return

        table = self.ui.tableWidgetStarPairs
        if self._active_star_pair_row is not None:
            active_star_id = self._star_pair_star_id(self._active_star_pair_row)
            if active_star_id in auto_star_ids:
                self._leave_star_pick_mode()

        for row in range(table.rowCount()):
            if self._star_pair_star_id(row) in auto_star_ids:
                self._clear_star_pair_position_data(row)

        deleted_count = len(auto_star_ids)
        self._auto_match_reference_star_ids = []
        self._auto_match_constraint_by_star_id = {
            star_id: constraint
            for star_id, constraint in self._auto_match_constraint_by_star_id.items()
            if star_id not in set(auto_star_ids)
        }
        self._auto_match_group_expanded = False
        self._refresh_reference_stars_from_current_map()
        self._update_reference_alignment_transform()
        self.ui.statusbar.showMessage(f"已删除自动扩展匹配列表，共 {deleted_count} 颗星。")

    def _handle_star_pair_delete_key(self) -> bool:
        table = self.ui.tableWidgetStarPairs
        rows = sorted({index.row() for index in table.selectedIndexes()})
        if not rows and table.currentRow() >= 0:
            rows = [table.currentRow()]
        if not rows:
            return False
        if any(self._is_auto_match_group_row(row) for row in rows):
            self._delete_auto_match_group()
            return True
        rows_with_position = [row for row in rows if self._star_pair_position_text(row)]
        rows_without_position = [row for row in rows if not self._star_pair_position_text(row)]
        for row in rows_with_position:
            self._clear_star_pair_position(row)
        for row in sorted(rows_without_position, reverse=True):
            self._delete_star_pair_row(row)
        return True

    def _scene_radius_from_screen_radius(
        self,
        view: QGraphicsView,
        viewport_pos: QPoint,
        screen_radius_px: int,
    ) -> float:
        scene_center = view.mapToScene(viewport_pos)
        scene_edge = view.mapToScene(viewport_pos + QPoint(max(1, screen_radius_px), 0))
        return max(1.0, float(np.hypot(scene_edge.x() - scene_center.x(), scene_edge.y() - scene_center.y())))

    def _reference_pick_star_positions(self) -> list[tuple[str, str, float, float, float, float]]:
        star_map = self._current_star_map
        if star_map is None:
            return []

        transform = self._sky_alignment_transform if self.current_image_preview is not None else None
        positions: list[tuple[str, str, float, float, float, float]] = []
        if len(star_map) > 0:
            if transform is None:
                star_points = np.column_stack((star_map.x_px, star_map.y_px))
            else:
                star_points = transform.transform_radec_points(np.column_stack((star_map.ra_deg, star_map.dec_deg)))
            for star_index in range(len(star_map)):
                if not bool(star_map.above_horizon[star_index]):
                    continue
                x_value = float(star_points[star_index, 0])
                y_value = float(star_points[star_index, 1])
                if not np.isfinite(x_value) or not np.isfinite(y_value):
                    continue
                reference_star = self._reference_star_from_star_map_index(star_map, star_index, output_index=0)
                positions.append(
                    (
                        reference_star.star_id,
                        reference_star.name,
                        x_value,
                        y_value,
                        float(star_map.mag_v[star_index]),
                        max(1.0, float(star_map.radius_px[star_index]) * self._current_reference_element_scale()),
                    )
                )

        return positions

    def _current_reference_element_scale(self) -> float:
        if self._sky_alignment_transform is None or self.current_image_preview is None:
            base_scale = 1.0
        else:
            image = self.current_image_preview.image
            base_scale = self._aligned_star_element_scale((image.width(), image.height()))
        return base_scale * self.reference_star_map_item.star_radius_zoom_scale()

    def _nearest_reference_pick_star(
        self,
        viewport_pos: QPoint,
    ) -> tuple[str, str, float] | None:
        positions = self._reference_pick_star_positions()
        if not positions:
            return None

        scene_pos = self.ui.referenceImageView.mapToScene(viewport_pos)
        click_x = float(scene_pos.x())
        click_y = float(scene_pos.y())
        scene_radius = self._scene_radius_from_screen_radius(
            self.ui.referenceImageView,
            viewport_pos,
            REFERENCE_STAR_PICK_SCREEN_RADIUS_PX,
        )
        mag_values = np.asarray([position[4] for position in positions], dtype=np.float64)
        brightest_mag = float(np.nanmin(mag_values))
        faintest_mag = float(np.nanmax(mag_values))
        mag_span = max(faintest_mag - brightest_mag, 1e-6)

        candidates: list[tuple[float, float, float, str, str]] = []
        for star_id, name, x_value, y_value, mag_v, radius_px in positions:
            distance = float(np.hypot(x_value - click_x, y_value - click_y))
            search_radius = scene_radius + max(radius_px * 1.5, 2.0)
            if distance > search_radius:
                continue
            brightness_rank = (mag_v - brightest_mag) / mag_span
            score = distance / max(search_radius, 1.0) + brightness_rank * 0.22
            candidates.append((score, distance, mag_v, star_id, name))

        if not candidates:
            return None
        _score, distance, _mag_v, star_id, name = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return star_id, name, distance

    def _handle_reference_map_click(self, viewport_pos: QPoint) -> None:
        picked = self._nearest_reference_pick_star(viewport_pos)
        if picked is None:
            self.ui.statusbar.showMessage("参考星图点击位置附近没有可用亮星，请稍微靠近星点再试。")
            return

        star_id, star_name, distance_px = picked
        if star_id in self._excluded_reference_star_ids:
            self._excluded_reference_star_ids.remove(star_id)
        existing_row = self._select_star_pair_row_by_id(star_id)
        if existing_row is not None:
            self.ui.statusbar.showMessage(
                f"已选中参考星 {self._star_pair_label(existing_row)}；可在真实图像中点选对应星点。"
            )
            return

        if star_id not in self._manual_reference_star_ids:
            self._manual_reference_star_ids.append(star_id)
        self._refresh_reference_stars_from_current_map()
        row = self._select_star_pair_row_by_id(star_id)
        if row is None:
            self.ui.statusbar.showMessage(f"未能添加参考星 {star_name or star_id}，请检查当前星等上限和视野。")
            return
        self.ui.statusbar.showMessage(
            f"已添加待匹配参考星 {self._star_pair_label(row)}；点击偏差约 {distance_px:.1f} px。"
        )

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
        self._add_or_update_star_pair_annotation(row, fitted_position)
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

    def _auto_pair_search_radius_px(self, transform: SkyAlignmentTransform) -> int:
        if self.current_image_preview is None:
            return self.ui_config.star_pick_psf_max_radius_px
        image = self.current_image_preview.image
        min_dimension = min(image.width(), image.height())
        radius = max(
            MIN_PSF_RADIUS_PX,
            self.ui_config.star_pick_psf_max_radius_px,
            int(round(transform.rms_px * AUTO_PAIR_RMS_RADIUS_SCALE + 6.0)),
        )
        return min(radius, AUTO_PAIR_MAX_SEARCH_RADIUS_PX, max(MIN_PSF_RADIUS_PX, min_dimension // 4))

    def _auto_pair_star(self, row: int) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再自动配对星点。")
            return

        transform = self._sky_alignment_transform
        if transform is None:
            self._update_reference_alignment_transform()
            transform = self._sky_alignment_transform
        if transform is None:
            QMessageBox.information(
                self,
                "无法自动配对",
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or f"至少需要 {MIN_ALIGNMENT_PAIRS} 对已配准参考星。",
            )
            return

        reference_star = self._reference_star_for_row(row)
        if reference_star is None:
            QMessageBox.warning(self, "无法自动配对", "当前行没有对应的参考星。")
            return

        predicted_x, predicted_y = transform.transform_radec(reference_star.ra_deg, reference_star.dec_deg)
        image = self.current_image_preview.image
        if not (0.0 <= predicted_x < image.width() and 0.0 <= predicted_y < image.height()):
            self.ui.statusbar.showMessage(
                f"{self._star_pair_label(row)} 的预测位置在真实图像外，无法自动配对。"
            )
            return

        search_radius_px = self._auto_pair_search_radius_px(transform)
        try:
            fitted_position = fit_star_position(
                image,
                click_x=predicted_x,
                click_y=predicted_y,
                radius_px=search_radius_px,
            )
        except Exception as exc:  # noqa: BLE001 - 自动配对要把失败原因反馈给用户。
            self.ui.statusbar.showMessage(f"自动配对失败: {exc}")
            QMessageBox.warning(self, "自动配对失败", str(exc))
            return

        distance_px = ((fitted_position.x - predicted_x) ** 2 + (fitted_position.y - predicted_y) ** 2) ** 0.5
        self._set_star_pair_position(row, fitted_position)
        self._add_or_update_star_pair_annotation(row, fitted_position)
        self.ui.tableWidgetStarPairs.selectRow(row)
        self.ui.statusbar.showMessage(
            "{label} 自动配对完成: x={x:.2f}, y={y:.2f}；预测偏差 {distance:.2f} px，搜索半径 {radius} px。".format(
                label=self._star_pair_label(row),
                x=fitted_position.x,
                y=fitted_position.y,
                distance=distance_px,
                radius=search_radius_px,
            )
        )

    def _auto_match_required_mag_limit(self) -> float:
        return max(
            self.ui.doubleSpinBoxMagLimit.value(),
            AUTO_MATCH_SEARCH_MAG_LIMIT,
        )

    def _ensure_current_star_map_for_auto_match(self, mag_limit: float) -> None:
        if self.ui.doubleSpinBoxMagLimit.value() + 1e-6 < mag_limit:
            was_blocked = self.ui.doubleSpinBoxMagLimit.blockSignals(True)
            self.ui.doubleSpinBoxMagLimit.setValue(min(mag_limit, self.ui.doubleSpinBoxMagLimit.maximum()))
            self.ui.doubleSpinBoxMagLimit.blockSignals(was_blocked)
            self.render_now()
        elif self._current_star_map is None:
            self.render_now()

    def _auto_match_candidate_stars(
        self,
        transform: SkyAlignmentTransform,
    ) -> tuple[list[ReferenceStar], dict[str, tuple[float, float]]]:
        if self.current_image_preview is None:
            return [], {}

        mag_limit = self._auto_match_required_mag_limit()
        self._ensure_current_star_map_for_auto_match(mag_limit)
        star_map = self._current_star_map
        if star_map is None or len(star_map) <= 0:
            return [], {}

        image = self.current_image_preview.image
        ra_dec_points = np.column_stack((star_map.ra_deg, star_map.dec_deg))
        predicted = transform.transform_radec_points(ra_dec_points)
        finite = np.all(np.isfinite(predicted), axis=1)
        inside = (
            finite
            & (predicted[:, 0] >= 0.0)
            & (predicted[:, 0] < image.width())
            & (predicted[:, 1] >= 0.0)
            & (predicted[:, 1] < image.height())
            & star_map.above_horizon.astype(bool)
        )

        if self.current_sky_mask is not None:
            mask_allowed = np.zeros(len(star_map), dtype=bool)
            for index in np.where(inside)[0]:
                mask_allowed[index] = self._sky_mask_allows_point(float(predicted[index, 0]), float(predicted[index, 1]))
            inside &= mask_allowed

        candidate_indices = np.where(inside)[0]
        if candidate_indices.size <= 0:
            return [], {}

        order = np.argsort(star_map.mag_v[candidate_indices], kind="stable")
        candidate_indices = candidate_indices[order]
        existing_star_ids = {
            self._star_pair_star_id(row)
            for row in range(self.ui.tableWidgetStarPairs.rowCount())
            if self._star_pair_star_id(row)
        }
        blocked_star_ids = existing_star_ids | set(self._auto_match_reference_star_ids) | set(self._excluded_reference_star_ids)
        new_star_limit = max(1, int(self.ui.spinBoxAutoMatchCount.value()))

        candidates: list[ReferenceStar] = []
        predicted_by_id: dict[str, tuple[float, float]] = {}
        seen_star_ids: set[str] = set()
        for star_index in candidate_indices:
            reference_star = self._reference_star_from_star_map_index(star_map, int(star_index), output_index=0)
            star_id = reference_star.star_id.strip()
            if not star_id or star_id in seen_star_ids or star_id in blocked_star_ids:
                continue
            seen_star_ids.add(star_id)
            candidates.append(reference_star)
            predicted_by_id[star_id] = (float(predicted[star_index, 0]), float(predicted[star_index, 1]))
            if len(candidates) >= new_star_limit:
                break
        return candidates, predicted_by_id

    def _ensure_auto_match_candidates_visible(self, candidates: list[ReferenceStar]) -> set[str]:
        visible_star_ids = {
            self._star_pair_star_id(row)
            for row in range(self.ui.tableWidgetStarPairs.rowCount())
            if self._star_pair_star_id(row)
        }
        candidate_auto_star_ids: set[str] = set()
        added_any = False
        constraint_mode = self._auto_match_constraint_mode()
        fit_weight = self._auto_match_soft_weight()
        for reference_star in candidates:
            star_id = reference_star.star_id.strip()
            if (
                not star_id
                or star_id in visible_star_ids
                or star_id in self._auto_match_reference_star_ids
                or star_id in self._excluded_reference_star_ids
            ):
                continue
            self._auto_match_reference_star_ids.append(star_id)
            self._auto_match_constraint_by_star_id[star_id] = (constraint_mode, fit_weight)
            candidate_auto_star_ids.add(star_id)
            visible_star_ids.add(star_id)
            added_any = True
        if added_any:
            self._auto_match_group_expanded = False
            self._refresh_reference_stars_from_current_map()
        return candidate_auto_star_ids

    def _remove_unmatched_auto_match_candidates(self, candidate_star_ids: set[str]) -> int:
        if not candidate_star_ids:
            return 0
        matched_positions = self._collect_star_pair_positions()
        remove_star_ids = {
            star_id
            for star_id in candidate_star_ids
            if star_id in self._auto_match_reference_star_ids and star_id not in matched_positions
        }
        if not remove_star_ids:
            return 0

        self._auto_match_reference_star_ids = [
            star_id for star_id in self._auto_match_reference_star_ids if star_id not in remove_star_ids
        ]
        for star_id in remove_star_ids:
            self._auto_match_constraint_by_star_id.pop(star_id, None)
        self._refresh_reference_stars_from_current_map()
        return len(remove_star_ids)

    def _existing_matched_positions(self) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            position = self._parse_star_pair_position_text(row)
            if position is not None:
                positions.append(position)
        return positions

    def _position_is_duplicate(self, position: tuple[float, float], accepted_positions: list[tuple[float, float]]) -> bool:
        for accepted_x, accepted_y in accepted_positions:
            if float(np.hypot(position[0] - accepted_x, position[1] - accepted_y)) < AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX:
                return True
        return False

    def auto_match_field_stars(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再自动扩展匹配。")
            return

        transform = self._sky_alignment_transform
        if transform is None:
            self._update_reference_alignment_transform()
            transform = self._sky_alignment_transform
        if transform is None:
            QMessageBox.information(
                self,
                "无法自动扩展匹配",
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or f"至少需要 {MIN_ALIGNMENT_PAIRS} 对已配准参考星。",
            )
            return

        candidates, predicted_by_id = self._auto_match_candidate_stars(transform)
        if not candidates:
            QMessageBox.information(self, "没有可新增星", "当前视场、数量设置和蒙版下没有新的可匹配参考星。")
            return

        self._ensure_auto_match_candidates_visible(candidates)
        image = self.current_image_preview.image
        search_radius_px = self.ui.spinBoxAutoMatchRadius.value()
        annotate_matches = len(candidates) <= AUTO_MATCH_ANNOTATION_LIMIT
        accepted_positions = self._existing_matched_positions()
        matched_count = 0
        skipped_existing = 0
        skipped_mask = 0
        skipped_duplicate = 0
        failed_count = 0
        canceled = False
        progress = QProgressDialog(self)
        progress.setWindowTitle("正在自动扩展匹配")
        progress.setLabelText(f"正在对 {len(candidates)} 颗新增星做 PSF 拟合...")
        progress.setRange(0, len(candidates))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()

        table = self.ui.tableWidgetStarPairs
        signals_were_blocked = table.blockSignals(True)
        try:
            for candidate_index, reference_star in enumerate(candidates, start=1):
                if progress.wasCanceled():
                    canceled = True
                    break
                if candidate_index == 1 or candidate_index % 10 == 0:
                    progress.setValue(candidate_index - 1)
                    QApplication.processEvents()

                star_id = reference_star.star_id.strip()
                row = self._row_for_star_id(star_id)
                if row is None:
                    failed_count += 1
                    continue
                if self._star_pair_position_text(row):
                    skipped_existing += 1
                    continue

                predicted_position = predicted_by_id.get(star_id)
                if predicted_position is None:
                    failed_count += 1
                    continue
                predicted_x, predicted_y = predicted_position
                if not self._sky_mask_allows_point(predicted_x, predicted_y):
                    skipped_mask += 1
                    continue

                try:
                    fitted_position = fit_star_position(
                        image,
                        click_x=predicted_x,
                        click_y=predicted_y,
                        radius_px=search_radius_px,
                    )
                except Exception:
                    failed_count += 1
                    continue

                distance_px = float(np.hypot(fitted_position.x - predicted_x, fitted_position.y - predicted_y))
                if distance_px > float(search_radius_px) or fitted_position.amplitude < AUTO_MATCH_MIN_AMPLITUDE:
                    failed_count += 1
                    continue
                if not self._sky_mask_allows_point(fitted_position.x, fitted_position.y):
                    skipped_mask += 1
                    continue
                fitted_xy = (float(fitted_position.x), float(fitted_position.y))
                if self._position_is_duplicate(fitted_xy, accepted_positions):
                    skipped_duplicate += 1
                    continue

                self._set_star_pair_position(row, fitted_position, update_alignment=False)
                if annotate_matches:
                    self._add_or_update_star_pair_annotation(row, fitted_position)
                accepted_positions.append(fitted_xy)
                matched_count += 1
        finally:
            table.blockSignals(signals_were_blocked)
            progress.setValue(len(candidates))
            progress.close()

        self._refresh_star_pair_table_styles()
        self._update_reference_alignment_transform()
        status_prefix = "自动扩展匹配已取消" if canceled else "自动扩展匹配完成"
        self.ui.statusbar.showMessage(
            "{status_prefix}：本次新增 {candidate_count}，配对成功 {matched_count}，已有 {skipped_existing}，"
            "蒙版跳过 {skipped_mask}，重复跳过 {skipped_duplicate}，失败 {failed_count}。".format(
                status_prefix=status_prefix,
                candidate_count=len(candidates),
                matched_count=matched_count,
                skipped_existing=skipped_existing,
                skipped_mask=skipped_mask,
                skipped_duplicate=skipped_duplicate,
                failed_count=failed_count,
            )
        )
        if matched_count <= 0 and not canceled:
            QMessageBox.information(
                self,
                "自动扩展匹配完成",
                "没有新增配对。可以检查蒙版、搜索半径和新增数量，或先增加几颗手动配对星提高初始配准精度。",
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

    def _set_json_import_controls_enabled(self, enabled: bool) -> None:
        self.ui.pushButtonImportReferenceJson.setEnabled(enabled)
        self.ui.pushButtonImportStarPairs.setEnabled(enabled)
        self.ui.pushButtonExportStarPairs.setEnabled(enabled)
        self.ui.pushButtonClearStarPairs.setEnabled(enabled)

    def _apply_loaded_image_preview(self, preview: ImagePreview, clear_existing_pairs: bool = True) -> None:
        if clear_existing_pairs:
            self._clear_star_pair_positions_for_new_input("新的真实图像")
        self.current_image_preview = preview
        self._clear_sky_mask_if_size_mismatch(preview.image.width(), preview.image.height())
        self._display_real_image_preview(preview)
        self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        self._update_imported_image_labels(preview)
        self.ui.statusbar.showMessage(
            "已导入图像: {path}  原始: {width} x {height} px。右键配对表行选择“点选位置”。".format(
                path=Path(preview.path).expanduser().resolve(),
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

    def import_sky_mask(self) -> None:
        if self._mask_import_thread is not None:
            QMessageBox.information(self, "正在导入蒙版", "当前已有蒙版正在导入，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导入同尺寸蒙版。")
            return

        default_dir = Path(self.current_image_preview.path).expanduser().resolve().parent
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入星空区域蒙版",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        self.start_sky_mask_import(file_path)

    def load_sky_mask(self, file_path: str | Path) -> None:
        self.start_sky_mask_import(file_path)

    def start_sky_mask_import(self, file_path: str | Path) -> None:
        if self._mask_import_thread is not None:
            QMessageBox.information(self, "正在导入蒙版", "当前已有蒙版正在导入，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导入同尺寸蒙版。")
            return

        mask_path = Path(file_path).expanduser().resolve()
        image = self.current_image_preview.image
        source_path = Path(self.current_image_preview.path).expanduser().resolve()
        self._set_mask_import_controls_enabled(False)
        self.ui.statusbar.showMessage(f"正在导入蒙版并生成缓存预览: {mask_path}")

        progress = QProgressDialog(self)
        progress.setWindowTitle("正在导入蒙版")
        progress.setLabelText(f"正在读取蒙版并生成缓存预览...\n{mask_path}")
        progress.setRange(0, 0)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        thread = QThread(self)
        worker = SkyMaskLoadWorker(
            mask_path,
            expected_size=(image.width(), image.height()),
            source_image=image,
            source_path=source_path,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_sky_mask_import_finished)
        worker.failed.connect(self._handle_sky_mask_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_sky_mask_import)

        self._mask_import_thread = thread
        self._mask_import_worker = worker
        self._mask_import_progress = progress
        thread.start()

    def _handle_sky_mask_import_finished(self, result: object) -> None:
        if self._mask_import_progress is not None:
            self._mask_import_progress.close()
        try:
            mask_path, source_path, mask, masked_image = result  # type: ignore[misc]
            if not isinstance(mask_path, Path):
                mask_path = Path(mask_path)
            if not isinstance(source_path, Path):
                source_path = Path(source_path)
            if self.current_image_preview is None:
                raise ValueError("真实图像已关闭，无法应用蒙版。")
            current_source_path = Path(self.current_image_preview.path).expanduser().resolve()
            if current_source_path != source_path:
                raise ValueError("真实图像已改变，请重新导入蒙版。")
            image = self.current_image_preview.image
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != (image.height(), image.width()):
                raise ValueError("蒙版尺寸与当前真实图像不一致，请重新导入。")

            self.current_sky_mask_path = mask_path
            self.current_sky_mask = mask_array
            self.current_sky_masked_image = masked_image if isinstance(masked_image, QImage) else None
            self._update_sky_mask_status()
            self._update_reference_alignment_controls()
            self._refresh_real_image_display_for_mask()
            self.ui.statusbar.showMessage(f"已导入蒙版并缓存显示图: {mask_path}")
        except Exception as exc:  # noqa: BLE001 - 主线程应用蒙版时需要把状态错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导入蒙版失败: {exc}")
            QMessageBox.critical(self, "导入蒙版失败", str(exc))

    def _handle_sky_mask_import_failed(self, error_message: str) -> None:
        if self._mask_import_progress is not None:
            self._mask_import_progress.close()
        self.ui.statusbar.showMessage(f"导入蒙版失败: {error_message}")
        QMessageBox.critical(self, "导入蒙版失败", error_message)

    def _cleanup_sky_mask_import(self) -> None:
        if self._mask_import_progress is not None:
            self._mask_import_progress.close()
        self._mask_import_thread = None
        self._mask_import_worker = None
        self._mask_import_progress = None
        self._update_reference_alignment_controls()

    def clear_sky_mask(self) -> None:
        if self.current_sky_mask is None:
            self.ui.statusbar.showMessage("当前没有正在使用的蒙版。")
            return
        self._reset_sky_mask_status()
        self._update_reference_alignment_controls()
        self._refresh_real_image_display_for_mask()
        self.ui.statusbar.showMessage("已清除蒙版，后续自动匹配将使用整张图像。")

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
        self._handle_reference_label_options_changed()

    def _handle_reference_label_options_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_reference_label_controls()
        self._refresh_reference_stars_from_current_map()
        self.schedule_render()

    def _reference_star_from_star_map_index(
        self,
        star_map: ProjectedStarMap,
        star_index: int,
        output_index: int,
    ) -> ReferenceStar:
        star_id = str(star_map.star_ids[star_index]).strip()
        display_name = str(star_map.display_names[star_index]).strip()
        common_name = str(star_map.common_names[star_index]).strip()
        name = common_name or display_name or star_id
        return ReferenceStar(
            index=output_index,
            star_id=star_id,
            name=name,
            display_name=display_name,
            common_name=common_name,
            ra_deg=float(star_map.ra_deg[star_index]),
            dec_deg=float(star_map.dec_deg[star_index]),
            mag_v=float(star_map.mag_v[star_index]),
            sim_x=float(star_map.x_px[star_index]),
            sim_y=float(star_map.y_px[star_index]),
            alt_deg=float(star_map.alt_deg[star_index]),
            az_deg=float(star_map.az_deg[star_index]),
        )

    def _reference_star_with_index(self, star: ReferenceStar, index: int) -> ReferenceStar:
        return ReferenceStar(
            index=index,
            star_id=star.star_id,
            name=star.name,
            display_name=star.display_name,
            common_name=star.common_name,
            ra_deg=star.ra_deg,
            dec_deg=star.dec_deg,
            mag_v=star.mag_v,
            sim_x=star.sim_x,
            sim_y=star.sim_y,
            alt_deg=star.alt_deg,
            az_deg=star.az_deg,
            object_type=star.object_type,
        )

    def _projected_reference_star_lookup(self, star_map: ProjectedStarMap) -> dict[str, ReferenceStar]:
        lookup: dict[str, ReferenceStar] = {}
        for star_index in range(len(star_map)):
            if not bool(star_map.above_horizon[star_index]):
                continue
            reference_star = self._reference_star_from_star_map_index(star_map, star_index, output_index=0)
            if reference_star.star_id:
                lookup[reference_star.star_id] = reference_star

        return lookup

    def _select_current_reference_stars(self, star_map: ProjectedStarMap) -> tuple[ReferenceStar, ...]:
        if self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT:
            auto_reference_stars = select_reference_stars(
                star_map=star_map,
                max_count=None,
                mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            )
        else:
            auto_reference_stars = select_reference_stars(
                star_map=star_map,
                max_count=self.ui.spinBoxReferenceStarCount.value(),
            )

        # 手动点选的参考星以星表编号保存；每次渲染后用当前投影坐标重新生成行。
        ordered_stars: list[ReferenceStar] = []
        seen_star_ids: set[str] = set()
        excluded_star_ids = set(self._excluded_reference_star_ids)
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        for star in auto_reference_stars:
            star_id = star.star_id.strip()
            if not star_id or star_id in seen_star_ids or star_id in excluded_star_ids or star_id in auto_match_star_ids:
                continue
            seen_star_ids.add(star_id)
            ordered_stars.append(star)

        manual_lookup = self._projected_reference_star_lookup(star_map)
        for star_id in self._manual_reference_star_ids:
            if star_id in seen_star_ids or star_id in excluded_star_ids or star_id in auto_match_star_ids:
                continue
            manual_star = manual_lookup.get(star_id)
            if manual_star is None:
                continue
            seen_star_ids.add(star_id)
            ordered_stars.append(manual_star)

        for star_id in self._auto_match_reference_star_ids:
            if star_id in seen_star_ids or star_id in excluded_star_ids:
                continue
            auto_match_star = manual_lookup.get(star_id)
            if auto_match_star is None:
                continue
            seen_star_ids.add(star_id)
            ordered_stars.append(auto_match_star)

        return tuple(self._reference_star_with_index(star, index) for index, star in enumerate(ordered_stars, start=1))

    def _refresh_reference_stars_from_current_map(self) -> None:
        if self._current_star_map is None:
            return
        reference_stars = self._select_current_reference_stars(self._current_star_map)
        self._update_star_pair_table(reference_stars)

    def _row_for_star_id(self, star_id: str) -> int | None:
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            if self._star_pair_star_id(row) == star_id:
                return row
        return None

    def _select_star_pair_row_by_id(self, star_id: str) -> int | None:
        row = self._row_for_star_id(star_id)
        if row is not None:
            self.ui.tableWidgetStarPairs.selectRow(row)
            return row
        return None

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

    def _aligned_star_element_scale(self, target_size: tuple[int, int]) -> float:
        long_side = max(float(target_size[0]), float(target_size[1]), 1.0)
        base_scale = long_side / float(PREVIEW_LONG_SIDE_PX)
        # 对齐到真实图像后，场景尺寸通常远大于预览图；这里用配置倍率补偿 fitInView 后的视觉缩小。
        return max(0.75, min(12.0, base_scale * self.ui_config.aligned_reference_scale_multiplier))

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
        visible_mag_limit: float | None = None,
    ) -> tuple[ObserverSettings, CameraSettings, ViewSettings, float, ProjectedStarMap]:
        observer = self._observer_settings()
        camera = camera or self._preview_camera_settings()
        view = self._view_settings()
        mag_limit = self.ui.doubleSpinBoxMagLimit.value() if visible_mag_limit is None else float(visible_mag_limit)
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

    def _display_star_map(self, star_map: ProjectedStarMap, reference_stars: tuple[ReferenceStar, ...]) -> None:
        render_size = (star_map.width, star_map.height)
        should_fit = self._last_render_size != render_size or self.star_map_item.boundingRect().isEmpty()
        self.star_map_item.setPos(0, 0)
        self.star_map_item.set_star_map(
            star_map,
            reference_stars=reference_stars,
            element_scale=1.0,
            draw_common_names=False,
            number_reference_stars=False,
        )
        self.scene.setSceneRect(0, 0, star_map.width, star_map.height)
        self._last_render_size = render_size
        if should_fit:
            self.fit_star_map()

    def _fit_reference_map_if_display_changed(self, display_key: tuple[object, ...]) -> None:
        should_fit = self._last_reference_render_size != display_key or self.reference_star_map_item.boundingRect().isEmpty()
        self._last_reference_render_size = display_key
        if should_fit:
            self.fit_reference_map()

    def _display_real_image_preview(self, preview: ImagePreview) -> None:
        self._clear_star_pair_annotations()
        self.real_image_item.setPos(0, 0)
        self.real_image_item.set_image(self._real_image_for_current_mask_preview())
        self.real_image_scene.setSceneRect(0, 0, preview.image.width(), preview.image.height())
        self._restore_star_pair_annotations_from_table()
        self._update_reference_alignment_transform()
        self.fit_real_image()

    def render_now(self) -> None:
        try:
            _observer, _camera, view, _mag_limit, star_map = self._build_projected_star_map()
            self._current_star_map = star_map
            reference_stars = self._select_current_reference_stars(star_map)
            self._display_star_map(star_map, reference_stars)
            self._update_star_pair_table(reference_stars)
            if self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT:
                reference_mode_text = f"标注星等 <= {self.ui.doubleSpinBoxReferenceMagLimit.value():.1f} mag"
            else:
                reference_mode_text = f"标注星数 {self.ui.spinBoxReferenceStarCount.value()} 颗"
            manual_count = len(
                [
                    star_id
                    for star_id in self._manual_reference_star_ids
                    if any(star.star_id == star_id for star in reference_stars)
                ]
            )
            if manual_count:
                reference_mode_text = f"{reference_mode_text}，手动 {manual_count} 颗"
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
        if self._json_import_thread is not None:
            QMessageBox.information(self, "正在导入 JSON", "当前已有 JSON 正在导入，请稍候。")
            return
        json_path = Path(file_path)
        self._set_json_import_controls_enabled(False)
        self._json_import_progress = self._show_json_import_progress(
            title="正在导入预览 JSON",
            label_text=f"正在读取预览 JSON 并恢复星空模拟参数...\n{json_path}",
            status_text=f"正在导入预览 JSON: {json_path}",
        )

        thread = QThread(self)
        worker = ReferenceJsonImportWorker(json_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_reference_json_import_finished)
        worker.failed.connect(self._handle_reference_json_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_json_import)

        self._json_import_thread = thread
        self._json_import_worker = worker
        thread.start()

    def _handle_reference_json_import_finished(self, result: object) -> None:
        try:
            source_path, payload = result  # type: ignore[misc]
            if not isinstance(source_path, Path):
                source_path = Path(source_path)
            self._clear_star_pair_positions_for_new_input("新的预览 JSON")
            self._apply_reference_payload(payload, source_path)
        except Exception as exc:  # noqa: BLE001 - 主线程恢复界面时也需要把错误反馈给用户。
            self.ui.statusbar.showMessage(f"导入预览 JSON 失败: {exc}")
            QMessageBox.critical(self, "导入预览 JSON 失败", str(exc))

    def _handle_reference_json_import_failed(self, error_message: str) -> None:
        self.ui.statusbar.showMessage(f"导入预览 JSON 失败: {error_message}")
        QMessageBox.critical(self, "导入预览 JSON 失败", error_message)

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

        restored_manual_star_ids: list[str] = []
        manual_ids_payload = payload.get("manual_reference_star_ids")
        if isinstance(manual_ids_payload, list):
            for raw_star_id in manual_ids_payload:
                star_id = str(raw_star_id).strip()
                if star_id and star_id not in restored_manual_star_ids:
                    restored_manual_star_ids.append(star_id)
        self._manual_reference_star_ids = restored_manual_star_ids
        self._auto_match_reference_star_ids = []
        self._auto_match_constraint_by_star_id = {}
        self._auto_match_group_expanded = False
        self._excluded_reference_star_ids = []
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
                manual_reference_star_ids=tuple(self._manual_reference_star_ids),
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
        if not self.scene.sceneRect().isEmpty():
            self.ui.starMapView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._update_live_star_map_zoom_scale(self.ui.starMapView)

    def fit_reference_map(self) -> None:
        if not self.reference_scene.sceneRect().isEmpty():
            self.ui.referenceImageView.fitInView(self.reference_scene.sceneRect(), Qt.KeepAspectRatio)
            self._update_live_star_map_zoom_scale(self.ui.referenceImageView)

    def fit_real_image(self) -> None:
        if not self.real_image_item.isNull():
            self.ui.realImageView.fitInView(self.real_image_scene.sceneRect(), Qt.KeepAspectRatio)
            self._cap_graphics_view_to_max_scale(self.ui.realImageView)
            self._update_live_star_map_zoom_scale(self.ui.realImageView)

    def fit_all_graphics_views(self) -> None:
        self._set_equal_reference_preview_sizes()
        self.fit_star_map()
        if self._can_sync_reference_real_views():
            self.fit_real_image()
            self._sync_reference_real_view_from(self.ui.realImageView, force=True)
        else:
            self.fit_reference_map()
            self.fit_real_image()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "图像预览仍在生成，请等待导入完成后再关闭窗口。")
            event.ignore()
            return
        if self._json_import_thread is not None:
            QMessageBox.information(self, "正在导入 JSON", "JSON 仍在导入，请等待完成后再关闭窗口。")
            event.ignore()
            return
        if self._mask_import_thread is not None:
            QMessageBox.information(self, "正在导入蒙版", "蒙版仍在导入，请等待完成后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        QTimer.singleShot(0, self._set_equal_reference_preview_sizes)
        QTimer.singleShot(0, self._refresh_all_elided_labels)
        QTimer.singleShot(0, self.fit_all_graphics_views)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is self.ui.splitterReferenceAndRealImage:
            if event.type() in (QEvent.Resize, QEvent.Show):
                QTimer.singleShot(0, self._set_equal_reference_preview_sizes)
            return False
        if watched in (
            self.ui.labelImportedImagePath,
            self.ui.labelSkyMaskStatus,
            self.ui.labelAlignmentTransformStatus,
        ):
            if event.type() in (QEvent.Resize, QEvent.Show):
                QTimer.singleShot(0, lambda label=watched: self._refresh_elided_label(label))
            return False
        if watched is self.ui.tableWidgetStarPairs and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                return self._handle_star_pair_delete_key()
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
        if watched is self.ui.referenceImageView:
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.referenceImageView,
                event,
            ):
                return True
        if watched is self.ui.realImageView:
            if event.type() == QEvent.NativeGesture:
                if (
                    self._active_star_pair_row is not None
                    and event.modifiers() & Qt.ControlModifier
                    and self._handle_star_pick_native_zoom(event)
                ):
                    return True
                if self._handle_graphics_view_native_zoom(self.ui.realImageView, event):
                    return True
        if watched is self.ui.referenceImageView.viewport():
            if event.type() in (QEvent.Enter, QEvent.MouseMove, QEvent.KeyPress, QEvent.KeyRelease):
                self._update_reference_map_cursor(self._event_ctrl_pressed(event))
            if event.type() == QEvent.Leave:
                self._reference_pick_press_pos = None
                self.ui.referenceImageView.viewport().unsetCursor()
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if self._event_ctrl_pressed(event):
                    self._update_reference_map_cursor(True)
                    self._reference_pick_press_pos = event.pos()
                    return True
                self._update_reference_map_cursor(False)
                self._reference_pick_press_pos = None
                return False
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                press_pos = self._reference_pick_press_pos
                self._reference_pick_press_pos = None
                if press_pos is not None:
                    move_distance = (event.pos() - press_pos).manhattanLength()
                    if move_distance <= QApplication.startDragDistance():
                        self._handle_reference_map_click(event.pos())
                    return True
            if event.type() == QEvent.Wheel:
                self._apply_graphics_view_zoom(self.ui.referenceImageView, event.angleDelta().y())
                return True
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.referenceImageView,
                event,
            ):
                return True
        if watched is self.ui.realImageView.viewport():
            if self._active_star_pair_row is not None:
                if event.type() in (QEvent.Enter, QEvent.MouseMove, QEvent.KeyPress, QEvent.KeyRelease):
                    self._update_real_image_pick_cursor(self._event_ctrl_pressed(event))
                    self._show_star_pick_status_hint(self._active_star_pair_row)
                if event.type() == QEvent.Leave:
                    self.ui.realImageView.viewport().unsetCursor()
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    if self._event_ctrl_pressed(event):
                        self._handle_real_image_pick_click(event.pos())
                        return True
                    return False
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
                if (
                    event.type() == QEvent.NativeGesture
                    and event.modifiers() & Qt.ControlModifier
                    and self._handle_star_pick_native_zoom(event)
                ):
                    return True
                if event.type() == QEvent.KeyPress and self._handle_star_pick_key_press(event):
                    return True
            if event.type() == QEvent.Wheel:
                self._apply_graphics_view_zoom(self.ui.realImageView, event.angleDelta().y())
                return True
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.realImageView,
                event,
            ):
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
        self._apply_graphics_view_zoom_factor(view, factor)

    def _apply_graphics_view_zoom_factor(self, view: QGraphicsView, factor: float) -> None:
        if factor <= 0.0 or not math.isfinite(factor) or abs(factor - 1.0) <= 1e-4:
            return
        if factor < 1.0:
            min_scale = self._graphics_view_fit_scale(view)
            current_scale = self._graphics_view_current_scale(view)
            if current_scale * factor <= min_scale:
                self._fit_graphics_view_to_scene(view)
                self._sync_reference_real_view_from(view)
                return
        if factor > 1.0:
            max_scale = self._graphics_view_max_scale(view)
            current_scale = self._graphics_view_current_scale(view)
            if max_scale is not None and current_scale * factor >= max_scale:
                self._set_graphics_view_scale(view, max_scale)
                self._update_live_star_map_zoom_scale(view)
                self._sync_reference_real_view_from(view)
                return
        view.scale(factor, factor)
        self._update_live_star_map_zoom_scale(view)
        self._sync_reference_real_view_from(view)

    def _native_gesture_zoom_value(self, event) -> float:  # type: ignore[no-untyped-def]
        if event.type() != QEvent.NativeGesture:
            return 0.0

        zoom_gesture = getattr(Qt, "ZoomNativeGesture", None)
        if zoom_gesture is None or event.gestureType() != zoom_gesture:
            return 0.0

        try:
            value = float(event.value())
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value) or abs(value) <= 1e-6:
            return 0.0
        return value

    def _native_gesture_zoom_factor(self, event) -> float:  # type: ignore[no-untyped-def]
        value = self._native_gesture_zoom_value(event)
        if value == 0.0:
            return 1.0

        # macOS 触控板的原生缩放值是连续增量，用指数映射避免小幅捏合过钝或过猛。
        factor = math.exp(value * TOUCHPAD_ZOOM_SENSITIVITY)
        return max(TOUCHPAD_ZOOM_MIN_FACTOR, min(TOUCHPAD_ZOOM_MAX_FACTOR, factor))

    def _handle_graphics_view_native_zoom(self, view: QGraphicsView, event) -> bool:  # type: ignore[no-untyped-def]
        factor = self._native_gesture_zoom_factor(event)
        if abs(factor - 1.0) <= 1e-4:
            return False
        self._apply_graphics_view_zoom_factor(view, factor)
        return True

    def _handle_star_pick_native_zoom(self, event) -> bool:  # type: ignore[no-untyped-def]
        value = self._native_gesture_zoom_value(event)
        if value == 0.0:
            return False

        self._star_pick_native_zoom_remainder += value * STAR_PICK_TOUCHPAD_STEPS_PER_ZOOM_UNIT
        wheel_steps = int(self._star_pick_native_zoom_remainder)
        if wheel_steps != 0:
            self._star_pick_native_zoom_remainder -= wheel_steps
            self._adjust_star_pick_circle_diameter(wheel_steps)
        return True

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
        self._update_live_star_map_zoom_scale(view)

    def _graphics_view_max_scale(self, view: QGraphicsView) -> float | None:
        if view is self.ui.realImageView:
            return self._real_image_zoom_max_scale
        if view is self.ui.referenceImageView and self._can_sync_reference_real_views():
            return self._real_image_zoom_max_scale
        return None

    def _live_star_map_item_for_view(self, view: QGraphicsView) -> LiveStarMapGraphicsItem | None:
        if view is self.ui.starMapView:
            return self.star_map_item
        if view is self.ui.referenceImageView:
            return self.reference_star_map_item
        if view is self.ui.realImageView:
            return self.real_reference_overlay_item
        return None

    def _update_live_star_map_zoom_scale(self, view: QGraphicsView) -> None:
        item = self._live_star_map_item_for_view(view)
        if item is None:
            return
        fit_scale = max(self._graphics_view_fit_scale(view), 1e-6)
        current_scale = max(self._graphics_view_current_scale(view), 1e-6)
        item.set_view_zoom_scale(current_scale / fit_scale)

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
