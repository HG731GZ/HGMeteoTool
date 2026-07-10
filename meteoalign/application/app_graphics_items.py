from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QBrush, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QGraphicsItem

from ..renderer import StarMapRenderer
from ..simulator import ProjectedStarMap, ReferenceStar
from ..alignment.models import SkyAlignmentTransform

# 从 app.py 移出的常量，供 LiveStarMapGraphicsItem 使用
STAR_RADIUS_ZOOM_EXPONENT = 0.32
STAR_RADIUS_MIN_ZOOM_SCALE = 0.48


def _reference_star_index_text(reference_star: ReferenceStar) -> str:
    return str(getattr(reference_star, "index_label", "") or reference_star.index)


class GraphicsImageItem(QGraphicsItem):
    """用于在 QGraphicsScene 中显示 QImage 的自定义图形项。"""

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()

    def set_image(self, image: QImage) -> None:
        """替换当前显示的图像。"""
        self.prepareGeometryChange()
        self.image = image

    def isNull(self) -> bool:
        """返回当前是否没有有效图像。"""
        return self.image.isNull()

    def pixmap(self) -> QPixmap:
        """将当前图像转换为 QPixmap，便于其他地方使用。"""
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
    """实时星空模拟图的 QGraphicsItem，支持原始渲染和配准叠加两种模式。"""

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
        """设置当前要渲染的星图数据。"""
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
        """清空当前星图显示。"""
        self.set_star_map(None)

    def set_view_zoom_scale(self, zoom_scale: float) -> None:
        """设置视图缩放比例，用于自适应星点大小。"""
        new_zoom_scale = max(float(zoom_scale), 1.0)
        if abs(new_zoom_scale - self.view_zoom_scale) <= 1e-3:
            return
        self.view_zoom_scale = new_zoom_scale
        self.update()

    def star_radius_zoom_scale(self) -> float:
        """根据当前视图缩放比例计算星点半径缩放因子。"""
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

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

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

        label_scale = max(self.element_scale, 0.75) * self.star_radius_zoom_scale()
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

            label_text = f"{_reference_star_index_text(reference_star)}. {reference_star.name}"
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
