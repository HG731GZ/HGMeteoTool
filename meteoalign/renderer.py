from __future__ import annotations

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF

from .config import StarMapUiConfig
from .simulator import ProjectedStarMap, ReferenceStar


class StarMapRenderer:
    def __init__(self, ui_config: StarMapUiConfig | None = None) -> None:
        self.ui_config = ui_config or StarMapUiConfig()

    def render(
        self,
        star_map: ProjectedStarMap,
        common_name_mag_limit: float = 1.0,
        reference_stars: tuple[ReferenceStar, ...] = (),
    ) -> QImage:
        image = QImage(star_map.width, star_map.height, QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(88, 88, 88))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)

        self._draw_sky_background(painter, star_map)
        self._draw_milky_way(painter, star_map)
        painter.setPen(Qt.NoPen)

        order = star_map.radius_px.argsort()
        for index in order:
            red, green, blue = (int(value) for value in star_map.star_rgb[index])
            alpha = int(star_map.alpha[index])
            color = QColor(red, green, blue, alpha)
            painter.setBrush(color)
            radius = float(star_map.radius_px[index])
            center = QPointF(float(star_map.x_px[index]), float(star_map.y_px[index]))
            painter.drawEllipse(QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0))

        self._draw_horizon_shadow(painter, star_map)
        self._draw_grid(painter, star_map)
        if not reference_stars:
            self._draw_star_names(painter, star_map, common_name_mag_limit)
        self._draw_reference_stars(painter, star_map, reference_stars)
        self._draw_direction_labels(painter, star_map)
        painter.end()
        return image

    def _draw_sky_background(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0))
        if star_map.sky_circle_radius_px is None:
            painter.drawRect(QRectF(0.0, 0.0, float(star_map.width), float(star_map.height)))
            return

        radius = float(star_map.sky_circle_radius_px)
        center = QPointF(star_map.width * 0.5, star_map.height * 0.5)
        painter.drawEllipse(QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0))

    def _draw_horizon_shadow(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        if not star_map.horizon_shadow_rects:
            return

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(30, 30, 30, 150))
        for rect in star_map.horizon_shadow_rects:
            painter.drawRect(QRectF(rect.x_px, rect.y_px, rect.width_px, rect.height_px))

    def _draw_milky_way(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        if not star_map.milky_way_polygons:
            return

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(165, 165, 165, 52))
        for polygon in star_map.milky_way_polygons:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            has_ring = False
            for ring in polygon.rings:
                if len(ring) < 3:
                    continue
                first_x, first_y = ring[0]
                path.moveTo(first_x, first_y)
                for x_value, y_value in ring[1:]:
                    path.lineTo(x_value, y_value)
                path.closeSubpath()
                has_ring = True
            if has_ring:
                painter.drawPath(path)

    def _draw_grid(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        painter.setBrush(Qt.NoBrush)
        for line in star_map.grid_lines:
            if len(line.points) < 2:
                continue
            if line.kind == "horizon":
                pen = QPen(QColor(80, 255, 80, 235), 2.8)
            elif line.kind == "azimuth":
                pen = QPen(QColor(150, 170, 190, 95), 1.0)
                pen.setStyle(Qt.DashLine)
            else:
                pen = QPen(QColor(150, 170, 190, 80), 1.0)
            painter.setPen(pen)
            polygon = QPolygonF([QPointF(x_value, y_value) for x_value, y_value in line.points])
            painter.drawPolyline(polygon)

    def _draw_star_names(self, painter: QPainter, star_map: ProjectedStarMap, common_name_mag_limit: float) -> None:
        font = QFont()
        font.setPointSize(self.ui_config.star_name_font_size_pt)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        for index, common_name in enumerate(star_map.common_names):
            name = str(common_name).strip()
            if not name or float(star_map.mag_v[index]) > common_name_mag_limit:
                continue

            red, green, blue = (int(value) for value in star_map.star_rgb[index])
            alpha = 235 if bool(star_map.above_horizon[index]) else 165
            text_width = metrics.horizontalAdvance(name)
            text_height = metrics.height()
            x_value = min(max(float(star_map.x_px[index]) + 8.0, 4.0), star_map.width - text_width - 4.0)
            y_value = min(max(float(star_map.y_px[index]) - 8.0, text_height + 4.0), star_map.height - 4.0)
            point = QPointF(x_value, y_value)

            painter.setPen(QPen(QColor(0, 0, 0, 220), 3.0))
            painter.drawText(point, name)
            painter.setPen(QPen(QColor(red, green, blue, alpha), 1.0))
            painter.drawText(point, name)

    def _draw_reference_stars(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        reference_stars: tuple[ReferenceStar, ...],
    ) -> None:
        if not reference_stars:
            return

        number_font = QFont()
        number_font.setPointSize(self.ui_config.reference_label_font_size_pt)
        number_font.setBold(True)

        for reference_star in reference_stars:
            x_value = float(reference_star.sim_x)
            y_value = float(reference_star.sim_y)
            marker_radius = 17.0
            marker_rect = QRectF(
                x_value - marker_radius,
                y_value - marker_radius,
                marker_radius * 2.0,
                marker_radius * 2.0,
            )

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 230), 5.0))
            painter.drawEllipse(marker_rect)
            painter.setPen(QPen(QColor(255, 230, 80, 255), 2.4))
            painter.drawEllipse(marker_rect)

            painter.setFont(number_font)
            metrics = painter.fontMetrics()
            label_text = f"{reference_star.index}. {reference_star.name}"
            max_label_width = max(40.0, star_map.width - 8.0)
            visible_label_text = metrics.elidedText(label_text, Qt.ElideRight, int(max_label_width - 14.0))
            label_width = min(metrics.horizontalAdvance(visible_label_text) + 14.0, max_label_width)
            label_height = metrics.height() + 8.0
            preferred_x = x_value + marker_radius + 8.0
            if preferred_x + label_width > star_map.width - 4.0:
                preferred_x = x_value - marker_radius - label_width - 8.0
            label_x = min(max(preferred_x, 4.0), max(4.0, star_map.width - label_width - 4.0))
            label_y = min(max(y_value - label_height / 2.0, 4.0), max(4.0, star_map.height - label_height - 4.0))
            label_rect = QRectF(label_x, label_y, label_width, label_height)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(label_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor(255, 240, 130, 255), 1.0))
            painter.drawText(label_rect, Qt.AlignCenter, visible_label_text)

    def _draw_direction_labels(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        if not star_map.direction_labels:
            return

        font = QFont()
        font.setPointSize(self.ui_config.direction_label_font_size_pt)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        for label in star_map.direction_labels:
            text_rect = metrics.boundingRect(label.text)
            width = text_rect.width() + 12
            height = text_rect.height() + 8
            x_value = min(max(label.x_px - width / 2.0, 4.0), star_map.width - width - 4.0)
            y_value = min(max(label.y_px - height / 2.0, 4.0), star_map.height - height - 4.0)
            rect = QRectF(x_value, y_value, width, height)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QPen(QColor(245, 245, 230, 230), 1.0))
            painter.drawText(rect, Qt.AlignCenter, label.text)
