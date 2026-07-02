from __future__ import annotations

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF

from .simulator import ProjectedStarMap


DIRECTION_LABEL_FONT_SIZE_PT = 16
STAR_NAME_FONT_SIZE_PT = 11


class StarMapRenderer:
    def render(self, star_map: ProjectedStarMap) -> QImage:
        image = QImage(star_map.width, star_map.height, QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)

        self._draw_grid(painter, star_map)
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

        self._draw_star_names(painter, star_map)
        self._draw_direction_labels(painter, star_map)
        painter.end()
        return image

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

    def _draw_star_names(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        font = QFont()
        font.setPointSize(STAR_NAME_FONT_SIZE_PT)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        for index, common_name in enumerate(star_map.common_names):
            name = str(common_name).strip()
            if not name or float(star_map.mag_v[index]) > 1.0:
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

    def _draw_direction_labels(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        if not star_map.direction_labels:
            return

        font = QFont()
        font.setPointSize(DIRECTION_LABEL_FONT_SIZE_PT)
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
