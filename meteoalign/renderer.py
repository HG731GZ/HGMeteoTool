from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF

from .config import StarMapUiConfig
from .meteor_showers import projected_meteor_radiants, projected_meteors
from .simulator import ProjectedStarMap, ReferenceStar, STAR_STYLE_BASE_RADIUS_PX


def _reference_star_index_text(reference_star: ReferenceStar) -> str:
    return str(getattr(reference_star, "index_label", "") or reference_star.index)


def _meteor_brightness_profile(progress: float) -> float:
    """返回流星沿途的平滑亮度，峰值固定在总长度约四分之三处。"""

    safe_progress = max(0.0, min(1.0, float(progress)))
    if safe_progress <= 0.75:
        return (safe_progress / 0.75) ** 0.62
    return ((1.0 - safe_progress) / 0.25) ** 0.72


def _valid_meteor_point_runs(
    points: tuple[tuple[float, float, float, bool], ...],
) -> tuple[tuple[tuple[float, float, float, bool], ...], ...]:
    """按投影连续性拆分轨迹，画面外但坐标有限的点仍属于连续段。"""

    runs: list[tuple[tuple[float, float, float, bool], ...]] = []
    current: list[tuple[float, float, float, bool]] = []
    for point in points:
        if point[3]:
            current.append(point)
            continue
        if len(current) >= 2:
            runs.append(tuple(current))
        current = []
    if len(current) >= 2:
        runs.append(tuple(current))
    return tuple(runs)


def _meteor_run_length_px(
    run: tuple[tuple[float, float, float, bool], ...],
    width: int,
    height: int,
) -> float:
    """估算轨迹屏幕长度，并限制透视投影边缘的数值发散。"""

    diagonal = max(1.0, float(np.hypot(width, height)))
    segment_limit = diagonal * 0.25
    length = sum(
        min(float(np.hypot(end[0] - start[0], end[1] - start[1])), segment_limit)
        for start, end in zip(run, run[1:])
    )
    return min(length, diagonal * 2.0)


def _meteor_shape_path(
    run: tuple[tuple[float, float, float, bool], ...],
    maximum_width_px: float,
) -> QPainterPath:
    """由连续中心线生成宽度随亮度变化的单个梭形路径。"""

    coordinates = np.asarray([(point[0], point[1]) for point in run], dtype=np.float64)
    tangents = np.empty_like(coordinates)
    tangents[0] = coordinates[1] - coordinates[0]
    tangents[-1] = coordinates[-1] - coordinates[-2]
    if len(run) > 2:
        tangents[1:-1] = coordinates[2:] - coordinates[:-2]
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    safe_norms = np.where(norms > 1e-9, norms, 1.0)
    normals = np.column_stack((-tangents[:, 1] / safe_norms, tangents[:, 0] / safe_norms))
    half_widths = np.asarray(
        [_meteor_brightness_profile(point[2]) * maximum_width_px * 0.5 for point in run],
        dtype=np.float64,
    )
    left = coordinates + normals * half_widths[:, None]
    right = coordinates - normals * half_widths[:, None]

    path = QPainterPath(QPointF(float(left[0, 0]), float(left[0, 1])))
    for point in left[1:]:
        path.lineTo(float(point[0]), float(point[1]))
    for point in right[::-1]:
        path.lineTo(float(point[0]), float(point[1]))
    path.closeSubpath()
    return path


def _meteor_gradient(
    run: tuple[tuple[float, float, float, bool], ...],
    base_color: QColor,
    maximum_alpha: int,
) -> QLinearGradient:
    """生成连续透明度渐变，避免逐段圆头画笔产生亮斑。"""

    start = QPointF(run[0][0], run[0][1])
    end = QPointF(run[-1][0], run[-1][1])
    if abs(start.x() - end.x()) + abs(start.y() - end.y()) <= 1e-6:
        end = QPointF(start.x() + 1.0, start.y())
    gradient = QLinearGradient(start, end)
    start_progress = float(run[0][2])
    end_progress = float(run[-1][2])
    progress_span = max(end_progress - start_progress, 1e-9)
    stop_progresses = [start_progress]
    if start_progress < 0.75 < end_progress:
        stop_progresses.append(0.75)
    stop_progresses.append(end_progress)
    for progress in stop_progresses:
        stop_position = (progress - start_progress) / progress_span
        color = QColor(base_color)
        color.setAlpha(int(round(maximum_alpha * _meteor_brightness_profile(progress))))
        gradient.setColorAt(max(0.0, min(1.0, stop_position)), color)
    return gradient


class StarMapRenderer:
    def __init__(self, ui_config: StarMapUiConfig | None = None) -> None:
        self.ui_config = ui_config or StarMapUiConfig()

    def star_marker_radius(
        self,
        base_radius_px: float,
        element_scale: float,
        zoom_scale: float,
    ) -> float:
        """应用界面缩放、视图缩放和用户倍率，得到恒星最终渲染半径。"""

        adjusted_base_radius = max(
            0.05,
            float(base_radius_px) - STAR_STYLE_BASE_RADIUS_PX + self.ui_config.base_star_marker_radius_px,
        )
        return (
            adjusted_base_radius
            * float(element_scale)
            * float(zoom_scale)
            * self.ui_config.star_marker_size_multiplier
        )

    def render(
        self,
        star_map: ProjectedStarMap,
        reference_stars: tuple[ReferenceStar, ...] = (),
        element_scale: float = 1.0,
        draw_common_names: bool = True,
        number_reference_stars: bool = True,
        draw_background: bool = True,
        draw_horizon_shadow: bool = True,
        draw_grid: bool = True,
        draw_solar_system_labels: bool = True,
        draw_direction_labels: bool = True,
        star_radius_scale: float = 1.0,
        font_scale: float | None = None,
    ) -> QImage:
        element_scale = max(float(element_scale), 0.05)
        image = QImage(star_map.width, star_map.height, QImage.Format_ARGB32_Premultiplied)
        if draw_background:
            image.fill(QColor(88, 88, 88))
        else:
            image.fill(Qt.transparent)

        painter = QPainter(image)
        self.paint(
            painter,
            star_map,
            reference_stars=reference_stars,
            element_scale=element_scale,
            draw_common_names=draw_common_names,
            number_reference_stars=number_reference_stars,
            draw_background=draw_background,
            draw_horizon_shadow=draw_horizon_shadow,
            draw_grid=draw_grid,
            draw_solar_system_labels=draw_solar_system_labels,
            draw_direction_labels=draw_direction_labels,
            star_radius_scale=star_radius_scale,
            font_scale=font_scale,
        )
        painter.end()
        return image

    def paint(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        reference_stars: tuple[ReferenceStar, ...] = (),
        element_scale: float = 1.0,
        draw_common_names: bool = True,
        number_reference_stars: bool = True,
        draw_background: bool = True,
        draw_horizon_shadow: bool = True,
        draw_grid: bool = True,
        draw_solar_system_labels: bool = True,
        draw_direction_labels: bool = True,
        star_radius_scale: float = 1.0,
        font_scale: float | None = None,
    ) -> None:
        element_scale = max(float(element_scale), 0.05)
        star_radius_scale = max(float(star_radius_scale), 0.05)
        resolved_font_scale = element_scale if font_scale is None else max(float(font_scale), 0.05)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if draw_background:
            self._draw_sky_background(painter, star_map)
        self._draw_milky_way(painter, star_map)
        self.draw_constellations(
            painter,
            star_map,
            element_scale,
            star_radius_scale,
            resolved_font_scale,
        )
        painter.setPen(Qt.NoPen)

        order = star_map.radius_px.argsort()
        for index in order:
            red, green, blue = (int(value) for value in star_map.star_rgb[index])
            alpha = int(star_map.alpha[index])
            color = QColor(red, green, blue, alpha)
            painter.setBrush(color)
            radius = self.star_marker_radius(star_map.radius_px[index], element_scale, star_radius_scale)
            center = QPointF(float(star_map.x_px[index]), float(star_map.y_px[index]))
            painter.drawEllipse(QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0))

        self._draw_solar_system_objects(painter, star_map, element_scale, star_radius_scale)
        if not self.ui_config.meteor_radiant_only:
            self._draw_meteor_showers(painter, star_map)
        if draw_horizon_shadow:
            self._draw_horizon_shadow(painter, star_map)
        if draw_grid:
            self._draw_grid(painter, star_map, element_scale)
        if self.ui_config.meteor_radiant_only:
            self._draw_meteor_radiants(painter, star_map, resolved_font_scale)
        reference_object_ids = {reference_star.star_id for reference_star in reference_stars}
        if reference_stars and number_reference_stars:
            self._draw_reference_stars(painter, star_map, reference_stars, resolved_font_scale)
        elif reference_stars:
            self._draw_reference_star_names(painter, star_map, reference_stars, resolved_font_scale)
        elif draw_common_names:
            self._draw_star_names(painter, star_map, resolved_font_scale)
        if draw_solar_system_labels:
            self._draw_solar_system_labels(
                painter,
                star_map,
                resolved_font_scale,
                excluded_object_ids=reference_object_ids,
            )
        if draw_direction_labels:
            self._draw_direction_labels(painter, star_map, resolved_font_scale)

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

    def draw_constellations(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        element_scale: float,
        star_radius_scale: float,
        font_scale: float | None = None,
    ) -> None:
        """绘制星座线与中文名，并在每个星点圆面外留出间隙。"""

        show_lines = self.ui_config.show_constellation_lines
        show_names = self.ui_config.show_constellation_names
        if not star_map.constellations or not (show_lines or show_names):
            return

        resolved_font_scale = element_scale if font_scale is None else max(float(font_scale), 0.05)

        color = QColor(self.ui_config.constellation_line_color_hex)
        color.setAlphaF(self.ui_config.constellation_line_opacity)
        line_width = self.ui_config.constellation_line_width_px * element_scale
        pen = QPen(color, line_width)
        pen.setCapStyle(Qt.FlatCap)
        if show_lines:
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            node_padding = max(1.25 * element_scale, line_width * 0.75)
            for constellation in star_map.constellations:
                for segment in constellation.segments:
                    start = QPointF(*segment.start)
                    end = QPointF(*segment.end)
                    dx = end.x() - start.x()
                    dy = end.y() - start.y()
                    length = float(np.hypot(dx, dy))
                    start_gap = self.star_marker_radius(
                        segment.start_radius_px,
                        element_scale,
                        star_radius_scale,
                    ) + node_padding
                    end_gap = self.star_marker_radius(
                        segment.end_radius_px,
                        element_scale,
                        star_radius_scale,
                    ) + node_padding
                    if length <= start_gap + end_gap + 0.5:
                        continue
                    unit_x = dx / length
                    unit_y = dy / length
                    painter.drawLine(
                        QPointF(start.x() + unit_x * start_gap, start.y() + unit_y * start_gap),
                        QPointF(end.x() - unit_x * end_gap, end.y() - unit_y * end_gap),
                    )

        if not show_names:
            return

        font = QFont()
        font.setPointSizeF(max(1.0, self.ui_config.constellation_name_font_size_pt * resolved_font_scale))
        font.setBold(True)
        painter.setFont(font)
        for constellation in star_map.constellations:
            label = constellation.chinese_name.strip()
            if not label:
                continue
            point = QPointF(constellation.label_x_px, constellation.label_y_px)
            painter.setPen(QPen(QColor(0, 0, 0, 210), max(1.0, 2.5 * resolved_font_scale)))
            painter.drawText(point, label)
            painter.setPen(QPen(color, max(0.5, 0.8 * resolved_font_scale)))
            painter.drawText(point, label)

    def _draw_solar_system_objects(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        element_scale: float,
        star_radius_scale: float,
    ) -> None:
        if not star_map.solar_system_objects:
            return

        for solar_object in sorted(star_map.solar_system_objects, key=lambda item: item.radius_px):
            red, green, blue = solar_object.color_rgb
            alpha = int(solar_object.alpha)
            marker_scale = element_scale * star_radius_scale
            radius = solar_object.radius_px * marker_scale
            center = QPointF(float(solar_object.sim_x), float(solar_object.sim_y))
            marker_rect = QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)

            painter.setPen(QPen(QColor(0, 0, 0, min(alpha, 210)), max(1.0, 1.6 * marker_scale)))
            painter.setBrush(QColor(red, green, blue, alpha))
            painter.drawEllipse(marker_rect)
            painter.setPen(QPen(QColor(255, 255, 255, min(alpha, 210)), max(0.7, 1.0 * marker_scale)))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(marker_rect)

    def _draw_meteor_showers(self, painter: QPainter, star_map: ProjectedStarMap) -> None:
        """用连续渐变梭形绘制流星，并由画布裁剪完整轨迹。"""

        meteors = projected_meteors(star_map, self.ui_config)
        if not meteors:
            return
        opacity = float(self.ui_config.meteor_opacity)
        thickness_ratio = float(self.ui_config.meteor_thickness_ratio)
        painter.save()
        try:
            painter.setClipRect(QRectF(0.0, 0.0, float(star_map.width), float(star_map.height)), Qt.IntersectClip)
            if star_map.sky_circle_radius_px is not None:
                radius = float(star_map.sky_circle_radius_px)
                circle_clip = QPainterPath()
                circle_clip.addEllipse(
                    QRectF(
                        star_map.width * 0.5 - radius,
                        star_map.height * 0.5 - radius,
                        radius * 2.0,
                        radius * 2.0,
                    )
                )
                painter.setClipPath(circle_clip, Qt.IntersectClip)

            for meteor in meteors:
                base_color = QColor(meteor.color_hex)
                maximum_alpha = max(0, min(255, int(round(255.0 * opacity * meteor.brightness))))
                for run in _valid_meteor_point_runs(meteor.points):
                    run_length = _meteor_run_length_px(run, star_map.width, star_map.height)
                    if run_length <= 0.0:
                        continue
                    gradient = _meteor_gradient(run, base_color, maximum_alpha)
                    if thickness_ratio <= 0.0:
                        center_path = QPainterPath(QPointF(run[0][0], run[0][1]))
                        for point in run[1:]:
                            center_path.lineTo(point[0], point[1])
                        pen = QPen(QBrush(gradient), 0.0)
                        pen.setCapStyle(Qt.FlatCap)
                        pen.setJoinStyle(Qt.RoundJoin)
                        painter.setPen(pen)
                        painter.setBrush(Qt.NoBrush)
                        painter.drawPath(center_path)
                        continue
                    maximum_width = max(0.2, run_length * thickness_ratio)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(gradient))
                    painter.drawPath(_meteor_shape_path(run, maximum_width))
        finally:
            painter.restore()

    def _draw_meteor_radiants(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        font_scale: float,
    ) -> None:
        """用绿色米字符号及名称标注当前可见的流星雨辐射点。"""

        radiants = projected_meteor_radiants(star_map, self.ui_config)
        if not radiants:
            return
        scale = max(float(font_scale), 0.05)
        alpha = max(0, min(255, int(round(255.0 * float(self.ui_config.meteor_opacity)))))
        marker_color = QColor(80, 255, 80, alpha)
        marker_radius = max(5.0, 8.0 * scale)
        marker_pen = QPen(marker_color, max(1.2, 1.8 * scale))
        marker_pen.setCapStyle(Qt.RoundCap)

        font = QFont()
        font.setPointSizeF(self.ui_config.meteor_radiant_label_font_size_pt * scale)
        font.setBold(True)
        painter.save()
        try:
            painter.setClipRect(QRectF(0.0, 0.0, float(star_map.width), float(star_map.height)), Qt.IntersectClip)
            if star_map.sky_circle_radius_px is not None:
                radius = float(star_map.sky_circle_radius_px)
                circle_clip = QPainterPath()
                circle_clip.addEllipse(
                    QRectF(
                        star_map.width * 0.5 - radius,
                        star_map.height * 0.5 - radius,
                        radius * 2.0,
                        radius * 2.0,
                    )
                )
                painter.setClipPath(circle_clip, Qt.IntersectClip)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            edge_padding = max(2.0, 4.0 * scale)
            text_gap = max(3.0, 5.0 * scale)

            for radiant in radiants:
                center = QPointF(radiant.x_px, radiant.y_px)
                painter.setPen(marker_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(
                    QPointF(center.x() - marker_radius, center.y()),
                    QPointF(center.x() + marker_radius, center.y()),
                )
                painter.drawLine(
                    QPointF(center.x(), center.y() - marker_radius),
                    QPointF(center.x(), center.y() + marker_radius),
                )
                diagonal = marker_radius * 0.72
                painter.drawLine(
                    QPointF(center.x() - diagonal, center.y() - diagonal),
                    QPointF(center.x() + diagonal, center.y() + diagonal),
                )
                painter.drawLine(
                    QPointF(center.x() - diagonal, center.y() + diagonal),
                    QPointF(center.x() + diagonal, center.y() - diagonal),
                )

                text_width = metrics.horizontalAdvance(radiant.label)
                text_x = center.x() + marker_radius + text_gap
                if text_x + text_width > star_map.width - edge_padding:
                    text_x = center.x() - marker_radius - text_gap - text_width
                text_x = max(edge_padding, min(text_x, star_map.width - text_width - edge_padding))
                text_y = max(
                    metrics.ascent() + edge_padding,
                    min(center.y() - marker_radius - text_gap, star_map.height - metrics.descent() - edge_padding),
                )
                text_point = QPointF(text_x, text_y)
                painter.setPen(QPen(QColor(0, 0, 0, min(alpha, 220)), max(1.0, 2.5 * scale)))
                painter.drawText(text_point, radiant.label)
                painter.setPen(QPen(marker_color, max(0.7, 0.9 * scale)))
                painter.drawText(text_point, radiant.label)
        finally:
            painter.restore()

    def _draw_grid(self, painter: QPainter, star_map: ProjectedStarMap, element_scale: float) -> None:
        painter.setBrush(Qt.NoBrush)
        for line in star_map.grid_lines:
            if len(line.points) < 2:
                continue
            if line.kind == "horizon":
                pen = QPen(QColor(80, 255, 80, 235), 2.8 * element_scale)
            elif line.kind == "azimuth":
                pen = QPen(QColor(150, 170, 190, 95), 1.0 * element_scale)
                pen.setStyle(Qt.DashLine)
            else:
                pen = QPen(QColor(150, 170, 190, 80), 1.0 * element_scale)
            painter.setPen(pen)
            polygon = QPolygonF([QPointF(x_value, y_value) for x_value, y_value in line.points])
            painter.drawPolyline(polygon)

    def _draw_star_names(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        element_scale: float,
    ) -> None:
        font = QFont()
        font.setPointSizeF(self.ui_config.star_name_font_size_pt * element_scale)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        offset_px = 8.0 * element_scale
        edge_padding_px = 4.0 * element_scale

        for index, common_name in enumerate(star_map.common_names):
            name = str(common_name).strip()
            if not name:
                continue

            red, green, blue = (int(value) for value in star_map.star_rgb[index])
            alpha = 235 if bool(star_map.above_horizon[index]) else 165
            text_width = metrics.horizontalAdvance(name)
            text_height = metrics.height()
            x_value = min(
                max(float(star_map.x_px[index]) + offset_px, edge_padding_px),
                star_map.width - text_width - edge_padding_px,
            )
            y_value = min(
                max(float(star_map.y_px[index]) - offset_px, text_height + edge_padding_px),
                star_map.height - edge_padding_px,
            )
            point = QPointF(x_value, y_value)

            painter.setPen(QPen(QColor(0, 0, 0, 220), 3.0 * element_scale))
            painter.drawText(point, name)
            painter.setPen(QPen(QColor(red, green, blue, alpha), 1.0 * element_scale))
            painter.drawText(point, name)

    def _draw_solar_system_labels(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        element_scale: float,
        excluded_object_ids: set[str],
    ) -> None:
        if not star_map.solar_system_objects:
            return

        font = QFont()
        font.setPointSizeF(self.ui_config.star_name_font_size_pt * element_scale)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        offset_px = 10.0 * element_scale
        edge_padding_px = 4.0 * element_scale

        for solar_object in star_map.solar_system_objects:
            if solar_object.object_id in excluded_object_ids:
                continue
            label = solar_object.display_name.strip()
            if not label:
                continue

            red, green, blue = solar_object.color_rgb
            alpha = 245 if solar_object.above_horizon else 165
            text_width = metrics.horizontalAdvance(label)
            text_height = metrics.height()
            x_value = min(
                max(float(solar_object.sim_x) + offset_px, edge_padding_px),
                star_map.width - text_width - edge_padding_px,
            )
            y_value = min(
                max(float(solar_object.sim_y) - offset_px, text_height + edge_padding_px),
                star_map.height - edge_padding_px,
            )
            point = QPointF(x_value, y_value)

            painter.setPen(QPen(QColor(0, 0, 0, 230), 3.0 * element_scale))
            painter.drawText(point, label)
            painter.setPen(QPen(QColor(red, green, blue, alpha), 1.0 * element_scale))
            painter.drawText(point, label)

    def _draw_reference_star_names(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        reference_stars: tuple[ReferenceStar, ...],
        element_scale: float,
    ) -> None:
        font = QFont()
        font.setPointSizeF(self.ui_config.star_name_font_size_pt * element_scale)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        offset_px = 8.0 * element_scale
        edge_padding_px = 4.0 * element_scale

        for reference_star in reference_stars:
            name = reference_star.common_name.strip() or reference_star.name.strip()
            if not name:
                continue

            text_width = metrics.horizontalAdvance(name)
            text_height = metrics.height()
            x_value = min(
                max(float(reference_star.sim_x) + offset_px, edge_padding_px),
                star_map.width - text_width - edge_padding_px,
            )
            y_value = min(
                max(float(reference_star.sim_y) - offset_px, text_height + edge_padding_px),
                star_map.height - edge_padding_px,
            )
            point = QPointF(x_value, y_value)

            painter.setPen(QPen(QColor(0, 0, 0, 220), 3.0 * element_scale))
            painter.drawText(point, name)
            painter.setPen(QPen(QColor(255, 240, 130, 245), 1.0 * element_scale))
            painter.drawText(point, name)

    def _draw_reference_stars(
        self,
        painter: QPainter,
        star_map: ProjectedStarMap,
        reference_stars: tuple[ReferenceStar, ...],
        element_scale: float,
    ) -> None:
        if not reference_stars:
            return

        number_font = QFont()
        number_font.setPointSizeF(self.ui_config.reference_label_font_size_pt * element_scale)
        number_font.setBold(True)
        edge_padding_px = 4.0 * element_scale
        label_padding_x_px = 14.0 * element_scale
        label_padding_y_px = 8.0 * element_scale

        for reference_star in reference_stars:
            x_value = float(reference_star.sim_x)
            y_value = float(reference_star.sim_y)
            marker_radius = 17.0 * element_scale
            marker_rect = QRectF(
                x_value - marker_radius,
                y_value - marker_radius,
                marker_radius * 2.0,
                marker_radius * 2.0,
            )

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 230), 5.0 * element_scale))
            painter.drawEllipse(marker_rect)
            painter.setPen(QPen(QColor(255, 230, 80, 255), 2.4 * element_scale))
            painter.drawEllipse(marker_rect)

            painter.setFont(number_font)
            metrics = painter.fontMetrics()
            label_text = f"{_reference_star_index_text(reference_star)}. {reference_star.name}"
            max_label_width = max(40.0 * element_scale, star_map.width - edge_padding_px * 2.0)
            visible_label_text = metrics.elidedText(label_text, Qt.ElideRight, int(max_label_width - label_padding_x_px))
            label_width = min(metrics.horizontalAdvance(visible_label_text) + label_padding_x_px, max_label_width)
            label_height = metrics.height() + label_padding_y_px
            preferred_x = x_value + marker_radius + 8.0 * element_scale
            if preferred_x + label_width > star_map.width - edge_padding_px:
                preferred_x = x_value - marker_radius - label_width - 8.0 * element_scale
            label_x = min(
                max(preferred_x, edge_padding_px),
                max(edge_padding_px, star_map.width - label_width - edge_padding_px),
            )
            label_y = min(
                max(y_value - label_height / 2.0, edge_padding_px),
                max(edge_padding_px, star_map.height - label_height - edge_padding_px),
            )
            label_rect = QRectF(label_x, label_y, label_width, label_height)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(label_rect, 4.0 * element_scale, 4.0 * element_scale)
            painter.setPen(QPen(QColor(255, 240, 130, 255), 1.0 * element_scale))
            painter.drawText(label_rect, Qt.AlignCenter, visible_label_text)

    def _draw_direction_labels(self, painter: QPainter, star_map: ProjectedStarMap, element_scale: float) -> None:
        if not star_map.direction_labels:
            return

        font = QFont()
        font.setPointSizeF(self.ui_config.direction_label_font_size_pt * element_scale)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        padding_x_px = 12.0 * element_scale
        padding_y_px = 8.0 * element_scale
        edge_padding_px = 4.0 * element_scale

        for label in star_map.direction_labels:
            text_rect = metrics.boundingRect(label.text)
            width = text_rect.width() + padding_x_px
            height = text_rect.height() + padding_y_px
            x_value = min(max(label.x_px - width / 2.0, edge_padding_px), star_map.width - width - edge_padding_px)
            y_value = min(max(label.y_px - height / 2.0, edge_padding_px), star_map.height - height - edge_padding_px)
            rect = QRectF(x_value, y_value, width, height)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(rect, 4.0 * element_scale, 4.0 * element_scale)
            painter.setPen(QPen(QColor(245, 245, 230, 230), 1.0 * element_scale))
            painter.drawText(rect, Qt.AlignCenter, label.text)
