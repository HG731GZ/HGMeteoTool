from __future__ import annotations

from .app_star_pair_table_common import *  # noqa: F401, F403

class StarPairAnnotationsMixin:
    """星对图像标注、视图聚焦和拾取光标。"""

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

    # ---- 标注管理 ----

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
        visible = self._show_real_image_annotations()
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
        auto_index_by_group: dict[str, int] = {}
        signals_were_blocked = table.blockSignals(True)
        for row in range(table.rowCount()):
            if self._is_manual_match_group_row(row):
                index_item = table.item(row, STAR_PAIR_INDEX_COLUMN)
                if index_item is not None:
                    index_item.setText("▼" if self._manual_match_group_expanded else "▶")
                continue
            if self._is_auto_match_group_row(row):
                group_id = self._row_auto_match_group_id(row)
                index_item = table.item(row, STAR_PAIR_INDEX_COLUMN)
                if index_item is not None:
                    expanded = self._auto_match_group_expanded_by_id.get(group_id, True)
                    index_item.setText("▼" if expanded else "▶")
                continue

            star_id = self._star_pair_star_id(row)
            reference_star = star_lookup.get(star_id)
            index_item = table.item(row, STAR_PAIR_INDEX_COLUMN)
            if index_item is None:
                index_item = self._read_only_table_item("")
                table.setItem(row, STAR_PAIR_INDEX_COLUMN, index_item)
            if self._is_auto_match_row(row):
                group_id = self._row_auto_match_group_id(row) or self._auto_match_group_by_star_id.get(star_id, "A")
                auto_index = auto_index_by_group.get(group_id, 1)
                index_text = f"{group_id}{auto_index}"
                index_item.setText(index_text)
                auto_index_by_group[group_id] = auto_index + 1
            else:
                index_text = str(regular_index)
                index_item.setText(index_text)
                regular_index += 1
            if reference_star is not None:
                renumbered_stars.append(
                    self._reference_star_with_index(reference_star, len(renumbered_stars) + 1, index_text)
                )
        table.blockSignals(signals_were_blocked)

        self._current_reference_stars = tuple(renumbered_stars)
        self._update_auto_match_group_row_text()
        self._sync_star_pair_annotations_to_table()
        self._refresh_star_pair_table_styles()

    def _restore_star_pair_annotations_from_table(self) -> None:
        if self.current_image_preview is None:
            return
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            if self._is_star_pair_group_row(row):
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

        # 真实星点位置一旦确认，就用黄色配对标注替代临时的蓝色聚焦提示。
        self._clear_focused_star_annotations()
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

    # ---- 聚焦与右键菜单 ----

    def _create_focus_annotation_items(
        self,
        scene: QGraphicsScene,
        point: QPointF,
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

        for item in (shadow_item, ellipse_item):
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

    def _set_reference_real_sync_checked(self) -> None:
        self._update_reference_alignment_controls()
        if self.ui.checkBoxSyncReferenceAndRealView.isEnabled() and not self.ui.checkBoxSyncReferenceAndRealView.isChecked():
            self.ui.checkBoxSyncReferenceAndRealView.setChecked(True)

    def _focus_star_pair_image_point(self, row: int, image_x: float, image_y: float) -> None:
        if self._active_star_pair_row is not None:
            self._leave_star_pick_mode()
        self._clear_focused_star_annotations()
        self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        focus_point = QPointF(float(image_x), float(image_y))

        self._set_reference_real_sync_checked()
        self._update_reference_alignment_display()
        self._focus_reference_real_views_on_point(focus_point)
        self._create_focus_annotation_items(self.reference_scene, focus_point)
        self._create_focus_annotation_items(self.real_image_scene, focus_point)
        self.ui.tableWidgetStarPairs.selectRow(row)

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

        self._focus_star_pair_image_point(row, predicted_x, predicted_y)
        self.ui.statusbar.showMessage(
            f"已聚焦理论位置: x={predicted_x:.2f}, y={predicted_y:.2f}。切换标注选项可重置蓝圈。"
        )

