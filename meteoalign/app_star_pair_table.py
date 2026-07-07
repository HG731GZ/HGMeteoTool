from __future__ import annotations
from .app_constants import *

import math

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTableWidgetItem,
)

from .config import StarMapUiConfig
from .star_fitting import FittedStarPosition

from .app_constants import (
    STAR_PAIR_INDEX_COLUMN,
    STAR_PAIR_NAME_COLUMN,
    STAR_PAIR_POSITION_COLUMN,
    STAR_PAIR_RESIDUAL_COLUMN,
    STAR_PAIR_RESIDUAL_WIDTH_SAMPLE,
    STAR_PAIR_SORT_KEY_INDEX,
    STAR_PAIR_SORT_KEY_RESIDUAL,
    STAR_PAIR_SORTABLE_COLUMNS,
    STAR_PAIR_ROW_TYPE_ROLE,
    STAR_PAIR_FIT_ROLE,
    STAR_PAIR_CONSTRAINT_MODE_ROLE,
    STAR_PAIR_FIT_WEIGHT_ROLE,
    STAR_PAIR_POSITION_ROLE,
    STAR_PAIR_AUTO_GROUP_ROLE,
    STAR_PAIR_ROW_TYPE_MANUAL,
    STAR_PAIR_ROW_TYPE_MANUAL_GROUP,
    STAR_PAIR_ROW_TYPE_AUTO_GROUP,
    STAR_PAIR_ROW_TYPE_AUTO_MATCH,
    STAR_PAIR_MANUAL_GROUP_LABEL,
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_CONSTRAINT_MODES,
    STAR_PAIR_FOCUS_MIN_MATCHED_COUNT,
    STAR_PAIR_FOCUS_ZOOM_FIT_SCALE,
    STAR_PAIR_FOCUS_MARKER_RADIUS_PX,
    STAR_PICK_CIRCLE_STEP_PX,
    MIN_PSF_RADIUS_PX,
    STAR_ANNOTATION_PSF_SIGMA_SCALE,
    STAR_ANNOTATION_MIN_RADIUS_PX,
    STAR_ANNOTATION_FALLBACK_RADIUS_PX,
    STAR_ANNOTATION_MAX_RADIUS_PX,
    REFERENCE_STAR_PICK_SCREEN_RADIUS_PX,
)


class StarPairTableMixin:
    """星对表格管理 Mixin：表格列、分组、排序、约束、标注、右键菜单、删除。"""

    # These attributes are provided by MainWindow
    ui: object
    ui_config: StarMapUiConfig
    _current_reference_stars: tuple
    _active_star_pair_row: int | None
    _star_pick_cursor: QCursor | None
    _star_pick_circle_diameter_px: int
    _star_pick_previous_drag_mode: object
    _star_pair_annotations: dict
    _focused_star_annotations: list
    _manual_match_group_expanded: bool
    _auto_match_reference_star_ids: list
    _auto_match_constraint_by_star_id: dict
    _auto_match_group_order: list
    _auto_match_group_by_star_id: dict
    _auto_match_group_expanded_by_id: dict
    _auto_match_next_group_index: int
    _star_pair_sort_key: str | None
    _star_pair_sort_descending: bool
    _sky_alignment_transform: SkyAlignmentTransform | None
    _sky_alignment_error_message: str
    _reference_alignment_error_message: str
    _current_star_map: object | None
    _manual_reference_star_ids: list
    _excluded_reference_star_ids: list
    current_image_preview: object | None
    real_image_scene: object
    reference_scene: object
    _syncing_reference_real_views: bool
    _show_real_image_annotations: object  # method
    _star_pair_alignment_residual: object  # method
    _residual_warning_thresholds: object  # method
    _update_reference_alignment_transform: object  # method
    _update_reference_alignment_controls: object  # method
    _update_reference_alignment_display: object  # method
    _graphics_view_current_scale: object  # method
    _graphics_view_fit_scale: object  # method
    _graphics_view_max_scale: object  # method
    _cap_graphics_view_to_max_scale: object  # method
    _update_live_star_map_zoom_scale: object  # method
    _reference_star_lookup: object  # method
    _reference_star_for_row: object  # method
    _refresh_reference_stars_from_current_map: object  # method
    _reference_star_with_index: object  # method
    _normalized_auto_match_constraint: object  # method
    _update_auto_match_group_row_text: object  # method
    _update_star_pair_table: object  # method
    _auto_pair_star: object  # method
    _refresh_star_pair_mode_cell: object  # method
    _make_star_pair_table_item: object  # method
    _sync_star_pair_annotations_to_table: object  # method
    _refresh_star_pair_table_styles: object  # method
    _restore_star_pair_annotations_from_table: object  # method
    _remove_star_pair_annotation: object  # method

    # ---- 表格列配置 ----

    def _star_pair_residual_column_width(self) -> int:
        table = self.ui.tableWidgetStarPairs
        digit_width = table.fontMetrics().horizontalAdvance(STAR_PAIR_RESIDUAL_WIDTH_SAMPLE)
        header_item = table.horizontalHeaderItem(STAR_PAIR_RESIDUAL_COLUMN)
        header_text = header_item.text() if header_item is not None else "残差"
        header_width = table.horizontalHeader().fontMetrics().horizontalAdvance(header_text)
        return max(digit_width + 18, header_width + 18, 56)

    def _apply_star_pair_table_column_widths(self) -> None:
        self.ui.tableWidgetStarPairs.setColumnWidth(
            STAR_PAIR_RESIDUAL_COLUMN,
            self._star_pair_residual_column_width(),
        )

    def _configure_star_pair_table_columns(self) -> None:
        table = self.ui.tableWidgetStarPairs
        header = table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(STAR_PAIR_INDEX_COLUMN, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(STAR_PAIR_NAME_COLUMN, QHeaderView.Stretch)
        header.setSectionResizeMode(STAR_PAIR_POSITION_COLUMN, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(STAR_PAIR_RESIDUAL_COLUMN, QHeaderView.Fixed)
        self._apply_star_pair_table_column_widths()
        self._update_star_pair_sort_indicator()

    def _update_star_pair_sort_indicator(self) -> None:
        header = self.ui.tableWidgetStarPairs.horizontalHeader()
        column_by_sort_key = {
            STAR_PAIR_SORT_KEY_INDEX: STAR_PAIR_INDEX_COLUMN,
            STAR_PAIR_SORT_KEY_RESIDUAL: STAR_PAIR_RESIDUAL_COLUMN,
        }
        column = column_by_sort_key.get(self._star_pair_sort_key or "")
        if column is None:
            header.setSortIndicatorShown(False)
            return
        # Qt 的表头三角方向在当前样式下与排序直觉相反，这里只反转显示，不改变实际排序逻辑。
        sort_order = Qt.AscendingOrder if self._star_pair_sort_descending else Qt.DescendingOrder
        header.setSortIndicator(column, sort_order)
        header.setSortIndicatorShown(True)

    def _handle_star_pair_header_clicked(self, column: int) -> None:
        sort_key = STAR_PAIR_SORTABLE_COLUMNS.get(column)
        if sort_key is None:
            return
        if self._star_pair_sort_key == sort_key:
            self._star_pair_sort_descending = not self._star_pair_sort_descending
        else:
            self._star_pair_sort_key = sort_key
            self._star_pair_sort_descending = True
        self._update_star_pair_sort_indicator()
        if self._current_reference_stars:
            self._update_star_pair_table(self._current_reference_stars)

        header_item = self.ui.tableWidgetStarPairs.horizontalHeaderItem(column)
        header_text = header_item.text() if header_item is not None else ""
        order_text = "从大到小" if self._star_pair_sort_descending else "从小到大"
        self.ui.statusbar.showMessage(f"已按\"{header_text}\"{order_text}排序；自动匹配组保持不变。")

    # ---- 行类型判断 ----

    def _star_pair_row_type(self, row: int) -> str:
        item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_INDEX_COLUMN)
        if item is None:
            return STAR_PAIR_ROW_TYPE_MANUAL
        row_type = str(item.data(STAR_PAIR_ROW_TYPE_ROLE) or STAR_PAIR_ROW_TYPE_MANUAL)
        if row_type not in {
            STAR_PAIR_ROW_TYPE_MANUAL,
            STAR_PAIR_ROW_TYPE_MANUAL_GROUP,
            STAR_PAIR_ROW_TYPE_AUTO_GROUP,
            STAR_PAIR_ROW_TYPE_AUTO_MATCH,
        }:
            return STAR_PAIR_ROW_TYPE_MANUAL
        return row_type

    def _is_manual_match_group_row(self, row: int) -> bool:
        return self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_MANUAL_GROUP

    def _is_auto_match_group_row(self, row: int) -> bool:
        return self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_AUTO_GROUP

    def _is_star_pair_group_row(self, row: int) -> bool:
        return self._star_pair_row_type(row) in {
            STAR_PAIR_ROW_TYPE_MANUAL_GROUP,
            STAR_PAIR_ROW_TYPE_AUTO_GROUP,
        }

    def _is_auto_match_row(self, row: int) -> bool:
        return self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_AUTO_MATCH

    def _row_auto_match_group_id(self, row: int) -> str:
        item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_INDEX_COLUMN)
        if item is None:
            return ""
        return str(item.data(STAR_PAIR_AUTO_GROUP_ROLE) or "").strip()

    def _auto_match_group_label(self, group_id: str) -> str:
        group_text = str(group_id or "").strip()
        return f"自动{group_text}" if group_text else "自动匹配"

    def _ensure_auto_match_group(self, group_id: str, expanded: bool = True) -> None:
        group_text = str(group_id or "").strip()
        if not group_text:
            return
        if group_text not in self._auto_match_group_order:
            self._auto_match_group_order.append(group_text)
        self._auto_match_group_expanded_by_id.setdefault(group_text, bool(expanded))
        if len(group_text) == 1 and "A" <= group_text <= "Z":
            self._auto_match_next_group_index = max(
                self._auto_match_next_group_index,
                ord(group_text) - ord("A") + 1,
            )

    def _create_auto_match_group(self) -> str:
        group_index = self._auto_match_next_group_index
        group_id = chr(ord("A") + group_index)
        self._auto_match_next_group_index = group_index + 1
        self._ensure_auto_match_group(group_id, expanded=True)
        return group_id

    def _auto_match_group_star_ids(self, group_id: str) -> list[str]:
        return [
            star_id
            for star_id in self._auto_match_reference_star_ids
            if self._auto_match_group_by_star_id.get(star_id) == group_id
        ]

    def _normalize_auto_match_groups(self) -> None:
        if not self._auto_match_reference_star_ids:
            self._auto_match_group_order = []
            self._auto_match_group_by_star_id = {}
            self._auto_match_group_expanded_by_id = {}
            self._auto_match_next_group_index = 0
            return

        group_order: list[str] = []
        for star_id in self._auto_match_reference_star_ids:
            group_id = self._auto_match_group_by_star_id.get(star_id, "").strip()
            if not group_id:
                group_id = "A"
                self._auto_match_group_by_star_id[star_id] = group_id
            if group_id not in group_order:
                group_order.append(group_id)

        # 保留已有显示顺序，并把新发现的旧数据组补在后面。
        ordered_groups = [group_id for group_id in self._auto_match_group_order if group_id in group_order]
        ordered_groups.extend(group_id for group_id in group_order if group_id not in ordered_groups)
        self._auto_match_group_order = ordered_groups
        self._auto_match_group_expanded_by_id = {
            group_id: self._auto_match_group_expanded_by_id.get(group_id, True)
            for group_id in self._auto_match_group_order
        }
        for group_id in self._auto_match_group_order:
            if len(group_id) == 1 and "A" <= group_id <= "Z":
                self._auto_match_next_group_index = max(
                    self._auto_match_next_group_index,
                    ord(group_id) - ord("A") + 1,
                )

    # ---- 约束管理 ----

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
        star_id = self._star_pair_star_id(row)
        if star_id and self._is_auto_match_row(row):
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

    def _star_pair_mode_display_text(self, row: int) -> str:
        if self._parse_star_pair_position_text(row) is None:
            return ""
        mode, fit_weight = self._star_pair_fit_constraint(row)
        if mode == AUTO_MATCH_CONSTRAINT_SOFT:
            return f"软约束({fit_weight:.2f})"
        return "锚点"

    def _star_pair_constraint_tooltip(self, row: int) -> str:
        mode, fit_weight = self._star_pair_fit_constraint(row)
        if mode == AUTO_MATCH_CONSTRAINT_SOFT:
            return f"投影拟合：软约束，权重 {fit_weight:.2f}。"
        return "投影拟合：硬锚点。"

    def _refresh_star_pair_mode_cell(self, row: int) -> None:
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount():
            return
        if self._is_star_pair_group_row(row):
            return
        table = self.ui.tableWidgetStarPairs
        position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
        if position_item is None:
            position_item = self._make_star_pair_table_item("", self._star_pair_row_type(row), self._star_pair_star_id(row))
            table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
        position_item.setText(self._star_pair_mode_display_text(row))

    def _set_star_pair_constraint(self, row: int, mode: str, fit_weight: float | None = None) -> None:
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount() or self._is_star_pair_group_row(row):
            return

        normalized_mode, normalized_weight = self._normalized_auto_match_constraint(mode, fit_weight)
        table = self.ui.tableWidgetStarPairs
        star_id = self._star_pair_star_id(row)
        if self._is_auto_match_row(row) and star_id:
            self._auto_match_constraint_by_star_id[star_id] = (normalized_mode, normalized_weight)

        tooltip = (
            f"投影拟合：软约束，权重 {normalized_weight:.2f}。"
            if normalized_mode == AUTO_MATCH_CONSTRAINT_SOFT
            else "投影拟合：硬锚点。"
        )
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is None:
                continue
            item.setData(STAR_PAIR_CONSTRAINT_MODE_ROLE, normalized_mode)
            item.setData(STAR_PAIR_FIT_WEIGHT_ROLE, normalized_weight)
            item.setToolTip(tooltip)
        self._refresh_star_pair_mode_cell(row)

    def _set_star_pair_constraints_for_rows(
        self,
        rows: list[int],
        mode: str,
        fit_weight: float | None = None,
    ) -> int:
        changed_count = 0
        signals_were_blocked = self.ui.tableWidgetStarPairs.blockSignals(True)
        try:
            for row in rows:
                if self._is_star_pair_group_row(row) or self._parse_star_pair_position_text(row) is None:
                    continue
                self._set_star_pair_constraint(row, mode, fit_weight)
                changed_count += 1
        finally:
            self.ui.tableWidgetStarPairs.blockSignals(signals_were_blocked)

        if changed_count > 0:
            self._update_auto_match_group_row_text()
            self._update_reference_alignment_transform()
        return changed_count

    def _set_star_pair_item_row_type(self, item: QTableWidgetItem, row_type: str) -> QTableWidgetItem:
        item.setData(STAR_PAIR_ROW_TYPE_ROLE, row_type)
        return item

    # ---- 拟合有效载荷 ----

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
            position = self._parse_star_pair_position_text(row)
            if star_id and position is not None:
                mode, fit_weight = self._star_pair_fit_constraint(row)
                states[star_id] = {
                    "position": position,
                    "fit_payload": self._star_pair_fit_payload(row),
                    "fit_constraint_mode": mode,
                    "fit_weight": fit_weight,
                }
        return states

    def _collect_star_pair_positions(self) -> dict[str, str]:
        positions: dict[str, str] = {}
        for star_id, state in self._collect_star_pair_states().items():
            position = state.get("position")
            if isinstance(position, (tuple, list)) and len(position) == 2:
                positions[star_id] = f"{float(position[0]):.2f}, {float(position[1]):.2f}"
        return positions

    def _star_pair_position_count(self) -> int:
        return len(self._collect_star_pair_positions())

    # ---- 表格填充 ----

    def _read_only_table_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _manual_match_group_row(self) -> int | None:
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            if self._is_manual_match_group_row(row):
                return row
        return None

    def _manual_match_group_counts(self) -> tuple[int, int]:
        total_count = 0
        paired_count = 0
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            if self._star_pair_row_type(row) != STAR_PAIR_ROW_TYPE_MANUAL:
                continue
            total_count += 1
            if self._star_pair_position_text(row):
                paired_count += 1
        return total_count, paired_count

    def _update_manual_match_group_row_text(self) -> None:
        group_row = self._manual_match_group_row()
        if group_row is None:
            return
        table = self.ui.tableWidgetStarPairs
        total_count, paired_count = self._manual_match_group_counts()
        arrow_text = "▼" if self._manual_match_group_expanded else "▶"
        mode_text = f"已匹配 {paired_count}/{total_count}"
        values = {
            STAR_PAIR_INDEX_COLUMN: arrow_text,
            STAR_PAIR_NAME_COLUMN: STAR_PAIR_MANUAL_GROUP_LABEL,
            STAR_PAIR_POSITION_COLUMN: mode_text,
            STAR_PAIR_RESIDUAL_COLUMN: "",
        }
        signals_were_blocked = table.blockSignals(True)
        for column, text in values.items():
            item = table.item(group_row, column)
            if item is not None:
                item.setText(text)
        table.blockSignals(signals_were_blocked)

    def _auto_match_group_row(self, group_id: str | None = None) -> int | None:
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            if not self._is_auto_match_group_row(row):
                continue
            if group_id is None or self._row_auto_match_group_id(row) == group_id:
                return row
        return None

    def _auto_match_group_counts(self, group_id: str) -> tuple[int, int]:
        total_count = 0
        paired_count = 0
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            if not self._is_auto_match_row(row) or self._row_auto_match_group_id(row) != group_id:
                continue
            total_count += 1
            if self._star_pair_position_text(row):
                paired_count += 1
        return total_count, paired_count

    def _update_auto_match_group_row_text(self, group_id: str | None = None) -> None:
        if group_id is None:
            self._update_manual_match_group_row_text()
            for current_group_id in self._auto_match_group_order:
                self._update_auto_match_group_row_text(current_group_id)
            return
        group_row = self._auto_match_group_row(group_id)
        if group_row is None:
            return
        table = self.ui.tableWidgetStarPairs
        total_count, paired_count = self._auto_match_group_counts(group_id)
        arrow_text = "▼" if self._auto_match_group_expanded_by_id.get(group_id, True) else "▶"
        mode_text = f"已匹配 {paired_count}/{total_count}"
        values = {
            STAR_PAIR_INDEX_COLUMN: arrow_text,
            STAR_PAIR_NAME_COLUMN: self._auto_match_group_label(group_id),
            STAR_PAIR_POSITION_COLUMN: mode_text,
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
        for row in range(table.rowCount()):
            if self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_MANUAL:
                table.setRowHidden(row, not self._manual_match_group_expanded)
            elif self._is_auto_match_row(row):
                group_id = self._row_auto_match_group_id(row)
                table.setRowHidden(row, not self._auto_match_group_expanded_by_id.get(group_id, True))
            else:
                table.setRowHidden(row, False)
        self._update_auto_match_group_row_text()

    def _toggle_manual_match_group(self) -> None:
        if self._manual_match_group_row() is None:
            return
        self._manual_match_group_expanded = not self._manual_match_group_expanded
        self._apply_auto_match_group_visibility()

    def _collapse_manual_match_group(self) -> None:
        if self._manual_match_group_row() is None:
            return
        self._manual_match_group_expanded = False
        self._apply_auto_match_group_visibility()

    def _toggle_auto_match_group(self, group_id: str) -> None:
        if self._auto_match_group_row(group_id) is None:
            return
        self._auto_match_group_expanded_by_id[group_id] = not self._auto_match_group_expanded_by_id.get(group_id, True)
        self._apply_auto_match_group_visibility()

    def _collapse_auto_match_group(self, group_id: str) -> None:
        if not group_id:
            return
        self._auto_match_group_expanded_by_id[group_id] = False
        self._apply_auto_match_group_visibility()

    def _handle_star_pair_cell_clicked(self, row: int, _column: int) -> None:
        if self._is_manual_match_group_row(row):
            self._toggle_manual_match_group()
        elif self._is_auto_match_group_row(row):
            self._toggle_auto_match_group(self._row_auto_match_group_id(row))

    def _handle_star_pair_cell_double_clicked(self, row: int, _column: int) -> None:
        if self._is_star_pair_group_row(row):
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
        group_id: str = "",
    ) -> None:
        table = self.ui.tableWidgetStarPairs
        star_id = star.star_id.strip()
        star_name = star.common_name.strip() or star_id
        saved_state = saved_states.get(star_id, {})
        position_value = saved_state.get("position")
        position: tuple[float, float] | None = None
        if isinstance(position_value, (tuple, list)) and len(position_value) == 2:
            position = (float(position_value[0]), float(position_value[1]))

        index_item = self._make_star_pair_table_item(index_text, row_type, star_id)
        name_item = self._make_star_pair_table_item(star_name, row_type, star_id)
        position_item = self._make_star_pair_table_item("", row_type, star_id)
        fit_payload = saved_state.get("fit_payload")
        position_item.setData(STAR_PAIR_FIT_ROLE, fit_payload if isinstance(fit_payload, dict) else None)
        position_item.setData(STAR_PAIR_POSITION_ROLE, position)
        residual_item = self._make_star_pair_table_item("", row_type, star_id)
        if saved_state:
            mode, fit_weight = self._normalized_auto_match_constraint(
                saved_state.get("fit_constraint_mode"),
                saved_state.get("fit_weight", 1.0),
            )
        elif row_type == STAR_PAIR_ROW_TYPE_AUTO_MATCH:
            mode, fit_weight = self._auto_match_constraint_for_star_id(star_id)
        else:
            mode, fit_weight = AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0
        if row_type == STAR_PAIR_ROW_TYPE_AUTO_MATCH and star_id:
            self._auto_match_constraint_by_star_id[star_id] = (mode, fit_weight)
        constraint_tip = (
            f"投影拟合：软约束，权重 {fit_weight:.2f}。"
            if mode == AUTO_MATCH_CONSTRAINT_SOFT
            else "投影拟合：硬锚点。"
        )
        for item in (index_item, name_item, position_item, residual_item):
            item.setData(STAR_PAIR_CONSTRAINT_MODE_ROLE, mode)
            item.setData(STAR_PAIR_FIT_WEIGHT_ROLE, fit_weight)
            if group_id:
                item.setData(STAR_PAIR_AUTO_GROUP_ROLE, group_id)
            item.setToolTip(constraint_tip)
        if position is not None:
            position_item.setText(f"软约束({fit_weight:.2f})" if mode == AUTO_MATCH_CONSTRAINT_SOFT else "锚点")

        table.setItem(row, STAR_PAIR_INDEX_COLUMN, index_item)
        table.setItem(row, STAR_PAIR_NAME_COLUMN, name_item)
        table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
        table.setItem(row, STAR_PAIR_RESIDUAL_COLUMN, residual_item)

    def _set_auto_match_group_table_row(self, row: int, group_id: str) -> None:
        table = self.ui.tableWidgetStarPairs
        for column, text in (
            (STAR_PAIR_INDEX_COLUMN, "▶"),
            (STAR_PAIR_NAME_COLUMN, self._auto_match_group_label(group_id)),
            (STAR_PAIR_POSITION_COLUMN, ""),
            (STAR_PAIR_RESIDUAL_COLUMN, ""),
        ):
            item = self._make_star_pair_table_item(text, STAR_PAIR_ROW_TYPE_AUTO_GROUP)
            item.setData(STAR_PAIR_AUTO_GROUP_ROLE, group_id)
            font = QFont(item.font())
            font.setBold(True)
            item.setFont(font)
            table.setItem(row, column, item)

    def _set_manual_match_group_table_row(self, row: int) -> None:
        table = self.ui.tableWidgetStarPairs
        for column, text in (
            (STAR_PAIR_INDEX_COLUMN, "▶"),
            (STAR_PAIR_NAME_COLUMN, STAR_PAIR_MANUAL_GROUP_LABEL),
            (STAR_PAIR_POSITION_COLUMN, ""),
            (STAR_PAIR_RESIDUAL_COLUMN, ""),
        ):
            item = self._make_star_pair_table_item(text, STAR_PAIR_ROW_TYPE_MANUAL_GROUP)
            font = QFont(item.font())
            font.setBold(True)
            item.setFont(font)
            table.setItem(row, column, item)

    def _saved_star_pair_position(self, saved_state: dict[str, object]) -> tuple[float, float] | None:
        position_value = saved_state.get("position")
        if not isinstance(position_value, (tuple, list)) or len(position_value) != 2:
            return None
        try:
            image_x = float(position_value[0])
            image_y = float(position_value[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(image_x) or not math.isfinite(image_y):
            return None
        return image_x, image_y

    def _star_pair_residual_distance_for_entry(
        self,
        star: ReferenceStar,
        saved_state: dict[str, object],
    ) -> float | None:
        transform = self._sky_alignment_transform
        if transform is None:
            return None
        target_position = self._saved_star_pair_position(saved_state)
        if target_position is None:
            return None
        predicted_x, predicted_y = transform.transform_radec(star.ra_deg, star.dec_deg)
        if not all(math.isfinite(value) for value in (predicted_x, predicted_y)):
            return None
        return float(np.hypot(predicted_x - target_position[0], predicted_y - target_position[1]))

    def _sort_star_pair_entries(
        self,
        entries: list[tuple[int, ReferenceStar, str, str, str]],
        saved_states: dict[str, dict[str, object]],
    ) -> list[tuple[int, ReferenceStar, str, str, str]]:
        if self._star_pair_sort_key not in {STAR_PAIR_SORT_KEY_INDEX, STAR_PAIR_SORT_KEY_RESIDUAL}:
            return entries

        if self._star_pair_sort_key == STAR_PAIR_SORT_KEY_INDEX:
            return sorted(
                entries,
                key=lambda entry: -entry[0] if self._star_pair_sort_descending else entry[0],
            )

        def residual_sort_key(entry: tuple[int, ReferenceStar, str, str, str]) -> tuple[int, float, int]:
            sequence_number, star, _index_text, _row_type, _group_id = entry
            residual = self._star_pair_residual_distance_for_entry(star, saved_states.get(star.star_id.strip(), {}))
            if residual is None:
                return 1, 0.0, sequence_number
            sort_value = -residual if self._star_pair_sort_descending else residual
            return 0, sort_value, sequence_number

        return sorted(entries, key=residual_sort_key)

    def _update_star_pair_table(self, reference_stars: tuple[ReferenceStar, ...]) -> None:
        self._current_reference_stars = tuple(reference_stars)
        table = self.ui.tableWidgetStarPairs
        saved_states = self._collect_star_pair_states()
        self._normalize_auto_match_groups()
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        regular_stars = [star for star in reference_stars if star.star_id.strip() not in auto_match_star_ids]
        regular_entries = [
            (display_index, star, str(display_index), STAR_PAIR_ROW_TYPE_MANUAL, "")
            for display_index, star in enumerate(regular_stars, start=1)
        ]
        regular_entries = self._sort_star_pair_entries(regular_entries, saved_states)
        auto_match_stars_by_group: dict[str, list[ReferenceStar]] = {
            group_id: [] for group_id in self._auto_match_group_order
        }
        for star in reference_stars:
            star_id = star.star_id.strip()
            if star_id not in auto_match_star_ids:
                continue
            group_id = self._auto_match_group_by_star_id.get(star_id, "A")
            auto_match_stars_by_group.setdefault(group_id, []).append(star)

        visible_groups = [
            group_id for group_id in self._auto_match_group_order if auto_match_stars_by_group.get(group_id)
        ]
        row_count = len(regular_stars) + sum(1 + len(auto_match_stars_by_group[group_id]) for group_id in visible_groups)
        if regular_stars:
            row_count += 1

        signals_were_blocked = table.blockSignals(True)
        table.setRowCount(row_count)
        row = 0
        if regular_entries:
            self._set_manual_match_group_table_row(row)
            row += 1
        for _sequence_number, star, index_text, row_type, group_id in regular_entries:
            self._set_star_pair_table_row(
                row,
                star,
                index_text,
                row_type,
                saved_states,
                group_id=group_id,
            )
            row += 1

        for group_id in visible_groups:
            self._set_auto_match_group_table_row(row, group_id)
            row += 1
            auto_match_stars = auto_match_stars_by_group[group_id]
            auto_entries = [
                (auto_index, star, f"{group_id}{auto_index}", STAR_PAIR_ROW_TYPE_AUTO_MATCH, group_id)
                for auto_index, star in enumerate(auto_match_stars, start=1)
            ]
            auto_entries = self._sort_star_pair_entries(auto_entries, saved_states)
            for _sequence_number, star, index_text, row_type, entry_group_id in auto_entries:
                self._set_star_pair_table_row(
                    row,
                    star,
                    index_text,
                    row_type,
                    saved_states,
                    group_id=entry_group_id,
                )
                row += 1
        table.blockSignals(signals_were_blocked)
        table.resizeColumnToContents(STAR_PAIR_INDEX_COLUMN)
        table.resizeColumnToContents(STAR_PAIR_POSITION_COLUMN)
        self._apply_star_pair_table_column_widths()
        self._apply_auto_match_group_visibility()
        self._sync_star_pair_annotations_to_table()
        self._refresh_star_pair_table_styles()
        self._restore_star_pair_annotations_from_table()
        self._update_reference_alignment_transform()

    def _handle_star_pair_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != STAR_PAIR_POSITION_COLUMN:
            return
        if self._is_star_pair_group_row(item.row()):
            return
        signals_were_blocked = self.ui.tableWidgetStarPairs.blockSignals(True)
        item.setData(STAR_PAIR_POSITION_ROLE, None)
        item.setData(STAR_PAIR_FIT_ROLE, None)
        self.ui.tableWidgetStarPairs.blockSignals(signals_were_blocked)
        star_id = self._star_pair_star_id(item.row())
        if star_id and not item.text().strip():
            self._remove_star_pair_annotation(star_id)
        self._refresh_star_pair_row_style(item.row())
        self._update_reference_alignment_transform()

    # ---- 表格数据访问 ----

    def _star_pair_star_id(self, row: int) -> str:
        name_item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_NAME_COLUMN)
        if name_item is None:
            return ""
        return str(name_item.data(Qt.UserRole) or "")

    def _star_pair_position_text(self, row: int) -> str:
        position = self._parse_star_pair_position_text(row)
        if position is None:
            return ""
        return f"{position[0]:.2f}, {position[1]:.2f}"

    def _parse_star_pair_position_text(self, row: int) -> tuple[float, float] | None:
        position_item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_POSITION_COLUMN)
        if position_item is None:
            return None

        position_payload = position_item.data(STAR_PAIR_POSITION_ROLE)
        if isinstance(position_payload, (tuple, list)) and len(position_payload) == 2:
            try:
                image_x = float(position_payload[0])
                image_y = float(position_payload[1])
            except (TypeError, ValueError):
                image_x = float("nan")
                image_y = float("nan")
            if math.isfinite(image_x) and math.isfinite(image_y):
                return image_x, image_y

        fit_payload = self._star_pair_fit_payload(row)
        if fit_payload is not None and "x" in fit_payload and "y" in fit_payload:
            image_x = float(fit_payload["x"])
            image_y = float(fit_payload["y"])
            if math.isfinite(image_x) and math.isfinite(image_y):
                return image_x, image_y

        position_text = position_item.text().strip()
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
        if self._is_manual_match_group_row(row):
            return STAR_PAIR_MANUAL_GROUP_LABEL
        if self._is_auto_match_group_row(row):
            return self._auto_match_group_label(self._row_auto_match_group_id(row))
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
        if self._is_star_pair_group_row(row):
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

        if self._active_star_pair_row == row and not self._is_star_pair_group_row(row):
            background = QColor(255, 242, 153)

        signals_were_blocked = table.blockSignals(True)
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is not None:
                item.setBackground(QBrush(background))
        table.blockSignals(signals_were_blocked)

    # ---- 选星光标与 PSF ----

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

    def _selected_star_pair_rows(self) -> list[int]:
        table = self.ui.tableWidgetStarPairs
        rows = sorted({index.row() for index in table.selectionModel().selectedRows()})
        if not rows:
            rows = sorted({index.row() for index in table.selectedIndexes()})
        if not rows and table.currentRow() >= 0:
            rows = [table.currentRow()]
        return rows

    def _default_soft_constraint_weight(self) -> float:
        return max(0.01, min(1.0, float(self.ui.doubleSpinBoxAutoMatchSoftWeight.value())))

    def _prompt_and_apply_soft_weight(self, rows: list[int]) -> None:
        soft_weights = [
            self._star_pair_fit_constraint(row)[1]
            for row in rows
            if not self._is_star_pair_group_row(row)
            and self._parse_star_pair_position_text(row) is not None
            and self._star_pair_fit_constraint(row)[0] == AUTO_MATCH_CONSTRAINT_SOFT
        ]
        current_weight = soft_weights[0] if len(soft_weights) == 1 else self._default_soft_constraint_weight()
        weight, accepted = QInputDialog.getDouble(
            self,
            "修改软约束权重",
            "权重（0.01-1.00）：",
            current_weight,
            0.01,
            1.0,
            2,
        )
        if not accepted:
            return
        changed_count = self._set_star_pair_constraints_for_rows(rows, AUTO_MATCH_CONSTRAINT_SOFT, weight)
        if changed_count > 0:
            self.ui.statusbar.showMessage(f"已修改 {changed_count} 行为软约束模式，权重 {weight:.2f}。")

    def _show_star_pair_context_menu(self, point: QPoint) -> None:
        table = self.ui.tableWidgetStarPairs
        row = table.rowAt(point.y())
        if row < 0:
            return

        selected_rows = self._selected_star_pair_rows()
        if row not in selected_rows:
            table.selectRow(row)
            selected_rows = [row]

        menu = QMenu(self)
        if self._is_manual_match_group_row(row) and len(selected_rows) == 1:
            toggle_action = menu.addAction("折叠手动匹配表" if self._manual_match_group_expanded else "展开手动匹配表")
            selected_action = menu.exec_(table.viewport().mapToGlobal(point))
            if selected_action is toggle_action:
                self._toggle_manual_match_group()
            return
        if self._is_auto_match_group_row(row) and len(selected_rows) == 1:
            group_id = self._row_auto_match_group_id(row)
            expanded = self._auto_match_group_expanded_by_id.get(group_id, True)
            toggle_action = menu.addAction("折叠自动匹配表" if expanded else "展开自动匹配表")
            delete_group_action = menu.addAction("删除自动匹配表")
            selected_action = menu.exec_(table.viewport().mapToGlobal(point))
            if selected_action is toggle_action:
                self._toggle_auto_match_group(group_id)
            elif selected_action is delete_group_action:
                deleted_count = self._delete_auto_match_group(group_id)
                if deleted_count > 0:
                    self.ui.statusbar.showMessage(f"已删除 {self._auto_match_group_label(group_id)}，共 {deleted_count} 颗星。")
            return

        action_rows = self._rows_expanded_from_groups(selected_rows)
        normal_rows = [
            selected_row
            for selected_row in action_rows
            if 0 <= selected_row < table.rowCount() and not self._is_star_pair_group_row(selected_row)
        ]
        matched_rows = [selected_row for selected_row in normal_rows if self._parse_star_pair_position_text(selected_row) is not None]

        pick_action = None
        if len(normal_rows) == 1:
            pick_action = menu.addAction("点选位置")
            pick_action.setEnabled(self.current_image_preview is not None)
        auto_pair_action = None
        if len(normal_rows) == 1 and not self._star_pair_position_text(normal_rows[0]):
            auto_pair_action = menu.addAction("自动配对")
            auto_pair_action.setEnabled(self._sky_alignment_transform is not None and self.current_image_preview is not None)
        clear_action = None
        if matched_rows:
            clear_action = menu.addAction("清除选中配对" if len(matched_rows) > 1 else "清除配对")

        change_to_soft_action = None
        change_to_anchor_action = None
        change_weight_action = None
        if matched_rows:
            selected_modes = {self._star_pair_fit_constraint(selected_row)[0] for selected_row in matched_rows}
            mode_menu = menu.addMenu("更改匹配模式")
            if selected_modes != {AUTO_MATCH_CONSTRAINT_SOFT}:
                change_to_soft_action = mode_menu.addAction("更改为软约束")
            if AUTO_MATCH_CONSTRAINT_SOFT in selected_modes:
                change_to_anchor_action = mode_menu.addAction("更改为锚点模式")
                change_weight_action = mode_menu.addAction("修改权重")

        clicked_group_id = self._row_auto_match_group_id(row) if self._is_auto_match_row(row) else ""
        collapse_manual_group_action = None
        if (
            0 <= row < table.rowCount()
            and self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_MANUAL
            and self._manual_match_group_expanded
        ):
            collapse_manual_group_action = menu.addAction("折叠手动匹配表")
        collapse_group_action = None
        if clicked_group_id and self._auto_match_group_expanded_by_id.get(clicked_group_id, True):
            collapse_group_action = menu.addAction("折叠自动匹配表")

        delete_action = menu.addAction("删除选中行" if len(normal_rows) > 1 else "删除该行")
        selected_action = menu.exec_(table.viewport().mapToGlobal(point))
        if pick_action is not None and selected_action is pick_action:
            self._enter_star_pick_mode(normal_rows[0])
        elif auto_pair_action is not None and selected_action is auto_pair_action:
            self._auto_pair_star(normal_rows[0])
        elif clear_action is not None and selected_action is clear_action:
            cleared_count = self._clear_star_pair_positions_for_rows(matched_rows)
            if cleared_count > 0:
                self.ui.statusbar.showMessage(f"已清除 {cleared_count} 个星点匹配。")
        elif change_to_soft_action is not None and selected_action is change_to_soft_action:
            weight = self._default_soft_constraint_weight()
            changed_count = self._set_star_pair_constraints_for_rows(matched_rows, AUTO_MATCH_CONSTRAINT_SOFT, weight)
            if changed_count > 0:
                self.ui.statusbar.showMessage(f"已将 {changed_count} 行更改为软约束模式，权重 {weight:.2f}。")
        elif change_to_anchor_action is not None and selected_action is change_to_anchor_action:
            changed_count = self._set_star_pair_constraints_for_rows(matched_rows, AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0)
            if changed_count > 0:
                self.ui.statusbar.showMessage(f"已将 {changed_count} 行更改为锚点模式。")
        elif change_weight_action is not None and selected_action is change_weight_action:
            self._prompt_and_apply_soft_weight(matched_rows)
        elif collapse_manual_group_action is not None and selected_action is collapse_manual_group_action:
            self._collapse_manual_match_group()
        elif collapse_group_action is not None and selected_action is collapse_group_action:
            self._collapse_auto_match_group(clicked_group_id)
        elif selected_action is delete_action:
            deleted_count = self._delete_star_pair_rows(selected_rows)
            if deleted_count > 0:
                self.ui.statusbar.showMessage(f"已删除 {deleted_count} 行参考星，后续序号已重新排列。")

    # ---- 选星模式 ----

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
        position_item.setData(STAR_PAIR_POSITION_ROLE, (float(fitted_position.x), float(fitted_position.y)))
        position_item.setData(STAR_PAIR_FIT_ROLE, self._fit_payload_from_position(fitted_position))
        position_item.setText(self._star_pair_mode_display_text(row))
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
            position_item.setData(STAR_PAIR_POSITION_ROLE, None)
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
                position_item.setData(STAR_PAIR_POSITION_ROLE, None)
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
        star_label = self._star_pair_label(row)
        cleared_count = self._clear_star_pair_positions_for_rows([row])
        if cleared_count > 0:
            self.ui.statusbar.showMessage(f"已清除 {star_label} 的真实图像配对。右键该行可重新点选位置。")

    def _delete_star_pair_row(self, row: int) -> None:
        deleted_count = self._delete_star_pair_rows([row])
        if deleted_count > 0:
            self.ui.statusbar.showMessage(f"已删除 {deleted_count} 行参考星，后续序号已重新排列。")

    # ---- 删除操作 ----

    def _rows_expanded_from_groups(self, rows: list[int]) -> list[int]:
        table = self.ui.tableWidgetStarPairs
        expanded_rows = set(rows)
        manual_group_selected = any(
            0 <= row < table.rowCount() and self._is_manual_match_group_row(row)
            for row in rows
        )
        selected_group_ids = {
            self._row_auto_match_group_id(row)
            for row in rows
            if 0 <= row < table.rowCount() and self._is_auto_match_group_row(row)
        }
        if manual_group_selected:
            for row in range(table.rowCount()):
                if self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_MANUAL:
                    expanded_rows.add(row)
        if selected_group_ids:
            for row in range(table.rowCount()):
                if self._is_auto_match_row(row) and self._row_auto_match_group_id(row) in selected_group_ids:
                    expanded_rows.add(row)
        return sorted(expanded_rows)

    def _clear_star_pair_positions_for_rows(self, rows: list[int]) -> int:
        table = self.ui.tableWidgetStarPairs
        target_rows = [
            row
            for row in self._rows_expanded_from_groups(sorted(set(rows)))
            if 0 <= row < table.rowCount() and not self._is_star_pair_group_row(row)
        ]
        if not target_rows:
            return 0
        if self._active_star_pair_row is not None and self._active_star_pair_row in target_rows:
            self._leave_star_pick_mode()

        cleared_count = 0
        for row in target_rows:
            if self._parse_star_pair_position_text(row) is None:
                continue
            self._clear_star_pair_position_data(row)
            self._refresh_star_pair_mode_cell(row)
            self._refresh_star_pair_row_style(row)
            cleared_count += 1
        if cleared_count > 0:
            self._update_auto_match_group_row_text()
            self._update_reference_alignment_transform()
        return cleared_count

    def _delete_auto_match_group(self, group_id: str) -> int:
        if not group_id:
            return 0
        table = self.ui.tableWidgetStarPairs
        group_rows = [
            row
            for row in range(table.rowCount())
            if self._is_auto_match_group_row(row) and self._row_auto_match_group_id(row) == group_id
        ]
        if not group_rows:
            self.ui.statusbar.showMessage("当前没有可删除的自动匹配表。")
            return 0
        return self._delete_star_pair_rows(group_rows)

    def _delete_star_pair_rows(self, rows: list[int]) -> int:
        table = self.ui.tableWidgetStarPairs
        valid_rows = [row for row in sorted(set(rows)) if 0 <= row < table.rowCount()]
        if not valid_rows:
            return 0
        rows_to_remove = self._rows_expanded_from_groups(valid_rows)

        group_ids_to_delete = {
            self._row_auto_match_group_id(row)
            for row in valid_rows
            if self._is_auto_match_group_row(row)
        }
        manual_group_selected = any(self._is_manual_match_group_row(row) for row in valid_rows)
        auto_star_ids_to_delete: set[str] = set()
        manual_star_ids_to_delete: set[str] = set()
        if manual_group_selected:
            for row in range(table.rowCount()):
                if self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_MANUAL:
                    star_id = self._star_pair_star_id(row)
                    if star_id:
                        manual_star_ids_to_delete.add(star_id)
        for group_id in group_ids_to_delete:
            auto_star_ids_to_delete.update(self._auto_match_group_star_ids(group_id))

        for row in valid_rows:
            if self._is_star_pair_group_row(row):
                continue
            star_id = self._star_pair_star_id(row)
            if not star_id:
                continue
            if self._is_auto_match_row(row):
                auto_star_ids_to_delete.add(star_id)
            else:
                manual_star_ids_to_delete.add(star_id)

        if not auto_star_ids_to_delete and not manual_star_ids_to_delete:
            return 0

        deleted_star_ids = auto_star_ids_to_delete | manual_star_ids_to_delete
        if self._active_star_pair_row is not None:
            active_star_id = self._star_pair_star_id(self._active_star_pair_row)
            if active_star_id in deleted_star_ids:
                self._leave_star_pick_mode()

        for star_id in deleted_star_ids:
            self._remove_star_pair_annotation(star_id)
            if star_id not in self._excluded_reference_star_ids:
                self._excluded_reference_star_ids.append(star_id)

        self._auto_match_reference_star_ids = [
            star_id for star_id in self._auto_match_reference_star_ids if star_id not in auto_star_ids_to_delete
        ]
        for star_id in auto_star_ids_to_delete:
            self._auto_match_constraint_by_star_id.pop(star_id, None)
            self._auto_match_group_by_star_id.pop(star_id, None)

        self._manual_reference_star_ids = [
            star_id for star_id in self._manual_reference_star_ids if star_id not in manual_star_ids_to_delete
        ]

        # 先从当前表格中移除旧行，再重建参考星列表。否则刷新逻辑会把带匹配坐标的已删行当作
        # “需要保留的已匹配星”重新加入，自动匹配行也会因为失去自动分组而落回手动匹配表。
        signals_were_blocked = table.blockSignals(True)
        try:
            for row in sorted(rows_to_remove, reverse=True):
                if 0 <= row < table.rowCount():
                    table.removeRow(row)
        finally:
            table.blockSignals(signals_were_blocked)

        self._normalize_auto_match_groups()
        if self._current_star_map is None:
            self._renumber_star_pair_rows_from_table()
        else:
            self._refresh_reference_stars_from_current_map()
        self._update_reference_alignment_transform()
        deleted_count = len(deleted_star_ids)
        if table.rowCount() > 0:
            table.selectRow(min(valid_rows[0], table.rowCount() - 1))
        return deleted_count

    def _handle_star_pair_delete_key(self) -> bool:
        table = self.ui.tableWidgetStarPairs
        rows = sorted({index.row() for index in table.selectedIndexes()})
        if not rows and table.currentRow() >= 0:
            rows = [table.currentRow()]
        if not rows:
            return False
        rows_with_position = [row for row in rows if self._star_pair_position_text(row)]
        rows_without_position = [row for row in rows if not self._star_pair_position_text(row)]
        cleared_count = self._clear_star_pair_positions_for_rows(rows_with_position)
        deleted_count = self._delete_star_pair_rows(rows_without_position)
        if cleared_count > 0 or deleted_count > 0:
            self.ui.statusbar.showMessage(f"已清除 {cleared_count} 个匹配，删除 {deleted_count} 行参考星。")
        return True
