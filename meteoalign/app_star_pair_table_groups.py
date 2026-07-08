from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem

from .app_constants import (
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_MODES,
    AUTO_MATCH_CONSTRAINT_SOFT,
    STAR_PAIR_AUTO_GROUP_ROLE,
    STAR_PAIR_CONSTRAINT_MODE_ROLE,
    STAR_PAIR_FIT_ROLE,
    STAR_PAIR_FIT_WEIGHT_ROLE,
    STAR_PAIR_INDEX_COLUMN,
    STAR_PAIR_MANUAL_GROUP_LABEL,
    STAR_PAIR_NAME_COLUMN,
    STAR_PAIR_POSITION_COLUMN,
    STAR_PAIR_POSITION_ROLE,
    STAR_PAIR_RESIDUAL_COLUMN,
    STAR_PAIR_RESIDUAL_WIDTH_SAMPLE,
    STAR_PAIR_ROW_TYPE_AUTO_GROUP,
    STAR_PAIR_ROW_TYPE_AUTO_MATCH,
    STAR_PAIR_ROW_TYPE_MANUAL,
    STAR_PAIR_ROW_TYPE_MANUAL_GROUP,
    STAR_PAIR_ROW_TYPE_ROLE,
    STAR_PAIR_SORTABLE_COLUMNS,
    STAR_PAIR_SORT_KEY_INDEX,
    STAR_PAIR_SORT_KEY_RESIDUAL,
)
from .simulator import ReferenceStar
from .star_fitting import FittedStarPosition

class StarPairTableGroupsMixin:
    """星对表格列、分组、排序和约束显示。"""

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
