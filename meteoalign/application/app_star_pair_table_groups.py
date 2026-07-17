from __future__ import annotations

import math
from functools import partial

import numpy as np
from PyQt5.QtCore import QItemSelectionModel, QTimer, Qt
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem

from .app_constants import (
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_MODES,
    AUTO_MATCH_CONSTRAINT_SOFT,
    STAR_PAIR_ANNOTATION_COLUMN,
    STAR_PAIR_AUTO_GROUP_ROLE,
    STAR_PAIR_INDEX_COLUMN,
    STAR_PAIR_INDEX_WIDTH_SAMPLE,
    STAR_PAIR_MANUAL_GROUP_LABEL,
    STAR_PAIR_MODE_WIDTH_SAMPLE,
    STAR_PAIR_NAME_COLUMN,
    STAR_PAIR_POSITION_COLUMN,
    STAR_PAIR_QUALITY_COLUMN,
    STAR_PAIR_QUALITY_WIDTH_SAMPLE,
    STAR_PAIR_RESIDUAL_COLUMN,
    STAR_PAIR_RESIDUAL_WIDTH_SAMPLE,
    STAR_PAIR_ROW_TYPE_AUTO_GROUP,
    STAR_PAIR_ROW_TYPE_AUTO_MATCH,
    STAR_PAIR_ROW_TYPE_MANUAL,
    STAR_PAIR_ROW_TYPE_MANUAL_GROUP,
    STAR_PAIR_ROW_TYPE_ROLE,
    STAR_PAIR_SORTABLE_COLUMNS,
    STAR_PAIR_SORT_KEY_INDEX,
    STAR_PAIR_SORT_KEY_QUALITY,
    STAR_PAIR_SORT_KEY_RESIDUAL,
)
from ..simulator import ReferenceStar
from ..star_fitting import FittedStarPosition

class StarPairTableGroupsMixin:
    """星对表格列、分组、排序和约束显示。"""

    def _star_pair_residual_column_width(self) -> int:
        table = self.ui.tableWidgetStarPairs
        digit_width = table.fontMetrics().horizontalAdvance(STAR_PAIR_RESIDUAL_WIDTH_SAMPLE)
        header_item = table.horizontalHeaderItem(STAR_PAIR_RESIDUAL_COLUMN)
        header_text = header_item.text() if header_item is not None else "残差"
        header_width = table.horizontalHeader().fontMetrics().horizontalAdvance(header_text)
        return max(digit_width + 8, header_width + 8, 56)

    def _star_pair_quality_column_width(self) -> int:
        table = self.ui.tableWidgetStarPairs
        digit_width = table.fontMetrics().horizontalAdvance(STAR_PAIR_QUALITY_WIDTH_SAMPLE)
        header_item = table.horizontalHeaderItem(STAR_PAIR_QUALITY_COLUMN)
        header_text = header_item.text() if header_item is not None else "PSF"
        header_width = table.horizontalHeader().fontMetrics().horizontalAdvance(header_text)
        return max(digit_width + 6, header_width + 6, 44)

    def _star_pair_index_column_width(self) -> int:
        table = self.ui.tableWidgetStarPairs
        sample_width = table.fontMetrics().horizontalAdvance(STAR_PAIR_INDEX_WIDTH_SAMPLE)
        header_item = table.horizontalHeaderItem(STAR_PAIR_INDEX_COLUMN)
        header_text = header_item.text() if header_item is not None else "序号"
        header_width = table.horizontalHeader().fontMetrics().horizontalAdvance(header_text)
        return max(sample_width + 4, header_width + 4, 36)

    def _star_pair_mode_column_width(self) -> int:
        """把模式列限制为约七个英文字符，给标注列腾出空间。"""

        table = self.ui.tableWidgetStarPairs
        sample_width = table.fontMetrics().horizontalAdvance(STAR_PAIR_MODE_WIDTH_SAMPLE)
        header_item = table.horizontalHeaderItem(STAR_PAIR_POSITION_COLUMN)
        header_text = header_item.text() if header_item is not None else "模式"
        header_width = table.horizontalHeader().fontMetrics().horizontalAdvance(header_text)
        return max(sample_width + 4, header_width + 4, 52)

    def _star_pair_annotation_column_width(self) -> int:
        table = self.ui.tableWidgetStarPairs
        header_item = table.horizontalHeaderItem(STAR_PAIR_ANNOTATION_COLUMN)
        header_text = header_item.text() if header_item is not None else "标注"
        header_width = table.horizontalHeader().fontMetrics().horizontalAdvance(header_text)
        return max(header_width + 4, 38)

    def _apply_star_pair_table_column_widths(self) -> None:
        self.ui.tableWidgetStarPairs.setColumnWidth(
            STAR_PAIR_INDEX_COLUMN,
            self._star_pair_index_column_width(),
        )
        self.ui.tableWidgetStarPairs.setColumnWidth(
            STAR_PAIR_POSITION_COLUMN,
            self._star_pair_mode_column_width(),
        )
        self.ui.tableWidgetStarPairs.setColumnWidth(
            STAR_PAIR_QUALITY_COLUMN,
            self._star_pair_quality_column_width(),
        )
        self.ui.tableWidgetStarPairs.setColumnWidth(
            STAR_PAIR_RESIDUAL_COLUMN,
            self._star_pair_residual_column_width(),
        )
        self.ui.tableWidgetStarPairs.setColumnWidth(
            STAR_PAIR_ANNOTATION_COLUMN,
            self._star_pair_annotation_column_width(),
        )

    def _configure_star_pair_table_columns(self) -> None:
        table = self.ui.tableWidgetStarPairs
        header = table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(STAR_PAIR_INDEX_COLUMN, QHeaderView.Fixed)
        header.setSectionResizeMode(STAR_PAIR_NAME_COLUMN, QHeaderView.Stretch)
        header.setSectionResizeMode(STAR_PAIR_POSITION_COLUMN, QHeaderView.Fixed)
        header.setSectionResizeMode(STAR_PAIR_QUALITY_COLUMN, QHeaderView.Fixed)
        header.setSectionResizeMode(STAR_PAIR_RESIDUAL_COLUMN, QHeaderView.Fixed)
        header.setSectionResizeMode(STAR_PAIR_ANNOTATION_COLUMN, QHeaderView.Fixed)
        self._apply_star_pair_table_column_widths()
        self._update_star_pair_sort_indicator()

    def _update_star_pair_sort_indicator(self) -> None:
        header = self.ui.tableWidgetStarPairs.horizontalHeader()
        column_by_sort_key = {
            STAR_PAIR_SORT_KEY_INDEX: STAR_PAIR_INDEX_COLUMN,
            STAR_PAIR_SORT_KEY_QUALITY: STAR_PAIR_QUALITY_COLUMN,
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
            if self._auto_match_group_id_for_star_id(star_id) == group_id
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
            group_id = self._auto_match_group_id_for_star_id(star_id)
            if not group_id:
                group_id = "A"
                if self._star_pair_record_for_star_id(star_id) is None:
                    self._auto_match_group_by_star_id[star_id] = group_id
            elif self._star_pair_record_for_star_id(star_id) is not None:
                self._auto_match_group_by_star_id.pop(star_id, None)
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

    def _star_pair_record_for_star_id(self, star_id: str):
        store = getattr(self, "_star_pair_store", None)
        if store is None or not star_id:
            return None
        return store.get(star_id)

    def _star_pair_record_for_row(self, row: int):
        return self._star_pair_record_for_star_id(self._star_pair_star_id(row))

    def _auto_match_group_id_for_star_id(self, star_id: str) -> str:
        record = self._star_pair_record_for_star_id(star_id)
        if record is not None and record.group_id:
            return str(record.group_id).strip()
        return str(self._auto_match_group_by_star_id.get(star_id, "")).strip()

    def _auto_match_constraint_for_star_id(self, star_id: str) -> tuple[str, float]:
        record = self._star_pair_record_for_star_id(star_id)
        if record is not None:
            mode = record.fit_constraint_mode
            weight = record.fit_weight
        elif hasattr(self, "_auto_match_constraint_mode") and hasattr(self, "_auto_match_soft_weight"):
            mode = self._auto_match_constraint_mode()
            weight = self._auto_match_soft_weight()
        else:
            mode, weight = AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0
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
        record = self._star_pair_record_for_row(row)
        if record is None:
            return AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0
        mode = str(record.fit_constraint_mode or AUTO_MATCH_CONSTRAINT_ANCHOR)
        weight = float(record.fit_weight)
        if mode not in AUTO_MATCH_CONSTRAINT_MODES:
            mode = AUTO_MATCH_CONSTRAINT_ANCHOR
        return (mode, max(0.01, min(1.0, weight))) if mode == AUTO_MATCH_CONSTRAINT_SOFT else (mode, 1.0)

    def _star_pair_mode_display_text(self, row: int) -> str:
        if self._parse_star_pair_position_text(row) is None:
            return ""
        mode, fit_weight = self._star_pair_fit_constraint(row)
        if mode == AUTO_MATCH_CONSTRAINT_SOFT:
            return f"({fit_weight:.2f})"
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

    def _star_pair_quality_score(self, row: int) -> float | None:
        record = self._star_pair_record_for_row(row)
        if record is None or record.psf is None:
            return None
        quality_score = float(record.psf.quality_score)
        return quality_score if math.isfinite(quality_score) else None

    def _refresh_star_pair_quality_cell(self, row: int) -> None:
        """用当前 PSF 记录刷新质量单元格；未拟合时保持空白。"""

        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount():
            return
        if self._is_star_pair_group_row(row):
            return
        table = self.ui.tableWidgetStarPairs
        quality_item = table.item(row, STAR_PAIR_QUALITY_COLUMN)
        if quality_item is None:
            quality_item = self._make_star_pair_table_item(
                "",
                self._star_pair_row_type(row),
                self._star_pair_star_id(row),
            )
            table.setItem(row, STAR_PAIR_QUALITY_COLUMN, quality_item)
        quality_score = self._star_pair_quality_score(row)
        quality_item.setText("" if quality_score is None else f"{quality_score:.2f}")

    def _set_star_pair_constraint(self, row: int, mode: str, fit_weight: float | None = None) -> None:
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount() or self._is_star_pair_group_row(row):
            return

        normalized_mode, normalized_weight = self._normalized_auto_match_constraint(mode, fit_weight)
        table = self.ui.tableWidgetStarPairs
        star_id = self._star_pair_star_id(row)

        store = getattr(self, "_star_pair_store", None)
        if store is not None and star_id:
            store.set_constraint(star_id, normalized_mode, normalized_weight)

        tooltip = (
            f"投影拟合：软约束，权重 {normalized_weight:.2f}。"
            if normalized_mode == AUTO_MATCH_CONSTRAINT_SOFT
            else "投影拟合：硬锚点。"
        )
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is None:
                continue
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

    def _star_pair_fit_payload(self, row: int) -> dict[str, float | bool] | None:
        record = self._star_pair_record_for_row(row)
        if record is None or record.psf is None:
            return None
        payload = record.psf.to_table_payload()

        fit_payload: dict[str, float | bool] = {}
        for key in (
            "x",
            "y",
            "amplitude",
            "background",
            "sigma_x",
            "sigma_y",
            "theta_rad",
            "fwhm_x",
            "fwhm_y",
            "snr",
            "fit_error",
            "saturation_fraction",
            "quality_score",
        ):
            try:
                value = float(payload[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                fit_payload[key] = value
        for key in ("saturated", "blended"):
            if key in payload:
                fit_payload[key] = bool(payload[key])
        return fit_payload or None

    def _fit_payload_from_position(self, fitted_position: FittedStarPosition) -> dict[str, float | bool]:
        return {
            "x": float(fitted_position.x),
            "y": float(fitted_position.y),
            "amplitude": float(fitted_position.amplitude),
            "background": float(fitted_position.background),
            "sigma_x": float(fitted_position.sigma_x),
            "sigma_y": float(fitted_position.sigma_y),
            "theta_rad": float(fitted_position.theta_rad),
            "fwhm_x": float(fitted_position.fwhm_x),
            "fwhm_y": float(fitted_position.fwhm_y),
            "snr": float(fitted_position.snr),
            "fit_error": float(fitted_position.fit_error),
            "saturated": bool(fitted_position.saturated),
            "saturation_fraction": float(fitted_position.saturation_fraction),
            "blended": bool(fitted_position.blended),
            "quality_score": float(fitted_position.quality_score),
        }

    def _fitted_position_for_row(self, row: int) -> FittedStarPosition | None:
        record = self._star_pair_record_for_row(row)
        if record is None:
            return None
        return record.fitted_position

    def _star_pair_store_states(self) -> dict[str, dict[str, object]]:
        states: dict[str, dict[str, object]] = {}
        store = getattr(self, "_star_pair_store", None)
        if store is None:
            return states
        for record in store.snapshot():
            star_id = record.star_id
            if not star_id:
                continue
            states[star_id] = {
                "position": record.position,
                "fit_payload": None if record.psf is None else record.psf.to_table_payload(),
                "fit_constraint_mode": record.fit_constraint_mode,
                "fit_weight": record.fit_weight,
                "pair_origin": record.pair_origin,
                "group_id": record.group_id,
                "group_name": record.group_name,
                "residual_px": record.residual_px,
            }
        return states

    def _star_pair_position_texts_from_store(self) -> dict[str, str]:
        positions: dict[str, str] = {}
        store = getattr(self, "_star_pair_store", None)
        if store is None:
            return positions
        for star_id, position in store.positions().items():
            positions[star_id] = f"{float(position[0]):.2f}, {float(position[1]):.2f}"
        return positions

    def _star_pair_position_count(self) -> int:
        return len(self._star_pair_position_texts_from_store())

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

    def _update_manual_match_group_row_text(self) -> None:
        group_row = self._manual_match_group_row()
        if group_row is None:
            return
        table = self.ui.tableWidgetStarPairs
        arrow_text = "▼" if self._manual_match_group_expanded else "▶"
        values = {
            STAR_PAIR_INDEX_COLUMN: arrow_text,
            STAR_PAIR_NAME_COLUMN: STAR_PAIR_MANUAL_GROUP_LABEL,
            STAR_PAIR_POSITION_COLUMN: "",
            STAR_PAIR_QUALITY_COLUMN: "",
            STAR_PAIR_RESIDUAL_COLUMN: "",
            STAR_PAIR_ANNOTATION_COLUMN: "",
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
        arrow_text = "▼" if self._auto_match_group_expanded_by_id.get(group_id, True) else "▶"
        values = {
            STAR_PAIR_INDEX_COLUMN: arrow_text,
            STAR_PAIR_NAME_COLUMN: self._auto_match_group_label(group_id),
            STAR_PAIR_POSITION_COLUMN: "",
            STAR_PAIR_QUALITY_COLUMN: "",
            STAR_PAIR_RESIDUAL_COLUMN: "",
            STAR_PAIR_ANNOTATION_COLUMN: "",
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
        if (
            self.ui_config.double_click_focus_auto_pair_enabled
            and not self._star_pair_position_text(row)
        ):
            self._auto_pair_star(row, silent_failure=True)
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

    def _hidden_star_pair_annotations(self) -> set[str]:
        """返回当前会话中被用户隐藏标注的星表编号集合。"""

        hidden_ids = getattr(self, "_hidden_star_pair_annotation_ids", None)
        if not isinstance(hidden_ids, set):
            hidden_ids = set(hidden_ids or ())
            self._hidden_star_pair_annotation_ids = hidden_ids
        return hidden_ids

    def _star_pair_annotation_is_enabled_for_id(self, star_id: str) -> bool:
        return bool(star_id) and star_id not in self._hidden_star_pair_annotations()

    def _make_star_pair_annotation_item(self, row_type: str, star_id: str) -> QTableWidgetItem:
        item = self._make_star_pair_table_item("", row_type, star_id)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(
            Qt.Checked if self._star_pair_annotation_is_enabled_for_id(star_id) else Qt.Unchecked
        )
        item.setTextAlignment(Qt.AlignCenter)
        item.setToolTip("控制该星在参考星图和真实图像中的标注显示。")
        return item

    def _star_pair_annotation_is_enabled(self, row: int) -> bool:
        if row < 0 or row >= self.ui.tableWidgetStarPairs.rowCount():
            return False
        if self._is_star_pair_group_row(row):
            return False
        item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_ANNOTATION_COLUMN)
        if item is not None and item.flags() & Qt.ItemIsUserCheckable:
            return item.checkState() == Qt.Checked
        return self._star_pair_annotation_is_enabled_for_id(self._star_pair_star_id(row))

    def _selected_star_pair_annotation_rows(self, changed_row: int) -> list[int]:
        table = self.ui.tableWidgetStarPairs
        selected_rows = sorted({index.row() for index in table.selectionModel().selectedRows()})
        if not selected_rows:
            selected_rows = sorted({index.row() for index in table.selectedIndexes()})
        if changed_row not in selected_rows:
            selected_rows = [changed_row]
        return [
            row
            for row in selected_rows
            if 0 <= row < table.rowCount() and not self._is_star_pair_group_row(row)
        ]

    def _restore_star_pair_annotation_selection(
        self,
        star_ids: tuple[str, ...],
        current_star_id: str,
    ) -> None:
        """复选框点击完成后恢复原多选行，避免 Qt 在鼠标释放时收窄选区。"""

        table = self.ui.tableWidgetStarPairs
        try:
            selection_model = table.selectionModel()
            model = table.model()
        except RuntimeError:
            return
        requested_ids = set(star_ids)
        rows_by_star_id = {
            self._star_pair_star_id(row): row
            for row in range(table.rowCount())
            if self._star_pair_star_id(row) in requested_ids
        }
        if len(rows_by_star_id) <= 1:
            return

        selection_model.clearSelection()
        for star_id in star_ids:
            row = rows_by_star_id.get(star_id)
            if row is None:
                continue
            selection_model.select(
                model.index(row, STAR_PAIR_INDEX_COLUMN),
                QItemSelectionModel.Select | QItemSelectionModel.Rows,
            )
        current_row = rows_by_star_id.get(current_star_id)
        if current_row is not None:
            selection_model.setCurrentIndex(
                model.index(current_row, STAR_PAIR_ANNOTATION_COLUMN),
                QItemSelectionModel.NoUpdate,
            )

    def _set_star_pair_annotations_for_rows(self, rows: list[int], visible: bool) -> int:
        """统一更新行级勾选状态，并立即刷新两侧图像标注。"""

        table = self.ui.tableWidgetStarPairs
        hidden_ids = self._hidden_star_pair_annotations()
        changed_count = 0
        signals_were_blocked = table.blockSignals(True)
        try:
            for row in sorted(set(rows)):
                if row < 0 or row >= table.rowCount() or self._is_star_pair_group_row(row):
                    continue
                star_id = self._star_pair_star_id(row)
                if not star_id:
                    continue
                was_visible = star_id not in hidden_ids
                if visible:
                    hidden_ids.discard(star_id)
                else:
                    hidden_ids.add(star_id)
                item = table.item(row, STAR_PAIR_ANNOTATION_COLUMN)
                if item is not None:
                    item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
                if was_visible != visible:
                    changed_count += 1
        finally:
            table.blockSignals(signals_were_blocked)

        update_real_annotations = getattr(self, "_update_star_pair_annotation_visibility", None)
        if callable(update_real_annotations):
            update_real_annotations()
        update_reference_annotations = getattr(self, "_update_reference_alignment_display", None)
        if callable(update_reference_annotations):
            update_reference_annotations()
        return changed_count

    def _star_pair_group_annotation_rows(self, group_row: int) -> list[int]:
        table = self.ui.tableWidgetStarPairs
        if self._is_manual_match_group_row(group_row):
            return [
                row
                for row in range(table.rowCount())
                if self._star_pair_row_type(row) == STAR_PAIR_ROW_TYPE_MANUAL
            ]
        if self._is_auto_match_group_row(group_row):
            group_id = self._row_auto_match_group_id(group_row)
            return [
                row
                for row in range(table.rowCount())
                if self._is_auto_match_row(row) and self._row_auto_match_group_id(row) == group_id
            ]
        return []

    def _star_pair_group_annotations_are_hidden(self, group_row: int) -> bool:
        rows = self._star_pair_group_annotation_rows(group_row)
        return bool(rows) and all(not self._star_pair_annotation_is_enabled(row) for row in rows)

    def _star_pair_group_annotation_action_text(self, group_row: int) -> str:
        if self._star_pair_group_annotations_are_hidden(group_row):
            return "显示列表所有标注"
        return "隐藏列表所有标注"

    def _toggle_star_pair_group_annotations(self, group_row: int) -> int:
        rows = self._star_pair_group_annotation_rows(group_row)
        if not rows:
            return 0
        visible = self._star_pair_group_annotations_are_hidden(group_row)
        self._set_star_pair_annotations_for_rows(rows, visible)
        group_label = self._star_pair_label(group_row)
        state_text = "显示" if visible else "隐藏"
        self.ui.statusbar.showMessage(f"已{state_text}{group_label}的全部 {len(rows)} 个标注。")
        return len(rows)

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
        quality_item = self._make_star_pair_table_item("", row_type, star_id)
        residual_item = self._make_star_pair_table_item("", row_type, star_id)
        annotation_item = self._make_star_pair_annotation_item(row_type, star_id)
        if saved_state:
            mode, fit_weight = self._normalized_auto_match_constraint(
                saved_state.get("fit_constraint_mode"),
                saved_state.get("fit_weight", 1.0),
            )
        elif row_type == STAR_PAIR_ROW_TYPE_AUTO_MATCH:
            mode, fit_weight = self._auto_match_constraint_for_star_id(star_id)
        else:
            mode, fit_weight = AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0
        constraint_tip = (
            f"投影拟合：软约束，权重 {fit_weight:.2f}。"
            if mode == AUTO_MATCH_CONSTRAINT_SOFT
            else "投影拟合：硬锚点。"
        )
        for item in (index_item, name_item, position_item, quality_item, residual_item):
            if group_id:
                item.setData(STAR_PAIR_AUTO_GROUP_ROLE, group_id)
            item.setToolTip(constraint_tip)
        if position is not None:
            position_item.setText(f"({fit_weight:.2f})" if mode == AUTO_MATCH_CONSTRAINT_SOFT else "锚点")
        fit_payload = saved_state.get("fit_payload")
        if isinstance(fit_payload, dict):
            try:
                quality_score = float(fit_payload["quality_score"])
            except (KeyError, TypeError, ValueError):
                quality_score = float("nan")
            if math.isfinite(quality_score):
                quality_item.setText(f"{quality_score:.2f}")

        table.setItem(row, STAR_PAIR_INDEX_COLUMN, index_item)
        table.setItem(row, STAR_PAIR_NAME_COLUMN, name_item)
        table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
        table.setItem(row, STAR_PAIR_QUALITY_COLUMN, quality_item)
        table.setItem(row, STAR_PAIR_RESIDUAL_COLUMN, residual_item)
        table.setItem(row, STAR_PAIR_ANNOTATION_COLUMN, annotation_item)

    def _set_auto_match_group_table_row(self, row: int, group_id: str) -> None:
        table = self.ui.tableWidgetStarPairs
        for column, text in (
            (STAR_PAIR_INDEX_COLUMN, "▶"),
            (STAR_PAIR_NAME_COLUMN, self._auto_match_group_label(group_id)),
            (STAR_PAIR_POSITION_COLUMN, ""),
            (STAR_PAIR_QUALITY_COLUMN, ""),
            (STAR_PAIR_RESIDUAL_COLUMN, ""),
            (STAR_PAIR_ANNOTATION_COLUMN, ""),
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
            (STAR_PAIR_QUALITY_COLUMN, ""),
            (STAR_PAIR_RESIDUAL_COLUMN, ""),
            (STAR_PAIR_ANNOTATION_COLUMN, ""),
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
        if self._star_pair_sort_key not in {
            STAR_PAIR_SORT_KEY_INDEX,
            STAR_PAIR_SORT_KEY_QUALITY,
            STAR_PAIR_SORT_KEY_RESIDUAL,
        }:
            return entries

        if self._star_pair_sort_key == STAR_PAIR_SORT_KEY_INDEX:
            return sorted(
                entries,
                key=lambda entry: -entry[0] if self._star_pair_sort_descending else entry[0],
            )

        if self._star_pair_sort_key == STAR_PAIR_SORT_KEY_QUALITY:
            def quality_sort_key(entry: tuple[int, ReferenceStar, str, str, str]) -> tuple[int, float, int]:
                sequence_number, star, _index_text, _row_type, _group_id = entry
                fit_payload = saved_states.get(star.star_id.strip(), {}).get("fit_payload")
                if not isinstance(fit_payload, dict):
                    return 1, 0.0, sequence_number
                try:
                    quality_score = float(fit_payload["quality_score"])
                except (KeyError, TypeError, ValueError):
                    return 1, 0.0, sequence_number
                if not math.isfinite(quality_score):
                    return 1, 0.0, sequence_number
                sort_value = -quality_score if self._star_pair_sort_descending else quality_score
                return 0, sort_value, sequence_number

            return sorted(entries, key=quality_sort_key)

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
        saved_states = self._star_pair_store_states()
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
            group_id = self._auto_match_group_id_for_star_id(star_id) or "A"
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
        self._apply_star_pair_table_column_widths()
        self._apply_auto_match_group_visibility()
        self._sync_star_pair_annotations_to_table()
        self._refresh_star_pair_table_styles()
        self._restore_star_pair_annotations_from_table()
        self._update_reference_alignment_transform()

    def _handle_star_pair_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == STAR_PAIR_ANNOTATION_COLUMN:
            if self._is_star_pair_group_row(item.row()):
                return
            rows = self._selected_star_pair_annotation_rows(item.row())
            selected_star_ids = tuple(
                star_id
                for row in rows
                if (star_id := self._star_pair_star_id(row))
            )
            current_star_id = self._star_pair_star_id(item.row())
            visible = item.checkState() == Qt.Checked
            changed_count = self._set_star_pair_annotations_for_rows(rows, visible)
            if len(selected_star_ids) > 1:
                QTimer.singleShot(
                    0,
                    partial(
                        self._restore_star_pair_annotation_selection,
                        selected_star_ids,
                        current_star_id,
                    ),
                )
            if changed_count > 0:
                state_text = "显示" if visible else "隐藏"
                self.ui.statusbar.showMessage(f"已{state_text} {changed_count} 行星点标注。")
            return
        if item.column() != STAR_PAIR_POSITION_COLUMN:
            return
        if self._is_star_pair_group_row(item.row()):
            return
        self._refresh_star_pair_row_style(item.row())
        self._update_reference_alignment_transform()

    # ---- 表格数据访问 ----

    def _star_pair_star_id(self, row: int) -> str:
        name_item = self.ui.tableWidgetStarPairs.item(row, STAR_PAIR_NAME_COLUMN)
        if name_item is None:
            return ""
        return str(name_item.data(Qt.UserRole) or "")

    def _reference_stars_with_visible_annotations(self) -> tuple[ReferenceStar, ...]:
        """仅过滤绘图标注，不改变参与匹配和拟合的参考星集合。"""

        hidden_ids = self._hidden_star_pair_annotations()
        return tuple(
            star
            for star in self._current_reference_stars
            if star.star_id.strip() not in hidden_ids
        )

    def _star_pair_position_text(self, row: int) -> str:
        position = self._parse_star_pair_position_text(row)
        if position is None:
            return ""
        return f"{position[0]:.2f}, {position[1]:.2f}"

    def _parse_star_pair_position_text(self, row: int) -> tuple[float, float] | None:
        record = self._star_pair_record_for_row(row)
        if record is None:
            return None
        image_x, image_y = record.position
        if not (math.isfinite(image_x) and math.isfinite(image_y)):
            return None
        return image_x, image_y

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
