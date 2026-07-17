from __future__ import annotations

from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtWidgets import QApplication, QInputDialog, QMenu, QMessageBox, QTableWidgetItem, QWidget

from .app_constants import (
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_SOFT,
    STAR_PAIR_NAME_COLUMN,
    STAR_PAIR_POSITION_COLUMN,
    STAR_PAIR_ROW_TYPE_MANUAL,
)
from ..auto_match_quality import AUTO_MATCH_QUALITY_FIELD_KEYS
from ..star_fitting import FittedStarPosition
from ..star_pair_model import (
    PAIR_ORIGIN_AUTO_MATCH,
    PAIR_ORIGIN_MANUAL,
    PsfFit,
    StarPairRecord,
)

class StarPairActionsMixin:
    """星对右键菜单、拾取模式、位置编辑和删除动作。"""

    def _star_pair_assistant_message_parent(self) -> QWidget:
        """优先让匹配助手内发起的消息框依附于助手，避免主窗口遮住助手。"""

        assistant = getattr(self, "star_pair_assistant", None)
        return assistant if isinstance(assistant, QWidget) else self

    def _reactivate_star_pair_assistant(self) -> None:
        """确认操作结束后让仍然可见的匹配助手继续位于当前交互层。"""

        assistant = getattr(self, "star_pair_assistant", None)
        if isinstance(assistant, QWidget) and assistant.isVisible():
            assistant.raise_()
            assistant.activateWindow()

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
        if self._is_manual_match_group_row(row):
            toggle_action = menu.addAction("折叠手动匹配表" if self._manual_match_group_expanded else "展开手动匹配表")
            annotation_action = menu.addAction(self._star_pair_group_annotation_action_text(row))
            selected_action = menu.exec_(table.viewport().mapToGlobal(point))
            if selected_action is toggle_action:
                self._toggle_manual_match_group()
            elif selected_action is annotation_action:
                self._toggle_star_pair_group_annotations(row)
            return
        if self._is_auto_match_group_row(row):
            group_id = self._row_auto_match_group_id(row)
            expanded = self._auto_match_group_expanded_by_id.get(group_id, True)
            toggle_action = menu.addAction("折叠自动匹配表" if expanded else "展开自动匹配表")
            delete_group_action = menu.addAction("删除自动匹配表")
            annotation_action = menu.addAction(self._star_pair_group_annotation_action_text(row))
            selected_action = menu.exec_(table.viewport().mapToGlobal(point))
            if selected_action is toggle_action:
                self._toggle_auto_match_group(group_id)
            elif selected_action is annotation_action:
                self._toggle_star_pair_group_annotations(row)
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
            auto_pair_action = menu.addAction("自动匹配")
            auto_pair_action.setEnabled(self._sky_alignment_transform is not None and self.current_image_preview is not None)
        clear_action = None
        if matched_rows:
            clear_action = menu.addAction("清除选中匹配" if len(matched_rows) > 1 else "清除匹配")

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
            self.ui.statusbar.showMessage("手动匹配失败：请先导入真实图像，再点选星点位置。")
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

        star_id = self._star_pair_star_id(row)
        if not star_id:
            return

        store = getattr(self, "_star_pair_store", None)
        if store is None:
            return

        psf = PsfFit.from_fitted_position(fitted_position)
        if star_id in store:
            store.update_position(star_id, fitted_position.x, fitted_position.y, psf=psf)
        else:
            reference_star = self._reference_star_for_row(row)
            if reference_star is None:
                return
            pair_origin = PAIR_ORIGIN_AUTO_MATCH if self._is_auto_match_row(row) else PAIR_ORIGIN_MANUAL
            constraint_mode, fit_weight = (
                self._auto_match_constraint_for_star_id(star_id)
                if self._is_auto_match_row(row)
                else (AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0)
            )
            group_id = self._row_auto_match_group_id(row) if self._is_auto_match_row(row) else None
            group_name = self._auto_match_group_label(group_id) if group_id else None
            record = StarPairRecord(
                reference_star=reference_star,
                image_x_px=float(fitted_position.x),
                image_y_px=float(fitted_position.y),
                psf=psf,
                pair_origin=pair_origin,
                group_id=group_id or None,
                group_name=group_name,
                fit_constraint_mode=constraint_mode,
                fit_weight=float(fit_weight),
            )
            store.add(record)

        # 位置变化后旧的自动扩展质量已经失效；批量自动扩展会在本次拟合完成后写入新值。
        store.remove_extra_fields(star_id, AUTO_MATCH_QUALITY_FIELD_KEYS)

        if self._is_auto_match_row(row):
            self._auto_match_group_by_star_id.pop(star_id, None)

        position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
        if position_item is None:
            position_item = QTableWidgetItem()
            table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
        signals_were_blocked = table.blockSignals(True)
        position_item.setText(self._star_pair_mode_display_text(row))
        table.blockSignals(signals_were_blocked)
        self._refresh_star_pair_quality_cell(row)
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
        if star_id and self._is_auto_match_row(row):
            group_id = self._row_auto_match_group_id(row) or self._auto_match_group_id_for_star_id(star_id) or "A"
            self._auto_match_group_by_star_id[star_id] = group_id
        store = getattr(self, "_star_pair_store", None)
        if store is not None and star_id:
            store.remove(star_id)
        if position_item is not None:
            signals_were_blocked = table.blockSignals(True)
            position_item.setText("")
            table.blockSignals(signals_were_blocked)
        self._refresh_star_pair_quality_cell(row)
        if star_id:
            self._remove_star_pair_annotation(star_id)
        return star_id

    def _clear_star_pair_positions(self) -> int:
        cleared_count = self._star_pair_position_count()
        if self._active_star_pair_row is not None:
            self._leave_star_pick_mode()
        store = getattr(self, "_star_pair_store", None)
        if store is not None:
            store.clear()
        table = self.ui.tableWidgetStarPairs
        table.blockSignals(True)
        for row in range(table.rowCount()):
            position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
            if position_item is not None:
                position_item.setText("")
            self._refresh_star_pair_quality_cell(row)
            if self._is_auto_match_row(row):
                star_id = self._star_pair_star_id(row)
                group_id = self._row_auto_match_group_id(row) or self._auto_match_group_id_for_star_id(star_id) or "A"
                if star_id:
                    self._auto_match_group_by_star_id[star_id] = group_id
        table.blockSignals(False)

        self._clear_star_pair_annotations()
        self._refresh_star_pair_table_styles()
        self._update_auto_match_group_row_text()
        self._update_reference_alignment_transform()
        return cleared_count

    def clear_all_star_pair_positions(self) -> None:
        pair_count = self._star_pair_position_count()
        if pair_count <= 0:
            self.ui.statusbar.showMessage("当前没有可清除的星点匹配。")
            return
        reply = QMessageBox.question(
            self._star_pair_assistant_message_parent(),
            "确认清除所有匹配",
            f"确定要清除当前全部 {pair_count} 个星点匹配吗？\n\n此操作会删除手工和自动匹配记录。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self.ui.statusbar.showMessage("已取消清除所有星点匹配。")
            return
        cleared_count = self._clear_star_pair_positions()
        self.ui.statusbar.showMessage(f"已清除 {cleared_count} 个星点匹配。")
        self._reactivate_star_pair_assistant()

    def delete_all_star_pair_rows(self) -> None:
        """确认后重置匹配表，并按当前参考星图范围恢复启动时的参考星列表。"""

        table = self.ui.tableWidgetStarPairs
        row_count = sum(1 for row in range(table.rowCount()) if not self._is_star_pair_group_row(row))
        if row_count > 0:
            reply = QMessageBox.question(
                self._star_pair_assistant_message_parent(),
                "确认重置匹配列表",
                (
                    f"确定要重置当前匹配列表吗？\n\n"
                    f"列表中的 {row_count} 行参考星及其匹配将被清空，随后会按当前参考星图范围和"
                    "星空模拟的标注设置重新生成初始列表。"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.ui.statusbar.showMessage("已取消重置匹配列表。")
                return

        _pair_count, deleted_count, rebuilt_count = self.reset_reference_star_list()
        self.ui.statusbar.showMessage(
            f"已重置匹配列表：清空 {deleted_count} 行参考星，并按当前参考星图重新添加 "
            f"{rebuilt_count} 颗标注星。"
        )
        self._reactivate_star_pair_assistant()

    def _clear_star_pair_position(self, row: int) -> None:
        star_label = self._star_pair_label(row)
        cleared_count = self._clear_star_pair_positions_for_rows([row])
        if cleared_count > 0:
            self.ui.statusbar.showMessage(f"已清除 {star_label} 的真实图像匹配。右键该行可重新点选位置。")

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
        self._hidden_star_pair_annotations().difference_update(deleted_star_ids)
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
            self._auto_match_group_by_star_id.pop(star_id, None)

        self._manual_reference_star_ids = [
            star_id for star_id in self._manual_reference_star_ids if star_id not in manual_star_ids_to_delete
        ]
        store = getattr(self, "_star_pair_store", None)
        if store is not None:
            store.remove_records(deleted_star_ids)

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

        # 分组行代表其全部子行，判断本次按键意图前先展开；只要其中存在一个匹配，
        # 整个选区就进入“清除匹配”模式，未匹配行必须保持不动。
        action_rows = [
            row
            for row in self._rows_expanded_from_groups(rows)
            if 0 <= row < table.rowCount() and not self._is_star_pair_group_row(row)
        ]
        rows_with_position = [
            row for row in action_rows if self._parse_star_pair_position_text(row) is not None
        ]
        if rows_with_position:
            cleared_count = self._clear_star_pair_positions_for_rows(rows_with_position)
            if cleared_count > 0:
                unchanged_count = len(action_rows) - len(rows_with_position)
                unchanged_text = f"；{unchanged_count} 个未匹配行保持不变" if unchanged_count > 0 else ""
                self.ui.statusbar.showMessage(f"已清除 {cleared_count} 个匹配{unchanged_text}。")
            return True

        deleted_count = self._delete_star_pair_rows(rows)
        if deleted_count > 0:
            self.ui.statusbar.showMessage(f"已删除 {deleted_count} 行参考星，后续序号已重新排列。")
        return True
