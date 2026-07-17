from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QMessageBox,
    QSpinBox,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetSelectionRange,
)

from meteoalign.application.app_constants import (
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_MODES,
    AUTO_MATCH_CONSTRAINT_SOFT,
    STAR_PAIR_ANNOTATION_COLUMN,
    STAR_PAIR_NAME_COLUMN,
    STAR_PAIR_POSITION_COLUMN,
    STAR_PAIR_ROW_TYPE_MANUAL,
)
from meteoalign.application.app_auto_match import AutoMatchMixin
from meteoalign.application import app_star_pair_actions as star_pair_actions_module
from meteoalign.application.app_star_pair_table import StarPairTableMixin
from meteoalign.application.star_pair_assistant_dialog import StarPairAssistantDialog
from meteoalign.simulator import ReferenceStar
from meteoalign.star_pair_model import StarPairRecord
from meteoalign.star_pair_store import StarPairStore


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _QT_APP = app
    return app


def _reference_star(star_id: str, index: int) -> ReferenceStar:
    return ReferenceStar(
        index=index,
        star_id=star_id,
        name=star_id,
        display_name=star_id,
        common_name=star_id,
        ra_deg=float(index),
        dec_deg=float(index),
        mag_v=1.0,
        sim_x=10.0 * index,
        sim_y=20.0 * index,
        alt_deg=45.0,
        az_deg=90.0,
    )


class _Ui:
    def __init__(self) -> None:
        self.tableWidgetStarPairs = QTableWidget()
        self.tableWidgetStarPairs.setColumnCount(6)
        self.tableWidgetStarPairs.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tableWidgetStarPairs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.spinBoxReferenceStarCount = QSpinBox()
        self.spinBoxReferenceStarCount.setRange(3, 40)
        self.spinBoxReferenceStarCount.setValue(12)
        self.statusbar = _StatusBar()


class _StatusBar:
    """记录 Delete/Backspace 操作反馈的状态栏替身。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.messages.append(message)


class _StarPairTableHarness(StarPairTableMixin):
    def __init__(self, reference_stars: tuple[ReferenceStar, ...]) -> None:
        _qapp()
        self.ui = _Ui()
        self.reference_stars = reference_stars
        self._current_reference_stars: tuple[ReferenceStar, ...] = ()
        self._active_star_pair_row = None
        self._star_pair_annotations = {}
        self._hidden_star_pair_annotation_ids: set[str] = set()
        self._focused_star_annotations = []
        self._manual_match_group_expanded = True
        self._auto_match_reference_star_ids: list[str] = []
        self._auto_match_group_order: list[str] = []
        self._auto_match_group_by_star_id: dict[str, str] = {}
        self._auto_match_group_expanded_by_id: dict[str, bool] = {}
        self._auto_match_next_group_index = 0
        self._star_pair_sort_key = None
        self._star_pair_sort_descending = True
        self._sky_alignment_transform = None
        self._current_star_map = object()
        self._manual_reference_star_ids: list[str] = []
        self._imported_reference_star_by_id: dict[str, ReferenceStar] = {}
        self._excluded_reference_star_ids: list[str] = []
        self._mask_excluded_reference_star_ids: set[str] = set()
        self.current_image_preview = None
        self._star_pair_store = StarPairStore()

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

    def _reference_star_lookup(self) -> dict[str, ReferenceStar]:
        return {star.star_id: star for star in self._current_reference_stars}

    def _reference_star_for_row(self, row: int) -> ReferenceStar | None:
        star_id = self._star_pair_star_id(row)
        if not star_id:
            return None
        return self._reference_star_lookup().get(star_id)

    def _reference_star_with_index(self, star: ReferenceStar, index: int) -> ReferenceStar:
        return replace(star, index=index)

    def _matched_reference_star_ids_from_table(self) -> set[str]:
        matched_star_ids: set[str] = set()
        table = self.ui.tableWidgetStarPairs
        for row in range(table.rowCount()):
            star_id = self._star_pair_star_id(row)
            if star_id and self._parse_star_pair_position_text(row) is not None:
                matched_star_ids.add(star_id)
        return matched_star_ids

    def _refresh_reference_stars_from_current_map(self) -> None:
        auto_star_ids = set(self._auto_match_reference_star_ids)
        excluded_star_ids = set(self._excluded_reference_star_ids) | set(self._mask_excluded_reference_star_ids)
        matched_star_ids = self._matched_reference_star_ids_from_table()
        visible_stars = tuple(
            star
            for star in self.reference_stars
            if star.star_id in auto_star_ids
            or star.star_id in matched_star_ids
            or star.star_id not in excluded_star_ids
        )
        self._update_star_pair_table(visible_stars)

    def _star_pair_alignment_residual(self, _row: int):  # type: ignore[no-untyped-def]
        return None

    def _update_reference_alignment_transform(self) -> None:
        return

    def _remove_star_pair_annotation(self, star_id: str) -> None:
        self._star_pair_annotations.pop(star_id, None)

    def _clear_star_pair_annotations(self) -> None:
        self._star_pair_annotations.clear()

    def _show_real_image_annotations(self) -> bool:
        return True


def _row_for_star_id(harness: _StarPairTableHarness, star_id: str) -> int:
    table = harness.ui.tableWidgetStarPairs
    for row in range(table.rowCount()):
        item = table.item(row, STAR_PAIR_NAME_COLUMN)
        if item is not None and item.data(Qt.UserRole) == star_id:
            return row
    raise AssertionError(f"表格中没有星 {star_id}")


def _visible_star_ids(harness: _StarPairTableHarness) -> list[str]:
    table = harness.ui.tableWidgetStarPairs
    star_ids: list[str] = []
    for row in range(table.rowCount()):
        if harness._star_pair_row_type(row) != STAR_PAIR_ROW_TYPE_MANUAL and not harness._is_auto_match_row(row):
            continue
        star_id = harness._star_pair_star_id(row)
        if star_id:
            star_ids.append(star_id)
    return star_ids


def _set_matched_position(harness: _StarPairTableHarness, star_id: str) -> None:
    row = _row_for_star_id(harness, star_id)
    reference_star = harness._reference_star_for_row(row)
    assert reference_star is not None
    group_id = harness._row_auto_match_group_id(row) if harness._is_auto_match_row(row) else None
    harness._star_pair_store.add(
        StarPairRecord(
            reference_star=reference_star,
            image_x_px=123.0,
            image_y_px=456.0,
            group_id=group_id,
        )
    )


def _select_star_rows(harness: _StarPairTableHarness, *star_ids: str) -> None:
    """按星表编号多选完整表格行。"""

    table = harness.ui.tableWidgetStarPairs
    table.clearSelection()
    for star_id in star_ids:
        row = _row_for_star_id(harness, star_id)
        table.setRangeSelected(
            QTableWidgetSelectionRange(row, 0, row, table.columnCount() - 1),
            True,
        )


def test_delete_key_mixed_selection_only_clears_matched_rows() -> None:
    """混选已匹配和未匹配行时，只清除匹配，未匹配行不得被删除。"""

    harness = _StarPairTableHarness(
        (_reference_star("matched-1", 1), _reference_star("unmatched-1", 2))
    )
    harness._update_star_pair_table(harness.reference_stars)
    _set_matched_position(harness, "matched-1")
    _select_star_rows(harness, "matched-1", "unmatched-1")

    handled = harness._handle_star_pair_delete_key()

    assert handled
    assert len(harness._star_pair_store) == 0
    assert set(_visible_star_ids(harness)) == {"matched-1", "unmatched-1"}
    assert harness._excluded_reference_star_ids == []
    assert harness.ui.statusbar.messages[-1] == "已清除 1 个匹配；1 个未匹配行保持不变。"


def test_delete_key_all_unmatched_selection_deletes_rows() -> None:
    """选区全部未匹配时，Delete/Backspace 才删除所选行。"""

    harness = _StarPairTableHarness(
        (_reference_star("unmatched-1", 1), _reference_star("unmatched-2", 2))
    )
    harness._update_star_pair_table(harness.reference_stars)
    _select_star_rows(harness, "unmatched-1", "unmatched-2")

    handled = harness._handle_star_pair_delete_key()

    assert handled
    assert _visible_star_ids(harness) == []
    assert set(harness._excluded_reference_star_ids) == {"unmatched-1", "unmatched-2"}
    assert harness.ui.statusbar.messages[-1] == "已删除 2 行参考星，后续序号已重新排列。"


def test_deleting_auto_match_group_does_not_move_children_to_manual_group() -> None:
    harness = _StarPairTableHarness(
        (
            _reference_star("manual-1", 1),
            _reference_star("auto-1", 2),
            _reference_star("auto-2", 3),
        )
    )
    harness._auto_match_reference_star_ids = ["auto-1", "auto-2"]
    harness._auto_match_group_order = ["A"]
    harness._auto_match_group_by_star_id = {"auto-1": "A", "auto-2": "A"}
    harness._auto_match_group_expanded_by_id = {"A": True}
    harness._update_star_pair_table(harness.reference_stars)
    _set_matched_position(harness, "auto-1")

    deleted_count = harness._delete_auto_match_group("A")

    assert deleted_count == 2
    assert _visible_star_ids(harness) == ["manual-1"]
    assert "auto-1" in harness._excluded_reference_star_ids
    assert "auto-2" in harness._excluded_reference_star_ids


def test_deleting_last_auto_match_group_reuses_its_letter() -> None:
    """删除末尾的自动 C 表后，下一次自动扩展应继续生成 C，而不是跳到 D。"""

    harness = _StarPairTableHarness(
        (
            _reference_star("auto-a", 1),
            _reference_star("auto-b", 2),
            _reference_star("auto-c", 3),
        )
    )
    harness._auto_match_reference_star_ids = ["auto-a", "auto-b", "auto-c"]
    harness._auto_match_group_order = ["A", "B", "C"]
    harness._auto_match_group_by_star_id = {
        "auto-a": "A",
        "auto-b": "B",
        "auto-c": "C",
    }
    harness._auto_match_group_expanded_by_id = {"A": True, "B": True, "C": True}
    harness._auto_match_next_group_index = 3
    harness._update_star_pair_table(harness.reference_stars)

    deleted_count = harness._delete_auto_match_group("C")
    next_group_id = harness._create_auto_match_group()

    assert deleted_count == 1
    assert next_group_id == "C"
    assert harness._auto_match_group_order == ["A", "B", "C"]


def test_reset_all_rows_rebuilds_startup_reference_list() -> None:
    """重置匹配列表后应清空派生分组与排除项，再恢复当前范围内的初始标注星。"""

    harness = _StarPairTableHarness(
        (
            _reference_star("manual-1", 1),
            _reference_star("manual-2", 2),
            _reference_star("auto-1", 3),
        )
    )
    harness._auto_match_reference_star_ids = ["auto-1"]
    harness._auto_match_group_order = ["A"]
    harness._auto_match_group_by_star_id = {"auto-1": "A"}
    harness._auto_match_group_expanded_by_id = {"A": False}
    harness._excluded_reference_star_ids = ["manual-2"]
    stale_aligned_reference_map = object()
    harness._current_reference_star_map = stale_aligned_reference_map
    harness._update_star_pair_table((harness.reference_stars[0], harness.reference_stars[2]))
    _set_matched_position(harness, "manual-1")
    _set_matched_position(harness, "auto-1")

    pair_count, deleted_count, rebuilt_count = harness.reset_reference_star_list()

    assert pair_count == 2
    assert deleted_count == 2
    assert rebuilt_count == 3
    assert _visible_star_ids(harness) == ["manual-1", "manual-2", "auto-1"]
    assert len(harness._star_pair_store) == 0
    assert harness._manual_reference_star_ids == []
    assert harness._auto_match_reference_star_ids == []
    assert harness._auto_match_group_order == []
    assert harness._auto_match_group_by_star_id == {}
    assert harness._auto_match_group_expanded_by_id == {}
    assert harness._excluded_reference_star_ids == []
    assert harness._mask_excluded_reference_star_ids == set()
    assert harness._current_reference_star_map is harness._current_star_map
    assert harness._current_reference_star_map is not stale_aligned_reference_map
    assert harness.ui.spinBoxReferenceStarCount.value() == 12


def test_reset_match_list_confirmation_uses_reset_wording(monkeypatch) -> None:
    """重置入口及确认提示不应继续使用“删除所有匹配”的旧名称。"""

    harness = _StarPairTableHarness((_reference_star("manual-1", 1),))
    harness._update_star_pair_table(harness.reference_stars)
    captured: list[tuple[str, str]] = []

    def confirm(_parent, title, message, *_args):  # type: ignore[no-untyped-def]
        captured.append((title, message))
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)

    harness.delete_all_star_pair_rows()

    assert captured == [
        (
            "确认重置匹配列表",
            "确定要重置当前匹配列表吗？\n\n"
            "列表中的 1 行参考星及其匹配将被清空，随后会按当前参考星图范围和星空模拟的标注设置重新生成初始列表。",
        )
    ]
    assert harness.ui.statusbar.messages[-1] == "已重置匹配列表：清空 1 行参考星，并按当前参考星图重新添加 1 颗标注星。"


def test_reset_confirmation_keeps_star_pair_assistant_visible(monkeypatch) -> None:
    """确认重置后助手应保持可见，确认框也必须依附于助手而非主窗口。"""

    app = _qapp()
    assistant = StarPairAssistantDialog()
    assistant.show()
    app.processEvents()
    harness = _StarPairTableHarness((_reference_star("manual-1", 1),))
    harness._update_star_pair_table(harness.reference_stars)
    harness.star_pair_assistant = assistant
    message_parents: list[object] = []
    reactivation_calls: list[str] = []
    monkeypatch.setattr(assistant, "raise_", lambda: reactivation_calls.append("raise"))
    monkeypatch.setattr(assistant, "activateWindow", lambda: reactivation_calls.append("activate"))

    def confirm(parent, *_args):  # type: ignore[no-untyped-def]
        message_parents.append(parent)
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)

    harness.delete_all_star_pair_rows()
    app.processEvents()

    assert message_parents == [assistant]
    assert assistant.isVisible()
    assert reactivation_calls == ["raise", "activate"]
    assistant.close()


def test_deleting_matched_manual_row_removes_it_instead_of_preserving_matched_star() -> None:
    harness = _StarPairTableHarness((_reference_star("manual-1", 1),))
    harness._update_star_pair_table(harness.reference_stars)
    _set_matched_position(harness, "manual-1")

    deleted_count = harness._delete_star_pair_rows([_row_for_star_id(harness, "manual-1")])

    assert deleted_count == 1
    assert _visible_star_ids(harness) == []
    assert "manual-1" in harness._excluded_reference_star_ids


def test_deleted_auto_match_row_can_reenter_auto_match_candidate_pool() -> None:
    reference_star = _reference_star("auto-1", 1)
    harness = _StarPairTableHarness((reference_star,))
    harness._auto_match_reference_star_ids = ["auto-1"]
    harness._auto_match_group_order = ["A"]
    harness._auto_match_group_by_star_id = {"auto-1": "A"}
    harness._auto_match_group_expanded_by_id = {"A": True}
    harness._update_star_pair_table(harness.reference_stars)

    deleted_count = harness._delete_star_pair_rows([_row_for_star_id(harness, "auto-1")])

    assert deleted_count == 1
    assert "auto-1" in harness._excluded_reference_star_ids
    assert "auto-1" not in AutoMatchMixin._auto_match_blocked_reference_star_ids(harness)

    harness._auto_match_constraint_mode = lambda: AUTO_MATCH_CONSTRAINT_SOFT  # type: ignore[attr-defined]
    harness._auto_match_soft_weight = lambda: 0.3  # type: ignore[attr-defined]
    candidate_ids = AutoMatchMixin._ensure_auto_match_candidates_visible(harness, [reference_star], "B")

    assert candidate_ids == {"auto-1"}
    assert "auto-1" not in harness._excluded_reference_star_ids
    assert "auto-1" in harness._auto_match_reference_star_ids
    assert harness._is_auto_match_row(_row_for_star_id(harness, "auto-1"))


def test_deleted_manual_row_can_reenter_auto_match_candidate_pool() -> None:
    """删除手动行只影响当前列表，不应永久阻止自动扩展再次选中该星。"""

    reference_star = _reference_star("manual-1", 1)
    harness = _StarPairTableHarness((reference_star,))
    harness._update_star_pair_table(harness.reference_stars)

    deleted_count = harness._delete_star_pair_rows([_row_for_star_id(harness, "manual-1")])

    assert deleted_count == 1
    assert "manual-1" in harness._excluded_reference_star_ids
    assert "manual-1" not in AutoMatchMixin._auto_match_blocked_reference_star_ids(harness)

    harness._auto_match_constraint_mode = lambda: AUTO_MATCH_CONSTRAINT_SOFT  # type: ignore[attr-defined]
    harness._auto_match_soft_weight = lambda: 0.3  # type: ignore[attr-defined]
    candidate_ids = AutoMatchMixin._ensure_auto_match_candidates_visible(harness, [reference_star], "B")

    assert candidate_ids == {"manual-1"}
    assert "manual-1" not in harness._excluded_reference_star_ids
    assert "manual-1" in harness._auto_match_reference_star_ids
    assert harness._is_auto_match_row(_row_for_star_id(harness, "manual-1"))


def test_only_mask_exclusions_remain_blocked_after_unmatched_candidate_cleanup() -> None:
    failed_star = _reference_star("failed-1", 1)
    masked_star = _reference_star("masked-1", 2)
    harness = _StarPairTableHarness((failed_star, masked_star))
    harness._auto_match_reference_star_ids = ["failed-1", "masked-1"]
    harness._auto_match_group_order = ["A"]
    harness._auto_match_group_by_star_id = {"failed-1": "A", "masked-1": "A"}
    harness._auto_match_group_expanded_by_id = {"A": True}
    harness._mask_excluded_reference_star_ids = {"masked-1"}
    harness._update_star_pair_table(harness.reference_stars)

    removed_count = AutoMatchMixin._remove_unmatched_auto_match_candidates(
        harness,
        {"failed-1", "masked-1"},
    )
    blocked_star_ids = AutoMatchMixin._auto_match_blocked_reference_star_ids(harness)

    assert removed_count == 2
    assert "failed-1" not in blocked_star_ids
    assert "masked-1" in blocked_star_ids
    assert "failed-1" in harness._excluded_reference_star_ids
    assert "masked-1" in harness._excluded_reference_star_ids


def test_annotation_checkbox_applies_to_all_selected_rows_and_both_previews() -> None:
    """多选行切换一次复选框后，两类标注都必须遵循相同的行级状态。"""

    class _VisibleItem:
        def __init__(self) -> None:
            self.visible = True

        def setVisible(self, visible: bool) -> None:  # noqa: N802 - 模拟 Qt 接口。
            self.visible = visible

    harness = _StarPairTableHarness(
        (
            _reference_star("manual-1", 1),
            _reference_star("manual-2", 2),
            _reference_star("manual-3", 3),
        )
    )
    harness._update_star_pair_table(harness.reference_stars)
    _select_star_rows(harness, "manual-1", "manual-2")

    changed_row = _row_for_star_id(harness, "manual-1")
    changed_item = harness.ui.tableWidgetStarPairs.item(changed_row, STAR_PAIR_ANNOTATION_COLUMN)
    assert changed_item is not None
    changed_item.setCheckState(Qt.Unchecked)
    harness._handle_star_pair_item_changed(changed_item)

    assert harness._hidden_star_pair_annotation_ids == {"manual-1", "manual-2"}
    for star_id in ("manual-1", "manual-2"):
        item = harness.ui.tableWidgetStarPairs.item(
            _row_for_star_id(harness, star_id),
            STAR_PAIR_ANNOTATION_COLUMN,
        )
        assert item is not None and item.checkState() == Qt.Unchecked
    visible_item = harness.ui.tableWidgetStarPairs.item(
        _row_for_star_id(harness, "manual-3"),
        STAR_PAIR_ANNOTATION_COLUMN,
    )
    assert visible_item is not None and visible_item.checkState() == Qt.Checked
    assert [star.star_id for star in harness._reference_stars_with_visible_annotations()] == ["manual-3"]

    annotation_items = {
        star_id: (_VisibleItem(), _VisibleItem())
        for star_id in ("manual-1", "manual-2", "manual-3")
    }
    harness._star_pair_annotations = annotation_items
    harness._update_star_pair_annotation_visibility()
    assert not annotation_items["manual-1"][0].visible
    assert not annotation_items["manual-2"][1].visible
    assert annotation_items["manual-3"][0].visible


def test_clicking_checkbox_keeps_multirow_batch_behavior() -> None:
    """真实鼠标点击复选框时也应把目标状态应用到当前多选行。"""

    app = _qapp()
    harness = _StarPairTableHarness(
        (_reference_star("manual-1", 1), _reference_star("manual-2", 2))
    )
    harness._update_star_pair_table(harness.reference_stars)
    harness._configure_star_pair_table_columns()
    table = harness.ui.tableWidgetStarPairs
    table.itemChanged.connect(harness._handle_star_pair_item_changed)
    table.resize(480, 240)
    table.show()
    app.processEvents()
    _select_star_rows(harness, "manual-1", "manual-2")

    row = _row_for_star_id(harness, "manual-1")
    item = table.item(row, STAR_PAIR_ANNOTATION_COLUMN)
    assert item is not None
    table.scrollToItem(item)
    app.processEvents()
    item_index = table.model().index(row, STAR_PAIR_ANNOTATION_COLUMN)
    option = QStyleOptionViewItem()
    option.initFrom(table)
    option.rect = table.visualItemRect(item)
    QStyledItemDelegate(table).initStyleOption(option, item_index)
    check_rect = table.style().subElementRect(QStyle.SE_ItemViewItemCheckIndicator, option, table)
    QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=check_rect.center())
    app.processEvents()

    assert harness._hidden_star_pair_annotation_ids == {"manual-1", "manual-2"}
    selected_star_ids = {
        harness._star_pair_star_id(index.row())
        for index in table.selectionModel().selectedRows()
    }
    assert selected_star_ids == {"manual-1", "manual-2"}
    table.close()


def test_group_annotation_toggle_hides_and_restores_only_that_list() -> None:
    """分组头菜单文案和批量状态应在隐藏、显示之间切换。"""

    harness = _StarPairTableHarness(
        (
            _reference_star("manual-1", 1),
            _reference_star("manual-2", 2),
            _reference_star("auto-1", 3),
        )
    )
    harness._auto_match_reference_star_ids = ["auto-1"]
    harness._auto_match_group_order = ["A"]
    harness._auto_match_group_by_star_id = {"auto-1": "A"}
    harness._auto_match_group_expanded_by_id = {"A": True}
    harness._update_star_pair_table(harness.reference_stars)
    group_row = harness._manual_match_group_row()
    assert group_row is not None

    assert harness._star_pair_group_annotation_action_text(group_row) == "隐藏列表所有标注"
    assert harness._toggle_star_pair_group_annotations(group_row) == 2
    assert harness._star_pair_group_annotation_action_text(group_row) == "显示列表所有标注"
    assert harness._hidden_star_pair_annotation_ids == {"manual-1", "manual-2"}
    auto_item = harness.ui.tableWidgetStarPairs.item(
        _row_for_star_id(harness, "auto-1"),
        STAR_PAIR_ANNOTATION_COLUMN,
    )
    assert auto_item is not None and auto_item.checkState() == Qt.Checked

    assert harness._toggle_star_pair_group_annotations(group_row) == 2
    assert harness._star_pair_group_annotation_action_text(group_row) == "隐藏列表所有标注"
    assert harness._hidden_star_pair_annotation_ids == set()


def test_auto_group_annotation_action_is_last_in_context_menu(monkeypatch) -> None:
    """自动匹配分组菜单中的整组标注动作应固定在最下面。"""

    menu_labels: list[str] = []

    class _Menu:
        def __init__(self, _parent: object) -> None:
            return

        def addAction(self, text: str):  # noqa: N802 - 模拟 Qt 接口。
            menu_labels.append(text)
            return object()

        def exec_(self, _point: object) -> None:
            return None

    monkeypatch.setattr(star_pair_actions_module, "QMenu", _Menu)
    harness = _StarPairTableHarness((_reference_star("auto-1", 1),))
    harness._auto_match_reference_star_ids = ["auto-1"]
    harness._auto_match_group_order = ["A"]
    harness._auto_match_group_by_star_id = {"auto-1": "A"}
    harness._auto_match_group_expanded_by_id = {"A": True}
    harness._update_star_pair_table(harness.reference_stars)
    group_row = harness._auto_match_group_row("A")
    assert group_row is not None
    table = harness.ui.tableWidgetStarPairs
    point = QPoint(4, table.rowViewportPosition(group_row) + table.rowHeight(group_row) // 2)

    harness._show_star_pair_context_menu(point)

    assert menu_labels == ["折叠自动匹配表", "删除自动匹配表", "隐藏列表所有标注"]


def test_soft_constraint_and_group_mode_text_stay_compact() -> None:
    """软约束只显示权重，折叠表头的模式列保持空白。"""

    harness = _StarPairTableHarness((_reference_star("manual-1", 1),))
    harness._update_star_pair_table(harness.reference_stars)
    _set_matched_position(harness, "manual-1")
    harness._star_pair_store.set_constraint("manual-1", AUTO_MATCH_CONSTRAINT_SOFT, 0.3)
    harness._update_star_pair_table(harness.reference_stars)

    row = _row_for_star_id(harness, "manual-1")
    group_row = harness._manual_match_group_row()
    assert group_row is not None
    assert harness.ui.tableWidgetStarPairs.item(row, STAR_PAIR_POSITION_COLUMN).text() == "(0.30)"
    assert harness.ui.tableWidgetStarPairs.item(group_row, STAR_PAIR_POSITION_COLUMN).text() == ""
