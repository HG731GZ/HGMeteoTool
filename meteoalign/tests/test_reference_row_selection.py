from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QItemSelectionModel, QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from meteoalign.app_auto_match import AutoMatchMixin
from meteoalign.app_rendering import RenderingMixin


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _QT_APP = app
    return app


class _StatusBar:
    def __init__(self) -> None:
        self.message = ""

    def showMessage(self, message: str) -> None:  # noqa: N802 - Qt 控件接口命名
        self.message = message


class _ReferenceClickSurface(QWidget):
    def __init__(self, click_handler) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self._click_handler = click_handler
        self.setFocusPolicy(Qt.StrongFocus)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            self._click_handler(event.pos())
        super().mouseReleaseEvent(event)


class _ReferenceClickHarness(AutoMatchMixin, RenderingMixin):
    def __init__(self, star_ids: list[str], picked_star_id: str) -> None:
        app = _qapp()
        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        self.table = QTableWidget(0, 1)
        self.reference_click_surface = _ReferenceClickSurface(self._handle_reference_map_click)
        layout.addWidget(self.table)
        layout.addWidget(self.reference_click_surface)
        self.ui = SimpleNamespace(
            tableWidgetStarPairs=self.table,
            statusbar=_StatusBar(),
        )
        self._manual_reference_star_ids = list(star_ids)
        self._excluded_reference_star_ids: list[str] = []
        self._picked_star_id = picked_star_id
        self._set_table_stars(star_ids)
        self.container.show()
        self.reference_click_surface.setFocus(Qt.OtherFocusReason)
        app.processEvents()

    def _set_table_stars(self, star_ids: list[str]) -> None:
        self.table.setRowCount(len(star_ids))
        for row, star_id in enumerate(star_ids):
            item = QTableWidgetItem(star_id)
            item.setData(Qt.UserRole, star_id)
            self.table.setItem(row, 0, item)

    def _star_pair_star_id(self, row: int) -> str:
        item = self.table.item(row, 0)
        return "" if item is None else str(item.data(Qt.UserRole) or "")

    def _nearest_reference_pick_star(self, _viewport_pos: QPoint) -> tuple[str, str, float]:
        return self._picked_star_id, self._picked_star_id, 0.0

    def _refresh_reference_stars_from_current_map(self) -> None:
        self._set_table_stars(self._manual_reference_star_ids)

    def _star_pair_label(self, row: int) -> str:
        return self._star_pair_star_id(row)

    def select_table_rows(self, rows: list[int]) -> None:
        selection_model = self.table.selectionModel()
        selection_model.clearSelection()
        for row in rows:
            selection_model.select(
                self.table.model().index(row, 0),
                QItemSelectionModel.Select | QItemSelectionModel.Rows,
            )


def _assert_active_row(harness: _ReferenceClickHarness, star_id: str) -> None:
    app = _qapp()
    app.processEvents()
    selected_rows = [index.row() for index in harness.table.selectionModel().selectedRows()]
    expected_row = harness._row_for_star_id(star_id)
    assert expected_row is not None
    assert selected_rows == [expected_row]
    assert harness.table.currentRow() == expected_row
    assert app.focusWidget() is harness.table


def test_reference_click_activates_existing_star_row() -> None:
    harness = _ReferenceClickHarness(["HR1", "HR2", "HR3"], "HR3")
    try:
        harness.select_table_rows([0, 1])

        QTest.mouseClick(harness.reference_click_surface, Qt.LeftButton, Qt.ControlModifier)

        _assert_active_row(harness, "HR3")
        assert harness._manual_reference_star_ids == ["HR1", "HR2", "HR3"]
    finally:
        harness.container.close()


def test_reference_click_activates_new_star_row() -> None:
    harness = _ReferenceClickHarness(["HR1", "HR2"], "HR3")
    try:
        harness.select_table_rows([0, 1])

        QTest.mouseClick(harness.reference_click_surface, Qt.LeftButton, Qt.ControlModifier)

        _assert_active_row(harness, "HR3")
        assert harness._manual_reference_star_ids == ["HR1", "HR2", "HR3"]
    finally:
        harness.container.close()
