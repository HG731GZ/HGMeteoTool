from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QTableWidget

from meteoalign.app_star_pair_json_task import StarPairJsonTaskMixin
from meteoalign.star_pair_store import StarPairStore


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
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - Qt 风格 API。
        self.messages.append(message)


class _Ui:
    def __init__(self) -> None:
        self.tableWidgetStarPairs = QTableWidget()
        self.tableWidgetStarPairs.setColumnCount(4)
        self.statusbar = _StatusBar()


class _Harness(StarPairJsonTaskMixin):
    def __init__(self) -> None:
        _qapp()
        self.ui = _Ui()
        self._star_pair_store = StarPairStore()
        self._active_star_pair_row = 3
        self._manual_match_group_expanded = False
        self._manual_reference_star_ids = ["manual-old"]
        self._imported_reference_star_by_id = {"manual-old": object()}
        self._auto_match_reference_star_ids = ["auto-old"]
        self._auto_match_group_order = ["A"]
        self._auto_match_group_by_star_id = {"auto-old": "A"}
        self._auto_match_group_expanded_by_id = {"A": False}
        self._auto_match_next_group_index = 1
        self._excluded_reference_star_ids = ["deleted-old"]
        self._sky_alignment_transform = object()
        self._source_astrometric_model = object()
        self._reference_alignment_error_message = "旧参考错误"
        self._sky_alignment_error_message = "旧配准错误"
        self._source_model_error_message = "旧模型错误"
        self._current_star_map = object()
        self._current_reference_star_map = object()
        self._clear_annotations_called = False
        self._leave_pick_called = False
        self._refresh_reference_called = False
        self._update_alignment_called = False

    def _star_pair_position_count(self) -> int:
        return len(self._star_pair_store)

    def _clear_star_pair_annotations(self) -> None:
        self._clear_annotations_called = True

    def _leave_star_pick_mode(self) -> None:
        self._leave_pick_called = True
        self._active_star_pair_row = None

    def _refresh_reference_stars_from_current_map(self) -> None:
        self._refresh_reference_called = True

    def _update_reference_alignment_transform(self) -> None:
        self._update_alignment_called = True


def test_new_input_reset_clears_unmatched_reference_state() -> None:
    harness = _Harness()
    harness.ui.tableWidgetStarPairs.setRowCount(5)

    cleared_count = harness._clear_star_pair_positions_for_new_input("新的真实图像")

    assert cleared_count == 0
    assert harness.ui.tableWidgetStarPairs.rowCount() == 0
    assert harness._manual_reference_star_ids == []
    assert harness._imported_reference_star_by_id == {}
    assert harness._auto_match_reference_star_ids == []
    assert harness._auto_match_group_order == []
    assert harness._auto_match_group_by_star_id == {}
    assert harness._auto_match_group_expanded_by_id == {}
    assert harness._auto_match_next_group_index == 0
    assert harness._excluded_reference_star_ids == []
    assert harness._manual_match_group_expanded is True
    assert harness._sky_alignment_transform is None
    assert harness._source_astrometric_model is None
    assert harness._reference_alignment_error_message == ""
    assert harness._sky_alignment_error_message == ""
    assert harness._source_model_error_message == ""
    assert harness._current_reference_star_map is harness._current_star_map
    assert harness._clear_annotations_called is True
    assert harness._leave_pick_called is True
    assert harness._refresh_reference_called is True
    assert harness._update_alignment_called is True
