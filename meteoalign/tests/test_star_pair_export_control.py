"""星点匹配 JSON 导出按钮状态测试。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QPushButton

from meteoalign.application.app_star_pair_session import StarPairSessionMixin


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _QT_APP = app
    return app


class _Harness(StarPairSessionMixin):
    """只提供导出按钮状态计算所需的最小对象。"""

    def __init__(self, pair_count: int) -> None:
        _qapp()
        self.ui = type("Ui", (), {"pushButtonExportStarPairs": QPushButton()})()
        self._pair_count = pair_count
        self._json_import_thread = None

    def _star_pair_position_count(self) -> int:
        return self._pair_count


@pytest.mark.parametrize(
    ("pair_count", "expected_enabled"),
    ((0, False), (1, False), (2, False), (3, False), (4, True), (5, True)),
)
def test_export_button_requires_four_matches(pair_count: int, expected_enabled: bool) -> None:
    """0 至 3 对匹配必须禁用，达到 4 对后才可导出。"""

    harness = _Harness(pair_count)
    harness._update_star_pair_export_control()

    assert harness.ui.pushButtonExportStarPairs.isEnabled() is expected_enabled


def test_export_button_stays_disabled_while_json_controls_are_busy() -> None:
    """即使已有四对匹配，JSON 导入期间也不得重新启用导出按钮。"""

    harness = _Harness(4)
    harness._json_import_thread = object()
    harness._update_star_pair_export_control()

    assert not harness.ui.pushButtonExportStarPairs.isEnabled()

    harness._json_import_thread = None
    harness._update_star_pair_export_control(controls_enabled=False)

    assert not harness.ui.pushButtonExportStarPairs.isEnabled()
