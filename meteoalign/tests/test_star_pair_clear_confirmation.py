from __future__ import annotations

from types import SimpleNamespace

from PyQt5.QtWidgets import QMessageBox

from meteoalign.application.app_star_pair_actions import StarPairActionsMixin


class _StatusBar:
    """记录状态提示的最小状态栏替身。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.messages.append(message)


class _ClearAllHarness(StarPairActionsMixin):
    """隔离验证清除全部匹配前的确认流程。"""

    def __init__(self, pair_count: int) -> None:
        self.ui = SimpleNamespace(statusbar=_StatusBar())
        self.pair_count = pair_count
        self.clear_called = False

    def _star_pair_position_count(self) -> int:
        return self.pair_count

    def _clear_star_pair_positions(self) -> int:
        self.clear_called = True
        return self.pair_count


def test_clear_all_star_pairs_does_not_modify_data_when_user_cancels(monkeypatch) -> None:
    """用户取消确认后不得执行任何清除操作。"""

    harness = _ClearAllHarness(pair_count=5)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    harness.clear_all_star_pair_positions()

    assert not harness.clear_called
    assert harness.ui.statusbar.messages[-1] == "已取消清除所有星点匹配。"


def test_clear_all_star_pairs_clears_data_after_user_confirms(monkeypatch) -> None:
    """用户确认后才执行清除操作。"""

    harness = _ClearAllHarness(pair_count=5)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    harness.clear_all_star_pair_positions()

    assert harness.clear_called
    assert harness.ui.statusbar.messages[-1] == "已清除 5 个星点匹配。"
