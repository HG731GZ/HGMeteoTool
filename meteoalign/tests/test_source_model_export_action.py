"""导出映射时同步保存星点匹配 JSON 的动作测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from meteoalign.application.app_source_model_export import SourceModelExportMixin


class _StatusBar:
    """记录最后一次状态栏消息。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.messages.append(message)


class _ExportHarness(SourceModelExportMixin):
    """只提供组合导出动作所需的最小状态。"""

    def __init__(self, model_path: Path, star_pair_path: Path) -> None:
        self.current_image_preview = object()
        self.ui = SimpleNamespace(statusbar=_StatusBar())
        self.model_path = model_path
        self.star_pair_path = star_pair_path
        self.calls: list[str] = []

    def _write_current_source_model(self):  # type: ignore[no-untyped-def]
        self.calls.append("model")
        return self.model_path, 8, 0.42, True

    def _write_current_star_pair_session(self):  # type: ignore[no-untyped-def]
        self.calls.append("star_pairs")
        return self.star_pair_path, 8


def test_export_mapping_also_saves_star_pair_json(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """点击导出映射后应顺带覆盖保存默认同名的星点匹配 JSON。"""

    harness = _ExportHarness(tmp_path / "frame_model.json", tmp_path / "frame_starpairs.json")
    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "meteoalign.application.app_source_model_export.QMessageBox.information",
        lambda _parent, title, message: dialogs.append((title, message)),
    )

    harness.export_source_model_json()

    assert harness.calls == ["model", "star_pairs"]
    assert "已导出映射并保存星点匹配 JSON" in harness.ui.statusbar.messages[-1]
    assert dialogs[0][0] == "映射与匹配 JSON 已导出"
    assert str(harness.model_path) in dialogs[0][1]
    assert str(harness.star_pair_path) in dialogs[0][1]


def test_export_mapping_reports_star_pair_save_failure_separately(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """映射已写入后若匹配保存失败，不应把已经成功的映射误报为失败。"""

    harness = _ExportHarness(tmp_path / "frame_model.json", tmp_path / "frame_starpairs.json")

    def fail_star_pair_save():  # type: ignore[no-untyped-def]
        raise OSError("测试写入失败")

    harness._write_current_star_pair_session = fail_star_pair_save
    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "meteoalign.application.app_source_model_export.QMessageBox.critical",
        lambda _parent, title, message: dialogs.append((title, message)),
    )

    harness.export_source_model_json()

    assert harness.calls == ["model"]
    assert "映射 JSON 已导出，但星点匹配 JSON 保存失败" in harness.ui.statusbar.messages[-1]
    assert dialogs[0][0] == "星点匹配 JSON 保存失败"
    assert str(harness.model_path) in dialogs[0][1]
