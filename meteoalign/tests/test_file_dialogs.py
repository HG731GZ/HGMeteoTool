"""跨平台文件对话框回归测试。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog, QTreeView

from meteoalign.application.file_dialogs import (
    get_multiple_open_file_names,
    multiple_file_dialog_options,
    supported_suffixes_from_name_filter,
)


_APPLICATION: QApplication | None = None


def _application() -> QApplication:
    global _APPLICATION
    _APPLICATION = QApplication.instance() or QApplication([])
    return _APPLICATION


def test_macos_multiple_file_dialog_uses_qt_dialog() -> None:
    options = multiple_file_dialog_options(platform_name="darwin")

    assert options & QFileDialog.DontUseNativeDialog


def test_long_filter_details_can_be_hidden_independently() -> None:
    options = multiple_file_dialog_options(
        hide_name_filter_details=True,
        platform_name="win32",
    )

    assert options & QFileDialog.HideNameFilterDetails
    assert not options & QFileDialog.DontUseNativeDialog


def test_windows_keeps_native_multiple_file_dialog() -> None:
    options = multiple_file_dialog_options(platform_name="win32")

    assert not options & QFileDialog.DontUseNativeDialog


def test_name_filter_suffixes_are_extracted_case_insensitively() -> None:
    suffixes = supported_suffixes_from_name_filter(
        "图像 (*.tif *.TIFF *.jpg);;所有文件 (*)"
    )

    assert suffixes == frozenset({".tif", ".tiff", ".jpg"})


def test_dialog_result_drops_unsupported_paths_on_native_platform(monkeypatch) -> None:
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: (
            ["first.TIF", "metadata.json", "second.JPG", "notes.txt"],
            "图像文件 (*.tif *.jpg)",
        ),
    )

    paths, _selected_filter = get_multiple_open_file_names(
        None,
        "导入图像",
        "",
        "图像文件 (*.tif *.jpg)",
        platform_name="win32",
    )

    assert paths == ["first.TIF", "second.JPG"]


def test_macos_command_a_selects_every_visible_file(monkeypatch) -> None:
    """Qt 多文件对话框应显式把 Meta+A 映射到文件视图全选。"""

    app = _application()
    selected_row_counts: list[int] = []

    def fake_exec(dialog: QFileDialog) -> int:
        dialog.show()
        app.processEvents()
        view = dialog.findChild(QTreeView, "treeView")
        model = QStandardItemModel(view)
        for name in ("first.ARW", "second.ARW", "third.ARW"):
            model.appendRow(QStandardItem(name))
        view.setModel(model)
        view.setSelectionMode(QTreeView.ExtendedSelection)
        view.setCurrentIndex(model.index(0, 0))
        view.setFocus()
        QTest.keyClick(view, Qt.Key_A, Qt.MetaModifier)
        app.processEvents()
        selected_row_counts.append(len(view.selectionModel().selectedRows()))
        return QFileDialog.Rejected

    monkeypatch.setattr(QFileDialog, "exec_", fake_exec)

    get_multiple_open_file_names(
        None,
        "导入流星图片",
        "",
        "RAW (*.ARW)",
        hide_name_filter_details=True,
        platform_name="darwin",
    )

    assert selected_row_counts == [3]
