"""跨平台多文件对话框与快捷键兼容。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAbstractItemView, QFileDialog, QFileSystemModel, QShortcut


_SUFFIX_PATTERN = re.compile(r"\*\.([A-Za-z0-9]+)")


def supported_suffixes_from_name_filter(file_filter: str) -> frozenset[str]:
    """从 Qt 名称过滤器提取业务允许的文件后缀。"""

    return frozenset(f".{match.group(1).casefold()}" for match in _SUFFIX_PATTERN.finditer(file_filter))


def _supported_selected_paths(paths: list[str], file_filter: str) -> list[str]:
    """再次过滤对话框结果，避免平台差异或手工输入绕过文件类型限制。"""

    allowed_suffixes = supported_suffixes_from_name_filter(file_filter)
    if not allowed_suffixes:
        return list(paths)
    return [path for path in paths if Path(path).suffix.casefold() in allowed_suffixes]


def multiple_file_dialog_options(
    *,
    hide_name_filter_details: bool = False,
    platform_name: str | None = None,
) -> QFileDialog.Options:
    """返回多文件选择选项；macOS 使用可控的 Qt 对话框。"""

    options = QFileDialog.Options()
    if hide_name_filter_details:
        options |= QFileDialog.HideNameFilterDetails
    if (platform_name or sys.platform) == "darwin":
        options |= QFileDialog.DontUseNativeDialog
    return options


def _select_all_file_dialog_items(dialog: QFileDialog) -> None:
    """全选对话框当前文件视图，不影响侧边栏和文件类型列表。"""

    for view in dialog.findChildren(QAbstractItemView):
        if not view.isVisible() or view.objectName() not in {"listView", "treeView"}:
            continue
        if view.selectionMode() not in (
            QAbstractItemView.MultiSelection,
            QAbstractItemView.ExtendedSelection,
        ):
            continue
        view.selectAll()


def get_multiple_open_file_names(
    parent,
    caption: str,
    directory: str,
    file_filter: str,
    *,
    hide_name_filter_details: bool = False,
    platform_name: str | None = None,
) -> tuple[list[str], str]:
    """显示多文件对话框；macOS Qt 对话框显式支持 Command+A。"""

    resolved_platform = platform_name or sys.platform
    options = multiple_file_dialog_options(
        hide_name_filter_details=hide_name_filter_details,
        platform_name=resolved_platform,
    )
    if resolved_platform != "darwin":
        selected_paths, selected_filter = QFileDialog.getOpenFileNames(
            parent,
            caption,
            directory,
            file_filter,
            options=options,
        )
        return _supported_selected_paths(selected_paths, file_filter), selected_filter

    dialog = QFileDialog(parent, caption, directory, file_filter)
    dialog.setOptions(options)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setFileMode(QFileDialog.ExistingFiles)
    file_system_model = dialog.findChild(QFileSystemModel)
    if file_system_model is not None:
        # QFileSystemModel 默认只把不匹配项禁用；这里改为直接隐藏，目录仍保持可导航。
        file_system_model.setNameFilterDisables(False)
    select_all_shortcut = QShortcut(QKeySequence("Meta+A"), dialog)
    select_all_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
    select_all_shortcut.activated.connect(lambda: _select_all_file_dialog_items(dialog))

    if dialog.exec_() != QFileDialog.Accepted:
        return [], dialog.selectedNameFilter()
    return (
        _supported_selected_paths(dialog.selectedFiles(), file_filter),
        dialog.selectedNameFilter(),
    )


__all__ = [
    "get_multiple_open_file_names",
    "multiple_file_dialog_options",
    "supported_suffixes_from_name_filter",
]
