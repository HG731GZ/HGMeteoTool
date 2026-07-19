from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QStatusBar


STATUS_MESSAGE_NORMAL = "normal"
STATUS_MESSAGE_ERROR = "error"
STATUS_MESSAGE_HINT = "hint"

_ERROR_MARKERS = (
    "已强制记录",
    "错误",
    "异常",
    "不可用",
    "无法",
    "未能",
    "中断",
    "拒绝",
    "缺少",
    "无效",
)
_HINT_MARKERS = (
    "请",
    "Ctrl+",
    "右键",
    "左键",
    "尚未",
    "至少需要",
    "当前没有",
    "无需",
    "不在",
    "暂不",
    "等待",
    "可以检查",
)
_FOCUS_HINT_MARKERS = (
    "双击聚焦",
    "无法聚焦",
    "可聚焦",
    "理论位置",
)
_QUOTED_ABSOLUTE_PATH = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:[\\/]|/)[^'\"]+)(?P=quote)"
)


def _path_name(path_text: str) -> str:
    """从 Windows 或 POSIX 路径中提取文件名。"""

    if re.match(r"^[A-Za-z]:[\\/]", path_text) or "\\" in path_text:
        return PureWindowsPath(path_text).name
    return PurePosixPath(path_text).name


def compact_import_status_paths(message: str) -> str:
    """把导入错误中带引号的绝对路径压缩为文件名。"""

    text = str(message)
    if not any(marker in text for marker in ("导入", "载入", "读取", "预览")):
        return text

    def replace_path(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{quote}{_path_name(match.group('path'))}{quote}"

    return _QUOTED_ABSOLUTE_PATH.sub(replace_path, text)


def classify_status_message(message: str) -> str:
    """根据状态文字判断普通、失败或操作提示。"""

    text = str(message).strip()
    if not text:
        return STATUS_MESSAGE_NORMAL
    # 聚焦流程中的坐标或前置条件问题用于指导用户，不作为处理失败。
    if any(marker in text for marker in _FOCUS_HINT_MARKERS):
        return STATUS_MESSAGE_HINT
    if any(marker in text for marker in _ERROR_MARKERS):
        return STATUS_MESSAGE_ERROR
    if re.search(r"失败(?:\s*[1-9]\d*\s*(?:张|个|项|次|颗)|\s*[：:]|[。；]?$)", text):
        return STATUS_MESSAGE_ERROR
    if any(marker in text for marker in _HINT_MARKERS):
        return STATUS_MESSAGE_HINT
    return STATUS_MESSAGE_NORMAL


class AppStatusBar(QStatusBar):
    """根据消息语义自动切换底色的主窗口状态栏。"""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._normal_palette = QPalette(self.palette())
        self.messageChanged.connect(self._apply_message_style)
        self._apply_message_style("")

    def _apply_message_style(self, message: str) -> None:
        """为当前左侧消息应用统一且跨平台的状态颜色。"""

        message_kind = classify_status_message(message)
        palette = QPalette(self._normal_palette)
        if message_kind == STATUS_MESSAGE_ERROR:
            palette.setColor(QPalette.Window, QColor("#f8d7da"))
            palette.setColor(QPalette.WindowText, QColor("#842029"))
        elif message_kind == STATUS_MESSAGE_HINT:
            palette.setColor(QPalette.Window, QColor("#fff3cd"))
            palette.setColor(QPalette.WindowText, QColor("#664d03"))
        self.setPalette(palette)
        self.setAutoFillBackground(message_kind != STATUS_MESSAGE_NORMAL)
        self.setProperty("messageKind", message_kind)

    def showMessage(self, message: str, timeout: int = 0) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        """显示状态消息，并隐藏导入错误中的完整文件路径。"""

        super().showMessage(compact_import_status_paths(message), timeout)
