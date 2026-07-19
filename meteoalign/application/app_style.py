from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QApplication

from ..runtime_paths import runtime_stylesheet_path
from ..ui import ui_resources as _ui_resources  # noqa: F401  # 注册 QSS 使用的矢量资源


def apply_application_stylesheet(
    app: QApplication,
    stylesheet_path: Path | None = None,
) -> Path:
    """加载 Win/macOS 共用的轻量界面样式，并返回实际资源路径。"""

    path = stylesheet_path or runtime_stylesheet_path()
    app.setStyleSheet(path.read_text(encoding="utf-8"))
    return path
