from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from ..runtime_paths import runtime_stylesheet_path
from ..ui import ui_resources as _ui_resources  # noqa: F401  # 注册 QSS 使用的矢量资源


def apply_application_stylesheet(
    app: QApplication,
    stylesheet_path: Path | None = None,
) -> Path | None:
    """Apply the custom UI stylesheet on macOS only.

    Windows and Linux deliberately keep Qt's native platform style.
    """

    if sys.platform != "darwin":
        return None

    path = stylesheet_path or runtime_stylesheet_path()
    app.setStyleSheet(path.read_text(encoding="utf-8"))
    return path
