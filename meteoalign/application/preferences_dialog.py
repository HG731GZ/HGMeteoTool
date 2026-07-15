from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QWidget

from ..ui.ui_preferences_dialog import Ui_PreferencesDialog
from ..ui.ui_preferences_launcher import Ui_PreferencesLauncher
from .preferences_page import PreferencesPage


class PreferencesDialog(QDialog):
    """承载软件参数页面的单实例非模态弹窗。"""

    def __init__(self, parent: QWidget | None = None, *, preference_path: Path | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_PreferencesDialog()
        self.ui.setupUi(self)
        self.setModal(False)
        self.preferences_page = PreferencesPage(self, preference_path=preference_path)
        self.ui.verticalLayoutPreferencesContainer.addWidget(self.preferences_page)
        self.preferences_page.close_requested.connect(self.close)


class PreferencesLauncher(QWidget):
    """放置在主标签栏右上角的软件选项入口。"""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_PreferencesLauncher()
        self.ui.setupUi(self)
        self.ui.pushButtonOpenPreferences.clicked.connect(self.clicked.emit)


__all__ = ["PreferencesDialog", "PreferencesLauncher"]
