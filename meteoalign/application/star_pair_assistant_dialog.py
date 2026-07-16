from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QWidget

from ..ui.ui_star_pair_assistant_dialog import Ui_StarPairAssistantDialog


STAR_PAIR_ASSISTANT_UI_NAMES = (
    "groupBoxStarPairs",
    "verticalLayoutStarPairs",
    "tableWidgetStarPairs",
    "horizontalLayoutStarPairSessionButtons",
    "pushButtonExportStarPairs",
    "pushButtonImportStarPairs",
    "horizontalLayoutStarPairResetButtons",
    "pushButtonDeleteStarPairs",
    "pushButtonClearStarPairs",
    "groupBoxAutoMatch",
    "verticalLayoutAutoMatch",
    "formLayoutAutoMatch",
    "labelAutoMatchCount",
    "spinBoxAutoMatchCount",
    "labelAutoMatchConstraintMode",
    "comboBoxAutoMatchConstraintMode",
    "labelAutoMatchSoftWeight",
    "doubleSpinBoxAutoMatchSoftWeight",
    "labelAutoMatchRadius",
    "spinBoxAutoMatchRadius",
    "pushButtonAutoMatchFieldStars",
)


class StarPairAssistantDialog(QDialog):
    """承载星点匹配列表和自动扩展工具的单实例非模态窗口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # 使用普通顶层窗口，确保主窗口激活后可以按正常层级挡住本窗口。
        window_flags = (self.windowFlags() & ~Qt.WindowType_Mask) | Qt.Window
        self.setWindowFlags(window_flags)
        self.ui = Ui_StarPairAssistantDialog()
        self.ui.setupUi(self)
        self.setModal(False)

    def bind_controls_to(self, target_ui: object) -> None:
        """把迁出的控件接口挂到主 UI，供现有业务模块继续共享同一组控件。"""

        for name in STAR_PAIR_ASSISTANT_UI_NAMES:
            setattr(target_ui, name, getattr(self.ui, name))


__all__ = ["StarPairAssistantDialog"]
