from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QWidget

from ..ui.ui_star_pair_assistant_dialog import Ui_StarPairAssistantDialog


STAR_PAIR_ASSISTANT_UI_NAMES = (
    "checkBoxStarPairAssistantAlwaysOnTop",
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
        self.ui.checkBoxStarPairAssistantAlwaysOnTop.toggled.connect(self.set_always_on_top)

    def set_always_on_top(self, enabled: bool) -> None:
        """切换普通顶层窗口的置顶标志，并避免可见窗口重新显示。"""

        enabled = bool(enabled)
        checkbox = self.ui.checkBoxStarPairAssistantAlwaysOnTop
        if checkbox.isChecked() != enabled:
            signals_were_blocked = checkbox.blockSignals(True)
            checkbox.setChecked(enabled)
            checkbox.blockSignals(signals_were_blocked)
        current_enabled = bool(self.windowFlags() & Qt.WindowStaysOnTopHint)
        if current_enabled == enabled:
            return
        if self.isVisible():
            window_handle = self.windowHandle()
            if window_handle is not None:
                # QWidget.setWindowFlag() 会隐藏已显示的顶层窗口，随后重新 show
                # 会产生明显闪烁。直接更新原生窗口句柄可保持窗口持续可见。
                updated_flags = self.windowFlags()
                if enabled:
                    updated_flags |= Qt.WindowStaysOnTopHint
                else:
                    updated_flags &= ~Qt.WindowStaysOnTopHint
                window_handle.setFlag(Qt.WindowStaysOnTopHint, enabled)
                # 只同步 QWidget 保存的标志，不再通知窗口系统或重建窗口。
                self.overrideWindowFlags(updated_flags)
                return

        # 窗口尚未显示时不会发生闪烁，保留标准方式以保证首次显示生效。
        self.setWindowFlag(Qt.WindowStaysOnTopHint, enabled)

    def always_on_top(self) -> bool:
        """返回助手窗口当前是否启用了固定前端显示。"""

        return bool(self.windowFlags() & Qt.WindowStaysOnTopHint)

    def bind_controls_to(self, target_ui: object) -> None:
        """把迁出的控件接口挂到主 UI，供现有业务模块继续共享同一组控件。"""

        for name in STAR_PAIR_ASSISTANT_UI_NAMES:
            setattr(target_ui, name, getattr(self.ui, name))


__all__ = ["StarPairAssistantDialog"]
