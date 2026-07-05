from __future__ import annotations

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel

from .config import StarMapUiConfig


class AppWidgetMixin:
    """提供 UI 控件字体、标签省略号等辅助方法。"""

    ui: object  # 由 MainWindow 提供

    def _apply_ui_font_config(self, ui_config: StarMapUiConfig) -> None:
        """根据配置设置全局控件字体和状态栏字体。"""
        controls_font = QFont(self.font())
        controls_font.setPointSize(ui_config.controls_font_size_pt)
        self.setFont(controls_font)
        self.ui.centralwidget.setFont(controls_font)

        status_font = QFont(self.ui.statusbar.font())
        status_font.setPointSize(ui_config.status_bar_font_size_pt)
        self.ui.statusbar.setFont(status_font)

    def _set_plain_label_text(self, label: QLabel, text: str, tooltip: str | None = None) -> None:
        """设置标签纯文本，并同步设置 tooltip。"""
        display_text = text.strip()
        label.setText(display_text)
        label.setToolTip((tooltip or display_text).strip())

    def _refresh_elided_label(self, label: QLabel) -> None:
        """根据标签当前宽度刷新省略号显示。"""
        full_text = str(label.property("fullText") or "")
        if not full_text:
            return
        available_width = max(12, label.contentsRect().width() - 2)
        label.setText(label.fontMetrics().elidedText(full_text, Qt.ElideRight, available_width))

    def _set_elided_label_text(self, label: QLabel, text: str, tooltip: str | None = None) -> None:
        """设置带省略号的标签文本，超宽时自动截断。"""
        full_text = text.strip()
        label.setProperty("fullText", full_text)
        label.setToolTip((tooltip or full_text).strip())
        self._refresh_elided_label(label)
        QTimer.singleShot(0, lambda label=label: self._refresh_elided_label(label))

    def _refresh_all_elided_labels(self) -> None:
        """刷新所有带省略号标签的显示。"""
        self._refresh_elided_label(self.ui.labelImportedImagePath)
        self._refresh_elided_label(self.ui.labelSkyMaskStatus)
        self._refresh_elided_label(self.ui.labelAlignmentTransformStatus)
