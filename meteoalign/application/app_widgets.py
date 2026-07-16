from __future__ import annotations

from html import escape

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QAbstractSlider, QAbstractSpinBox, QComboBox, QLabel, QWidget

from ..config import StarMapUiConfig


class AppWidgetMixin:
    """提供 UI 控件字体、标签省略号等辅助方法。"""

    ui: object  # 由 MainWindow 提供

    def _apply_ui_font_config(self, ui_config: StarMapUiConfig) -> None:
        """根据配置设置全局控件字体和状态栏字体。"""
        controls_font = QFont(self.font())
        controls_font.setPointSize(ui_config.controls_font_size_pt)
        self.setFont(controls_font)
        self.ui.centralwidget.setFont(controls_font)
        star_pair_assistant = getattr(self, "star_pair_assistant", None)
        if star_pair_assistant is not None:
            star_pair_assistant.setFont(controls_font)

        statusbar = getattr(self.ui, "statusbar", None)
        if statusbar is None:
            statusbar = self.statusBar()
            statusbar.setObjectName("statusbar")
            self.ui.statusbar = statusbar

        status_font = QFont(statusbar.font())
        status_font.setPointSize(ui_config.status_bar_font_size_pt)
        statusbar.setFont(status_font)

        image_context_label = getattr(self.ui, "labelStatusImageContext", None)
        if image_context_label is not None and not bool(statusbar.property("imageContextWidgetAdded")):
            # 控件固定在状态栏右侧，具体可见性由当前主选项卡控制。
            statusbar.addPermanentWidget(image_context_label)
            statusbar.setProperty("imageContextWidgetAdded", True)

        self._install_value_control_wheel_filters()

    def _install_value_control_wheel_filters(self) -> None:
        """拦截所有会因滚轮改变值的控件，避免误改参数。"""

        roots: list[QWidget] = [self]
        star_pair_assistant = getattr(self, "star_pair_assistant", None)
        if star_pair_assistant is not None:
            roots.append(star_pair_assistant)
        for root in roots:
            value_controls = (
                *root.findChildren(QAbstractSpinBox),
                *root.findChildren(QComboBox),
                *root.findChildren(QAbstractSlider),
            )
            for control in value_controls:
                control.installEventFilter(self)

    @staticmethod
    def _is_wheel_value_control(widget: object) -> bool:
        """判断控件是否会默认将滚轮解释为数值或选项变更。"""

        inherits = getattr(widget, "inherits", None)
        if not callable(inherits):
            return False
        # PyQt 的部分包装对象在 isinstance 检查时可能递归调用 Python 元类型，必须使用 Qt 元对象判断。
        return any(
            bool(inherits(class_name))
            for class_name in ("QAbstractSpinBox", "QComboBox", "QAbstractSlider")
        )

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
        suffix_plain = str(label.property("richSuffixPlain") or "")
        suffix_html = str(label.property("richSuffixHtml") or "")
        available_width = max(12, label.contentsRect().width() - 2)
        if suffix_plain and suffix_html:
            suffix_width = label.fontMetrics().horizontalAdvance(suffix_plain)
            prefix_width = max(12, available_width - suffix_width)
            prefix_text = label.fontMetrics().elidedText(full_text, Qt.ElideRight, prefix_width)
            label.setText(f"{escape(prefix_text)}{suffix_html}")
            return
        label.setText(label.fontMetrics().elidedText(full_text, Qt.ElideRight, available_width))

    def _set_elided_label_text(self, label: QLabel, text: str, tooltip: str | None = None) -> None:
        """设置带省略号的标签文本，超宽时自动截断。"""
        full_text = text.strip()
        label.setTextFormat(Qt.PlainText)
        label.setProperty("fullText", full_text)
        label.setProperty("richSuffixPlain", "")
        label.setProperty("richSuffixHtml", "")
        label.setToolTip((tooltip or full_text).strip())
        self._refresh_elided_label(label)
        QTimer.singleShot(0, lambda label=label: self._refresh_elided_label(label))

    def _set_elided_label_text_with_html_suffix(
        self,
        label: QLabel,
        text: str,
        suffix_plain: str,
        suffix_html: str,
        tooltip: str | None = None,
    ) -> None:
        """设置带 HTML 后缀的省略标签，后缀尽量保持可见。"""
        full_text = text.strip()
        label.setTextFormat(Qt.RichText)
        label.setProperty("fullText", full_text)
        label.setProperty("richSuffixPlain", suffix_plain.strip())
        label.setProperty("richSuffixHtml", suffix_html.strip())
        label.setToolTip((tooltip or f"{full_text}{suffix_plain}").strip())
        self._refresh_elided_label(label)
        QTimer.singleShot(0, lambda label=label: self._refresh_elided_label(label))

    def _refresh_all_elided_labels(self) -> None:
        """刷新所有带省略号标签的显示。"""
        self._refresh_elided_label(self.ui.labelImportedImagePath)
        if hasattr(self.ui, "labelAdjacentImageModel"):
            self._refresh_elided_label(self.ui.labelAdjacentImageModel)
        if hasattr(self.ui, "labelAdjacentFramingStatus"):
            self._refresh_elided_label(self.ui.labelAdjacentFramingStatus)
        if hasattr(self.ui, "labelImageSequenceStatus"):
            self._refresh_elided_label(self.ui.labelImageSequenceStatus)
        if hasattr(self.ui, "labelImageSequenceSummary"):
            self._refresh_elided_label(self.ui.labelImageSequenceSummary)
        if hasattr(self.ui, "labelImageSequencePreviewTitle"):
            self._refresh_elided_label(self.ui.labelImageSequencePreviewTitle)
        self._refresh_elided_label(self.ui.labelSkyMaskStatus)
        self._refresh_elided_label(self.ui.labelAlignmentTransformStatus)
