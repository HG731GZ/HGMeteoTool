from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication, QComboBox, QMainWindow, QScrollArea, QSpinBox, QVBoxLayout, QWidget

from meteoalign.application.app_view_controls import ViewControlsMixin
from meteoalign.application.app_widgets import AppWidgetMixin


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _QT_APP = app
    return app


class _WheelFilterHarness(QMainWindow, AppWidgetMixin, ViewControlsMixin):
    """验证参数控件滚轮被拦截并转交外层滚动区。"""

    def __init__(self) -> None:
        super().__init__()
        self.ui = SimpleNamespace()
        self.scroll_area = QScrollArea(self)
        content = QWidget()
        content.setMinimumHeight(800)
        layout = QVBoxLayout(content)
        self.spin_box = QSpinBox(content)
        self.spin_box.setRange(0, 100)
        self.spin_box.setValue(50)
        self.combo_box = QComboBox(content)
        self.combo_box.addItems(["A", "B", "C"])
        layout.addWidget(self.spin_box)
        layout.addWidget(self.combo_box)
        self.scroll_area.setWidget(content)
        self.scroll_area.setWidgetResizable(False)
        self.setCentralWidget(self.scroll_area)
        self._install_value_control_wheel_filters()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        return ViewControlsMixin.eventFilter(self, watched, event)


def _wheel_event(widget: QWidget) -> QWheelEvent:
    global_pos = widget.mapToGlobal(widget.rect().center())
    return QWheelEvent(
        QPointF(widget.rect().center()),
        QPointF(global_pos),
        QPoint(),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )


def test_wheel_does_not_change_spinbox_or_combobox_value() -> None:
    """鼠标位于参数控件上滚动时，不得改动其数值或当前选项。"""

    app = _qapp()
    harness = _WheelFilterHarness()
    harness.resize(240, 180)
    harness.show()
    app.processEvents()

    scrollbar = harness.scroll_area.verticalScrollBar()
    scrollbar.setValue(0)
    QApplication.sendEvent(harness.spin_box, _wheel_event(harness.spin_box))
    assert scrollbar.value() > 0
    QApplication.sendEvent(harness.combo_box, _wheel_event(harness.combo_box))
    app.processEvents()

    assert harness.spin_box.value() == 50
    assert harness.combo_box.currentIndex() == 0


def test_wheel_value_control_detection_uses_qt_meta_object_without_instancecheck() -> None:
    """Qt 控件类型判断不得触发 PyQt 包装对象的递归 isinstance。"""

    harness = _WheelFilterHarness()

    assert harness._is_wheel_value_control(harness.spin_box)
    assert harness._is_wheel_value_control(harness.combo_box)
    assert not harness._is_wheel_value_control(harness.scroll_area)
