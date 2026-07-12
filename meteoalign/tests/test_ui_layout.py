"""跨平台 UI 布局回归测试。"""

from __future__ import annotations

import os

# 让无显示服务器的 CI 也能创建 Qt 窗口；用户桌面运行不受此测试环境变量影响。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFormLayout, QMainWindow, QPushButton, QSizePolicy

from meteoalign.ui.ui_main_window import Ui_MainWindow


DYNAMIC_INFORMATION_LABELS = (
    "labelImageSequenceSummary",
    "labelImageSequencePreviewTitle",
    "labelImportedImagePath",
    "labelImageSequenceStatus",
    "labelImportedCameraProfile",
    "labelSkyMaskStatus",
    "labelAdjacentImageModel",
    "labelAdjacentFramingStatus",
    "labelAlignmentTransformStatus",
    "labelMosaicModelPath",
    "labelMosaicSourceImage",
    "labelMosaicModelInfo",
    "labelMosaicViewInfo",
    "labelMosaicBatchFramingPath",
)

FORM_LAYOUTS = (
    "formLayoutObserver",
    "formLayoutCamera",
    "formLayoutView",
    "formLayoutReference",
    "formLayoutImageSequenceStatus",
    "formLayoutImportedImage",
    "formLayoutCameraProfileReuse",
    "formLayoutAdjacentImageFraming",
    "formLayoutAutoMatch",
    "formLayoutMosaicSourceModel",
    "formLayoutMosaicObserver",
    "formLayoutMosaicProjection",
    "formLayoutMosaicOutputSize",
    "formLayoutMosaicCrop",
    "formLayoutMosaicBatchSettings",
)


def test_dynamic_information_labels_are_not_collapsed_by_layout() -> None:
    """动态信息标签必须保留宽度，避免在 macOS 上不可见。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.resize(1280, 820)
    window.show()
    app.processEvents()

    for name in DYNAMIC_INFORMATION_LABELS:
        label = getattr(ui, name)
        assert label.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
        assert label.width() > 0, f"{name} was collapsed by its layout"

    for name in FORM_LAYOUTS:
        layout = getattr(ui, name)
        assert layout.fieldGrowthPolicy() == QFormLayout.AllNonFixedFieldsGrow
        assert layout.labelAlignment() == Qt.AlignLeft | Qt.AlignVCenter

    ui.tabWidgetMain.setCurrentWidget(ui.tabReferenceImage)
    app.processEvents()
    title_label = ui.formLayoutImportedImage.itemAt(0, QFormLayout.LabelRole).widget()
    value_label = ui.formLayoutImportedImage.itemAt(0, QFormLayout.FieldRole).widget()
    assert title_label.x() < value_label.x()
    assert value_label.width() > title_label.width()

    window.close()


def test_adjacent_alignment_settings_button_is_right_of_calculation_button() -> None:
    """粗略取景参数齿轮必须紧随计算按钮，且保留无障碍名称。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.show()
    app.processEvents()

    layout = ui.horizontalLayoutAdjacentImageFramingButtons
    assert layout.indexOf(ui.pushButtonCalculateAdjacentFraming) < layout.indexOf(
        ui.toolButtonAdjacentAlignmentSettings
    )
    assert ui.toolButtonAdjacentAlignmentSettings.text() == "设定"
    assert ui.toolButtonAdjacentAlignmentSettings.minimumSize().width() == 56
    assert ui.toolButtonAdjacentAlignmentSettings.maximumSize().width() == 56
    assert ui.toolButtonAdjacentAlignmentSettings.height() == ui.pushButtonCalculateAdjacentFraming.height()
    assert ui.toolButtonAdjacentAlignmentSettings.accessibleName() == "粗略取景参数设置"
    assert isinstance(ui.toolButtonAdjacentAlignmentSettings, QPushButton)

    window.close()
