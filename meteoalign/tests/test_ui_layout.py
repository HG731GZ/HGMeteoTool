"""跨平台 UI 布局回归测试。"""

from __future__ import annotations

import os
from types import SimpleNamespace

# 让无显示服务器的 CI 也能创建 Qt 窗口；用户桌面运行不受此测试环境变量影响。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFormLayout, QHeaderView, QMainWindow, QPushButton, QSizePolicy, QTableWidget

from meteoalign.application.app_sequence_table_preview import SequenceTablePreviewMixin
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


def test_sequence_refinement_controls_and_columns_are_available() -> None:
    """序列页应提供两种单帧精修方式及三类 RMS 列。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.show()
    app.processEvents()

    assert not ui.comboBoxSequenceRefinementMode.isEnabled()
    assert not ui.pushButtonRefineSequenceFrames.isEnabled()
    assert ui.comboBoxSequenceRefinementMode.itemText(0) == "优化取景角度"
    assert ui.comboBoxSequenceRefinementMode.itemText(1) == "单帧重新拟合"
    assert ui.tableWidgetImageSequence.columnCount() == 5
    assert ui.tableWidgetImageSequence.horizontalHeaderItem(2).text() == "δt RMS"
    assert ui.tableWidgetImageSequence.horizontalHeaderItem(3).text() == "δt + pose RMS"
    assert ui.tableWidgetImageSequence.horizontalHeaderItem(4).text() == "重拟合RMS"

    window.close()


def test_sequence_table_columns_can_be_resized_by_user() -> None:
    """序列表应提供足够宽的文件名列，且不在刷新时覆盖用户调整。"""

    app = QApplication.instance() or QApplication([])
    table = QTableWidget(0, 5)

    class SequenceTable(SequenceTablePreviewMixin):
        def __init__(self) -> None:
            self.ui = SimpleNamespace(tableWidgetImageSequence=table)
            self._image_sequence_sort_key = "index"
            self._image_sequence_sort_descending = False

    sequence_table = SequenceTable()
    sequence_table._configure_image_sequence_table_columns()
    header = table.horizontalHeader()

    for column in range(table.columnCount()):
        assert header.sectionResizeMode(column) == QHeaderView.Interactive
    expected_name_width = table.fontMetrics().horizontalAdvance("abcdefghijklmnopqrstuvwxy") + 16
    assert table.columnWidth(1) == expected_name_width

    table.setColumnWidth(1, 420)
    sequence_table._refresh_image_sequence_table()
    assert table.columnWidth(1) == 420
