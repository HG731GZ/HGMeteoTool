"""跨平台 UI 布局回归测试。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

# 让无显示服务器的 CI 也能创建 Qt 窗口；用户桌面运行不受此测试环境变量影响。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFormLayout,
    QHeaderView,
    QMainWindow,
    QSizePolicy,
    QTableWidget,
)

from meteoalign.application.app_sequence_table_preview import SequenceTablePreviewMixin
from meteoalign.application.app_mosaic import MosaicProjectionMixin, MosaicSourceItem
from meteoalign.application.app_rendering import RenderingMixin
from meteoalign.application.app_star_pair_table_groups import StarPairTableGroupsMixin
from meteoalign.application.main_window import MainWindow
from meteoalign.application.star_pair_assistant_dialog import StarPairAssistantDialog
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
    "formLayoutSourceModel",
    "formLayoutMosaicSourceModel",
    "formLayoutMosaicObserver",
    "formLayoutMosaicProjection",
    "formLayoutMosaicOutputSize",
    "formLayoutMosaicCrop",
    "formLayoutMosaicBatchSettings",
)


def test_status_image_context_is_visible_only_on_star_matching_tab() -> None:
    """状态栏右侧图像上下文不得出现在星点匹配以外的页面。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    class StatusContextHost(RenderingMixin):
        def __init__(self) -> None:
            self.ui = ui

        def fit_all_graphics_views(self) -> None:
            return

    host = StatusContextHost()

    ui.tabWidgetMain.setCurrentWidget(ui.tabSimulator)
    host._handle_tab_changed()
    assert ui.labelStatusImageContext.isHidden()

    ui.tabWidgetMain.setCurrentWidget(ui.tabReferenceImage)
    host._handle_tab_changed()
    assert not ui.labelStatusImageContext.isHidden()
    assert "background-color: #cfe2ff" in ui.labelStatusImageContext.styleSheet()

    ui.tabWidgetMain.setCurrentWidget(ui.tabMeteorSelection)
    host._handle_tab_changed()
    assert ui.labelStatusImageContext.isHidden()

    window.close()


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


def test_star_matching_side_panel_has_no_horizontal_scroll_range() -> None:
    """星点匹配左栏在较宽字体下也必须适配原宽度，并且只能纵向滚动。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    enlarged_font = QFont(ui.widgetReferencePickSidePanel.font())
    enlarged_font.setPointSize(15)
    ui.widgetReferencePickSidePanel.setFont(enlarged_font)
    window.resize(1280, 820)
    ui.tabWidgetMain.setCurrentWidget(ui.tabReferenceImage)
    window.show()
    app.processEvents()

    scroll_area = ui.scrollAreaReferencePickControls
    side_panel = ui.widgetReferencePickSidePanel
    horizontal_bar = scroll_area.horizontalScrollBar()
    assert scroll_area.minimumWidth() == 350
    assert scroll_area.maximumWidth() == 390
    assert scroll_area.width() <= 390
    assert scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert side_panel.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert side_panel.width() == scroll_area.viewport().width()
    assert horizontal_bar.minimum() == 0
    assert horizontal_bar.maximum() == 0
    horizontal_bar.setValue(100)
    assert horizontal_bar.value() == 0

    right_edge_controls = (
        ui.toolButtonAdjacentAlignmentSettings,
        ui.pushButtonOpenStarPairAssistant,
        ui.comboBoxSkyAlignmentModel,
        ui.pushButtonExportSourceModel,
    )
    for control in right_edge_controls:
        assert control.geometry().right() <= control.parentWidget().contentsRect().right()

    window.close()


def test_adjacent_framing_controls_are_compact_aligned_and_not_clipped() -> None:
    """参考图像控件应保持同行对齐，且窄侧栏中的长文本按钮不得被裁切。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    ui.scrollAreaReferencePickControls.setMaximumWidth(350)
    window.resize(1280, 820)
    ui.tabWidgetMain.setCurrentWidget(ui.tabReferenceImage)
    window.show()
    app.processEvents()

    assert ui.horizontalLayoutReferenceImage.spacing() == 6
    assert ui.verticalLayoutReferencePickSidePanel.spacing() == 6
    assert ui.verticalLayoutImportedImage.spacing() == 6
    assert ui.verticalLayoutAdjacentImageFraming.spacing() == 6
    assert ui.formLayoutAdjacentImageFraming.horizontalSpacing() == 6
    assert ui.formLayoutAdjacentImageFraming.verticalSpacing() == 6
    assert ui.formLayoutAdjacentImageFraming.rowWrapPolicy() == QFormLayout.DontWrapRows
    assert ui.horizontalLayoutAdjacentImageInfo.spacing() == 6
    assert ui.horizontalLayoutAdjacentImageFramingButtons.spacing() == 4
    assert ui.groupBoxImportedImage.height() >= ui.groupBoxImportedImage.minimumSizeHint().height()

    aligned_controls = (
        ui.labelAdjacentImageModelTitle,
        ui.labelAdjacentImageModel,
        ui.pushButtonPreviewAdjacentImage,
    )
    control_centers = [
        control.mapTo(ui.groupBoxAdjacentImageFraming, control.rect().center()).y()
        for control in aligned_controls
    ]
    assert max(control_centers) - min(control_centers) <= 1
    assert ui.labelAdjacentImageModel.width() >= ui.labelAdjacentImageModel.sizeHint().width()
    assert ui.pushButtonPreviewAdjacentImage.width() >= ui.pushButtonPreviewAdjacentImage.sizeHint().width()
    assert ui.pushButtonPreviewAdjacentImage.height() >= ui.pushButtonPreviewAdjacentImage.sizeHint().height()

    for button in (
        ui.pushButtonImportAdjacentImage,
        ui.pushButtonCalculateAdjacentFraming,
        ui.toolButtonAdjacentAlignmentSettings,
    ):
        assert button.fontMetrics().horizontalAdvance(button.text()) < button.contentsRect().width()

    window.close()


def test_star_pair_assistant_owns_moved_controls_and_uses_normal_window_layer() -> None:
    """匹配控件应位于窄幅普通顶层窗口，并可绑定给主窗口业务层使用。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    main_ui = Ui_MainWindow()
    main_ui.setupUi(window)
    dialog = StarPairAssistantDialog()
    dialog.bind_controls_to(main_ui)

    assert not dialog.isModal()
    assert dialog.parentWidget() is None
    assert dialog.windowFlags() & Qt.WindowType_Mask == Qt.Window
    assert not bool(dialog.windowFlags() & Qt.WindowStaysOnTopHint)
    assert dialog.windowTitle() == "星点匹配助手"
    assert dialog.width() == 390
    assert dialog.minimumWidth() == 350
    assert dialog.maximumWidth() == 390
    assert dialog.ui.verticalLayoutStarPairAssistant.itemAt(0).widget() is dialog.ui.checkBoxStarPairAssistantAlwaysOnTop
    assert dialog.ui.checkBoxStarPairAssistantAlwaysOnTop.text() == "固定前端显示"
    assert not dialog.ui.checkBoxStarPairAssistantAlwaysOnTop.isChecked()
    assert not dialog.always_on_top()
    assert main_ui.checkBoxStarPairAssistantAlwaysOnTop is dialog.ui.checkBoxStarPairAssistantAlwaysOnTop
    assert main_ui.pushButtonOpenStarPairAssistant.text() == "星点匹配助手"
    assert main_ui.tableWidgetStarPairs is dialog.ui.tableWidgetStarPairs
    assert dialog.ui.pushButtonDeleteStarPairs.text() == "重置匹配列表"
    assert dialog.ui.pushButtonClearStarPairs.text() == "清除所有匹配"
    assert not dialog.ui.pushButtonExportStarPairs.isEnabled()
    assert dialog.ui.horizontalLayoutStarPairSessionButtons.itemAt(0).widget() is dialog.ui.pushButtonImportStarPairs
    assert dialog.ui.horizontalLayoutStarPairSessionButtons.itemAt(1).widget() is dialog.ui.pushButtonExportStarPairs
    assert dialog.ui.horizontalLayoutStarPairResetButtons.itemAt(0).widget() is dialog.ui.pushButtonClearStarPairs
    assert dialog.ui.horizontalLayoutStarPairResetButtons.itemAt(1).widget() is dialog.ui.pushButtonDeleteStarPairs
    assert dialog.ui.formLayoutAutoMatch.fieldGrowthPolicy() == QFormLayout.AllNonFixedFieldsGrow
    assert dialog.ui.tableWidgetStarPairs.columnCount() == 6
    assert dialog.ui.tableWidgetStarPairs.horizontalHeaderItem(3).text() == "质量"
    assert dialog.ui.tableWidgetStarPairs.horizontalHeaderItem(5).text() == "标注"
    assert dialog.ui.tableWidgetStarPairs.verticalHeader().isHidden()

    table_host = StarPairTableGroupsMixin()
    table_host.ui = dialog.ui
    table_host._star_pair_sort_key = None
    table_host._star_pair_sort_descending = True
    table_host._configure_star_pair_table_columns()
    dialog.show()
    app.processEvents()
    window_handle = dialog.windowHandle()
    assert window_handle is not None
    original_show = dialog.show
    reshow_calls: list[str] = []
    dialog.show = lambda: reshow_calls.append("show")  # type: ignore[method-assign]
    dialog.ui.checkBoxStarPairAssistantAlwaysOnTop.setChecked(True)
    app.processEvents()
    assert dialog.isVisible()
    assert dialog.always_on_top()
    assert dialog.windowHandle() is window_handle
    assert bool(window_handle.flags() & Qt.WindowStaysOnTopHint)
    dialog.ui.checkBoxStarPairAssistantAlwaysOnTop.setChecked(False)
    app.processEvents()
    assert dialog.isVisible()
    assert not dialog.always_on_top()
    assert dialog.windowHandle() is window_handle
    assert not bool(window_handle.flags() & Qt.WindowStaysOnTopHint)
    assert reshow_calls == []
    dialog.show = original_show  # type: ignore[method-assign]
    table = dialog.ui.tableWidgetStarPairs
    assert table.columnWidth(0) >= table.fontMetrics().horizontalAdvance("A000") + 8
    assert table.columnWidth(3) == table_host._star_pair_quality_column_width()
    assert table.columnWidth(3) >= table.fontMetrics().horizontalAdvance("0.00") + 6
    assert table.columnWidth(0) < table.columnWidth(4)
    assert table.columnWidth(2) < table.columnWidth(4)
    assert table.columnWidth(5) < table.columnWidth(4)
    assert table.columnWidth(1) > table.columnWidth(4)
    assert table.horizontalScrollBar().maximum() == 0

    calls: list[str] = []
    dialog.show = lambda: calls.append("show")  # type: ignore[method-assign]
    dialog.raise_ = lambda: calls.append("raise")  # type: ignore[method-assign]
    dialog.activateWindow = lambda: calls.append("activate")  # type: ignore[method-assign]
    host = SimpleNamespace(star_pair_assistant=dialog)
    MainWindow._show_star_pair_assistant(host)  # type: ignore[arg-type]
    MainWindow._show_star_pair_assistant(host)  # type: ignore[arg-type]
    assert calls == ["show", "raise", "activate"] * 2

    dialog.close()
    window.close()


def test_horizontal_import_export_buttons_use_consistent_order() -> None:
    """横向并列的导入、导出入口必须统一为导入在左、导出在右。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.horizontalLayoutMosaicFramingIo.itemAt(0).widget() is ui.pushButtonImportMosaicFraming
    assert ui.horizontalLayoutMosaicFramingIo.itemAt(1).widget() is ui.pushButtonExportMosaicFraming

    window.close()


def test_mosaic_projection_omits_fixed_preview_controls() -> None:
    """全景构图不应再显示已固定的极限星等和网格开关。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert not hasattr(ui, "labelMosaicMagLimit")
    assert not hasattr(ui, "doubleSpinBoxMosaicMagLimit")
    assert not hasattr(ui, "checkBoxMosaicShowGrid")
    assert ui.formLayoutMosaicProjection.itemAt(5, QFormLayout.LabelRole).widget() is ui.labelMosaicGridPrecision

    window.close()


def test_mosaic_batch_page_contains_panorama_preview() -> None:
    """批处理页右侧应提供独立的全景图预览和基础信息栏。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.horizontalLayoutMosaicBatch.itemAt(1).widget() is ui.groupBoxMosaicBatchPreview
    assert ui.verticalLayoutMosaicBatchPreview.itemAt(1).widget() is ui.mosaicBatchPreviewView
    assert ui.labelMosaicBatchPreviewTitle.text() == "全景图预览"
    assert "完整画布" in ui.labelMosaicBatchPreviewInfo.text()
    assert "裁剪输出" in ui.labelMosaicBatchPreviewInfo.text()

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


def test_sequence_processing_buttons_and_table_selection_layout() -> None:
    """处理按钮应位于蒙版按钮下方，序列表应支持多行右键操作。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    controls_layout = ui.verticalLayoutImageSequenceControls
    assert controls_layout.itemAt(2).layout() is ui.horizontalLayoutImageSequenceMaskButtons
    assert controls_layout.itemAt(3).layout() is ui.horizontalLayoutImageSequenceProcessButtons
    assert ui.horizontalLayoutImageSequenceProcessButtons.itemAt(0).widget() is ui.pushButtonProcessImageSequence
    assert ui.horizontalLayoutImageSequenceProcessButtons.itemAt(1).widget() is ui.pushButtonContinueImageSequence
    assert ui.pushButtonProcessImageSequence.text() == "开始处理"
    assert ui.pushButtonContinueImageSequence.text() == "继续处理"
    assert not ui.pushButtonContinueImageSequence.isEnabled()
    assert ui.tableWidgetImageSequence.selectionMode() == QAbstractItemView.ExtendedSelection
    assert ui.tableWidgetImageSequence.contextMenuPolicy() == Qt.CustomContextMenu

    window.close()


def test_mosaic_source_file_table_selection_sync() -> None:
    """全景源文件表应跟随单图预览定位。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    mixin = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    mixin.ui = ui
    mixin.schedule_mosaic_render = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mixin._configure_mosaic_source_table()

    def source_item(name: str, projection: str, meteor_count: int) -> MosaicSourceItem:
        source_model = SimpleNamespace(
            json_path=Path(f"{name}.json"),
            source_image_path=Path(f"{name}.tif"),
            source_image_text=f"{name}.tif",
            model=SimpleNamespace(
                camera_calibration_profile=SimpleNamespace(base_projection_type=projection),
            ),
        )
        return MosaicSourceItem(
            source_model=source_model,  # type: ignore[arg-type]
            meteor_boxes=tuple(SimpleNamespace() for _index in range(meteor_count)),  # type: ignore[arg-type]
        )

    mixin._mosaic_source_items = [
        source_item("first", "rectilinear", 2),
        source_item("second", "azimuthal_equidistant_tangent", 1),
    ]
    mixin._update_mosaic_display_model_combo()

    table = ui.tableWidgetMosaicSourceFiles
    assert table.verticalHeader().isHidden()
    assert table.item(0, 0).text() == "1"
    assert table.item(0, 1).text() == "first.tif"
    assert table.item(0, 2).text() == "TAN"
    assert table.item(0, 3).text() == "2"
    assert table.item(1, 2).text() == "插值"

    table.setColumnWidth(1, 260)
    mixin._refresh_mosaic_source_table()
    assert table.columnWidth(1) == 260

    ui.comboBoxMosaicDisplayModel.setCurrentIndex(2)
    mixin._handle_mosaic_display_model_changed()
    assert table.currentRow() == 1

    mixin._handle_mosaic_source_table_double_clicked(0, 1)
    assert ui.comboBoxMosaicDisplayModel.currentIndex() == 1
    assert mixin._selected_mosaic_source_items() == [mixin._mosaic_current_source_items()[0]]

    wheel_calls: list[object] = []
    mixin._handle_table_wheel = lambda source_table, event: wheel_calls.append((source_table, event)) or True  # type: ignore[method-assign]
    wheel_event = SimpleNamespace(type=lambda: QEvent.Wheel)
    assert mixin._handle_mosaic_event_filter(table, wheel_event)
    assert mixin._handle_mosaic_event_filter(table.viewport(), wheel_event)
    assert [call[0] for call in wheel_calls] == [table, table]

    removed_rows: list[int] = []
    mixin._remove_mosaic_source_row = removed_rows.append  # type: ignore[method-assign]
    table.selectRow(1)
    for key in (Qt.Key_Delete, Qt.Key_Backspace):
        key_event = SimpleNamespace(type=lambda: QEvent.KeyPress, key=lambda key=key: key)
        assert mixin._handle_mosaic_event_filter(table, key_event)
    assert removed_rows == [1, 1]

    window.close()
