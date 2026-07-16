"""跨平台 UI 布局回归测试。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

# 让无显示服务器的 CI 也能创建 Qt 窗口；用户桌面运行不受此测试环境变量影响。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QApplication, QFormLayout, QHeaderView, QMainWindow, QSizePolicy, QTableWidget

from meteoalign.application.app_sequence_table_preview import SequenceTablePreviewMixin
from meteoalign.application.app_mosaic import MosaicProjectionMixin, MosaicSourceItem
from meteoalign.application.app_rendering import RenderingMixin
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
