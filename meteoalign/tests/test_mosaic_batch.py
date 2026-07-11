from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from meteoalign.application.app_mosaic_batch import MosaicBatchMixin
from meteoalign.meteor_selection import MeteorBox, save_meteor_selection


class _Control:
    def __init__(self, value: int = 0) -> None:
        self.value = int(value)
        self.enabled = True
        self.text = ""
        self.tooltip = ""

    def currentIndex(self) -> int:  # noqa: N802 - Qt 控件接口命名
        return int(self.value)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt 控件接口命名
        self.enabled = bool(enabled)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt 控件接口命名
        self.text = str(text)

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt 控件接口命名
        self.tooltip = str(text)


class _Table:
    def __init__(self) -> None:
        self.row_count = -1

    def setRowCount(self, row_count: int) -> None:  # noqa: N802 - Qt 控件接口命名
        self.row_count = int(row_count)


class _StatusBar:
    def __init__(self) -> None:
        self.message = ""

    def showMessage(self, message: str) -> None:  # noqa: N802 - Qt 控件接口命名
        self.message = str(message)


class _CheckBox:
    def __init__(self, checked: bool) -> None:
        self.checked = bool(checked)

    def isChecked(self) -> bool:  # noqa: N802 - Qt 控件接口命名
        return self.checked


def _batch_window(mode_index: int) -> MosaicBatchMixin:
    window = MosaicBatchMixin.__new__(MosaicBatchMixin)
    window.ui = SimpleNamespace(
        comboBoxMosaicBatchMode=_Control(mode_index),
        labelMosaicBatchFramingPathTitle=_Control(),
        labelMosaicBatchFramingPath=_Control(),
        labelMosaicBatchImageCount=_Control(),
        pushButtonImportMosaicBatchFramingJson=_Control(),
        pushButtonImportMosaicBatchImageJson=_Control(),
        pushButtonClearMosaicBatchImports=_Control(),
        pushButtonStartMosaicBatch=_Control(),
        tableWidgetMosaicBatchImages=_Table(),
        statusbar=_StatusBar(),
    )
    window._mosaic_batch_framing = None
    window._mosaic_batch_base_model = None
    window._mosaic_batch_items = []
    window._set_elided_label_text = lambda label, text, tooltip="": label.setText(text)  # type: ignore[attr-defined]
    return window


def test_mosaic_batch_base_mode_updates_text_and_locks_after_any_import() -> None:
    window = _batch_window(1)

    window._update_mosaic_batch_controls()

    assert window.ui.labelMosaicBatchFramingPathTitle.text == "底图"
    assert window.ui.pushButtonImportMosaicBatchFramingJson.text == "导入底图JSON"
    assert window.ui.pushButtonImportMosaicBatchFramingJson.enabled
    assert window.ui.pushButtonImportMosaicBatchImageJson.enabled
    assert window.ui.comboBoxMosaicBatchMode.enabled
    assert not window.ui.pushButtonStartMosaicBatch.enabled

    # 只要先导入一张待贴图像，模式也必须锁定；目标未导入时仍不能开始。
    window._mosaic_batch_items = [object()]
    window._update_mosaic_batch_controls()

    assert not window.ui.comboBoxMosaicBatchMode.enabled
    assert not window.ui.pushButtonStartMosaicBatch.enabled


def test_mosaic_batch_base_geometry_enables_export_and_clear_unlocks_mode() -> None:
    window = _batch_window(1)
    window._mosaic_batch_base_model = SimpleNamespace(
        json_path=Path("base_model.json"),
        image_width_px=6000,
        image_height_px=4000,
        model=object(),
    )
    window._mosaic_batch_items = [object()]

    window._update_mosaic_batch_controls()
    geometry = window._mosaic_batch_target_geometry()

    assert geometry is not None
    assert geometry.output_width_px == 6000
    assert geometry.output_height_px == 4000
    assert window.ui.pushButtonStartMosaicBatch.enabled
    assert not window.ui.comboBoxMosaicBatchMode.enabled

    window.clear_mosaic_batch_imports()

    assert window._mosaic_batch_base_model is None
    assert not window._mosaic_batch_items
    assert window.ui.tableWidgetMosaicBatchImages.row_count == 0
    assert window.ui.comboBoxMosaicBatchMode.enabled
    assert not window.ui.pushButtonStartMosaicBatch.enabled


def test_mosaic_batch_detects_sibling_meteor_selection_json(tmp_path) -> None:
    """导入源图模型时应自动读取同名流星框选 JSON。"""

    image_path = tmp_path / "IMG_0001.TIF"
    save_meteor_selection(image_path, 6000, 4000, [MeteorBox(10.0, 20.0, 100.0, 200.0)])
    source_model = SimpleNamespace(source_image_path=image_path)

    item = MosaicBatchMixin._mosaic_batch_item_from_source_model(source_model)
    window = _batch_window(0)
    text, tooltip = window._mosaic_batch_meteor_selection_text(item)

    assert item.meteor_selection_path == tmp_path / "IMG_0001_Meteor.json"
    assert item.meteor_boxes == (MeteorBox(10.0, 20.0, 100.0, 200.0),)
    assert text == "有（1）"
    assert "IMG_0001_Meteor.json" in tooltip


def test_mosaic_batch_meteor_only_uses_detected_box_regions() -> None:
    """流星区域模式只向导出层传递框选源图矩形。"""

    window = _batch_window(0)
    window.ui.checkBoxMosaicBatchMeteorOnly = _CheckBox(True)
    item = SimpleNamespace(meteor_boxes=(MeteorBox(10.2, 20.4, 100.6, 200.8),))

    assert window._mosaic_batch_item_source_regions(item) == ((10, 20, 101, 201),)
