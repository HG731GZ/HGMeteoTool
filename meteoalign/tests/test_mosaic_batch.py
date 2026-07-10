from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from meteoalign.app_mosaic_batch import MosaicBatchMixin


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

