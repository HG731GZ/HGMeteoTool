from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtGui import QImage

import meteoalign.application.app_mosaic_batch as app_mosaic_batch
from meteoalign.application.app_mosaic_batch import (
    MosaicBatchExportTask,
    MosaicBatchExportWorker,
    MosaicBatchImageItem,
    MosaicBatchMixin,
)
from meteoalign.mosaic.export.geometry import MosaicExportGeometry
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


def _worker_task(row: int, tmp_path: Path) -> MosaicBatchExportTask:
    source_model = SimpleNamespace(
        json_path=tmp_path / f"source_{row}.json",
        source_image_path=tmp_path / f"source_{row}.tif",
    )
    return MosaicBatchExportTask(
        row=row,
        item=MosaicBatchImageItem(source_model=source_model),  # type: ignore[arg-type]
        geometry=MosaicExportGeometry(10, 8, 0, 0, 10, 8),
        output_path=tmp_path / f"output_{row}.tif",
        framing=None,
        base_model=None,
        block_rows=8,
        map_tile_size_px=4,
        exact_remap_repair=False,
        tiff_lzw_compression=False,
        source_pixel_regions=None,
    )


def test_mosaic_batch_worker_continues_after_single_file_failure(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """后台批处理应记录单张失败，并继续导出后续图像。"""

    tasks = tuple(_worker_task(row, tmp_path) for row in range(3))

    def fake_write(task, progress_callback) -> None:  # type: ignore[no-untyped-def]
        progress_callback("测试导出", 1, 2)
        if task.row == 1:
            raise ValueError("测试失败")
        progress_callback("测试导出", 2, 2)

    monkeypatch.setattr(app_mosaic_batch, "_write_mosaic_batch_export_task", fake_write)
    worker = MosaicBatchExportWorker(tasks)
    started_rows: list[int] = []
    succeeded_rows: list[int] = []
    failed_rows: list[tuple[int, str]] = []
    completed: list[tuple[int, int, bool]] = []
    worker.item_started.connect(started_rows.append)
    worker.item_succeeded.connect(lambda row, _path: succeeded_rows.append(row))
    worker.item_failed.connect(lambda row, message: failed_rows.append((row, message)))
    worker.completed.connect(lambda success, failed, canceled: completed.append((success, failed, canceled)))

    worker.run()

    assert started_rows == [0, 1, 2]
    assert succeeded_rows == [0, 2]
    assert failed_rows == [(1, "测试失败")]
    assert completed == [(2, 1, False)]


def test_mosaic_batch_worker_stops_cooperatively_after_cancel(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """取消请求应在下一个进度检查点停止，且不能启动下一张图。"""

    tasks = tuple(_worker_task(row, tmp_path) for row in range(2))
    worker = MosaicBatchExportWorker(tasks)

    def fake_write(_task, progress_callback) -> None:  # type: ignore[no-untyped-def]
        progress_callback("开始", 0, 10)
        worker.request_cancel()
        progress_callback("继续", 1, 10)

    monkeypatch.setattr(app_mosaic_batch, "_write_mosaic_batch_export_task", fake_write)
    started_rows: list[int] = []
    completed: list[tuple[int, int, bool]] = []
    worker.item_started.connect(started_rows.append)
    worker.completed.connect(lambda success, failed, canceled: completed.append((success, failed, canceled)))

    worker.run()

    assert started_rows == [0]
    assert completed == [(0, 0, True)]


def test_mosaic_batch_controls_are_locked_while_worker_is_active() -> None:
    """后台任务运行时不得再次导入、清空或重复启动。"""

    window = _batch_window(0)
    window._mosaic_batch_thread = object()
    window._mosaic_batch_framing = SimpleNamespace(
        json_path=Path("framing.json"),
        geometry=MosaicExportGeometry(10, 8, 0, 0, 10, 8),
    )
    window._mosaic_batch_items = [object()]

    window._update_mosaic_batch_controls()

    assert not window.ui.comboBoxMosaicBatchMode.enabled
    assert not window.ui.pushButtonImportMosaicBatchFramingJson.enabled
    assert not window.ui.pushButtonImportMosaicBatchImageJson.enabled
    assert not window.ui.pushButtonClearMosaicBatchImports.enabled
    assert not window.ui.pushButtonStartMosaicBatch.enabled


def test_mosaic_batch_preview_info_contains_projection_and_pixel_counts() -> None:
    """批处理预览标题应只显示投影和最终输出像素数量。"""

    window = _batch_window(0)
    window.ui.labelMosaicBatchPreviewInfo = _Control()
    window._mosaic_batch_framing = SimpleNamespace(
        payload={"projection": {"model": "fisheye_equidistant", "display_name": "等距鱼眼(ARC)"}},
        geometry=MosaicExportGeometry(8000, 4000, 1000, 500, 6000, 3000),
    )

    window._update_mosaic_batch_preview_info()

    assert "投影：等距鱼眼(ARC)" in window.ui.labelMosaicBatchPreviewInfo.text
    assert "输出尺寸：6000 × 3000 px（18.00 MP）" in window.ui.labelMosaicBatchPreviewInfo.text
    assert "完整画布" not in window.ui.labelMosaicBatchPreviewInfo.text


def test_mosaic_batch_base_preview_is_8bit_and_cached(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """底图模式应生成 8-bit 预览，并避免窗口刷新时重复解码底图。"""

    image_path = tmp_path / "panorama.tif"
    image_path.write_bytes(b"preview fixture")
    source_image = QImage(120, 60, QImage.Format_ARGB32)
    source_image.fill(0xFF304050)
    load_calls: list[tuple[Path, int | None]] = []

    def fake_load(path, max_long_side_px=None):  # type: ignore[no-untyped-def]
        load_calls.append((Path(path), max_long_side_px))
        return SimpleNamespace(image=source_image)

    monkeypatch.setattr(app_mosaic_batch, "load_image_preview", fake_load)
    window = _batch_window(1)
    window._mosaic_batch_base_model = SimpleNamespace(source_image_path=image_path)
    window._mosaic_batch_base_preview_path = None
    window._mosaic_batch_base_preview_image = None

    first = window._load_mosaic_batch_base_preview()
    second = window._load_mosaic_batch_base_preview()

    assert first.format() == QImage.Format_RGB888
    assert second.cacheKey() == first.cacheKey()
    assert len(load_calls) == 1


def test_mosaic_batch_progress_close_does_not_restore_canceling_status() -> None:
    """取消完成后关闭进度弹窗，不得再次把状态栏改回“正在取消”。"""

    class _Signal:
        def __init__(self) -> None:
            self.slot = None

        def connect(self, slot) -> None:  # type: ignore[no-untyped-def]
            self.slot = slot

        def disconnect(self, slot) -> None:  # type: ignore[no-untyped-def]
            if self.slot != slot:
                raise TypeError("信号未连接")
            self.slot = None

        def emit(self) -> None:
            if self.slot is not None:
                self.slot()

    class _Progress:
        def __init__(self) -> None:
            self.canceled = _Signal()

        def setCancelButton(self, _button) -> None:  # noqa: N802 - Qt 控件接口命名
            return

        def setLabelText(self, _text: str) -> None:  # noqa: N802 - Qt 控件接口命名
            return

        def close(self) -> None:
            # 模拟部分平台上关闭 QProgressDialog 时再次发出 canceled。
            self.canceled.emit()

    class _Worker:
        def __init__(self) -> None:
            self.cancel_count = 0

        def request_cancel(self) -> None:
            self.cancel_count += 1

    window = _batch_window(0)
    progress = _Progress()
    worker = _Worker()
    window._mosaic_batch_progress = progress
    window._mosaic_batch_worker = worker
    window._mosaic_batch_cancel_requested = False
    window._mosaic_batch_terminal_handled = False
    progress.canceled.connect(window._cancel_mosaic_batch_processing)

    progress.canceled.emit()
    assert worker.cancel_count == 1
    assert window.ui.statusbar.message == "正在取消全景图批处理..."

    window._handle_mosaic_batch_completed(0, 0, True)

    assert worker.cancel_count == 1
    assert window.ui.statusbar.message == "全景图批处理已取消：已成功导出 0 张。"
