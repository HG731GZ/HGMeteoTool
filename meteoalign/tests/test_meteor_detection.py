"""MetDet worker 配置、协议与页面联动测试。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

from meteoalign.application.app_meteor_selection import MeteorSelectionMixin
from meteoalign.application.metdet_worker_client import MetDetWorkerClient
from meteoalign.meteor_detection import (
    MeteorDetectionOptions,
    load_meteor_detection_options,
    resolve_meteor_worker_invocation,
    save_meteor_detection_options,
)
from meteoalign.meteor_selection import MeteorBox, meteor_json_path, save_meteor_selection
from meteoalign.ui.ui_main_window import Ui_MainWindow


_APPLICATION: QApplication | None = None


def _application() -> QApplication:
    global _APPLICATION
    _APPLICATION = QApplication.instance() or QApplication([])
    return _APPLICATION


class _MeteorDetectionHost(QMainWindow, MeteorSelectionMixin):
    """提供流星检测页面所需的最小宿主。"""

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._init_meteor_selection_page()

    def _import_dialog_directory(self, fallback):  # type: ignore[no-untyped-def]
        return fallback

    def _remember_import_path(self, _selected) -> None:  # type: ignore[no-untyped-def]
        return


def test_meteor_detection_options_round_trip(tmp_path) -> None:
    """检测参数应完整写入并从 preference.json 恢复。"""

    preference_path = tmp_path / "preference.json"
    expected = MeteorDetectionOptions(
        engine_path=str(tmp_path / "worker"),
        model_path=str(tmp_path / "model.onnx"),
        confidence_threshold=0.31,
        nms_threshold=0.52,
        multiscale=3,
        partition=4,
        provider="cpu",
        box_expansion_ratio=0.2,
    )
    assert save_meteor_detection_options(expected, preference_path)
    loaded = load_meteor_detection_options(preference_path)
    assert loaded == expected
    assert loaded.worker_options()["overwrite"] is True


def test_meteor_detection_worker_options_include_optional_mask_path(tmp_path) -> None:
    """流星蒙版应作为单次任务参数传给 worker，且不写入持久检测配置。"""

    mask_path = tmp_path / "sky mask.png"
    options = MeteorDetectionOptions()

    assert "mask_path" not in options.worker_options()
    assert options.worker_options(mask_path)["mask_path"] == str(mask_path.resolve())


def test_worker_client_writes_mask_path_into_detect_options(tmp_path, monkeypatch) -> None:
    """客户端发送 detect 消息时应把蒙版路径放在 options 内。"""

    _application()
    client = MetDetWorkerClient()
    monkeypatch.setattr(MetDetWorkerClient, "is_ready", property(lambda _client: True))
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(client, "_write_message", messages.append)
    mask_path = tmp_path / "mask.tif"

    client.detect([str(tmp_path / "image.tif")], MeteorDetectionOptions(), mask_path)

    assert messages[-1]["command"] == "detect"
    assert messages[-1]["options"]["mask_path"] == str(mask_path.resolve())  # type: ignore[index]


def test_source_worker_directory_resolves_to_current_python(tmp_path) -> None:
    """开发环境允许把引擎位置指向包含 metdet_worker.py 的源码目录。"""

    worker_path = tmp_path / "metdet_worker.py"
    worker_path.write_text("print('ready')\n", encoding="utf-8")
    invocation = resolve_meteor_worker_invocation(str(tmp_path))
    assert invocation.arguments == ("-u", str(worker_path))
    assert invocation.resolved_path == worker_path
    assert invocation.working_directory == str(tmp_path)


def test_worker_client_rejects_incompatible_protocol_and_accepts_ready() -> None:
    """客户端必须只接受 metdet.jsonl v1 的 ready 握手。"""

    _application()
    client = MetDetWorkerClient()
    errors: list[str] = []
    ready_messages: list[dict[str, object]] = []
    client.workerError.connect(errors.append)
    client.ready.connect(ready_messages.append)

    client._handle_protocol_line(b'{"type":"ready","protocol":"metdet.jsonl","protocol_version":2}')
    assert errors and "协议版本不兼容" in errors[-1]
    assert not ready_messages

    client._handle_protocol_line(b'{"type":"ready","protocol":"metdet.jsonl","protocol_version":1}')
    assert ready_messages[-1]["protocol_version"] == 1


def test_detection_result_focuses_completed_image_and_loads_boxes(tmp_path) -> None:
    """处理第 i 张结束后应显示该结果，使下一张推理时持续展示上一张。"""

    app = _application()
    host = _MeteorDetectionHost()
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    image = QImage(120, 80, QImage.Format_RGB32)
    image.fill(Qt.black)
    assert image.save(str(first_path))
    assert image.save(str(second_path))
    save_meteor_selection(second_path, 120, 80, [MeteorBox(10, 12, 90, 60)])
    host._meteor_selection_paths = [first_path, second_path]
    host._meteor_selection_boxes_by_path = {first_path: [], second_path: []}
    host._meteor_selection_current_index = 0
    host._meteor_detection_active = True
    host._meteor_detection_job_paths = [first_path, second_path]

    host._apply_meteor_detection_result({"index": 2, "error": None})
    app.processEvents()

    assert host._meteor_selection_current_index == 1
    assert host._meteor_selection_boxes_by_path[second_path] == [MeteorBox(10, 12, 90, 60)]
    assert host.ui.tableWidgetMeteorSelectionImages.currentRow() == 1
    assert host.ui.meteorSelectionView.boxes() == [MeteorBox(10, 12, 90, 60)]
    host._shutdown_meteor_detection_worker()
    host.close()


def test_raw_detection_result_queues_preview_without_sync_decode(tmp_path, monkeypatch) -> None:
    """RAW 结果回调只能更新列表并排队后台预览，不能在 GUI 线程调用 rawpy。"""

    _application()
    host = _MeteorDetectionHost()
    raw_path = tmp_path / "second.ARW"
    raw_path.write_bytes(b"raw")
    host._meteor_selection_paths = [raw_path]
    host._meteor_selection_boxes_by_path = {raw_path: []}
    host._meteor_selection_current_index = 0
    host._meteor_detection_active = True
    host._meteor_detection_job_paths = [raw_path]
    queued_paths: list[Path] = []
    monkeypatch.setattr(host, "_queue_meteor_selection_preview_load", queued_paths.append)

    def fail_on_synchronous_decode(_image_path):  # type: ignore[no-untyped-def]
        raise AssertionError("检测结果回调不应同步解码 RAW")

    monkeypatch.setattr(host, "_meteor_selection_preview_for_path", fail_on_synchronous_decode)

    host._apply_meteor_detection_result({"index": 1, "error": None, "meteor_boxes": []})

    assert queued_paths == [raw_path]
    assert host._meteor_selection_current_index == 0
    assert host.ui.tableWidgetMeteorSelectionImages.currentRow() == 0
    assert "正在后台读取 RAW 预览" in host.ui.labelMeteorSelectionCaptureTime.text()
    host._shutdown_meteor_detection_worker()
    host.close()


def test_raw_preview_queue_preserves_detection_order(tmp_path) -> None:
    """检测快于预览时也应保留中间 RAW，而不是只留下最后一张。"""

    _application()
    host = _MeteorDetectionHost()
    paths = [tmp_path / f"image-{index}.ARW" for index in range(3)]
    host._meteor_selection_paths = paths
    host._meteor_selection_preview_load_thread = object()

    for path in paths:
        host._queue_meteor_selection_preview_load(path)
    host._queue_meteor_selection_preview_load(paths[1])

    assert list(host._meteor_selection_preview_queue) == paths
    assert host._meteor_selection_preview_queued_paths == set(paths)
    host._meteor_selection_preview_load_thread = None
    host._meteor_selection_preview_queue.clear()
    host._meteor_selection_preview_queued_paths.clear()
    host._shutdown_meteor_detection_worker()
    host.close()


def test_detection_result_updates_only_completed_table_row(tmp_path, monkeypatch) -> None:
    """大批量检测的单张结果不能重新创建整张图片表格。"""

    _application()
    host = _MeteorDetectionHost()
    paths = [tmp_path / f"image-{index}.ARW" for index in range(3)]
    host._meteor_selection_paths = paths
    host._meteor_selection_boxes_by_path = {path: [] for path in paths}
    host._meteor_selection_current_index = 0
    host._meteor_detection_active = True
    host._meteor_detection_job_paths = paths
    host._refresh_meteor_selection_table()
    updated_rows: list[int] = []
    monkeypatch.setattr(host, "_refresh_meteor_selection_table_row", updated_rows.append)
    monkeypatch.setattr(host, "_show_meteor_selection_current_image", lambda **_kwargs: None)

    def fail_full_refresh() -> None:
        raise AssertionError("逐图结果不应重建整张表格")

    monkeypatch.setattr(host, "_refresh_meteor_selection_table", fail_full_refresh)

    host._apply_meteor_detection_result({"index": 2, "error": "decode failed"})

    assert updated_rows == [1]
    host._shutdown_meteor_detection_worker()
    host.close()


def test_detection_disables_manual_controls_until_finished() -> None:
    """自动检测期间应禁用框选、切图及其他会改变列表的操作。"""

    _application()
    host = _MeteorDetectionHost()
    host._meteor_selection_paths = [Path("example.tif")]
    host._meteor_selection_current_index = 0
    host._meteor_detection_active = True
    host._update_meteor_selection_controls()

    assert not host.ui.pushButtonImportMeteorImages.isEnabled()
    assert not host.ui.pushButtonMeteorDetectionOptions.isEnabled()
    assert not host.ui.toolButtonMeteorSelectionPrevious.isEnabled()
    assert not host.ui.toolButtonMeteorSelectionNext.isEnabled()
    assert not host.ui.pushButtonClearMeteorBoxes.isEnabled()
    assert not host.ui.meteorSelectionView._box_editing_enabled
    assert host.ui.pushButtonAutoDetectMeteors.text() == "取消检测"
    assert host.ui.pushButtonAutoDetectMeteors.isEnabled()
    host._shutdown_meteor_detection_worker()
    host.close()


def test_detection_preparation_does_not_persist_pending_manual_boxes(tmp_path) -> None:
    """检测准备阶段不得提前保存仍在内存中的人工框选。"""

    _application()
    host = _MeteorDetectionHost()
    image_path = tmp_path / "manual.png"
    image = QImage(100, 60, QImage.Format_RGB32)
    image.fill(Qt.black)
    assert image.save(str(image_path))
    boxes = [MeteorBox(2, 3, 40, 50)]
    host._meteor_selection_paths = [image_path]
    host._meteor_selection_boxes_by_path = {image_path: boxes}
    host._meteor_selection_image_sizes = {image_path: (100, 60)}
    host._meteor_selection_dirty_paths = {image_path}

    assert host._prepare_meteor_detection_paths() == [image_path]
    assert not meteor_json_path(image_path).exists()
    host._shutdown_meteor_detection_worker()
    host.close()


def test_detection_always_clears_old_boxes_when_result_is_empty(tmp_path) -> None:
    """自动检测结果为零时应删除旧 JSON，并让列表显示零个框。"""

    _application()
    host = _MeteorDetectionHost()
    image_path = tmp_path / "old.png"
    image = QImage(100, 60, QImage.Format_RGB32)
    image.fill(Qt.black)
    assert image.save(str(image_path))
    old_boxes = [MeteorBox(2, 3, 40, 50)]
    save_meteor_selection(image_path, 100, 60, old_boxes)
    host._meteor_selection_paths = [image_path]
    host._meteor_selection_boxes_by_path = {image_path: old_boxes}
    host._meteor_selection_current_index = 0
    host._meteor_detection_job_paths = [image_path]

    host._apply_meteor_detection_result({"index": 1, "error": None, "meteor_boxes": []})

    assert not meteor_json_path(image_path).exists()
    assert host._meteor_selection_boxes_by_path[image_path] == []
    host._shutdown_meteor_detection_worker()
    host.close()


def test_move_meteor_files_also_moves_all_corresponding_json(tmp_path, monkeypatch) -> None:
    """移动流星图片时应移动同主文件名的全部关联 JSON，且不误移相似前缀。"""

    _application()
    host = _MeteorDetectionHost()
    source_directory = tmp_path / "source"
    target_directory = tmp_path / "target"
    source_directory.mkdir()
    target_directory.mkdir()
    image_path = source_directory / "meteor.tif"
    image_path.write_bytes(b"test-image")
    boxes = [MeteorBox(1, 2, 30, 40)]
    save_meteor_selection(image_path, 100, 80, boxes)
    corresponding_names = {
        "meteor.json",
        "meteor_starpairs.json",
        "meteor_model.json",
        "meteor.model.json",
        "meteor-extra.JSON",
    }
    for json_name in corresponding_names:
        (source_directory / json_name).write_text("{}\n", encoding="utf-8")
    unrelated_names = {"meteorology.json", "meteor2_model.json", "unrelated.json"}
    for json_name in unrelated_names:
        (source_directory / json_name).write_text("{}\n", encoding="utf-8")
    host._meteor_selection_paths = [image_path]
    host._meteor_selection_boxes_by_path = {image_path: boxes}
    host._meteor_selection_image_sizes = {image_path: (100, 80)}
    host._meteor_selection_current_index = -1
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(target_directory))

    host.move_meteor_files()

    moved_image = target_directory / image_path.name
    assert moved_image.exists()
    assert meteor_json_path(moved_image).exists()
    for json_name in corresponding_names:
        assert (target_directory / json_name).exists()
        assert not (source_directory / json_name).exists()
    for json_name in unrelated_names:
        assert (source_directory / json_name).exists()
        assert not (target_directory / json_name).exists()
    assert not image_path.exists()
    assert host._meteor_selection_paths == [moved_image]
    assert host._meteor_selection_boxes_by_path[moved_image] == boxes
    host._shutdown_meteor_detection_worker()
    host.close()
