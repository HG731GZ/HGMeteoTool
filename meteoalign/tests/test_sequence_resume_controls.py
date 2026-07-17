"""图像序列继续处理、列表移除和局部精修回归测试。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QComboBox, QPushButton

from meteoalign.application import app_sequence_processing, app_sequence_refinement
from meteoalign.application.app_sequence_processing import SequenceProcessingMixin
from meteoalign.application.app_sequence_refinement import SequenceRefinementMixin
from meteoalign.application.app_sequence_table_preview import SequenceTablePreviewMixin
from meteoalign.image_sequence import ImageSequenceItem


def _sequence_item(path: Path) -> ImageSequenceItem:
    return ImageSequenceItem(
        path=path,
        capture_datetime=datetime(2026, 8, 13, 0, 0, 0),
        capture_datetime_utc=None,
        capture_time_source="测试",
    )


def _starpair_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_starpairs.json")


def _model_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_model.json")


def _write_complete_outputs(item: ImageSequenceItem, *, delta_t_seconds: float = 0.0) -> None:
    _starpair_path(item.path).write_text(
        json.dumps({"sequence_timing": {"delta_t_seconds": delta_t_seconds}}),
        encoding="utf-8",
    )
    _model_path(item.path).write_text("{}", encoding="utf-8")


class _StatusBar:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:
        self.messages.append(message)


class _VisibleDialog:
    def __init__(self, *, visible: bool = True) -> None:
        self.visible = visible
        self.close_count = 0

    def isVisible(self) -> bool:
        return self.visible

    def close(self) -> None:
        self.visible = False
        self.close_count += 1


class _SequenceStateHost(SequenceTablePreviewMixin, SequenceRefinementMixin):
    def __init__(self, items: list[ImageSequenceItem]) -> None:
        self.ui = SimpleNamespace(
            pushButtonProcessImageSequence=QPushButton(),
            pushButtonContinueImageSequence=QPushButton(),
            comboBoxSequenceRefinementMode=QComboBox(),
            pushButtonRefineSequenceFrames=QPushButton(),
            statusbar=_StatusBar(),
        )
        self.ui.comboBoxSequenceRefinementMode.addItems(["优化取景角度", "单帧重新拟合"])
        self._image_sequence_items = items
        self._image_sequence_current_index = 0
        self._sequence_processing_active = False
        self._sequence_refinement_active = False
        self.refined_names: list[str] = []

    @staticmethod
    def _sequence_starpair_json_path(image_path: Path) -> Path:
        return _starpair_path(image_path)

    @staticmethod
    def _sequence_model_json_path(image_path: Path) -> Path:
        return _model_path(image_path)

    def _refresh_image_sequence_table(self) -> None:
        return

    def _refine_sequence_frame(self, *, starpair_path: Path, model_path: Path, mode: str) -> float:
        del model_path, mode
        self.refined_names.append(starpair_path.name)
        return 0.5


class _FakeProgressDialog:
    def __init__(self, _parent: object) -> None:
        self.value = 0
        self.canceled = False

    def setWindowTitle(self, _text: str) -> None:
        return

    def setRange(self, _minimum: int, _maximum: int) -> None:
        return

    def setWindowModality(self, _modality: object) -> None:
        return

    def setMinimumDuration(self, _duration: int) -> None:
        return

    def setAutoClose(self, _enabled: bool) -> None:
        return

    def setAutoReset(self, _enabled: bool) -> None:
        return

    def setLabelText(self, _text: str) -> None:
        return

    def setValue(self, value: int) -> None:
        self.value = value

    def show(self) -> None:
        return

    def close(self) -> None:
        return

    def wasCanceled(self) -> bool:
        return self.canceled


class _FakeMessageBox:
    Yes = 1
    No = 2

    @staticmethod
    def question(*_args: object, **_kwargs: object) -> int:
        return _FakeMessageBox.Yes

    @staticmethod
    def information(*_args: object, **_kwargs: object) -> None:
        return

    @staticmethod
    def warning(*_args: object, **_kwargs: object) -> None:
        return


def test_continue_and_refinement_controls_follow_complete_output_pairs(tmp_path) -> None:
    """继续处理关注未完成帧，单帧修正只要求至少两帧完整输出。"""

    app = QApplication.instance() or QApplication([])
    items = [_sequence_item(tmp_path / f"frame_{index}.jpg") for index in range(1, 5)]
    host = _SequenceStateHost(items)

    host._update_image_sequence_controls()
    assert host.ui.pushButtonProcessImageSequence.isEnabled()
    assert not host.ui.pushButtonContinueImageSequence.isEnabled()
    assert not host.ui.pushButtonRefineSequenceFrames.isEnabled()

    _write_complete_outputs(items[0])
    _write_complete_outputs(items[1])
    host._update_image_sequence_controls()
    assert host._sequence_first_unprocessed_index() == 2
    assert host.ui.pushButtonContinueImageSequence.isEnabled()
    assert host.ui.pushButtonRefineSequenceFrames.isEnabled()

    _starpair_path(items[2].path).write_text("{}", encoding="utf-8")
    assert host._sequence_first_unprocessed_index() == 2
    _model_path(items[2].path).write_text("{}", encoding="utf-8")
    assert host._sequence_first_unprocessed_index() == 3

    _write_complete_outputs(items[3])
    host._update_image_sequence_controls()
    assert host._sequence_first_unprocessed_index() is None
    assert not host.ui.pushButtonContinueImageSequence.isEnabled()
    assert host.ui.pushButtonRefineSequenceFrames.isEnabled()


def test_refinement_skips_unprocessed_images(tmp_path, monkeypatch) -> None:
    """精修入口只遍历同时存在两类 JSON 的图像。"""

    app = QApplication.instance() or QApplication([])
    items = [_sequence_item(tmp_path / f"frame_{index}.jpg") for index in range(1, 5)]
    _write_complete_outputs(items[0])
    _write_complete_outputs(items[2])
    host = _SequenceStateHost(items)
    host.star_pair_assistant = _VisibleDialog()
    host.image_preview_dialog = _VisibleDialog()
    monkeypatch.setattr(app_sequence_refinement, "QProgressDialog", _FakeProgressDialog)
    monkeypatch.setattr(app_sequence_refinement, "QMessageBox", _FakeMessageBox)

    host.refine_sequence_frames()

    assert host.refined_names == ["frame_1_starpairs.json", "frame_3_starpairs.json"]
    assert host.star_pair_assistant.close_count == 1
    assert host.image_preview_dialog.close_count == 1


class _ProcessingEntryHost(SequenceProcessingMixin, SequenceTablePreviewMixin):
    def __init__(self) -> None:
        self.star_pair_assistant = _VisibleDialog()
        self.image_preview_dialog = _VisibleDialog()
        self.runs: list[bool] = []

    def _sequence_can_continue(self) -> bool:
        return True

    def _run_image_sequence_processing(self, *, continue_only: bool) -> None:
        self.runs.append(continue_only)


def test_start_and_continue_close_visible_auxiliary_windows() -> None:
    """开始处理和继续处理都应先关闭星点匹配助手与图像预览窗口。"""

    host = _ProcessingEntryHost()
    host.process_image_sequence()
    assert host.runs == [False]
    assert host.star_pair_assistant.close_count == 1
    assert host.image_preview_dialog.close_count == 1

    host.star_pair_assistant.visible = True
    host.image_preview_dialog.visible = True
    host.continue_image_sequence()
    assert host.runs == [False, True]
    assert host.star_pair_assistant.close_count == 2
    assert host.image_preview_dialog.close_count == 2


class _RemovalHost(SequenceTablePreviewMixin):
    def __init__(self, items: list[ImageSequenceItem]) -> None:
        self._image_sequence_items = items
        self._image_sequence_current_index = 2
        self.ui = SimpleNamespace(statusbar=_StatusBar())
        self.reloaded = False
        self.reset = False

    def _clear_image_sequence_preview_cache(self) -> None:
        return

    def _update_imported_sequence_status(self) -> None:
        return

    def _refresh_image_sequence_table(self) -> None:
        return

    def _set_image_sequence_preview_index(self, index: int, *, sync_table_selection: bool = True) -> None:
        del sync_table_selection
        self._image_sequence_current_index = index

    def _reload_first_sequence_item_after_removal(self) -> None:
        self.reloaded = True

    def _update_image_sequence_controls(self) -> None:
        return

    def _reset_image_sequence_status(self) -> None:
        self.reset = True
        self._image_sequence_items = []
        self._image_sequence_current_index = -1


def test_removing_rows_reloads_changed_first_frame_and_keeps_nearest_preview(tmp_path) -> None:
    """移除首帧后应重载新基准，当前预览被移除时选择相邻剩余帧。"""

    items = [_sequence_item(tmp_path / f"frame_{index}.jpg") for index in range(1, 5)]
    host = _RemovalHost(items)

    host._remove_image_sequence_indices([0, 2])

    assert [item.path.name for item in host._image_sequence_items] == ["frame_2.jpg", "frame_4.jpg"]
    assert host._image_sequence_current_index == 1
    assert host.reloaded
    assert "重新载入新的第一帧" in host.ui.statusbar.messages[-1]


class _ProcessingHost(SequenceProcessingMixin, SequenceTablePreviewMixin):
    def __init__(self, items: list[ImageSequenceItem]) -> None:
        self._image_sequence_items = items
        self._sequence_processing_active = False
        self.current_image_preview = SimpleNamespace(image=SimpleNamespace(width=lambda: 64, height=lambda: 48))
        self.ui = SimpleNamespace(statusbar=_StatusBar())
        self.processed_names: list[str] = []
        self.initial_delta_by_name: dict[str, float] = {}

    @staticmethod
    def _sequence_starpair_json_path(image_path: Path) -> Path:
        return _starpair_path(image_path)

    @staticmethod
    def _sequence_model_json_path(image_path: Path) -> Path:
        return _model_path(image_path)

    def _ensure_first_sequence_session_loaded_for_processing(self) -> None:
        return

    def _ensure_sequence_ready_for_processing(self) -> None:
        return

    def _current_sequence_first_item(self) -> ImageSequenceItem:
        return self._image_sequence_items[0]

    def _ensure_first_sequence_output_jsons(self, _item: ImageSequenceItem) -> list[Path]:
        return []

    def _apply_sequence_observation_time(self, _item: ImageSequenceItem, *, emit_signal: bool) -> None:
        del emit_signal

    def render_now(self) -> None:
        return

    def _sequence_base_templates(self) -> list[object]:
        return [object()]

    def _sequence_pair_targets(self, _templates: list[object]) -> tuple[int, int]:
        return 1, 1

    def _fit_sequence_fixed_camera_model(self, _templates: list[object], _target_size: tuple[int, int]) -> object:
        return object()

    def _sequence_frame_matched_pairs(
        self,
        item: ImageSequenceItem,
        _preview: object,
        _templates: list[object],
        _fixed_model: object,
        _target_size: tuple[int, int],
        previous_delta_seconds: float,
        _desired_pair_count: int,
        _stats: dict[str, int],
    ) -> list[object]:
        self.initial_delta_by_name[item.path.name] = previous_delta_seconds
        return [object()]

    def _sequence_time_fit_for_pairs(self, item: ImageSequenceItem, _pairs: list[object], _fixed_model: object, *, initial_delta_seconds: float) -> object:
        del initial_delta_seconds
        return SimpleNamespace(delta_t_seconds=1.25 if item.path.stem == "frame_2" else 2.5)

    def _apply_sequence_time_fit(self, pairs: list[object], _time_fit: object, *, require_accepted: bool) -> list[object]:
        del require_accepted
        return pairs

    def _write_sequence_outputs(
        self,
        item: ImageSequenceItem,
        _first_item: ImageSequenceItem,
        _preview: object,
        _pairs: list[object],
        _fixed_model: object,
        time_fit: object,
    ) -> tuple[Path, Path]:
        self.processed_names.append(item.path.name)
        _write_complete_outputs(item, delta_t_seconds=float(time_fit.delta_t_seconds))
        return _starpair_path(item.path), _model_path(item.path)

    def _refresh_image_sequence_table(self) -> None:
        return

    def _set_image_sequence_processing_index(self, _index: int) -> None:
        return

    def _set_image_import_controls_enabled(self, _enabled: bool) -> None:
        return

    def _update_image_sequence_controls(self) -> None:
        return


def test_continue_processing_preserves_complete_frames_and_resumes_delta(tmp_path, monkeypatch) -> None:
    """继续处理应跳过完整输出，并从最近已有结果恢复时间偏移初值。"""

    app = QApplication.instance() or QApplication([])
    items = [_sequence_item(tmp_path / f"frame_{index}.jpg") for index in range(1, 5)]
    _write_complete_outputs(items[0])
    _write_complete_outputs(items[2], delta_t_seconds=7.0)
    host = _ProcessingHost(items)
    preview = SimpleNamespace(image=SimpleNamespace(width=lambda: 64, height=lambda: 48))
    monkeypatch.setattr(app_sequence_processing, "load_image_preview", lambda *_args, **_kwargs: preview)
    monkeypatch.setattr(app_sequence_processing, "QProgressDialog", _FakeProgressDialog)
    monkeypatch.setattr(app_sequence_processing, "QMessageBox", _FakeMessageBox)

    host._run_image_sequence_processing(continue_only=True)

    assert host.processed_names == ["frame_2.jpg", "frame_4.jpg"]
    assert host.initial_delta_by_name["frame_2.jpg"] == 0.0
    assert host.initial_delta_by_name["frame_4.jpg"] == 7.0
    assert host._sequence_first_unprocessed_index() is None
