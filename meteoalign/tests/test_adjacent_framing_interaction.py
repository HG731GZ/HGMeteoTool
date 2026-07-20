"""参考图像粗略取景交互测试。"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication, QCheckBox, QMainWindow, QMessageBox

from meteoalign.adjacent_alignment import AdjacentFramingResult
from meteoalign.application import app_adjacent_framing
from meteoalign.application.app_adjacent_framing import AdjacentFramingMixin
from meteoalign.application.image_preview_dialog import ImagePreviewDialog
from meteoalign.ui.ui_main_window import Ui_MainWindow


class _AdjacentFramingHost(AdjacentFramingMixin, QMainWindow):
    """只提供参考图像交互所需状态的轻量主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.current_image_preview = None
        self._adjacent_framing_thread = None
        self._adjacent_framing_worker = None
        self._adjacent_framing_progress = None
        self._image_group_paths: tuple[Path, ...] = ()
        self._image_import_thread = None
        self._json_import_thread = None
        self._sequence_import_thread = None
        self._sequence_processing_active = False
        self.image_preview_dialog = ImagePreviewDialog()
        self._init_adjacent_framing_defaults()

    def _set_elided_label_text(self, label, text: str, tooltip: str) -> None:  # type: ignore[no-untyped-def]
        label.setText(text)
        label.setToolTip(tooltip)

    def _update_reference_alignment_transform(self) -> None:
        return

    def _image_group_mode_active(self) -> bool:
        return len(self._image_group_paths) > 1

    def _image_group_controls_idle(self) -> bool:
        return (
            self._image_import_thread is None
            and self._json_import_thread is None
            and self._sequence_import_thread is None
            and not self._sequence_processing_active
        )


def _write_test_image(path: Path) -> None:
    image = QImage(48, 32, QImage.Format_RGB888)
    image.fill(QColor("#203050"))
    assert image.save(str(path))


def test_adjacent_image_controls_have_required_text_and_order() -> None:
    """主界面应把导入、计算和设定三个按钮并排放置。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.labelAdjacentImageModel.text() == "未导入参考图像"
    assert ui.horizontalLayoutAdjacentImageInfo.itemAt(0).widget() is ui.labelAdjacentImageModel
    assert ui.horizontalLayoutAdjacentImageInfo.itemAt(1).widget() is ui.pushButtonPreviewAdjacentImage
    assert ui.pushButtonPreviewAdjacentImage.text() == "预览"
    assert not ui.pushButtonPreviewAdjacentImage.isEnabled()
    assert (
        ui.horizontalLayoutAdjacentImageFramingButtons.itemAt(0).widget()
        is ui.pushButtonImportAdjacentImage
    )
    assert (
        ui.horizontalLayoutAdjacentImageFramingButtons.itemAt(1).widget()
        is ui.pushButtonCalculateAdjacentFraming
    )
    assert (
        ui.horizontalLayoutAdjacentImageFramingButtons.itemAt(2).widget()
        is ui.toolButtonAdjacentAlignmentSettings
    )
    assert ui.pushButtonImportAdjacentImage.text() == "导入参考图像"
    assert ui.pushButtonCalculateAdjacentFraming.text() == "计算粗略取景"
    assert "解除模拟星空视角同步" in ui.pushButtonCalculateAdjacentFraming.toolTip()
    assert ui.toolButtonAdjacentAlignmentSettings.text() == "设定"
    assert not hasattr(ui, "pushButtonSelectAdjacentImageFromGroup")
    window.close()


def test_image_without_model_shows_required_message(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """没有同名 model.json 的图像不得替换当前参考图像。"""

    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "without_model.png"
    _write_test_image(image_path)
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_adjacent_framing.QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )
    host = _AdjacentFramingHost()

    assert not host.load_adjacent_image(image_path)
    assert host._adjacent_image_path is None
    assert host.ui.labelAdjacentImageModel.text() == "未导入参考图像"
    assert messages and "已有模型(model.json)的图像" in messages[0][1]
    host.close()


def test_imported_image_name_and_preview_button_state(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """有效图像应按文件名显示，并解锁参考图像预览按钮。"""

    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "adjacent.png"
    model_path = tmp_path / "adjacent_model.json"
    _write_test_image(image_path)
    model_path.write_text("{}", encoding="utf-8")
    loaded: list[tuple[Path, Path]] = []

    def fake_load(model: str | Path, image: str | Path):
        loaded.append((Path(model), Path(image)))
        return object(), Path(image)

    monkeypatch.setattr(app_adjacent_framing, "load_adjacent_frame_model", fake_load)
    host = _AdjacentFramingHost()

    assert host.load_adjacent_image(image_path)
    assert loaded == [(model_path.resolve(), image_path.resolve())]
    assert host.ui.labelAdjacentImageModel.text() == image_path.name
    assert host.ui.pushButtonPreviewAdjacentImage.isEnabled()
    host.close()


def test_manual_reference_import_cancels_automatic_selection(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """手动参考导入成功后取消自动选择，自动导入则保持勾选。"""

    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "adjacent.png"
    model_path = tmp_path / "adjacent_model.json"
    _write_test_image(image_path)
    model_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        app_adjacent_framing,
        "load_adjacent_frame_model",
        lambda _model, image: (object(), Path(image)),
    )
    check_box = QCheckBox()
    selected_references: list[Path] = []
    host = _AdjacentFramingHost()
    host.image_group_assistant = SimpleNamespace(
        ui=SimpleNamespace(checkBoxAutoSelectReference=check_box),
        set_reference_image=lambda path: selected_references.append(Path(path)),
    )

    check_box.setChecked(True)
    assert host.load_adjacent_image(image_path)
    assert not check_box.isChecked()
    assert selected_references == [image_path.resolve()]

    check_box.setChecked(True)
    assert host.load_adjacent_image(image_path, automatic_selection=True)
    assert check_box.isChecked()
    host.close()


def test_current_image_cannot_be_used_as_adjacent_image(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """参考图像与当前图像相同时应显示提示并禁止计算。"""

    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "same.png"
    model_path = tmp_path / "same_model.json"
    _write_test_image(image_path)
    model_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        app_adjacent_framing,
        "load_adjacent_frame_model",
        lambda _model, image: (object(), Path(image)),
    )
    host = _AdjacentFramingHost()
    host.current_image_preview = SimpleNamespace(path=image_path)

    assert host.load_adjacent_image(image_path)
    assert host.ui.labelAdjacentImageModel.text() == "参考图像不能与为当前图像"
    assert not host.ui.pushButtonCalculateAdjacentFraming.isEnabled()

    host.current_image_preview = SimpleNamespace(path=tmp_path / "other.png")
    host._update_adjacent_framing_controls()
    assert host.ui.labelAdjacentImageModel.text() == image_path.name
    assert host.ui.pushButtonCalculateAdjacentFraming.isEnabled()
    host.close()


def test_landscape_time_warning_can_cancel_rough_framing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """地景模式下两图拍摄时间超过一分钟时，选择“否”不得创建后台任务。"""

    app = QApplication.instance() or QApplication([])
    host = _AdjacentFramingHost()
    reference_path = Path("reference.jpg").resolve()
    current_path = Path("current.jpg").resolve()
    host._adjacent_image_path = reference_path
    host._adjacent_model_json_path = Path("reference_model.json").resolve()
    host.current_image_preview = SimpleNamespace(path=current_path)
    host.ui.comboBoxAdjacentAlignmentMode.setCurrentIndex(1)
    capture_times = {
        reference_path: datetime(2026, 8, 12, 22, 0, 0),
        current_path: datetime(2026, 8, 12, 22, 5, 0),
    }

    def fake_read_capture_time(path: str | Path):
        resolved_path = Path(path).resolve()
        return SimpleNamespace(
            capture_datetime=capture_times[resolved_path],
            capture_datetime_utc=None,
        )

    questions: list[tuple[str, str]] = []

    def answer_no(_parent, title: str, message: str, *_args):  # type: ignore[no-untyped-def]
        questions.append((title, message))
        return QMessageBox.No

    monkeypatch.setattr(app_adjacent_framing, "read_image_capture_time", fake_read_capture_time)
    monkeypatch.setattr(app_adjacent_framing.QMessageBox, "question", answer_no)
    monkeypatch.setattr(
        app_adjacent_framing,
        "create_progress_dialog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应创建后台任务")),
    )

    host.toggle_adjacent_rough_framing()

    assert questions == [
        (
            "拍摄时间间隔较长",
            "参考图像与当前图像的拍摄时间间隔约为5分钟，粗略取景偏差可能较大，是否继续？",
        )
    ]
    assert host._adjacent_framing_thread is None
    assert host.ui.statusbar.currentMessage() == "已取消粗略取景计算。"
    host.close()


def test_calculated_framing_keeps_sync_and_reset_releases_current_view() -> None:
    """计算成功后保持视角同步，重置时解除接管但保留当前视角。"""

    app = QApplication.instance() or QApplication([])
    host = _AdjacentFramingHost()
    current_path = Path("current.jpg").resolve()
    host.current_image_preview = SimpleNamespace(path=current_path)

    def sync_to_rough_framing() -> None:
        host.ui.doubleSpinBoxAz.setValue(240.0)
        host.ui.doubleSpinBoxAlt.setValue(50.0)
        host.ui.doubleSpinBoxRoll.setValue(25.0)

    host._update_reference_alignment_transform = sync_to_rough_framing  # type: ignore[method-assign]
    transform = SimpleNamespace(rms_px=2.5)
    result = AdjacentFramingResult(
        model_json_path=Path("reference_model.json").resolve(),
        image_a_path=Path("reference.jpg").resolve(),
        image_b_path=current_path,
        mode="stars",
        correspondence_count=18,
        correspondence_rms_px=1.5,
        source_model=object(),  # type: ignore[arg-type]
        transform=transform,  # type: ignore[arg-type]
    )

    host._handle_adjacent_framing_finished(result)

    assert host.ui.pushButtonCalculateAdjacentFraming.text() == "重置粗略取景"
    assert host.ui.pushButtonCalculateAdjacentFraming.isEnabled()
    assert host.ui.doubleSpinBoxAz.value() == 240.0
    assert host.ui.doubleSpinBoxAlt.value() == 50.0
    assert host.ui.doubleSpinBoxRoll.value() == 25.0

    host.toggle_adjacent_rough_framing()

    assert host._adjacent_framing_result is None
    assert host._rough_alignment_transform is None
    assert host._rough_source_astrometric_model is None
    assert host.ui.pushButtonCalculateAdjacentFraming.text() == "计算粗略取景"
    assert host.ui.doubleSpinBoxAz.value() == 240.0
    assert host.ui.doubleSpinBoxAlt.value() == 50.0
    assert host.ui.doubleSpinBoxRoll.value() == 25.0
    assert host.ui.statusbar.currentMessage() == "已重置粗略取景，可手动调整模拟星空视角。"
    host.close()


def test_landscape_time_check_skips_prompt_when_exif_is_missing_or_within_one_minute(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """任一图无拍摄时间或间隔不超过一分钟时，不应弹出确认框。"""

    app = QApplication.instance() or QApplication([])
    host = _AdjacentFramingHost()
    reference_path = Path("reference.jpg").resolve()
    current_path = Path("current.jpg").resolve()
    close_times = iter(
        (
            SimpleNamespace(capture_datetime=datetime(2026, 8, 12, 22, 0, 0), capture_datetime_utc=None),
            SimpleNamespace(capture_datetime=datetime(2026, 8, 12, 22, 1, 0), capture_datetime_utc=None),
        )
    )
    monkeypatch.setattr(app_adjacent_framing, "read_image_capture_time", lambda _path: next(close_times))
    monkeypatch.setattr(
        app_adjacent_framing.QMessageBox,
        "question",
        lambda *_args: (_ for _ in ()).throw(AssertionError("不应弹出确认框")),
    )

    assert host._confirm_landscape_time_interval(reference_path, current_path)

    monkeypatch.setattr(
        app_adjacent_framing,
        "read_image_capture_time",
        lambda _path: (_ for _ in ()).throw(ValueError("没有 EXIF 时间")),
    )
    assert host._confirm_landscape_time_interval(reference_path, current_path)
    host.close()


def test_landscape_time_warning_allows_calculation_after_confirmation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """拍摄间隔较长时选择“是”，时间检查应允许调用方继续计算。"""

    app = QApplication.instance() or QApplication([])
    host = _AdjacentFramingHost()
    capture_times = iter(
        (
            SimpleNamespace(capture_datetime=datetime(2026, 8, 12, 22, 0, 0), capture_datetime_utc=None),
            SimpleNamespace(capture_datetime=datetime(2026, 8, 12, 22, 5, 0), capture_datetime_utc=None),
        )
    )
    monkeypatch.setattr(app_adjacent_framing, "read_image_capture_time", lambda _path: next(capture_times))
    monkeypatch.setattr(app_adjacent_framing.QMessageBox, "question", lambda *_args: QMessageBox.Yes)

    assert host._confirm_landscape_time_interval(Path("reference.jpg"), Path("current.jpg"))
    host.close()


def test_preview_button_reuses_and_refreshes_one_window(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """重复预览应重新读取图像，同时继续使用原来的窗口实例。"""

    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "adjacent.png"
    model_path = tmp_path / "adjacent_model.json"
    _write_test_image(image_path)
    model_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        app_adjacent_framing,
        "load_adjacent_frame_model",
        lambda _model, image: (object(), Path(image)),
    )
    original_loader = app_adjacent_framing.load_image_preview
    loaded_paths: list[Path] = []

    def counted_loader(path: str | Path):
        loaded_paths.append(Path(path).resolve())
        return original_loader(path)

    monkeypatch.setattr(app_adjacent_framing, "load_image_preview", counted_loader)
    host = _AdjacentFramingHost()
    assert host.load_adjacent_image(image_path)

    host.show_adjacent_image_preview()
    app.processEvents()
    first_dialog = host.image_preview_dialog
    host.show_adjacent_image_preview()
    app.processEvents()

    assert host.image_preview_dialog is first_dialog
    assert loaded_paths == [image_path.resolve(), image_path.resolve()]
    assert first_dialog.ui.labelImageName.text() == image_path.name
    assert first_dialog.windowTitle() == f"{image_path.name} 预览"
    assert first_dialog.image_path == image_path.resolve()
    assert first_dialog.parentWidget() is None
    assert first_dialog.windowFlags() & Qt.WindowType_Mask == Qt.Window
    first_dialog.close()
    host.close()
