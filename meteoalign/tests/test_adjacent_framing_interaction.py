"""参考图像粗略取景交互测试。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow

from meteoalign.application import app_adjacent_framing
from meteoalign.application.app_adjacent_framing import AdjacentFramingMixin
from meteoalign.application.image_group_assistant_dialog import (
    IMAGE_GROUP_PREVIEW_COLUMN,
    ImageGroupReferenceDialog,
)
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
    """主界面应显示图像导入、预览和预留的图像组选取入口。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.labelAdjacentImageModel.text() == "未导入参考图像"
    assert ui.horizontalLayoutAdjacentImageInfo.itemAt(0).widget() is ui.labelAdjacentImageModel
    assert ui.horizontalLayoutAdjacentImageInfo.itemAt(1).widget() is ui.pushButtonPreviewAdjacentImage
    assert ui.pushButtonPreviewAdjacentImage.text() == "预览"
    assert not ui.pushButtonPreviewAdjacentImage.isEnabled()
    assert ui.horizontalLayoutAdjacentImageFramingButtons.itemAt(0).widget() is ui.pushButtonImportAdjacentImage
    assert (
        ui.horizontalLayoutAdjacentImageFramingButtons.itemAt(1).widget()
        is ui.pushButtonSelectAdjacentImageFromGroup
    )
    assert ui.horizontalLayoutAdjacentImageFramingButtons.itemAt(2).widget() is ui.pushButtonCalculateAdjacentFraming
    assert ui.pushButtonImportAdjacentImage.text() == "导入参考图像"
    assert ui.pushButtonSelectAdjacentImageFromGroup.text() == "从组中选取"
    assert not ui.pushButtonSelectAdjacentImageFromGroup.isEnabled()
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


def test_imported_image_name_and_group_button_state(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """有效图像应按文件名显示，图像组可用时解锁占位按钮。"""

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
    assert not host.ui.pushButtonSelectAdjacentImageFromGroup.isEnabled()

    host._image_group_paths = (image_path, tmp_path / "other.png")
    host._update_adjacent_framing_controls()
    assert host.ui.pushButtonSelectAdjacentImageFromGroup.isEnabled()
    host.close()


def test_double_clicking_group_reference_keeps_dialog_open_and_switches_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """双击组内图像应完成导入、保留窗口，并把红色参考切换到新行。"""

    app = QApplication.instance() or QApplication([])
    reference_image = tmp_path / "reference.png"
    other_image = tmp_path / "other.png"
    invalid_image = tmp_path / "invalid.png"
    model_path = tmp_path / "reference_model.json"
    other_model_path = tmp_path / "other_model.json"
    _write_test_image(reference_image)
    _write_test_image(other_image)
    _write_test_image(invalid_image)
    model_path.write_text("{}", encoding="utf-8")
    other_model_path.write_text("{}", encoding="utf-8")
    loaded: list[tuple[Path, Path]] = []

    def fake_load(model: str | Path, image: str | Path):
        loaded.append((Path(model), Path(image)))
        return object(), Path(image)

    monkeypatch.setattr(app_adjacent_framing, "load_adjacent_frame_model", fake_load)
    messages: list[str] = []
    monkeypatch.setattr(
        app_adjacent_framing.QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )
    host = _AdjacentFramingHost()
    dialog = ImageGroupReferenceDialog(host.image_preview_dialog)
    host.image_group_reference_dialog = dialog
    dialog.image_activated.connect(host._handle_adjacent_reference_from_group_activated)
    host._image_group_paths = (
        reference_image.resolve(),
        other_image.resolve(),
        invalid_image.resolve(),
    )

    host.show_adjacent_reference_from_group()
    app.processEvents()
    assert dialog.isVisible()
    dialog.ui.tableWidgetImageGroup.cellDoubleClicked.emit(0, 0)
    app.processEvents()

    assert loaded == [(model_path.resolve(), reference_image.resolve())]
    assert host._adjacent_image_path == reference_image.resolve()
    assert host.ui.labelAdjacentImageModel.text() == reference_image.name
    assert dialog.isVisible()
    assert dialog.ui.tableWidgetImageGroup.item(0, 1).text() == "参考"
    assert dialog.ui.tableWidgetImageGroup.item(0, 2).text() == "参考"

    dialog.ui.tableWidgetImageGroup.cellDoubleClicked.emit(1, 0)
    app.processEvents()
    assert loaded == [
        (model_path.resolve(), reference_image.resolve()),
        (other_model_path.resolve(), other_image.resolve()),
    ]
    assert host._adjacent_image_path == other_image.resolve()
    assert dialog.isVisible()
    assert dialog.ui.tableWidgetImageGroup.item(0, 1).text() == ""
    assert dialog.ui.tableWidgetImageGroup.item(0, 2).text() == "已有"
    assert dialog.ui.tableWidgetImageGroup.item(1, 1).text() == "参考"
    assert dialog.ui.tableWidgetImageGroup.item(1, 2).text() == "参考"

    dialog.ui.tableWidgetImageGroup.cellDoubleClicked.emit(2, 0)
    app.processEvents()
    assert len(loaded) == 2
    assert messages and "model.json" in messages[-1]
    assert host._adjacent_image_path == other_image.resolve()
    assert dialog.ui.tableWidgetImageGroup.item(1, 1).text() == "参考"
    assert dialog.ui.tableWidgetImageGroup.item(1, 2).text() == "参考"
    assert dialog.ui.tableWidgetImageGroup.item(2, 1).text() == ""
    assert dialog.ui.tableWidgetImageGroup.item(2, 2).text() == ""
    dialog.close()
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


def test_preview_button_reuses_and_refreshes_one_window(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """重复预览应重新读取图像，同时继续使用原来的窗口实例。"""

    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "adjacent.png"
    group_image_path = tmp_path / "group.png"
    model_path = tmp_path / "adjacent_model.json"
    _write_test_image(image_path)
    _write_test_image(group_image_path)
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

    group_dialog = ImageGroupReferenceDialog(host.image_preview_dialog)
    group_dialog.set_image_paths((group_image_path,))
    group_dialog.ui.tableWidgetImageGroup.cellWidget(0, IMAGE_GROUP_PREVIEW_COLUMN).click()
    app.processEvents()
    assert host.image_preview_dialog.image_path == group_image_path.resolve()
    group_dialog.close()
    assert host.image_preview_dialog.isVisible()

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
    group_dialog.close()
    host.close()
