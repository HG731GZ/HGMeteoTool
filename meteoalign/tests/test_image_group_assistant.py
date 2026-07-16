"""图像组助手的界面与切图状态回归测试。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

from meteoalign.application.app_image_group import ImageGroupMixin
from meteoalign.application.image_group_assistant_dialog import (
    IMAGE_GROUP_CELL_HORIZONTAL_PADDING,
    IMAGE_GROUP_FILE_NAME_CHAR_COUNT,
    IMAGE_GROUP_FILE_NAME_WIDTH_SAMPLE,
    IMAGE_GROUP_PREVIEW_COLUMN,
    IMAGE_GROUP_PREVIEW_HORIZONTAL_PADDING,
    IMAGE_GROUP_PREVIEW_TEXT,
    IMAGE_GROUP_READY_COLOR,
    ImageGroupAssistantDialog,
)
from meteoalign.application.main_window import MainWindow
from meteoalign.ui.ui_main_window import Ui_MainWindow


class _ImageGroupHost(ImageGroupMixin):
    def __init__(self, dialog: ImageGroupAssistantDialog) -> None:
        self.ui = SimpleNamespace(pushButtonOpenImageGroupAssistant=QPushButton())
        self.image_group_assistant = dialog
        self._image_group_paths: tuple[Path, ...] = ()
        self._image_import_thread = None
        self._json_import_thread = None
        self._sequence_import_thread = None
        self._sequence_processing_active = False
        self._sequence_active = False
        self.current_image_preview = None
        self.started_imports: list[tuple[Path, bool]] = []
        self.pair_count = 0

    def _sequence_mode_active(self) -> bool:
        return self._sequence_active

    def _star_pair_position_count(self) -> int:
        return self.pair_count

    def _star_pair_session_path_for_image(self, image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}_starpairs.json")

    def _source_model_path_for_image(self, image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}_model.json")

    def start_single_image_import(
        self,
        image_path: Path,
        *,
        preserve_image_group_status: bool = False,
    ) -> None:
        self.started_imports.append((image_path, preserve_image_group_status))


def _write_test_image(path: Path) -> None:
    """写入可供真实预览加载器读取的小尺寸测试图像。"""

    image = QImage(48, 32, QImage.Format_RGB888)
    image.fill(QColor("#203050"))
    assert image.save(str(path))


def test_main_ui_places_image_group_assistant_left_of_star_pair_assistant() -> None:
    """主界面应提供多图导入入口和顺序正确的两个助手按钮。"""

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.pushButtonImportImages.text() == "导入图像"
    assert ui.horizontalLayoutMatchingAssistants.itemAt(0).widget() is ui.pushButtonOpenImageGroupAssistant
    assert ui.horizontalLayoutMatchingAssistants.itemAt(1).widget() is ui.pushButtonOpenStarPairAssistant
    assert not ui.pushButtonOpenImageGroupAssistant.isEnabled()
    window.close()


def test_image_group_dialog_marks_existing_outputs_green(tmp_path: Path) -> None:
    """匹配与映射文件存在时，对应状态单元格应显示绿色。"""

    app = QApplication.instance() or QApplication([])
    first_image = tmp_path / "first.tif"
    second_image = tmp_path / "second.tif"
    first_image.touch()
    second_image.touch()
    (tmp_path / "first_starpairs.json").write_text("{}", encoding="utf-8")
    (tmp_path / "first_Mask.png").touch()
    (tmp_path / "second_model.json").write_text("{}", encoding="utf-8")

    dialog = ImageGroupAssistantDialog()
    dialog.set_image_paths((first_image, second_image))
    # 主窗口启动后助手会先隐藏一段时间；隐藏状态不得用无效布局放大窗口。
    app.processEvents()
    assert not dialog.isVisible()
    assert dialog.maximumWidth() == 480
    dialog.show()
    app.processEvents()
    table = dialog.ui.tableWidgetImageGroup

    assert [table.horizontalHeaderItem(column).text() for column in range(5)] == [
        "文件名",
        "匹配",
        "映射",
        "蒙版",
        "预览",
    ]
    assert table.item(0, 0).text() == "first.tif"
    assert table.item(0, 1).text() == "已有"
    assert table.item(0, 1).background().color() == IMAGE_GROUP_READY_COLOR
    assert table.item(0, 2).text() == ""
    assert table.item(0, 3).text() == "已有"
    assert table.item(0, 3).background().color() == IMAGE_GROUP_READY_COLOR
    assert table.item(0, 3).toolTip() == str((tmp_path / "first_Mask.png").resolve())
    assert table.item(1, 1).text() == ""
    assert table.item(1, 2).text() == "已有"
    assert table.item(1, 2).background().color() == IMAGE_GROUP_READY_COLOR
    assert table.item(1, 3).text() == ""
    expected_file_width = (
        table.fontMetrics().horizontalAdvance(
            IMAGE_GROUP_FILE_NAME_WIDTH_SAMPLE[:IMAGE_GROUP_FILE_NAME_CHAR_COUNT]
        )
        + IMAGE_GROUP_CELL_HORIZONTAL_PADDING
    )
    assert table.columnWidth(0) == expected_file_width
    expected_preview_width = (
        table.fontMetrics().horizontalAdvance(IMAGE_GROUP_PREVIEW_TEXT)
        + IMAGE_GROUP_PREVIEW_HORIZONTAL_PADDING
    )
    assert table.columnWidth(IMAGE_GROUP_PREVIEW_COLUMN) == expected_preview_width
    assert table.cellWidget(0, IMAGE_GROUP_PREVIEW_COLUMN).width() == expected_preview_width
    assert dialog.width() < 480
    assert dialog.minimumWidth() == 250
    assert dialog.maximumWidth() == dialog.width()
    assert table.viewport().width() == sum(table.columnWidth(column) for column in range(5))
    dialog.close()


def test_long_file_name_is_elided_from_the_left(tmp_path: Path) -> None:
    """超长文件名应省略开头并保留末尾，完整路径继续放在悬浮提示中。"""

    app = QApplication.instance() or QApplication([])
    file_name = "very_long_image_name_for_meteor_shower_123456789.TIF"
    image_path = tmp_path / file_name
    image_path.touch()
    dialog = ImageGroupAssistantDialog()
    dialog.set_image_paths((image_path,))

    item = dialog.ui.tableWidgetImageGroup.item(0, 0)
    assert item.text().startswith("...")
    assert item.text().endswith("6789.TIF")
    assert item.toolTip() == str(image_path.resolve())
    dialog.close()


def test_image_group_mode_controls_button_and_double_click_loading(tmp_path: Path) -> None:
    """只有多图且非序列模式可打开助手，双击时应保留图像组。"""

    app = QApplication.instance() or QApplication([])
    dialog = ImageGroupAssistantDialog()
    host = _ImageGroupHost(dialog)
    first_image = tmp_path / "first.tif"
    second_image = tmp_path / "second.tif"
    first_image.touch()
    second_image.touch()

    host._set_image_group_paths((str(first_image), str(second_image)))
    assert host.ui.pushButtonOpenImageGroupAssistant.isEnabled()

    host.current_image_preview = SimpleNamespace(path=first_image)
    host._handle_image_group_image_activated(second_image)
    assert host.started_imports == [(second_image.resolve(), True)]

    host._sequence_active = True
    host._update_image_group_controls()
    assert not host.ui.pushButtonOpenImageGroupAssistant.isEnabled()

    host._sequence_active = False
    dialog.show()
    app.processEvents()
    assert dialog.isVisible()
    host._set_image_group_paths((str(first_image),))
    app.processEvents()
    assert not host.ui.pushButtonOpenImageGroupAssistant.isEnabled()
    assert dialog.ui.tableWidgetImageGroup.rowCount() == 0
    assert not dialog.isVisible()
    dialog.close()


def test_image_group_dialog_emits_path_for_any_double_clicked_column(tmp_path: Path) -> None:
    """双击列表任意列都应请求加载对应行的图像。"""

    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "frame.fit"
    image_path.touch()
    dialog = ImageGroupAssistantDialog()
    dialog.set_image_paths((image_path,))
    emitted: list[Path] = []
    dialog.image_activated.connect(emitted.append)

    dialog.ui.tableWidgetImageGroup.cellDoubleClicked.emit(0, 2)
    assert emitted == [image_path.resolve()]
    assert dialog.windowFlags() & Qt.WindowType_Mask == Qt.Window
    assert dialog.parentWidget() is None
    assert not dialog.isModal()
    dialog.close()


def test_preview_button_reuses_image_preview_dialog(tmp_path: Path) -> None:
    """各行预览按钮应刷新并复用同一个通用图像预览窗口。"""

    app = QApplication.instance() or QApplication([])
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    _write_test_image(first_image)
    _write_test_image(second_image)
    dialog = ImageGroupAssistantDialog()
    dialog.set_image_paths((first_image, second_image))
    activated: list[Path] = []
    dialog.image_activated.connect(activated.append)
    table = dialog.ui.tableWidgetImageGroup

    first_button = table.cellWidget(0, IMAGE_GROUP_PREVIEW_COLUMN)
    second_button = table.cellWidget(1, IMAGE_GROUP_PREVIEW_COLUMN)
    assert isinstance(first_button, QPushButton)
    assert isinstance(second_button, QPushButton)
    assert first_button.text() == "预览"

    first_button.click()
    app.processEvents()
    preview_dialog = dialog._image_preview_dialog
    assert preview_dialog is not None
    assert preview_dialog.image_path == first_image.resolve()

    second_button.click()
    app.processEvents()
    assert dialog._image_preview_dialog is preview_dialog
    assert preview_dialog.image_path == second_image.resolve()
    assert preview_dialog.ui.labelImageName.text() == second_image.name
    assert activated == []

    dialog.close()
    app.processEvents()
    assert not preview_dialog.isVisible()


def test_unsaved_switch_prompt_has_three_required_actions(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """有匹配但缺少输出时，切图确认框应提供指定的三个操作。"""

    app = QApplication.instance() or QApplication([])
    dialog = ImageGroupAssistantDialog()
    host = _ImageGroupHost(dialog)
    current_image = tmp_path / "current.tif"
    target_image = tmp_path / "target.tif"
    current_image.touch()
    target_image.touch()
    host.current_image_preview = SimpleNamespace(path=current_image)
    host.pair_count = 3

    class FakeMessageBox:
        Warning = 1
        AcceptRole = 2
        DestructiveRole = 3
        RejectRole = 4
        labels: list[str] = []

        def __init__(self, _parent) -> None:  # type: ignore[no-untyped-def]
            self._buttons: list[object] = []
            self._clicked_button = None

        def setIcon(self, _icon) -> None:  # type: ignore[no-untyped-def]
            return

        def setWindowTitle(self, _title: str) -> None:
            return

        def setText(self, _text: str) -> None:
            return

        def addButton(self, label: str, _role):  # type: ignore[no-untyped-def]
            button = object()
            self.labels.append(label)
            self._buttons.append(button)
            return button

        def setDefaultButton(self, _button) -> None:  # type: ignore[no-untyped-def]
            return

        def exec_(self) -> None:
            self._clicked_button = self._buttons[2]

        def clickedButton(self):  # type: ignore[no-untyped-def]
            return self._clicked_button

    monkeypatch.setattr("meteoalign.application.app_image_group.QMessageBox", FakeMessageBox)
    assert not host._confirm_image_group_switch(target_image)
    assert FakeMessageBox.labels == ["保存并跳转", "不保存", "取消"]
    dialog.close()


def test_main_window_close_also_closes_image_group_assistant(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """主窗口成功关闭后，所有独立助手窗口也必须一起关闭。"""

    closed: list[str] = []
    host = SimpleNamespace(
        _meteor_mask_import_thread=None,
        preferences_dialog=SimpleNamespace(close=lambda: closed.append("preferences")),
        star_pair_assistant=SimpleNamespace(close=lambda: closed.append("star_pair")),
        image_group_assistant=SimpleNamespace(close=lambda: closed.append("image_group")),
        _shutdown_meteor_detection_worker=lambda: closed.append("worker"),
    )
    event = SimpleNamespace(isAccepted=lambda: True)
    monkeypatch.setattr(
        "meteoalign.application.main_window.ViewControlsMixin.closeEvent",
        lambda _self, _event: None,
    )

    MainWindow.closeEvent(host, event)  # type: ignore[arg-type]
    assert closed == ["preferences", "star_pair", "image_group", "worker"]
