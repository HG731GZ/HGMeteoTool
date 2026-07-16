from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QDialog, QHeaderView, QMessageBox, QPushButton, QTableWidgetItem, QWidget

from ..image_path_resolution import companion_sky_mask_path
from ..image_preview import load_image_preview
from ..ui.ui_image_group_assistant_dialog import Ui_ImageGroupAssistantDialog
from .image_preview_dialog import ImagePreviewDialog


IMAGE_GROUP_READY_COLOR = QColor("#c8e6c9")
IMAGE_GROUP_READY_TEXT_COLOR = QColor("#1b5e20")
IMAGE_GROUP_FILE_NAME_CHAR_COUNT = 25
IMAGE_GROUP_FILE_NAME_WIDTH_SAMPLE = "abcdefghijklmnopqrstuvwxyz"
IMAGE_GROUP_CELL_HORIZONTAL_PADDING = 12
IMAGE_GROUP_MASK_COLUMN = 3
IMAGE_GROUP_PREVIEW_COLUMN = 4
IMAGE_GROUP_PREVIEW_TEXT = "预览"
IMAGE_GROUP_PREVIEW_HORIZONTAL_PADDING = 12


class ImageGroupAssistantDialog(QDialog):
    """显示多图像的匹配与映射文件状态。"""

    image_activated = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # 使用普通顶层窗口，主窗口激活后可以按正常窗口层级覆盖本窗口。
        window_flags = (self.windowFlags() & ~Qt.WindowType_Mask) | Qt.Window
        self.setWindowFlags(window_flags)
        self.ui = Ui_ImageGroupAssistantDialog()
        self.ui.setupUi(self)
        self.setModal(False)
        self._image_paths: tuple[Path, ...] = ()
        self._image_preview_dialog: ImagePreviewDialog | None = None

        table = self.ui.tableWidgetImageGroup
        header = table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(IMAGE_GROUP_MASK_COLUMN, QHeaderView.ResizeToContents)
        self._preview_column_width = (
            table.fontMetrics().horizontalAdvance(IMAGE_GROUP_PREVIEW_TEXT)
            + IMAGE_GROUP_PREVIEW_HORIZONTAL_PADDING
        )
        header.setSectionResizeMode(IMAGE_GROUP_PREVIEW_COLUMN, QHeaderView.Fixed)
        header.resizeSection(IMAGE_GROUP_PREVIEW_COLUMN, self._preview_column_width)
        file_column_width = (
            table.fontMetrics().horizontalAdvance(
                IMAGE_GROUP_FILE_NAME_WIDTH_SAMPLE[:IMAGE_GROUP_FILE_NAME_CHAR_COUNT]
            )
            + IMAGE_GROUP_CELL_HORIZONTAL_PADDING
        )
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.resizeSection(0, file_column_width)
        table.cellDoubleClicked.connect(self._handle_cell_double_clicked)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """窗口真正可见后再按有效布局收紧宽度。"""

        QDialog.showEvent(self, event)
        QTimer.singleShot(0, self._fit_default_width_to_columns)

    def _fit_default_width_to_columns(self) -> None:
        """按各列所需宽度收紧窗口，避免表格右侧出现空白。"""

        table = self.ui.tableWidgetImageGroup
        layout = self.layout()
        if layout is not None:
            layout.activate()
        window_chrome_width = max(0, self.width() - table.viewport().width())
        fitted_width = sum(table.columnWidth(column) for column in range(table.columnCount()))
        fitted_width += window_chrome_width
        fitted_width = max(self.minimumWidth(), fitted_width)
        self.setMaximumWidth(fitted_width)
        self.resize(fitted_width, self.height())

    @staticmethod
    def _star_pair_path(image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}_starpairs.json")

    @staticmethod
    def _model_path(image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}_model.json")

    def set_image_paths(self, image_paths: list[Path] | tuple[Path, ...]) -> None:
        """替换图像组列表，并立即刷新配套 JSON 状态。"""

        self._image_paths = tuple(Path(path).expanduser().resolve() for path in image_paths)
        table = self.ui.tableWidgetImageGroup
        table.setRowCount(len(self._image_paths))
        for row, image_path in enumerate(self._image_paths):
            file_item = QTableWidgetItem(self._display_file_name(image_path.name))
            file_item.setToolTip(str(image_path))
            table.setItem(row, 0, file_item)
            table.setItem(row, 1, QTableWidgetItem())
            table.setItem(row, 2, QTableWidgetItem())
            table.setItem(row, IMAGE_GROUP_MASK_COLUMN, QTableWidgetItem())
            preview_button = QPushButton(IMAGE_GROUP_PREVIEW_TEXT, table)
            preview_button.setAutoDefault(False)
            preview_button.setFixedWidth(self._preview_column_width)
            preview_button.setToolTip(f"预览图像：{image_path}")
            preview_button.clicked.connect(
                lambda _checked=False, path=image_path: self._show_image_preview(path)
            )
            table.setCellWidget(row, IMAGE_GROUP_PREVIEW_COLUMN, preview_button)
        self.refresh_file_statuses()

    def _show_image_preview(self, image_path: Path) -> None:
        """读取指定行的图像，并复用唯一的预览窗口显示。"""

        try:
            preview = load_image_preview(image_path)
        except Exception as exc:  # noqa: BLE001 - 单行预览失败需要直接向用户说明。
            QMessageBox.warning(self, "图像预览失败", str(exc))
            return

        if self._image_preview_dialog is None:
            self._image_preview_dialog = ImagePreviewDialog(self)
        self._image_preview_dialog.show_preview(preview)

    def _display_file_name(self, file_name: str) -> str:
        """超出文件名列宽时省略开头，尽量保留扩展名和末尾编号。"""

        table = self.ui.tableWidgetImageGroup
        available_width = max(
            0,
            table.columnWidth(0) - IMAGE_GROUP_CELL_HORIZONTAL_PADDING,
        )
        metrics = table.fontMetrics()
        if metrics.horizontalAdvance(file_name) <= available_width:
            return file_name

        prefix = "..."
        suffix = file_name
        while suffix and metrics.horizontalAdvance(prefix + suffix) > available_width:
            suffix = suffix[1:]
        return prefix + suffix

    def refresh_file_statuses(self) -> None:
        """按当前磁盘文件刷新“匹配”“映射”和“蒙版”状态。"""

        table = self.ui.tableWidgetImageGroup
        for row, image_path in enumerate(self._image_paths):
            self._set_status_cell(row, 1, self._star_pair_path(image_path).is_file())
            self._set_status_cell(row, 2, self._model_path(image_path).is_file())
            mask_path = companion_sky_mask_path(image_path)
            self._set_status_cell(row, IMAGE_GROUP_MASK_COLUMN, mask_path is not None)
            mask_item = table.item(row, IMAGE_GROUP_MASK_COLUMN)
            if mask_item is not None:
                mask_item.setToolTip(str(mask_path) if mask_path is not None else "")

    def _set_status_cell(self, row: int, column: int, ready: bool) -> None:
        item = self.ui.tableWidgetImageGroup.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.ui.tableWidgetImageGroup.setItem(row, column, item)
        item.setText("已有" if ready else "")
        item.setTextAlignment(Qt.AlignCenter)
        item.setBackground(QBrush(IMAGE_GROUP_READY_COLOR) if ready else QBrush())
        item.setForeground(QBrush(IMAGE_GROUP_READY_TEXT_COLOR) if ready else QBrush())

    def set_current_image(self, image_path: str | Path | None) -> None:
        """选中当前正在主窗口中匹配的图像。"""

        table = self.ui.tableWidgetImageGroup
        if image_path is None:
            table.clearSelection()
            return
        resolved_path = Path(image_path).expanduser().resolve()
        for row, candidate in enumerate(self._image_paths):
            if candidate == resolved_path:
                table.selectRow(row)
                table.scrollToItem(table.item(row, 0))
                return
        table.clearSelection()

    def _handle_cell_double_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._image_paths):
            self.image_activated.emit(self._image_paths[row])

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """关闭助手时一并关闭它创建的预览窗口。"""

        if self._image_preview_dialog is not None:
            self._image_preview_dialog.close()
        QDialog.closeEvent(self, event)


__all__ = ["ImageGroupAssistantDialog"]
