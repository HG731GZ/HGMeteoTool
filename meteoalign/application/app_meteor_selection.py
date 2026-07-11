"""流星框选页面的导入、预览、列表与保存控制。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPalette
from PyQt5.QtWidgets import QAbstractItemView, QFileDialog, QHeaderView, QTableWidgetItem

from ..image_preview import IMAGE_FILE_FILTER, load_image_preview
from ..image_sequence import read_image_capture_time, sequence_item_local_datetime
from ..meteor_selection import MeteorBox, load_meteor_selection, save_meteor_selection


METEOR_SELECTION_INDEX_COLUMN = 0
METEOR_SELECTION_NAME_COLUMN = 1
METEOR_SELECTION_COUNT_COLUMN = 2
METEOR_SELECTION_INDEX_ROLE = Qt.UserRole + 31
METEOR_SELECTION_ROW_GREEN = QColor(220, 252, 231)


class MeteorSelectionMixin:
    """管理流星图片列表和每张图像的框选数据。"""

    def _init_meteor_selection_page(self) -> None:
        """初始化页面状态和表格列宽。"""

        self._meteor_selection_paths: list[Path] = []
        self._meteor_selection_boxes_by_path: dict[Path, list[MeteorBox]] = {}
        self._meteor_selection_image_sizes: dict[Path, tuple[int, int]] = {}
        self._meteor_selection_current_index = -1
        table = self.ui.tableWidgetMeteorSelectionImages
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        self._keep_meteor_selection_table_highlight_active(table)
        header = table.horizontalHeader()
        header.setSectionResizeMode(METEOR_SELECTION_INDEX_COLUMN, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(METEOR_SELECTION_NAME_COLUMN, QHeaderView.Stretch)
        header.setSectionResizeMode(METEOR_SELECTION_COUNT_COLUMN, QHeaderView.ResizeToContents)
        self._reset_meteor_selection_page()

    def _keep_meteor_selection_table_highlight_active(self, table) -> None:  # type: ignore[no-untyped-def]
        """让 Windows 下失焦的当前预览行仍使用与鼠标点击相同的蓝色。"""

        palette = table.palette()
        active_highlight = palette.color(QPalette.Active, QPalette.Highlight)
        active_highlighted_text = palette.color(QPalette.Active, QPalette.HighlightedText)
        palette.setColor(QPalette.Inactive, QPalette.Highlight, active_highlight)
        palette.setColor(QPalette.Inactive, QPalette.HighlightedText, active_highlighted_text)
        table.setPalette(palette)

    def _connect_meteor_selection_inputs(self) -> None:
        """连接流星框选页面控件信号。"""

        self.ui.pushButtonImportMeteorImages.clicked.connect(self.import_meteor_images)
        self.ui.pushButtonClearMeteorBoxes.clicked.connect(self.clear_meteor_boxes)
        self.ui.pushButtonSaveAllMeteorBoxes.clicked.connect(self.save_all_meteor_boxes)
        self.ui.toolButtonMeteorSelectionPrevious.clicked.connect(self.show_previous_meteor_image)
        self.ui.toolButtonMeteorSelectionNext.clicked.connect(self.show_next_meteor_image)
        self.ui.tableWidgetMeteorSelectionImages.cellDoubleClicked.connect(
            self._handle_meteor_selection_table_double_clicked
        )
        self.ui.meteorSelectionView.boxesChanged.connect(self._handle_meteor_boxes_changed)
        self.ui.doubleSpinBoxUtcOffset.valueChanged.connect(self._handle_meteor_selection_time_context_changed)

    def _reset_meteor_selection_page(self) -> None:
        """清空页面显示与内存状态。"""

        self._meteor_selection_paths = []
        self._meteor_selection_boxes_by_path = {}
        self._meteor_selection_image_sizes = {}
        self._meteor_selection_current_index = -1
        self.ui.meteorSelectionView.clear_image()
        self.ui.labelMeteorSelectionPreviewTitle.setText("未导入流星图片")
        self.ui.labelMeteorSelectionPreviewTitle.setToolTip("")
        self.ui.labelMeteorSelectionCaptureTime.setText("拍摄时间：未读取")
        self.ui.labelMeteorSelectionCaptureTime.setToolTip("")
        self._refresh_meteor_selection_table()
        self._update_meteor_selection_controls()

    def import_meteor_images(self) -> None:
        """导入一张或多张需要标记流星的图像。"""

        fallback = self._meteor_selection_paths[0].parent if self._meteor_selection_paths else Path.cwd()
        selected_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "导入流星图片",
            str(self._import_dialog_directory(fallback)),
            IMAGE_FILE_FILTER,
        )
        if not selected_paths:
            return
        self._remember_import_path(selected_paths)

        paths: list[Path] = []
        seen_paths: set[Path] = set()
        for selected_path in selected_paths:
            try:
                image_path = Path(selected_path).expanduser().resolve()
            except OSError:
                image_path = Path(selected_path).expanduser()
            if image_path in seen_paths:
                continue
            seen_paths.add(image_path)
            paths.append(image_path)

        self._meteor_selection_paths = paths
        self._meteor_selection_boxes_by_path = {}
        self._meteor_selection_image_sizes = {}
        read_errors: list[str] = []
        for image_path in paths:
            try:
                self._meteor_selection_boxes_by_path[image_path] = load_meteor_selection(image_path)
            except ValueError as exc:
                self._meteor_selection_boxes_by_path[image_path] = []
                read_errors.append(f"{image_path.name}：{exc}")
        self._meteor_selection_current_index = 0 if paths else -1
        self._refresh_meteor_selection_table()
        self._show_meteor_selection_current_image()
        if read_errors:
            self.ui.statusbar.showMessage("已有框选文件无法读取，已按空框选导入：" + "；".join(read_errors), 12000)

    def show_previous_meteor_image(self) -> None:
        """显示列表中的上一张图像。"""

        self._set_meteor_selection_current_index(self._meteor_selection_current_index - 1)

    def show_next_meteor_image(self) -> None:
        """显示列表中的下一张图像。"""

        self._set_meteor_selection_current_index(self._meteor_selection_current_index + 1)

    def _set_meteor_selection_current_index(self, index: int) -> None:
        paths = self._meteor_selection_paths
        if not paths:
            return
        self._store_current_meteor_boxes()
        self._meteor_selection_current_index = max(0, min(int(index), len(paths) - 1))
        self._show_meteor_selection_current_image()

    def _show_meteor_selection_current_image(self) -> None:
        paths = self._meteor_selection_paths
        if not paths or self._meteor_selection_current_index < 0:
            self._update_meteor_selection_controls()
            return
        current_index = max(0, min(self._meteor_selection_current_index, len(paths) - 1))
        self._meteor_selection_current_index = current_index
        image_path = paths[current_index]
        self.ui.labelMeteorSelectionPreviewTitle.setText(f"{current_index + 1}/{len(paths)}  {image_path.name}")
        self.ui.labelMeteorSelectionPreviewTitle.setToolTip(str(image_path))

        try:
            preview = load_image_preview(image_path)
        except Exception as exc:  # noqa: BLE001 - 单图预览失败不能阻断其他图片的框选。
            self._meteor_selection_image_sizes.pop(image_path, None)
            self.ui.meteorSelectionView.clear_image()
            self.ui.labelMeteorSelectionCaptureTime.setText("拍摄时间：无法读取")
            self.ui.labelMeteorSelectionCaptureTime.setToolTip(str(exc))
            self.ui.statusbar.showMessage(f"流星图片预览读取失败：{image_path.name}：{exc}", 10000)
        else:
            image_size = (preview.original_width, preview.original_height)
            self._meteor_selection_image_sizes[image_path] = image_size
            self.ui.meteorSelectionView.set_image(preview.image, *image_size)
            self.ui.meteorSelectionView.set_boxes(self._meteor_selection_boxes_by_path.get(image_path, []))
            self._update_meteor_selection_capture_time(image_path)

        self._refresh_meteor_selection_table()
        self._select_current_meteor_selection_table_row()
        self._update_meteor_selection_controls()

    def _update_meteor_selection_capture_time(self, image_path: Path) -> None:
        """读取并显示当前图像的 EXIF/XMP 拍摄时间。"""

        try:
            item = read_image_capture_time(image_path)
            local_time = sequence_item_local_datetime(item, self.ui.doubleSpinBoxUtcOffset.value())
        except Exception as exc:  # noqa: BLE001 - 部分相机图像不含时间字段，仍允许框选和保存。
            self.ui.labelMeteorSelectionCaptureTime.setText("拍摄时间：未读取")
            self.ui.labelMeteorSelectionCaptureTime.setToolTip(str(exc))
            return
        self.ui.labelMeteorSelectionCaptureTime.setText(
            f"拍摄时间：{local_time.strftime('%Y-%m-%d %H:%M:%S')}（{item.capture_time_source}）"
        )
        self.ui.labelMeteorSelectionCaptureTime.setToolTip(str(image_path))

    def _handle_meteor_selection_time_context_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        """在 UTC 偏移变更后刷新当前图像显示的本地拍摄时间。"""

        if self._has_current_meteor_selection_image():
            self._update_meteor_selection_capture_time(
                self._meteor_selection_paths[self._meteor_selection_current_index]
            )

    def _store_current_meteor_boxes(self) -> None:
        """将视图中的框选同步到当前图像的内存状态。"""

        index = self._meteor_selection_current_index
        if 0 <= index < len(self._meteor_selection_paths):
            image_path = self._meteor_selection_paths[index]
            if image_path in self._meteor_selection_image_sizes:
                self._meteor_selection_boxes_by_path[image_path] = self.ui.meteorSelectionView.boxes()

    def _handle_meteor_boxes_changed(self, boxes: list[MeteorBox]) -> None:
        """响应专用视图的新增或清除框选操作。"""

        index = self._meteor_selection_current_index
        if not 0 <= index < len(self._meteor_selection_paths):
            return
        self._meteor_selection_boxes_by_path[self._meteor_selection_paths[index]] = list(boxes)
        self._refresh_meteor_selection_table()
        self._select_current_meteor_selection_table_row()
        self._update_meteor_selection_controls()

    def clear_meteor_boxes(self) -> None:
        """清除当前图像的所有流星框选。"""

        if not self._has_current_meteor_selection_image():
            return
        self.ui.meteorSelectionView.clear_boxes()

    def save_all_meteor_boxes(self) -> None:
        """将所有存在流星框选的图像写入各自同目录 JSON 文件。"""

        self._store_current_meteor_boxes()
        selected_paths = [
            image_path
            for image_path in self._meteor_selection_paths
            if self._meteor_selection_boxes_by_path.get(image_path, [])
        ]
        if not selected_paths:
            self.ui.statusbar.showMessage("没有流星框选需要保存。", 5000)
            return

        saved_count = 0
        saved_box_count = 0
        failures: list[str] = []
        for image_path in selected_paths:
            try:
                image_size = self._meteor_selection_image_size(image_path)
                save_meteor_selection(
                    image_path,
                    image_size[0],
                    image_size[1],
                    self._meteor_selection_boxes_by_path[image_path],
                )
            except (OSError, ValueError) as exc:
                failures.append(f"{image_path.name}：{exc}")
                continue
            saved_count += 1
            saved_box_count += len(self._meteor_selection_boxes_by_path[image_path])
        self._refresh_meteor_selection_table()
        if failures:
            self.ui.statusbar.showMessage(
                f"已保存 {saved_count} 张图片、{saved_box_count} 个流星框选；失败：" + "；".join(failures),
                12000,
            )
            return
        self.ui.statusbar.showMessage(f"已保存 {saved_count} 张图片、{saved_box_count} 个流星框选。", 8000)

    def _meteor_selection_image_size(self, image_path: Path) -> tuple[int, int]:
        """返回图像原始尺寸；未预览的图片会在保存时按需读取。"""

        image_size = self._meteor_selection_image_sizes.get(image_path)
        if image_size is not None:
            return image_size
        preview = load_image_preview(image_path)
        image_size = (preview.original_width, preview.original_height)
        self._meteor_selection_image_sizes[image_path] = image_size
        return image_size

    def _refresh_meteor_selection_table(self) -> None:
        """刷新左侧图片列表及每张图像的流星数。"""

        table = self.ui.tableWidgetMeteorSelectionImages
        old_state = table.blockSignals(True)
        try:
            table.setRowCount(len(self._meteor_selection_paths))
            for row, image_path in enumerate(self._meteor_selection_paths):
                image_index = row + 1
                count = len(self._meteor_selection_boxes_by_path.get(image_path, []))
                index_item = self._new_read_only_meteor_table_item(str(image_index))
                name_item = self._new_read_only_meteor_table_item(image_path.name)
                count_item = self._new_read_only_meteor_table_item(str(count))
                for item in (index_item, name_item, count_item):
                    item.setData(METEOR_SELECTION_INDEX_ROLE, row)
                    item.setToolTip(str(image_path))
                    if count:
                        item.setBackground(QBrush(METEOR_SELECTION_ROW_GREEN))
                table.setItem(row, METEOR_SELECTION_INDEX_COLUMN, index_item)
                table.setItem(row, METEOR_SELECTION_NAME_COLUMN, name_item)
                table.setItem(row, METEOR_SELECTION_COUNT_COLUMN, count_item)
        finally:
            table.blockSignals(old_state)

    def _new_read_only_meteor_table_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _handle_meteor_selection_table_double_clicked(self, row: int, _column: int) -> None:
        item = self.ui.tableWidgetMeteorSelectionImages.item(row, METEOR_SELECTION_INDEX_COLUMN)
        if item is None:
            return
        try:
            index = int(item.data(METEOR_SELECTION_INDEX_ROLE))
        except (TypeError, ValueError):
            return
        self._set_meteor_selection_current_index(index)

    def _select_current_meteor_selection_table_row(self) -> None:
        current_index = self._meteor_selection_current_index
        table = self.ui.tableWidgetMeteorSelectionImages
        if current_index < 0 or current_index >= table.rowCount():
            table.clearSelection()
            return
        table.selectRow(current_index)
        current_item = table.item(current_index, METEOR_SELECTION_NAME_COLUMN)
        if current_item is not None:
            table.scrollToItem(current_item, QAbstractItemView.PositionAtCenter)

    def _has_current_meteor_selection_image(self) -> bool:
        return 0 <= self._meteor_selection_current_index < len(self._meteor_selection_paths)

    def _update_meteor_selection_controls(self) -> None:
        has_current_image = self._has_current_meteor_selection_image()
        current_index = self._meteor_selection_current_index
        total_count = len(self._meteor_selection_paths)
        self.ui.toolButtonMeteorSelectionPrevious.setEnabled(has_current_image and current_index > 0)
        self.ui.toolButtonMeteorSelectionNext.setEnabled(has_current_image and current_index < total_count - 1)
        self.ui.pushButtonClearMeteorBoxes.setEnabled(has_current_image)
        self.ui.pushButtonSaveAllMeteorBoxes.setEnabled(
            any(self._meteor_selection_boxes_by_path.get(image_path, []) for image_path in self._meteor_selection_paths)
        )
