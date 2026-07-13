"""流星框选页面的导入、预览、列表与保存控制。"""

from __future__ import annotations

import shutil
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QBrush, QColor, QPalette
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QFileDialog, QHeaderView, QMessageBox, QTableWidgetItem

from ..image_sequence import read_image_capture_time, sequence_item_local_datetime
from ..meteor_detection import (
    MeteorDetectionOptions,
    load_meteor_detection_options,
    save_meteor_detection_options,
)
from ..meteor_selection import MeteorBox, load_meteor_selection, meteor_json_path, save_meteor_selection
from ..raw_image_preview import METEOR_IMAGE_FILE_FILTER, load_meteor_image_preview
from .metdet_worker_client import MetDetWorkerClient
from .meteor_detection_options_dialog import MeteorDetectionOptionsDialog


METEOR_SELECTION_INDEX_COLUMN = 0
METEOR_SELECTION_NAME_COLUMN = 1
METEOR_SELECTION_COUNT_COLUMN = 2
METEOR_SELECTION_INDEX_ROLE = Qt.UserRole + 31
METEOR_SELECTION_ROW_GREEN = QColor(220, 252, 231)
METEOR_SIDECAR_JSON_SEPARATORS = ("_", ".", "-")


def meteor_sidecar_json_paths(image_path: str | Path) -> list[Path]:
    """返回与图片主文件名对应的全部同目录 JSON，匹配时避免相似前缀误命中。"""

    path = Path(image_path).expanduser()
    image_stem = path.stem.casefold()
    sidecars: list[Path] = []
    for candidate in path.parent.iterdir():
        if not candidate.is_file() or candidate.suffix.casefold() != ".json":
            continue
        candidate_stem = candidate.stem.casefold()
        if candidate_stem == image_stem or any(
            candidate_stem.startswith(image_stem + separator)
            for separator in METEOR_SIDECAR_JSON_SEPARATORS
        ):
            sidecars.append(candidate)
    return sorted(sidecars, key=lambda item: item.name.casefold())


class MeteorSelectionMixin:
    """管理流星图片列表和每张图像的框选数据。"""

    def _init_meteor_selection_page(self) -> None:
        """初始化页面状态和表格列宽。"""

        self._meteor_selection_paths: list[Path] = []
        self._meteor_selection_boxes_by_path: dict[Path, list[MeteorBox]] = {}
        self._meteor_selection_image_sizes: dict[Path, tuple[int, int]] = {}
        self._meteor_selection_dirty_paths: set[Path] = set()
        self._meteor_selection_current_index = -1
        self._meteor_detection_options = load_meteor_detection_options()
        self._meteor_detection_client = MetDetWorkerClient(self)
        self._meteor_detection_active = False
        self._meteor_detection_request_id: str | None = None
        self._meteor_detection_job_paths: list[Path] = []
        self._meteor_detection_failures: list[str] = []
        self._meteor_detection_success_count = 0
        self._meteor_detection_detected_count = 0
        self._meteor_detection_active_provider = ""
        self._meteor_detection_engine_status = "正在启动检测引擎…"
        self.ui.meteorSelectionView.set_touchpad_pinch_zoom_enabled(
            bool(getattr(getattr(self, "ui_config", None), "touchpad_pinch_zoom_enabled", True))
        )
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
        self.ui.pushButtonClearMeteorImports.clicked.connect(self.clear_all_imported_meteor_images)
        self.ui.pushButtonClearMeteorBoxes.clicked.connect(self.clear_meteor_boxes)
        self.ui.pushButtonSaveAllMeteorBoxes.clicked.connect(self.save_all_meteor_boxes)
        self.ui.pushButtonMeteorDetectionOptions.clicked.connect(self.show_meteor_detection_options)
        self.ui.pushButtonAutoDetectMeteors.clicked.connect(self.toggle_automatic_meteor_detection)
        self.ui.pushButtonMoveMeteorFiles.clicked.connect(self.move_meteor_files)
        self.ui.toolButtonMeteorSelectionPrevious.clicked.connect(self.show_previous_meteor_image)
        self.ui.toolButtonMeteorSelectionNext.clicked.connect(self.show_next_meteor_image)
        self.ui.tableWidgetMeteorSelectionImages.cellDoubleClicked.connect(
            self._handle_meteor_selection_table_double_clicked
        )
        self.ui.meteorSelectionView.boxesChanged.connect(self._handle_meteor_boxes_changed)
        self.ui.doubleSpinBoxUtcOffset.valueChanged.connect(self._handle_meteor_selection_time_context_changed)
        self.ui.tableWidgetMeteorSelectionImages.cellClicked.connect(self._handle_meteor_selection_table_clicked)
        self._meteor_detection_client.ready.connect(self._handle_meteor_detection_worker_ready)
        self._meteor_detection_client.messageReceived.connect(self._handle_meteor_detection_message)
        self._meteor_detection_client.workerError.connect(self._handle_meteor_detection_worker_error)
        self._meteor_detection_client.workerStopped.connect(self._handle_meteor_detection_worker_stopped)
        QTimer.singleShot(0, self._start_meteor_detection_worker)

    def _reset_meteor_selection_page(self) -> None:
        """清空页面显示与内存状态。"""

        self._meteor_selection_paths = []
        self._meteor_selection_boxes_by_path = {}
        self._meteor_selection_image_sizes = {}
        self._meteor_selection_dirty_paths = set()
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

        if self._meteor_detection_active:
            return
        fallback = self._meteor_selection_paths[0].parent if self._meteor_selection_paths else Path.cwd()
        selected_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "导入流星图片",
            str(self._import_dialog_directory(fallback)),
            METEOR_IMAGE_FILE_FILTER,
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
        self._meteor_selection_dirty_paths = set()
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

    def clear_all_imported_meteor_images(self) -> None:
        """清空当前流星图片批次，但不删除或修改磁盘上的图片与 JSON。"""

        if self._meteor_detection_active or not self._meteor_selection_paths:
            return
        if self._meteor_selection_dirty_paths:
            answer = QMessageBox.question(
                self,
                "存在未保存的框选修改",
                "当前有尚未保存的流星框选修改。清除所有导入后，这些内存修改会丢失。\n\n是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._reset_meteor_selection_page()
        self.ui.statusbar.showMessage("已清除当前批次的所有导入，可以继续导入下一批图片。", 6000)

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
            preview = load_meteor_image_preview(image_path)
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
        image_path = self._meteor_selection_paths[index]
        self._meteor_selection_boxes_by_path[image_path] = list(boxes)
        self._meteor_selection_dirty_paths.add(image_path)
        self._refresh_meteor_selection_table()
        self._select_current_meteor_selection_table_row()
        self._update_meteor_selection_controls()
        self.ui.statusbar.showMessage("流星框选已修改，请点击“保存所有框选”写入文件。", 8000)

    def clear_meteor_boxes(self) -> None:
        """清除当前图像的所有流星框选。"""

        if not self._has_current_meteor_selection_image():
            return
        self.ui.meteorSelectionView.clear_boxes()

    def save_all_meteor_boxes(self) -> None:
        """保存全部框选；待保存图片没有框时删除其 Meteor JSON。"""

        self._store_current_meteor_boxes()
        selected_paths = [
            image_path
            for image_path in self._meteor_selection_paths
            if self._meteor_selection_boxes_by_path.get(image_path, [])
        ]
        empty_dirty_paths = [
            image_path
            for image_path in self._meteor_selection_paths
            if image_path in self._meteor_selection_dirty_paths
            and not self._meteor_selection_boxes_by_path.get(image_path, [])
        ]
        if not selected_paths and not empty_dirty_paths:
            self.ui.statusbar.showMessage("没有流星框选修改需要保存。", 5000)
            return

        saved_count = 0
        saved_box_count = 0
        deleted_json_count = 0
        completed_paths: set[Path] = set()
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
            completed_paths.add(image_path)
        for image_path in empty_dirty_paths:
            json_path = meteor_json_path(image_path)
            json_existed = json_path.exists()
            try:
                json_path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(f"{image_path.name}：无法删除无流星框选文件：{exc}")
                continue
            deleted_json_count += int(json_existed)
            completed_paths.add(image_path)
        self._meteor_selection_dirty_paths.difference_update(completed_paths)
        self._refresh_meteor_selection_table()
        self._update_meteor_selection_controls()
        summary = f"已保存 {saved_count} 张图片、{saved_box_count} 个流星框选"
        if empty_dirty_paths:
            summary += f"；已清理 {deleted_json_count} 个无流星 Meteor JSON"
        if failures:
            self.ui.statusbar.showMessage(
                summary + "；失败：" + "；".join(failures),
                12000,
            )
            return
        self.ui.statusbar.showMessage(summary + "。", 8000)

    def _meteor_selection_image_size(self, image_path: Path) -> tuple[int, int]:
        """返回图像原始尺寸；未预览的图片会在保存时按需读取。"""

        image_size = self._meteor_selection_image_sizes.get(image_path)
        if image_size is not None:
            return image_size
        preview = load_meteor_image_preview(image_path)
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
        if self._meteor_detection_active:
            self._select_current_meteor_selection_table_row()
            return
        item = self.ui.tableWidgetMeteorSelectionImages.item(row, METEOR_SELECTION_INDEX_COLUMN)
        if item is None:
            return
        try:
            index = int(item.data(METEOR_SELECTION_INDEX_ROLE))
        except (TypeError, ValueError):
            return
        self._set_meteor_selection_current_index(index)

    def _handle_meteor_selection_table_clicked(self, _row: int, _column: int) -> None:
        """检测期间阻止鼠标把高亮行移离右侧正在显示的图像。"""

        if self._meteor_detection_active:
            QTimer.singleShot(0, self._select_current_meteor_selection_table_row)

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
        detecting = self._meteor_detection_active
        self.ui.pushButtonImportMeteorImages.setEnabled(not detecting)
        self.ui.pushButtonClearMeteorImports.setEnabled(not detecting and bool(self._meteor_selection_paths))
        self.ui.pushButtonMeteorDetectionOptions.setEnabled(not detecting)
        self.ui.toolButtonMeteorSelectionPrevious.setEnabled(not detecting and has_current_image and current_index > 0)
        self.ui.toolButtonMeteorSelectionNext.setEnabled(
            not detecting and has_current_image and current_index < total_count - 1
        )
        self.ui.pushButtonClearMeteorBoxes.setEnabled(not detecting and has_current_image)
        self.ui.pushButtonSaveAllMeteorBoxes.setEnabled(
            not detecting
            and (
                bool(self._meteor_selection_dirty_paths)
                or any(
                    self._meteor_selection_boxes_by_path.get(image_path, [])
                    for image_path in self._meteor_selection_paths
                )
            )
        )
        has_meteor_files = any(
            self._meteor_selection_boxes_by_path.get(image_path, []) for image_path in self._meteor_selection_paths
        )
        self.ui.pushButtonMoveMeteorFiles.setEnabled(
            not detecting and not self._meteor_selection_dirty_paths and has_meteor_files
        )
        self.ui.meteorSelectionView.set_box_editing_enabled(not detecting)
        if detecting:
            self.ui.pushButtonAutoDetectMeteors.setText("取消检测")
            self.ui.pushButtonAutoDetectMeteors.setToolTip("终止当前检测；引擎会在后台重新启动。")
            self.ui.pushButtonAutoDetectMeteors.setEnabled(True)
        else:
            self.ui.pushButtonAutoDetectMeteors.setText("自动检测")
            if self._meteor_selection_dirty_paths:
                self.ui.pushButtonAutoDetectMeteors.setToolTip("请先保存当前流星框选修改。")
            else:
                self.ui.pushButtonAutoDetectMeteors.setToolTip(self._meteor_detection_engine_status)
            self.ui.pushButtonAutoDetectMeteors.setEnabled(
                bool(self._meteor_selection_paths)
                and not self._meteor_selection_dirty_paths
                and self._meteor_detection_client.is_ready
            )

    def show_meteor_detection_options(self) -> None:
        """显示自动检测选项，保存后重启 worker 使引擎位置立即生效。"""

        if self._meteor_detection_active:
            return
        dialog = MeteorDetectionOptionsDialog(self._meteor_detection_options, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        options = dialog.options()
        if not save_meteor_detection_options(options):
            QMessageBox.warning(self, "无法保存选项", "无法写入 preference.json，本次设置不会保留。")
        self._meteor_detection_options = options
        self._start_meteor_detection_worker()

    def _start_meteor_detection_worker(self) -> None:
        """按当前配置启动或重启检测 worker。"""

        if self._meteor_detection_active:
            return
        self._meteor_detection_engine_status = "正在启动检测引擎…"
        self._update_meteor_selection_controls()
        try:
            self._meteor_detection_client.start(self._meteor_detection_options.engine_path)
        except (OSError, RuntimeError) as exc:
            self._meteor_detection_engine_status = str(exc)
            self.ui.statusbar.showMessage(f"流星检测引擎不可用：{exc}", 12000)
            self._update_meteor_selection_controls()

    def _handle_meteor_detection_worker_ready(self, payload: dict[str, object]) -> None:
        """worker 完成协议握手后开放自动检测按钮。"""

        providers = payload.get("available_providers")
        provider_text = "、".join(str(item) for item in providers) if isinstance(providers, list) else "未知"
        self._meteor_detection_engine_status = f"检测引擎已就绪；可用 Provider：{provider_text}"
        self.ui.statusbar.showMessage(self._meteor_detection_engine_status, 6000)
        self._update_meteor_selection_controls()

    def toggle_automatic_meteor_detection(self) -> None:
        """开始检测；再次点击时按协议终止正在执行的 worker。"""

        if self._meteor_detection_active:
            self.cancel_automatic_meteor_detection()
            return
        if not self._meteor_selection_paths or not self._meteor_detection_client.is_ready:
            return
        if self._meteor_selection_dirty_paths:
            self.ui.statusbar.showMessage("请先点击“保存所有框选”，再开始自动检测。", 8000)
            return
        self._store_current_meteor_boxes()
        job_paths = self._prepare_meteor_detection_paths()
        self._meteor_detection_failures = []
        self._meteor_detection_success_count = 0
        self._meteor_detection_detected_count = 0
        self._meteor_detection_active_provider = ""
        try:
            request_id = self._meteor_detection_client.detect(
                [str(path) for path in job_paths],
                self._meteor_detection_options,
            )
        except RuntimeError as exc:
            self._handle_meteor_detection_worker_error(str(exc))
            return
        self._meteor_detection_request_id = request_id
        self._meteor_detection_job_paths = job_paths
        self._meteor_detection_active = True
        self.ui.statusbar.showMessage(f"准备检测 {len(job_paths)} 张图片…")
        self._update_meteor_selection_controls()

    def _prepare_meteor_detection_paths(self) -> list[Path]:
        """返回全部待检测图片；自动检测始终覆盖已有结果。"""

        return list(self._meteor_selection_paths)

    def cancel_automatic_meteor_detection(self) -> None:
        """取消检测并重启一个干净的 worker。"""

        if not self._meteor_detection_active:
            return
        self._meteor_detection_active = False
        self._meteor_detection_request_id = None
        self._meteor_detection_job_paths = []
        self._meteor_detection_engine_status = "检测已取消，正在重启引擎…"
        self._meteor_detection_client.cancel_active_job()
        self.ui.statusbar.showMessage(self._meteor_detection_engine_status, 6000)
        self._update_meteor_selection_controls()
        QTimer.singleShot(0, self._start_meteor_detection_worker)

    def _handle_meteor_detection_message(self, payload: dict[str, object]) -> None:
        """处理逐图进度、结果和任务结束消息。"""

        message_type = str(payload.get("type") or "")
        if message_type in {"ready", "pong", "shutdown_ack"}:
            return
        request_id = payload.get("request_id")
        if request_id != self._meteor_detection_request_id:
            return
        if message_type == "progress":
            index = payload.get("index", "?")
            total = payload.get("total", len(self._meteor_selection_paths))
            image_name = Path(str(payload.get("image_path") or "")).name
            self.ui.statusbar.showMessage(f"正在检测 {index}/{total}：{image_name}")
        elif message_type == "model_ready":
            provider = payload.get("provider") or "未知"
            self._meteor_detection_active_provider = str(provider)
            self.ui.statusbar.showMessage(f"流星模型已就绪（{provider}），开始逐张检测…")
        elif message_type == "result":
            self._apply_meteor_detection_result(payload)
        elif message_type == "job_finished":
            self._finish_automatic_meteor_detection(payload)
        elif message_type == "request_error":
            message = str(payload.get("message") or "未知协议错误")
            self._meteor_detection_failures.append(message)
            self._finish_automatic_meteor_detection({"failure_count": 1})

    def _apply_meteor_detection_result(self, payload: dict[str, object]) -> None:
        """重载当前结果 JSON，并把列表和预览定位到刚处理完的图片。"""

        try:
            index = int(payload.get("index", 0)) - 1
        except (TypeError, ValueError):
            return
        if not 0 <= index < len(self._meteor_detection_job_paths):
            return
        image_path = self._meteor_detection_job_paths[index]
        try:
            selection_index = self._meteor_selection_paths.index(image_path)
        except ValueError:
            return
        error = payload.get("error")
        if error:
            self._meteor_detection_failures.append(f"{image_path.name}：{error}")
        else:
            self._meteor_detection_success_count += 1
            detected_boxes = payload.get("meteor_boxes")
            if detected_boxes == []:
                try:
                    meteor_json_path(image_path).unlink(missing_ok=True)
                except OSError as exc:
                    self._meteor_detection_failures.append(f"{image_path.name}：无法清除旧框选：{exc}")
                    boxes = self._meteor_selection_boxes_by_path.get(image_path, [])
                else:
                    boxes = []
            else:
                try:
                    boxes = load_meteor_selection(image_path)
                except ValueError as exc:
                    self._meteor_detection_failures.append(f"{image_path.name}：{exc}")
                    boxes = self._meteor_selection_boxes_by_path.get(image_path, [])
            self._meteor_selection_boxes_by_path[image_path] = list(boxes)
            self._meteor_selection_dirty_paths.discard(image_path)
            if boxes:
                self._meteor_detection_detected_count += 1
        self._meteor_selection_current_index = selection_index
        self._show_meteor_selection_current_image()

    def _finish_automatic_meteor_detection(self, payload: dict[str, object]) -> None:
        """恢复手动交互，并汇总检测结果。"""

        if not self._meteor_detection_active:
            return
        self._meteor_detection_active = False
        self._meteor_detection_request_id = None
        self._meteor_detection_job_paths = []
        provider = self._meteor_detection_active_provider
        self._update_meteor_selection_controls()
        failure_count = len(self._meteor_detection_failures)
        summary = (
            f"自动检测完成：成功 {self._meteor_detection_success_count} 张，"
            f"检测到流星 {self._meteor_detection_detected_count} 张"
        )
        if provider:
            summary += f"，Provider：{provider}"
        if failure_count:
            summary += f"，失败 {failure_count} 张：" + "；".join(self._meteor_detection_failures[:5])
            self.ui.statusbar.showMessage(summary, 15000)
        else:
            self.ui.statusbar.showMessage(summary, 10000)

    def _handle_meteor_detection_worker_error(self, message: str) -> None:
        """显示进程级错误并恢复页面交互。"""

        was_active = self._meteor_detection_active
        self._meteor_detection_active = False
        self._meteor_detection_request_id = None
        self._meteor_detection_job_paths = []
        self._meteor_detection_engine_status = message
        self._update_meteor_selection_controls()
        self.ui.statusbar.showMessage(message, 15000)
        if was_active:
            QMessageBox.warning(self, "流星检测中断", message)

    def _handle_meteor_detection_worker_stopped(self) -> None:
        """worker 停止后撤销就绪状态；取消路径会另行安排重启。"""

        self._update_meteor_selection_controls()

    def move_meteor_files(self) -> None:
        """把有框选的图像及其全部关联 JSON 移到用户选择的目录。"""

        if self._meteor_detection_active:
            return
        if self._meteor_selection_dirty_paths:
            self.ui.statusbar.showMessage("请先点击“保存所有框选”，再移动流星文件。", 8000)
            return
        self._store_current_meteor_boxes()
        source_paths = [
            path for path in self._meteor_selection_paths if self._meteor_selection_boxes_by_path.get(path, [])
        ]
        if not source_paths:
            self.ui.statusbar.showMessage("没有检测到流星的图片可移动。", 5000)
            return
        fallback = source_paths[0].parent
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "选择流星文件目标文件夹",
            str(self._import_dialog_directory(fallback)),
        )
        if not selected_directory:
            return
        target_directory = Path(selected_directory).expanduser().resolve()
        self._remember_import_path(target_directory)

        moved_by_source: dict[Path, Path] = {}
        moved_json_count = 0
        failures: list[str] = []
        for source_path in source_paths:
            target_path = target_directory / source_path.name
            source_json = meteor_json_path(source_path)
            if target_path == source_path:
                continue
            if not source_json.exists():
                try:
                    width, height = self._meteor_selection_image_size(source_path)
                    save_meteor_selection(
                        source_path,
                        width,
                        height,
                        self._meteor_selection_boxes_by_path[source_path],
                    )
                except (OSError, ValueError) as exc:
                    failures.append(f"{source_path.name}：无法保存框选：{exc}")
                    continue
            try:
                sidecar_paths = meteor_sidecar_json_paths(source_path)
            except OSError as exc:
                failures.append(f"{source_path.name}：无法查找关联 JSON：{exc}")
                continue
            move_pairs = [(source_path, target_path)] + [
                (json_path, target_directory / json_path.name) for json_path in sidecar_paths
            ]
            conflicting_targets = [target for source, target in move_pairs if source != target and target.exists()]
            if conflicting_targets:
                conflict_names = "、".join(path.name for path in conflicting_targets)
                failures.append(f"{source_path.name}：目标位置已有同名文件：{conflict_names}")
                continue

            completed_pairs: list[tuple[Path, Path]] = []
            try:
                for source, target in move_pairs:
                    shutil.move(str(source), str(target))
                    completed_pairs.append((source, target))
            except OSError as exc:
                rollback_failures: list[str] = []
                for original, moved_path in reversed(completed_pairs):
                    if not moved_path.exists() or original.exists():
                        continue
                    try:
                        shutil.move(str(moved_path), str(original))
                    except OSError as rollback_exc:
                        rollback_failures.append(f"{moved_path.name}：{rollback_exc}")
                if target_path.exists() and not source_path.exists():
                    moved_by_source[source_path] = target_path
                rollback_text = ""
                if rollback_failures:
                    rollback_text = "；部分文件回滚失败：" + "；".join(rollback_failures)
                failures.append(f"{source_path.name}：{exc}{rollback_text}")
                continue
            moved_by_source[source_path] = target_path
            moved_json_count += len(sidecar_paths)

        if moved_by_source:
            self._replace_moved_meteor_paths(moved_by_source)
        message = f"已移动 {len(moved_by_source)} 张流星图片及 {moved_json_count} 个关联 JSON。"
        if failures:
            message += " 未移动：" + "；".join(failures)
            self.ui.statusbar.showMessage(message, 15000)
        else:
            self.ui.statusbar.showMessage(message, 10000)

    def _replace_moved_meteor_paths(self, moved_by_source: dict[Path, Path]) -> None:
        """移动完成后保留图片顺序、框选、尺寸与当前预览位置。"""

        old_boxes = self._meteor_selection_boxes_by_path
        old_sizes = self._meteor_selection_image_sizes
        old_dirty_paths = self._meteor_selection_dirty_paths
        new_paths = [moved_by_source.get(path, path) for path in self._meteor_selection_paths]
        self._meteor_selection_paths = new_paths
        self._meteor_selection_boxes_by_path = {
            moved_by_source.get(path, path): boxes for path, boxes in old_boxes.items()
        }
        self._meteor_selection_image_sizes = {
            moved_by_source.get(path, path): size for path, size in old_sizes.items()
        }
        self._meteor_selection_dirty_paths = {
            moved_by_source.get(path, path) for path in old_dirty_paths
        }
        self._refresh_meteor_selection_table()
        self._show_meteor_selection_current_image()

    def _shutdown_meteor_detection_worker(self) -> None:
        """主窗口退出时关闭外部 worker，避免遗留后台进程。"""

        self._meteor_detection_active = False
        self._meteor_detection_request_id = None
        self._meteor_detection_job_paths = []
        self._meteor_detection_client.stop()
