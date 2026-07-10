from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from ..alignment.constants import MIN_ALIGNMENT_PAIRS
from .app_workers import ImageSequenceCollectWorker
from ..image_preview import IMAGE_FILE_FILTER
from ..image_sequence import ImageSequenceItem, RejectedSequenceImage, sequence_item_time_delta_seconds
from ..qt_tasks import create_progress_dialog, start_qt_worker_task
from ..sequence_constants import IMAGE_SEQUENCE_IMPORT_PROGRESS_MIN_VISIBLE_MS

class SequenceImportMixin:
    """图像序列导入、首帧准备和导入状态管理。"""

    def import_image_sequence(self) -> None:
        if getattr(self, "_sequence_processing_active", False):
            QMessageBox.information(self, "正在处理序列", "图像序列仍在处理，请等待完成后再导入新序列。")
            return
        if getattr(self, "_sequence_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入序列", "当前已有图像序列正在导入，请稍候。")
            return
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return

        default_dir = Path(self.current_image_preview.path).parent if self.current_image_preview is not None else Path.cwd()
        default_dir = self._import_dialog_directory(default_dir)
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "导入序列图像",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_paths:
            return
        self._remember_import_path(file_paths)

        self.start_image_sequence_import(file_paths)

    def start_image_sequence_import(self, file_paths: list[str] | tuple[str, ...]) -> None:
        """后台读取序列图像 EXIF 时间，并显示导入进度弹窗。"""
        if getattr(self, "_sequence_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入序列", "当前已有图像序列正在导入，请稍候。")
            return

        self._set_image_import_controls_enabled(False)
        self.ui.statusbar.showMessage(f"正在导入序列图像并读取 EXIF: {len(file_paths)} 张")

        progress = create_progress_dialog(
            self,
            title="正在导入序列图像",
            label_text="正在读取序列图像 EXIF 拍摄时间...\n已选择 {count} 张图像".format(count=len(file_paths)),
            minimum=0,
            maximum=0,
        )
        progress.setValue(0)
        force_show = getattr(progress, "forceShow", None)
        if callable(force_show):
            force_show()
        if QApplication.platformName().lower() != "offscreen":
            progress.raise_()
            progress.activateWindow()
        progress.repaint()
        QApplication.processEvents()

        worker = ImageSequenceCollectWorker(file_paths)
        task = start_qt_worker_task(
            parent=self,
            worker=worker,
            finished_signal=worker.finished,
            failed_signal=worker.failed,
            on_finished=self._handle_image_sequence_import_finished,
            on_failed=self._handle_image_sequence_import_failed,
            on_cleanup=self._cleanup_image_sequence_import,
            progress_dialog=progress,
            start_delay_ms=50,
            minimum_visible_ms=IMAGE_SEQUENCE_IMPORT_PROGRESS_MIN_VISIBLE_MS,
        )

        self._sequence_import_thread = task.thread
        self._sequence_import_worker = task.worker
        self._sequence_import_progress = progress

    def _set_sequence_import_progress_label(self, text: str) -> None:
        progress = getattr(self, "_sequence_import_progress", None)
        if progress is None:
            return
        progress.setLabelText(text)
        progress.repaint()
        QApplication.processEvents()

    def _handle_image_sequence_import_finished(self, result: object) -> None:
        try:
            items, rejected = result  # type: ignore[misc]
            if not isinstance(items, list) or not isinstance(rejected, list):
                raise ValueError("序列导入结果格式无效。")
            self._apply_collected_image_sequence(items, rejected)
        except Exception as exc:  # noqa: BLE001 - 主线程恢复序列状态时要把错误反馈给用户。
            self.ui.statusbar.showMessage(f"序列导入失败: {exc}")
            QMessageBox.critical(self, "序列导入失败", str(exc))

    def _handle_image_sequence_import_failed(self, error_message: str) -> None:
        self.ui.statusbar.showMessage(f"序列导入失败: {error_message}")
        QMessageBox.critical(self, "序列导入失败", error_message)

    def _cleanup_image_sequence_import(self) -> None:
        if self._sequence_import_progress is not None:
            self._sequence_import_progress.close()
        self._sequence_import_thread = None
        self._sequence_import_worker = None
        self._sequence_import_progress = None
        self._set_image_import_controls_enabled(getattr(self, "_image_import_thread", None) is None)
        self._update_image_sequence_controls()

    def _apply_collected_image_sequence(
        self,
        items: list[ImageSequenceItem],
        rejected: list[RejectedSequenceImage],
    ) -> None:
        if not items:
            self._reset_image_sequence_status()
            message = "所选图像都没有读到 EXIF 拍摄时间，已全部跳过。"
            if rejected:
                message += "\n\n" + self._rejected_sequence_summary(rejected)
            QMessageBox.warning(self, "未导入序列图像", message)
            self.ui.statusbar.showMessage("序列导入失败：全部图像缺少可用 EXIF 拍摄时间。")
            return

        self._set_sequence_import_progress_label("正在整理序列表并生成第一帧预览...")
        self._clear_image_sequence_preview_cache()
        self._image_sequence_items = items
        self._image_sequence_current_index = 0
        self._update_imported_sequence_status(rejected)
        self._refresh_image_sequence_table()
        self._set_image_sequence_preview_index(0)
        if hasattr(self.ui, "tabImageSequence"):
            self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabImageSequence)
        QApplication.processEvents()
        if rejected:
            QMessageBox.warning(
                self,
                "部分图像已跳过",
                "有 {count} 张图像没有可用 EXIF 拍摄时间，未加入处理序列。\n\n{summary}".format(
                    count=len(rejected),
                    summary=self._rejected_sequence_summary(rejected),
                ),
            )

        span_seconds = sequence_item_time_delta_seconds(items[-1], items[0])
        if span_seconds > 24.0 * 3600.0:
            QMessageBox.warning(
                self,
                "序列时间跨度较长",
                f"当前序列时间跨度约 {span_seconds / 3600.0:.2f} 小时。当前批处理假定单个序列不超过一天。",
            )

        self._apply_sequence_observation_time(items[0], emit_signal=True)
        first_starpair_path = self._sequence_starpair_json_path(items[0].path)
        if first_starpair_path.exists():
            self._start_first_sequence_session_import(first_starpair_path, len(items))
            self.ui.statusbar.showMessage(
                f"已导入序列 {len(items)} 张，正在后台载入第一张配对 JSON: {first_starpair_path}"
            )
            return

        self.ui.statusbar.showMessage(
            f"已导入序列 {len(items)} 张，第一张没有配对 JSON，正在载入点选基准: {items[0].path}"
        )
        self.start_single_image_import(items[0].path, preserve_sequence_status=True)

    def _start_first_sequence_session_import(self, json_path: Path, sequence_count: int) -> None:
        if getattr(self, "_json_import_thread", None) is not None:
            self.ui.statusbar.showMessage(
                f"已导入序列 {sequence_count} 张；JSON 导入器正忙，暂未载入第一张配对 JSON: {json_path}"
            )
            return
        self.load_star_pair_session(
            json_path,
            switch_to_reference=False,
            show_progress=False,
            clear_input_name="第一帧配对 JSON",
        )

    def _rejected_sequence_summary(self, rejected: list[RejectedSequenceImage], limit: int = 10) -> str:
        lines = [f"{item.path.name}: {item.reason}" for item in rejected[:limit]]
        if len(rejected) > limit:
            lines.append(f"... 另有 {len(rejected) - limit} 张")
        return "\n".join(lines)

    def _current_sequence_first_item(self) -> ImageSequenceItem:
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            raise ValueError("请先导入图像序列。")
        return items[0]

    def _should_skip_auto_import_star_pair_session(self, image_path: Path) -> bool:
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            return False
        try:
            return Path(image_path).expanduser().resolve() == items[0].path.expanduser().resolve()
        except OSError:
            return False

    def _ensure_sequence_ready_for_processing(self) -> None:
        first_item = self._current_sequence_first_item()
        if self.current_image_preview is None:
            raise ValueError("请先让序列第一张图像导入完成，并完成第一张图的星点匹配。")
        current_path = Path(self.current_image_preview.path).expanduser().resolve()
        if current_path != first_item.path.expanduser().resolve():
            raise ValueError("当前真实图像不是序列第一张。请重新导入序列，或先载入序列第一张后再处理。")
        if self._star_pair_position_count() < MIN_ALIGNMENT_PAIRS:
            raise ValueError(f"第一张图至少需要 {MIN_ALIGNMENT_PAIRS} 对星点匹配后才能处理序列。")
        if self._source_astrometric_model is None:
            self._update_reference_alignment_transform()
        if self._source_astrometric_model is None:
            raise ValueError(self._source_model_error_message or "第一张图的源图映射尚未就绪。")
