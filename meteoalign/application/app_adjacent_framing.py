"""星点匹配页的参考图像导入与粗略取景交互。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

from ..adjacent_alignment import (
    ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
    ADJACENT_ALIGNMENT_MODE_STARS,
    AdjacentFramingResult,
    adjacent_alignment_mode_display_name,
    load_adjacent_frame_model,
)
from ..qt_tasks import create_progress_dialog, start_qt_worker_task
from ..adjacent_framing_worker import AdjacentFramingWorker
from ..image_preview import IMAGE_FILE_FILTER, load_image_preview
from ..image_sequence import read_image_capture_time, sequence_item_time_delta_seconds
from .adjacent_alignment_settings_dialog import AdjacentAlignmentSettingsDialog
from .image_preview_dialog import ImagePreviewDialog


class AdjacentFramingMixin:
    """管理相邻已标定图像 A 驱动当前图像 B 粗略取景的 UI 状态。"""

    ui: object
    current_image_preview: object | None
    _adjacent_image_path: Path | None
    _adjacent_model_json_path: Path | None
    _adjacent_framing_result: AdjacentFramingResult | None
    _adjacent_framing_thread: object | None
    _adjacent_framing_worker: object | None
    _adjacent_framing_progress: QProgressDialog | None
    _rough_alignment_transform: object | None
    _rough_source_astrometric_model: object | None
    image_preview_dialog: ImagePreviewDialog

    def _init_adjacent_framing_defaults(self) -> None:
        """初始化参考图像区域的界面与内存状态。"""

        self._adjacent_image_path = None
        self._adjacent_model_json_path = None
        self._adjacent_framing_result = None
        self._rough_alignment_transform = None
        self._rough_source_astrometric_model = None
        if hasattr(self.ui, "comboBoxAdjacentAlignmentMode"):
            self.ui.comboBoxAdjacentAlignmentMode.setCurrentIndex(0)
        self._set_elided_label_text(self.ui.labelAdjacentFramingStatus, "未计算粗略取景", "")
        self._update_adjacent_framing_controls()

    def _adjacent_alignment_mode(self) -> str:
        if not hasattr(self.ui, "comboBoxAdjacentAlignmentMode"):
            return ADJACENT_ALIGNMENT_MODE_STARS
        if self.ui.comboBoxAdjacentAlignmentMode.currentIndex() == 1:
            return ADJACENT_ALIGNMENT_MODE_LANDSCAPE
        return ADJACENT_ALIGNMENT_MODE_STARS

    def _handle_adjacent_alignment_mode_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        """切换模式后使旧结果失效，避免界面模式与实际粗略模型不一致。"""

        if getattr(self, "_adjacent_framing_result", None) is not None:
            self._clear_adjacent_rough_framing(
                status_text="工作模式已改变，请重新计算粗略取景",
                refresh_alignment=True,
            )
            return
        self._update_adjacent_framing_controls()

    def _update_adjacent_framing_controls(self, *unused) -> None:  # type: ignore[no-untyped-def]
        """根据当前图像、参考图像、图像组和后台任务状态更新操作按钮。"""

        if not hasattr(self.ui, "pushButtonImportAdjacentImage"):
            return
        adjacent_is_current = self._adjacent_image_is_current()
        self._update_adjacent_image_label(adjacent_is_current)
        idle = getattr(self, "_adjacent_framing_thread", None) is None
        self.ui.pushButtonImportAdjacentImage.setEnabled(idle)
        if hasattr(self.ui, "pushButtonPreviewAdjacentImage"):
            self.ui.pushButtonPreviewAdjacentImage.setEnabled(
                getattr(self, "_adjacent_image_path", None) is not None
            )
        self.ui.comboBoxAdjacentAlignmentMode.setEnabled(idle)
        self.ui.pushButtonCalculateAdjacentFraming.setEnabled(
            idle
            and getattr(self, "_adjacent_model_json_path", None) is not None
            and self.current_image_preview is not None
            and not adjacent_is_current
        )
        if hasattr(self.ui, "toolButtonAdjacentAlignmentSettings"):
            self.ui.toolButtonAdjacentAlignmentSettings.setEnabled(idle)

    def _adjacent_image_is_current(self) -> bool:
        """判断参考图像与当前正在处理的图像是否为同一文件。"""

        image_path = getattr(self, "_adjacent_image_path", None)
        current_preview = getattr(self, "current_image_preview", None)
        if image_path is None or current_preview is None:
            return False
        current_path = Path(current_preview.path).expanduser().resolve()
        return image_path == current_path

    def _update_adjacent_image_label(self, adjacent_is_current: bool | None = None) -> None:
        """按导入状态和当前图像关系刷新参考图像名称。"""

        image_path = getattr(self, "_adjacent_image_path", None)
        if image_path is None:
            self._set_elided_label_text(self.ui.labelAdjacentImageModel, "未导入参考图像", "")
            return
        model_path = getattr(self, "_adjacent_model_json_path", None)
        tooltip = f"参考图像：{image_path}"
        if model_path is not None:
            tooltip += f"\nmodel.json：{model_path}"
        if adjacent_is_current is None:
            adjacent_is_current = self._adjacent_image_is_current()
        display_text = "参考图像不能与为当前图像" if adjacent_is_current else image_path.name
        self._set_elided_label_text(self.ui.labelAdjacentImageModel, display_text, tooltip)

    def open_adjacent_alignment_settings(self) -> None:
        """打开当前粗略取景模式的超参数设置窗口。"""

        mode = self._adjacent_alignment_mode()
        dialog = AdjacentAlignmentSettingsDialog(mode, self)
        if not dialog.exec_():
            return
        self.ui.statusbar.showMessage(
            f"已保存{adjacent_alignment_mode_display_name(mode)}参数并立即生效；"
            "已生成的粗略取景结果保持不变。"
        )

    def _clear_adjacent_rough_framing(
        self,
        *,
        status_text: str = "未计算粗略取景",
        refresh_alignment: bool = False,
    ) -> None:
        """清除已生成的粗略模型；保留用户已选择的参考图像及其模型。"""

        self._adjacent_framing_result = None
        self._rough_alignment_transform = None
        self._rough_source_astrometric_model = None
        if hasattr(self.ui, "labelAdjacentFramingStatus"):
            self._set_elided_label_text(self.ui.labelAdjacentFramingStatus, status_text, status_text)
        self._update_adjacent_framing_controls()
        if refresh_alignment and self.current_image_preview is not None:
            self._update_reference_alignment_transform()

    def import_adjacent_image(self) -> None:
        """选择参考图像，并检查同目录下是否已有对应的 model.json。"""

        default_dir = Path.cwd()
        current_path = getattr(self, "_adjacent_image_path", None)
        if current_path is not None:
            default_dir = current_path.parent
        elif self.current_image_preview is not None:
            default_dir = Path(self.current_image_preview.path).expanduser().resolve().parent
        default_dir = self._import_dialog_directory(default_dir)
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入参考图像",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        self._remember_import_path(file_path)
        self.load_adjacent_image(file_path)

    @staticmethod
    def _adjacent_model_path_for_image(image_path: Path) -> Path:
        """返回图像导出源图映射时使用的固定 model.json 路径。"""

        return image_path.with_name(f"{image_path.stem}_model.json")

    def load_adjacent_image(
        self,
        file_path: str | Path,
        *,
        automatic_selection: bool = False,
    ) -> bool:
        """验证参考图像及其 model.json，并更新界面显示。"""

        image_path = Path(file_path).expanduser().resolve()
        model_path = self._adjacent_model_path_for_image(image_path)
        if not model_path.is_file():
            message = (
                "所选图像没有对应的 model.json。\n"
                "请导入已有模型(model.json)的图像。\n\n"
                f"需要的模型文件：{model_path}"
            )
            self.ui.statusbar.showMessage(f"参考图像缺少 model.json：{image_path}")
            QMessageBox.information(self, "参考图像缺少模型", message)
            return False

        try:
            load_adjacent_frame_model(model_path, image_path)
        except Exception as exc:  # noqa: BLE001 - 参考图像或模型错误需要立即给出明确反馈。
            self.ui.statusbar.showMessage(f"导入参考图像失败: {exc}")
            QMessageBox.critical(self, "导入参考图像失败", str(exc))
            return False

        self._adjacent_image_path = image_path
        self._adjacent_model_json_path = model_path
        image_group_assistant = getattr(self, "image_group_assistant", None)
        if image_group_assistant is not None:
            if not automatic_selection:
                image_group_assistant.ui.checkBoxAutoSelectReference.setChecked(False)
            image_group_assistant.set_reference_image(image_path)
        self._clear_adjacent_rough_framing(
            status_text="已导入参考图像，等待计算粗略取景",
            refresh_alignment=True,
        )
        self._update_adjacent_image_label()
        self.ui.statusbar.showMessage(f"已导入参考图像：{image_path}")
        self._update_adjacent_framing_controls()
        return True

    def show_adjacent_image_preview(self) -> None:
        """刷新并显示唯一的参考图像预览窗口。"""

        image_path = getattr(self, "_adjacent_image_path", None)
        if image_path is None:
            QMessageBox.information(self, "尚未导入参考图像", "请先导入参考图像。")
            return
        try:
            preview = load_image_preview(image_path)
        except Exception as exc:  # noqa: BLE001 - 预览读取错误需要直接展示给用户。
            self.ui.statusbar.showMessage(f"参考图像预览失败: {exc}")
            QMessageBox.warning(self, "参考图像预览失败", str(exc))
            return

        self.image_preview_dialog.show_preview(preview)

    def calculate_adjacent_rough_framing(self) -> None:
        """后台寻找参考图像 A↔当前图像 B 对应点，并用 A 的 Pixel↔ICRS 模型生成 B 的粗略取景。"""

        if getattr(self, "_adjacent_framing_thread", None) is not None:
            QMessageBox.information(self, "正在计算粗略取景", "当前已有粗略取景计算任务，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入图像。")
            return
        model_path = getattr(self, "_adjacent_model_json_path", None)
        image_a_path = getattr(self, "_adjacent_image_path", None)
        if model_path is None or image_a_path is None:
            QMessageBox.information(
                self,
                "尚未导入参考图像",
                "请先导入已有模型(model.json)的参考图像。",
            )
            return

        mode = self._adjacent_alignment_mode()
        current_path = Path(self.current_image_preview.path).expanduser().resolve()
        if image_a_path == current_path:
            QMessageBox.information(self, "参考图像无效", "参考图像不能与为当前图像。")
            self._update_adjacent_framing_controls()
            return
        if mode == ADJACENT_ALIGNMENT_MODE_LANDSCAPE and not self._confirm_landscape_time_interval(
            image_a_path,
            current_path,
        ):
            self.ui.statusbar.showMessage("已取消粗略取景计算。")
            return
        self._update_adjacent_framing_controls()
        self.ui.statusbar.showMessage(f"正在使用{adjacent_alignment_mode_display_name(mode)}计算粗略取景…")
        progress = create_progress_dialog(
            self,
            title="正在计算粗略取景",
            label_text=f"正在使用{adjacent_alignment_mode_display_name(mode)}寻找参考图像与当前图像的对应点…",
            minimum=0,
            maximum=0,
        )
        progress.repaint()
        QApplication.processEvents()
        worker = AdjacentFramingWorker(model_path, current_path, mode, image_a_path)
        task = start_qt_worker_task(
            parent=self,
            worker=worker,
            finished_signal=worker.finished,
            failed_signal=worker.failed,
            on_finished=self._handle_adjacent_framing_finished,
            on_failed=self._handle_adjacent_framing_failed,
            on_cleanup=self._cleanup_adjacent_framing,
            progress_dialog=progress,
        )
        self._adjacent_framing_thread = task.thread
        self._adjacent_framing_worker = task.worker
        self._adjacent_framing_progress = progress
        self._update_adjacent_framing_controls()

    def _confirm_landscape_time_interval(self, reference_path: Path, current_path: Path) -> bool:
        """两张图拍摄时间相差较大时，确认是否继续地景粗略取景。"""

        try:
            reference_item = read_image_capture_time(reference_path)
            current_item = read_image_capture_time(current_path)
        except Exception:  # noqa: BLE001 - 任一图缺少原始拍摄时间时直接沿用原流程。
            return True

        interval_seconds = abs(sequence_item_time_delta_seconds(current_item, reference_item))
        if interval_seconds <= 60.0:
            return True
        minutes_text = f"{interval_seconds / 60.0:.1f}".rstrip("0").rstrip(".")
        answer = QMessageBox.question(
            self,
            "拍摄时间间隔较长",
            f"参考图像与当前图像的拍摄时间间隔约为{minutes_text}分钟，粗略取景偏差可能较大，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _handle_adjacent_framing_finished(self, result: object) -> None:
        """启用粗略 FrameAstrometricModel，使参考星图无需四颗手工星即可叠放。"""

        if self._adjacent_framing_progress is not None:
            self._adjacent_framing_progress.close()
        if not isinstance(result, AdjacentFramingResult):
            self._handle_adjacent_framing_failed("后台计算返回了无效的粗略取景结果。")
            return
        if self.current_image_preview is None:
            return
        current_path = Path(self.current_image_preview.path).expanduser().resolve()
        if result.image_b_path != current_path:
            self._handle_adjacent_framing_failed("当前图像已改变，已丢弃旧图像的粗略取景结果。")
            return

        self._adjacent_framing_result = result
        self._rough_alignment_transform = result.transform
        self._rough_source_astrometric_model = result.source_model
        status_text = (
            f"已生成粗略取景：{adjacent_alignment_mode_display_name(result.mode)}，"
            f"{result.correspondence_count} 对，映射 RMS {result.transform.rms_px:.2f}px"
        )
        tooltip = (
            f"参考图像：{result.image_a_path}\n当前图像：{result.image_b_path}\n"
            f"{adjacent_alignment_mode_display_name(result.mode)}\n"
            f"PixelA↔PixelB：{result.correspondence_count} 对，配准 RMS {result.correspondence_rms_px:.2f} px\n"
            f"当前图像的冻结 Profile 姿态 RMS：{result.transform.rms_px:.2f} px"
        )
        self._set_elided_label_text(self.ui.labelAdjacentFramingStatus, status_text, tooltip)
        self._update_reference_alignment_transform()
        self._update_adjacent_framing_controls()
        self.ui.statusbar.showMessage(
            f"已生成当前图像的粗略 Pixel↔ICRS 取景：{result.correspondence_count} 对对应点，"
            f"RMS {result.transform.rms_px:.2f}px。"
        )

    def _handle_adjacent_framing_failed(self, error_message: str) -> None:
        """展示参考图像配准失败原因，并保留已导入图像及其模型便于重试。"""

        if self._adjacent_framing_progress is not None:
            self._adjacent_framing_progress.close()
        self._clear_adjacent_rough_framing(
            status_text="粗略取景计算失败",
            refresh_alignment=True,
        )
        self.ui.statusbar.showMessage(f"粗略取景计算失败: {error_message}")
        QMessageBox.warning(self, "粗略取景计算失败", error_message)

    def _cleanup_adjacent_framing(self) -> None:
        """释放后台任务引用，并恢复参考图像区域的操作按钮。"""

        if self._adjacent_framing_progress is not None:
            self._adjacent_framing_progress.close()
        self._adjacent_framing_thread = None
        self._adjacent_framing_worker = None
        self._adjacent_framing_progress = None
        self._update_adjacent_framing_controls()
