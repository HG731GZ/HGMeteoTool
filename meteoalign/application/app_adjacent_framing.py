"""星点匹配页的相邻图像导入与粗略取景交互。"""

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
from .adjacent_alignment_settings_dialog import AdjacentAlignmentSettingsDialog


MODEL_JSON_FILE_FILTER = "HoshinoPanoAssistant 源图映射 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)"


class AdjacentFramingMixin:
    """管理相邻已标定图像 A 驱动当前图像 B 粗略取景的 UI 状态。"""

    ui: object
    current_image_preview: object | None
    _adjacent_model_json_path: Path | None
    _adjacent_framing_result: AdjacentFramingResult | None
    _adjacent_framing_thread: object | None
    _adjacent_framing_worker: object | None
    _adjacent_framing_progress: QProgressDialog | None
    _rough_alignment_transform: object | None
    _rough_source_astrometric_model: object | None

    def _init_adjacent_framing_defaults(self) -> None:
        """初始化相邻图像区域的界面与内存状态。"""

        self._adjacent_model_json_path = None
        self._adjacent_framing_result = None
        self._rough_alignment_transform = None
        self._rough_source_astrometric_model = None
        if hasattr(self.ui, "comboBoxAdjacentAlignmentMode"):
            self.ui.comboBoxAdjacentAlignmentMode.setCurrentIndex(0)
        self._set_elided_label_text(self.ui.labelAdjacentImageModel, "未导入 model.json", "")
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
        """根据当前图像、model.json 和后台任务状态更新操作按钮。"""

        if not hasattr(self.ui, "pushButtonImportAdjacentImage"):
            return
        idle = getattr(self, "_adjacent_framing_thread", None) is None
        self.ui.pushButtonImportAdjacentImage.setEnabled(idle)
        self.ui.comboBoxAdjacentAlignmentMode.setEnabled(idle)
        self.ui.pushButtonCalculateAdjacentFraming.setEnabled(
            idle
            and getattr(self, "_adjacent_model_json_path", None) is not None
            and self.current_image_preview is not None
        )
        if hasattr(self.ui, "toolButtonAdjacentAlignmentSettings"):
            self.ui.toolButtonAdjacentAlignmentSettings.setEnabled(idle)

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
        """清除已生成的粗略模型；保留用户已选择的相邻 model.json。"""

        self._adjacent_framing_result = None
        self._rough_alignment_transform = None
        self._rough_source_astrometric_model = None
        if hasattr(self.ui, "labelAdjacentFramingStatus"):
            self._set_elided_label_text(self.ui.labelAdjacentFramingStatus, status_text, status_text)
        self._update_adjacent_framing_controls()
        if refresh_alignment and self.current_image_preview is not None:
            self._update_reference_alignment_transform()

    def import_adjacent_image(self) -> None:
        """选择相邻图像 A 已导出的 model.json，而非未标定的原始图像。"""

        default_dir = Path.cwd()
        current_path = getattr(self, "_adjacent_model_json_path", None)
        if current_path is not None:
            default_dir = current_path.parent
        elif self.current_image_preview is not None:
            default_dir = Path(self.current_image_preview.path).expanduser().resolve().parent
        default_dir = self._import_dialog_directory(default_dir)
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入相邻图像的 model.json",
            str(default_dir),
            MODEL_JSON_FILE_FILTER,
        )
        if not file_path:
            return
        self._remember_import_path(file_path)
        self.load_adjacent_model_json(file_path)

    def load_adjacent_model_json(self, file_path: str | Path) -> bool:
        """验证 model.json 与其原图 A 是否可用，并更新界面显示。"""

        try:
            model_path = Path(file_path).expanduser().resolve()
            _frame_model, image_path = load_adjacent_frame_model(model_path)
        except Exception as exc:  # noqa: BLE001 - 用户选择的 JSON 需要立即给出明确反馈。
            self.ui.statusbar.showMessage(f"导入相邻图像 model.json 失败: {exc}")
            QMessageBox.critical(self, "导入相邻图像模型失败", str(exc))
            return False

        self._adjacent_model_json_path = model_path
        self._clear_adjacent_rough_framing(
            status_text="已导入 model.json，等待计算粗略取景",
            refresh_alignment=True,
        )
        display_text = f"{model_path.name}（原图：{image_path.name}）"
        tooltip = f"model.json：{model_path}\n相邻图像 A：{image_path}"
        self._set_elided_label_text(self.ui.labelAdjacentImageModel, display_text, tooltip)
        self.ui.statusbar.showMessage(f"已导入相邻图像 model.json：{model_path}")
        self._update_adjacent_framing_controls()
        return True

    def calculate_adjacent_rough_framing(self) -> None:
        """后台寻找 相邻图像A↔当前图像B 对应点，并用 A 的 Pixel↔ICRS 模型生成 B 的粗略取景。"""

        if getattr(self, "_adjacent_framing_thread", None) is not None:
            QMessageBox.information(self, "正在计算粗略取景", "当前已有粗略取景计算任务，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入图像。")
            return
        model_path = getattr(self, "_adjacent_model_json_path", None)
        if model_path is None:
            QMessageBox.information(self, "尚未导入相邻图像模型", "请先导入相邻图像的 model.json。")
            return

        mode = self._adjacent_alignment_mode()
        current_path = Path(self.current_image_preview.path).expanduser().resolve()
        self._update_adjacent_framing_controls()
        self.ui.statusbar.showMessage(f"正在使用{adjacent_alignment_mode_display_name(mode)}计算粗略取景…")
        progress = create_progress_dialog(
            self,
            title="正在计算粗略取景",
            label_text=(
                f"正在使用{adjacent_alignment_mode_display_name(mode)}寻找相邻图像与当前图像的对应点…"
                #"计算完成后将自动叠放参考星图。"
            ),
            minimum=0,
            maximum=0,
        )
        progress.repaint()
        QApplication.processEvents()
        worker = AdjacentFramingWorker(model_path, current_path, mode)
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
            f"相邻图像：{result.image_a_path}\n当前图像：{result.image_b_path}\n"
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
        """展示相邻图像配准失败原因，并保留已导入的 model.json 便于重试。"""

        if self._adjacent_framing_progress is not None:
            self._adjacent_framing_progress.close()
        self._clear_adjacent_rough_framing(
            status_text="粗略取景计算失败",
            refresh_alignment=True,
        )
        self.ui.statusbar.showMessage(f"粗略取景计算失败: {error_message}")
        QMessageBox.warning(self, "粗略取景计算失败", error_message)

    def _cleanup_adjacent_framing(self) -> None:
        """释放后台任务引用，并恢复相邻图像区域的操作按钮。"""

        if self._adjacent_framing_progress is not None:
            self._adjacent_framing_progress.close()
        self._adjacent_framing_thread = None
        self._adjacent_framing_worker = None
        self._adjacent_framing_progress = None
        self._update_adjacent_framing_controls()
