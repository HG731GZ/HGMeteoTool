from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox, QProgressDialog

from ..catalog import project_root
from ..image_path_resolution import companion_sky_mask_path, is_reserved_mask_path
from ..image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from ..image_sequence import read_image_capture_time, sequence_item_local_datetime
from ..qt_tasks import create_progress_dialog, start_qt_worker_task
from ..reference import build_reference_payload, save_reference_outputs
from .app_utils import _image_with_binary_mask
from .app_workers import ImagePreviewLoadWorker, SkyMaskLoadWorker


class ImageMixin:
    """图像导入与蒙版管理 Mixin。"""

    ui: object
    _image_import_thread: object | None
    _image_import_worker: object | None
    _image_import_progress: QProgressDialog | None
    _mask_import_thread: object | None
    _mask_import_worker: object | None
    _mask_import_progress: QProgressDialog | None
    current_image_preview: ImagePreview | None
    current_sky_mask: np.ndarray | None
    current_sky_mask_path: Path | None
    current_sky_masked_image: QImage | None
    _mask_excluded_reference_star_ids: set
    _preserve_sequence_on_next_image_load: bool
    _preserve_image_group_on_next_image_load: bool
    _sky_alignment_transform: object | None
    _source_astrometric_model: object | None
    _reference_alignment_error_message: str
    _sky_alignment_error_message: str
    _source_model_error_message: str
    real_image_item: object
    real_image_scene: object

    def _reset_imported_image_labels(self) -> None:
        self.ui.labelImportedImagePath.setProperty("fullPath", "")
        self._set_elided_label_text(self.ui.labelImportedImagePath, "未导入", "")
        self.ui.labelImportedImageSize.setText("-")
        self._update_status_image_context()

    def _update_imported_image_labels(self, preview: ImagePreview) -> None:
        image_path = Path(preview.path).expanduser().resolve()
        image_path_text = str(image_path)
        self.ui.labelImportedImagePath.setProperty("fullPath", image_path_text)
        is_sequence_image = False
        if hasattr(self, "_is_sequence_image_path"):
            is_sequence_image = bool(self._is_sequence_image_path(image_path))
        if is_sequence_image:
            self._set_elided_label_text_with_html_suffix(
                self.ui.labelImportedImagePath,
                image_path.name,
                "(序列)",
                '<span style="color:#d32f2f;">(序列)</span>',
                image_path_text,
            )
        else:
            self._set_elided_label_text(self.ui.labelImportedImagePath, image_path.name, image_path_text)
        self.ui.labelImportedImageSize.setText(f"{preview.original_width} x {preview.original_height} px")
        self._update_status_image_context()

    def _update_status_image_context(self) -> None:
        """刷新状态栏右侧的当前图像文件名与蒙版使用状态。"""

        status_label = getattr(self.ui, "labelStatusImageContext", None)
        if status_label is None:
            return
        projection_combo = getattr(self.ui, "comboBoxSkyAlignmentModel", None)
        projection_text = ""
        if projection_combo is not None:
            projection_text = str(projection_combo.currentText()).strip()
        projection_context = f"  |  投影：{projection_text}" if projection_text else ""
        preview = self.current_image_preview
        if preview is None:
            status_label.setText(f"图像：未导入  |  蒙版：未使用{projection_context}")
            status_label.setToolTip("")
            return

        image_path = Path(preview.path).expanduser().resolve()
        if self.current_sky_mask is None:
            mask_text = "蒙版：未使用"
            mask_tooltip = "未使用蒙版"
        else:
            mask_text = "蒙版：已使用"
            mask_tooltip = (
                str(self.current_sky_mask_path)
                if self.current_sky_mask_path is not None
                else "正在使用内存蒙版"
            )
        status_label.setText(f"图像：{image_path.name}  |  {mask_text}{projection_context}")
        projection_tooltip = f"\n当前投影模型：{projection_text}" if projection_text else ""
        status_label.setToolTip(f"真实图像：{image_path}\n{mask_tooltip}{projection_tooltip}")

    def _apply_single_image_exif_observation_time(self, image_path: str | Path) -> str:
        """读取单张图像 EXIF 时间，并同步到星空模拟页的拍摄时间。"""

        try:
            capture_item = read_image_capture_time(image_path)
            local_dt = sequence_item_local_datetime(capture_item, self.ui.doubleSpinBoxUtcOffset.value())
            qt_datetime = QDateTime.fromString(local_dt.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd HH:mm:ss")
            if not qt_datetime.isValid():
                raise ValueError("EXIF 拍摄时间无法转换为界面时间。")
        except Exception:
            return ""

        self.ui.dateTimeEditObservation.setDateTime(qt_datetime)
        if hasattr(self, "schedule_render"):
            self.schedule_render(delay_ms=0)
        return f"  已应用 EXIF 拍摄时间: {capture_item.capture_datetime.isoformat()}。"

    def _show_imported_image_path_context_menu(self, position) -> None:  # type: ignore[no-untyped-def]
        """显示真实图像文件名右键菜单，用于复制完整路径。"""
        image_path_text = str(self.ui.labelImportedImagePath.property("fullPath") or "").strip()
        if not image_path_text:
            return
        menu = QMenu(self.ui.labelImportedImagePath)
        copy_action = menu.addAction("复制完整文件路径")
        selected_action = menu.exec_(self.ui.labelImportedImagePath.mapToGlobal(position))
        if selected_action is copy_action:
            QApplication.clipboard().setText(image_path_text)
            self.ui.statusbar.showMessage(f"已复制真实图像完整路径: {image_path_text}")

    def _reset_sky_mask_status(self) -> None:
        self.current_sky_mask_path = None
        self.current_sky_mask = None
        self.current_sky_masked_image = None
        self._mask_excluded_reference_star_ids = set()
        if hasattr(self, "_invalidate_image_sequence_mask_cache"):
            self._invalidate_image_sequence_mask_cache()
        self._set_elided_label_text(self.ui.labelSkyMaskStatus, "未使用蒙版", "")
        self._update_status_image_context()
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()

    def _update_sky_mask_status(self) -> None:
        if self.current_sky_mask is None:
            self._reset_sky_mask_status()
            return

        valid_fraction = float(np.count_nonzero(self.current_sky_mask)) / max(float(self.current_sky_mask.size), 1.0)
        path_text = str(self.current_sky_mask_path) if self.current_sky_mask_path is not None else "内存蒙版"
        self._set_elided_label_text(
            self.ui.labelSkyMaskStatus,
            f"蒙版有效区域 {valid_fraction * 100.0:.1f}%",
            path_text,
        )
        self._update_status_image_context()
        if hasattr(self, "_invalidate_image_sequence_mask_cache"):
            self._invalidate_image_sequence_mask_cache()
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()

    def _sky_mask_allows_point(self, x_px: float, y_px: float) -> bool:
        if self.current_sky_mask is None:
            return True
        if not (math.isfinite(x_px) and math.isfinite(y_px)):
            return False

        mask_height, mask_width = self.current_sky_mask.shape
        image_x = int(round(x_px))
        image_y = int(round(y_px))
        if image_x < 0 or image_x >= mask_width or image_y < 0 or image_y >= mask_height:
            return False
        return bool(self.current_sky_mask[image_y, image_x])

    def _clear_sky_mask_if_size_mismatch(self, image_width: int, image_height: int) -> None:
        if self.current_sky_mask is None:
            return
        mask_height, mask_width = self.current_sky_mask.shape
        if mask_width == image_width and mask_height == image_height:
            self.current_sky_masked_image = None
            return
        self._reset_sky_mask_status()
        self.ui.statusbar.showMessage("新的真实图像尺寸与已有蒙版不一致，已自动清除蒙版。")

    def _real_image_for_current_mask_preview(self) -> QImage:
        if self.current_image_preview is None:
            return QImage()
        image = self.current_image_preview.image
        if self.current_sky_mask is not None and self.ui.checkBoxShowSkyMask.isChecked():
            if self.current_sky_masked_image is None:
                self.current_sky_masked_image = _image_with_binary_mask(image, self.current_sky_mask)
            return self.current_sky_masked_image
        return image

    def _refresh_real_image_display_for_mask(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self.current_image_preview is None:
            return
        self.real_image_item.set_image(self._real_image_for_current_mask_preview())

    def _set_mask_import_controls_enabled(self, enabled: bool) -> None:
        sequence_mode = bool(hasattr(self, "_sequence_mode_active") and self._sequence_mode_active())
        self.ui.pushButtonImportSkyMask.setEnabled(
            enabled and not sequence_mode and self.current_image_preview is not None
        )
        self.ui.pushButtonClearSkyMask.setEnabled(enabled and not sequence_mode and self.current_sky_mask is not None)
        self.ui.checkBoxShowSkyMask.setEnabled(enabled and self.current_sky_mask is not None)
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()

    def _maybe_auto_import_sky_mask_for_image(self, image_path: str | Path) -> bool:
        """按“图像名_Mask.后缀”约定自动导入当前图像对应的蒙版。"""

        if self._mask_import_thread is not None:
            return False
        mask_path = companion_sky_mask_path(image_path)
        if mask_path is None:
            return False
        current_mask_path = self.current_sky_mask_path
        if (
            self.current_sky_mask is not None
            and current_mask_path is not None
            and current_mask_path.expanduser().resolve() == mask_path
        ):
            return False
        self.ui.statusbar.showMessage(f"发现同名蒙版，正在自动导入: {mask_path}")
        self.start_sky_mask_import(mask_path)
        return True

    def _image_paths_without_reserved_masks(
        self,
        file_paths: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        """排除通过“导入图像”选择的 _Mask 文件，无原图可导入时说明预留规则。"""

        image_paths = tuple(path for path in file_paths if not is_reserved_mask_path(path))
        mask_paths = tuple(path for path in file_paths if is_reserved_mask_path(path))
        if not mask_paths or image_paths:
            return image_paths

        displayed_names = "\n".join(f"• {Path(path).name}" for path in mask_paths[:5])
        if len(mask_paths) > 5:
            displayed_names += f"\n• 其余 {len(mask_paths) - 5} 个文件"
        QMessageBox.information(
            self,
            "请勿通过“导入图像”导入蒙版",
            "以下文件的主文件名以“_Mask”结尾，已被识别为蒙版：\n"
            f"{displayed_names}\n\n"
            "不要把蒙版作为普通图像导入；“_Mask”是蒙版预留字段。"
            "请导入对应原图，软件会自动关联蒙版。若需手动选择蒙版，请使用“导入蒙版”，该入口接受任意文件名。",
        )
        return image_paths

    def import_images(self) -> None:
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return
        default_dir = project_root() / "testimages"
        if not default_dir.exists():
            default_dir = project_root()
        default_dir = self._import_dialog_directory(default_dir)
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "导入图像",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_paths:
            return
        self._remember_import_path(file_paths)
        file_paths = list(self._image_paths_without_reserved_masks(file_paths))
        if not file_paths:
            return
        normalized_paths = self._set_image_group_paths(file_paths)
        if not normalized_paths:
            return
        preserve_image_group = len(normalized_paths) > 1
        self.start_single_image_import(
            normalized_paths[0],
            preserve_image_group_status=preserve_image_group,
        )

    def start_single_image_import(
        self,
        file_path: str | Path,
        *,
        preserve_sequence_status: bool = False,
        preserve_image_group_status: bool = False,
    ) -> None:
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return

        image_path = Path(file_path)
        self._preserve_sequence_on_next_image_load = bool(preserve_sequence_status)
        self._preserve_image_group_on_next_image_load = bool(preserve_image_group_status)
        self._set_image_import_controls_enabled(False)
        self.ui.statusbar.showMessage(f"正在读取整张图像并量化为 8-bit: {image_path}")

        progress = create_progress_dialog(
            self,
            title="正在导入图像",
            label_text=f"正在读取整张图像并量化为 8-bit 显示图...\n{image_path}",
            minimum=0,
            maximum=0,
        )
        worker = ImagePreviewLoadWorker(image_path, None)
        task = start_qt_worker_task(
            parent=self,
            worker=worker,
            finished_signal=worker.finished,
            failed_signal=worker.failed,
            on_finished=self._handle_single_image_import_finished,
            on_failed=self._handle_single_image_import_failed,
            on_cleanup=self._cleanup_single_image_import,
            progress_dialog=progress,
        )

        self._image_import_thread = task.thread
        self._image_import_worker = task.worker
        self._image_import_progress = progress
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()

    def load_single_image(self, file_path: str | Path) -> None:
        try:
            preview = load_image_preview(file_path, max_long_side_px=None)
            self._apply_loaded_image_preview(preview)
        except Exception as exc:  # noqa: BLE001 - 文件导入错误需要以对话框形式提示用户。
            self.ui.statusbar.showMessage(f"导入图像失败: {exc}")
            QMessageBox.critical(self, "导入图像失败", str(exc))

    def _reset_image_import_results(
        self,
        *,
        preserve_sequence_status: bool,
        preserve_image_group_status: bool,
    ) -> None:
        if not preserve_sequence_status and hasattr(self, "_reset_image_sequence_status"):
            self._reset_image_sequence_status()
        if not preserve_image_group_status and hasattr(self, "_reset_image_group_status"):
            self._reset_image_group_status()
        if hasattr(self, "_clear_adjacent_rough_framing"):
            self._clear_adjacent_rough_framing(
                status_text="当前图像已改变，请重新计算粗略取景",
                refresh_alignment=False,
            )
        self._reset_sky_mask_status()
        self._sky_alignment_transform = None
        self._source_astrometric_model = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""

    def _set_image_import_controls_enabled(self, enabled: bool) -> None:
        self.ui.pushButtonImportImages.setEnabled(enabled)
        self.ui.pushButtonImportImageSequence.setEnabled(enabled)
        if hasattr(self.ui, "pushButtonProcessImageSequence"):
            self._update_image_sequence_controls()
        if hasattr(self, "_update_image_group_controls"):
            self._update_image_group_controls()

    def _set_json_import_controls_enabled(self, enabled: bool) -> None:
        self.ui.pushButtonImportReferenceJson.setEnabled(enabled)
        self.ui.pushButtonImportStarPairs.setEnabled(enabled)
        self._update_star_pair_export_control(controls_enabled=enabled)
        self.ui.pushButtonDeleteStarPairs.setEnabled(enabled)
        self.ui.pushButtonClearStarPairs.setEnabled(enabled)
        if hasattr(self.ui, "pushButtonOpenImageGroupAssistant"):
            if enabled and hasattr(self, "_update_image_group_controls"):
                self._update_image_group_controls()
            else:
                self.ui.pushButtonOpenImageGroupAssistant.setEnabled(False)

    def _apply_loaded_image_preview(
        self,
        preview: ImagePreview,
        clear_existing_pairs: bool = True,
        *,
        switch_to_reference: bool = True,
    ) -> None:
        preserve_sequence_status = bool(getattr(self, "_preserve_sequence_on_next_image_load", False))
        self._preserve_sequence_on_next_image_load = False
        preserve_image_group_status = bool(
            getattr(self, "_preserve_image_group_on_next_image_load", False)
        )
        self._preserve_image_group_on_next_image_load = False
        previous_image_path = (
            Path(self.current_image_preview.path).expanduser().resolve()
            if self.current_image_preview is not None
            else None
        )
        if clear_existing_pairs:
            self._clear_star_pair_positions_for_new_input("新的真实图像")
            self._reset_image_import_results(
                preserve_sequence_status=preserve_sequence_status,
                preserve_image_group_status=preserve_image_group_status,
            )
        self.current_image_preview = preview
        if not clear_existing_pairs:
            current_image_path = Path(preview.path).expanduser().resolve()
            if previous_image_path is not None and previous_image_path != current_image_path:
                # 导入另一张图的匹配 JSON 时，即使尺寸相同也不能沿用旧图蒙版。
                self._reset_sky_mask_status()
            else:
                self._clear_sky_mask_if_size_mismatch(preview.image.width(), preview.image.height())
        self._display_real_image_preview(preview)
        if switch_to_reference:
            self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        self._update_imported_image_labels(preview)
        if hasattr(self, "_refresh_image_group_assistant_status"):
            self._refresh_image_group_assistant_status()
        exif_time_message = ""
        if clear_existing_pairs:
            exif_time_message = self._apply_single_image_exif_observation_time(preview.path)
        self.ui.statusbar.showMessage(
            "已导入图像: {path}  原始: {width} x {height} px。{exif}右键匹配表行选择“点选位置”。".format(
                path=Path(preview.path).expanduser().resolve(),
                width=preview.original_width,
                height=preview.original_height,
                exif=exif_time_message,
            )
        )
        skip_auto_import = (
            hasattr(self, "_should_skip_auto_import_star_pair_session")
            and self._should_skip_auto_import_star_pair_session(Path(preview.path))
        )
        if clear_existing_pairs and not skip_auto_import and hasattr(self, "_maybe_auto_import_star_pair_session_for_image"):
            self._maybe_auto_import_star_pair_session_for_image(Path(preview.path))
        if hasattr(self, "_update_reference_alignment_controls"):
            self._update_reference_alignment_controls()
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()
        if hasattr(self, "_update_adjacent_framing_controls"):
            self._update_adjacent_framing_controls()
        self._maybe_auto_import_sky_mask_for_image(preview.path)

    def _handle_single_image_import_finished(self, preview: object) -> None:
        if self._image_import_progress is not None:
            self._image_import_progress.close()
        self._apply_loaded_image_preview(preview)  # type: ignore[arg-type]

    def _handle_single_image_import_failed(self, error_message: str) -> None:
        self._preserve_sequence_on_next_image_load = False
        self._preserve_image_group_on_next_image_load = False
        if self._image_import_progress is not None:
            self._image_import_progress.close()
        self.ui.statusbar.showMessage(f"导入图像失败: {error_message}")
        QMessageBox.critical(self, "导入图像失败", error_message)

    def _cleanup_single_image_import(self) -> None:
        self._image_import_thread = None
        self._image_import_worker = None
        self._image_import_progress = None
        self._set_image_import_controls_enabled(True)

    def import_sky_mask(self) -> None:
        if self._mask_import_thread is not None:
            QMessageBox.information(self, "正在导入蒙版", "当前已有蒙版正在导入，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导入同尺寸蒙版。")
            return

        default_dir = Path(self.current_image_preview.path).expanduser().resolve().parent
        default_dir = self._import_dialog_directory(default_dir)
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入星空区域蒙版",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        self._remember_import_path(file_path)
        self.start_sky_mask_import(file_path)

    def load_sky_mask(self, file_path: str | Path) -> None:
        self.start_sky_mask_import(file_path)

    def start_sky_mask_import(self, file_path: str | Path) -> None:
        if self._mask_import_thread is not None:
            QMessageBox.information(self, "正在导入蒙版", "当前已有蒙版正在导入，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导入同尺寸蒙版。")
            return

        mask_path = Path(file_path).expanduser().resolve()
        image = self.current_image_preview.image
        source_path = Path(self.current_image_preview.path).expanduser().resolve()
        self._set_mask_import_controls_enabled(False)
        self.ui.statusbar.showMessage(f"正在导入蒙版并生成缓存预览: {mask_path}")

        progress = create_progress_dialog(
            self,
            title="正在导入蒙版",
            label_text=f"正在读取蒙版并生成缓存预览...\n{mask_path}",
            minimum=0,
            maximum=0,
        )
        worker = SkyMaskLoadWorker(
            mask_path,
            expected_size=(image.width(), image.height()),
            source_image=image,
            source_path=source_path,
        )
        task = start_qt_worker_task(
            parent=self,
            worker=worker,
            finished_signal=worker.finished,
            failed_signal=worker.failed,
            on_finished=self._handle_sky_mask_import_finished,
            on_failed=self._handle_sky_mask_import_failed,
            on_cleanup=self._cleanup_sky_mask_import,
            progress_dialog=progress,
        )

        self._mask_import_thread = task.thread
        self._mask_import_worker = task.worker
        self._mask_import_progress = progress

    def _handle_sky_mask_import_finished(self, result: object) -> None:
        if self._mask_import_progress is not None:
            self._mask_import_progress.close()
        try:
            mask_path, source_path, mask, masked_image = result  # type: ignore[misc]
            if not isinstance(mask_path, Path):
                mask_path = Path(mask_path)
            if not isinstance(source_path, Path):
                source_path = Path(source_path)
            if self.current_image_preview is None:
                raise ValueError("真实图像已关闭，无法应用蒙版。")
            current_source_path = Path(self.current_image_preview.path).expanduser().resolve()
            if current_source_path != source_path:
                raise ValueError("真实图像已改变，请重新导入蒙版。")
            image = self.current_image_preview.image
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != (image.height(), image.width()):
                raise ValueError("蒙版尺寸与当前真实图像不一致，请重新导入。")

            self._mask_excluded_reference_star_ids = set()
            self.current_sky_mask_path = mask_path
            self.current_sky_mask = mask_array
            self.current_sky_masked_image = masked_image if isinstance(masked_image, QImage) else None
            self._update_sky_mask_status()
            self._update_reference_alignment_controls()
            self._refresh_real_image_display_for_mask()
            if hasattr(self, "_update_image_sequence_preview"):
                self._update_image_sequence_preview()
            self.ui.statusbar.showMessage(f"已导入蒙版并缓存显示图: {mask_path}")
        except Exception as exc:  # noqa: BLE001 - 主线程应用蒙版时需要把状态错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导入蒙版失败: {exc}")
            QMessageBox.critical(self, "导入蒙版失败", str(exc))

    def _handle_sky_mask_import_failed(self, error_message: str) -> None:
        if self._mask_import_progress is not None:
            self._mask_import_progress.close()
        self.ui.statusbar.showMessage(f"导入蒙版失败: {error_message}")
        QMessageBox.critical(self, "导入蒙版失败", error_message)

    def _cleanup_sky_mask_import(self) -> None:
        if self._mask_import_progress is not None:
            self._mask_import_progress.close()
        self._mask_import_thread = None
        self._mask_import_worker = None
        self._mask_import_progress = None
        self._update_reference_alignment_controls()
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()

    def clear_sky_mask(self) -> None:
        if self.current_sky_mask is None:
            self.ui.statusbar.showMessage("当前没有正在使用的蒙版。")
            return
        self._reset_sky_mask_status()
        self._update_reference_alignment_controls()
        self._refresh_real_image_display_for_mask()
        if hasattr(self, "_update_image_sequence_preview"):
            self._update_image_sequence_preview()
        self.ui.statusbar.showMessage("已清除蒙版，后续自动匹配将使用整张图像。")

    def _next_reference_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return project_root() / "outputs" / f"reference_{timestamp}"

    def export_reference_map(self) -> None:
        try:
            output_camera = self._output_camera_settings()
            observer, camera, view, mag_limit, star_map = self._build_projected_star_map(camera=output_camera)
            reference_stars = self._select_current_reference_stars(star_map)
            if not reference_stars:
                QMessageBox.warning(self, "无法生成参考图", "当前视野内没有可用的地平线上参考星。")
                return

            image = self.renderer.render(
                star_map,
                reference_stars=reference_stars,
                element_scale=self._render_element_scale(camera),
                draw_common_names=False,
            )
            payload = build_reference_payload(
                star_map=star_map,
                reference_stars=reference_stars,
                observer=observer,
                camera=camera,
                view=view,
                visible_mag_limit=mag_limit,
                utc_offset_hours=self.ui.doubleSpinBoxUtcOffset.value(),
                reference_label_mode=self._reference_label_mode(),
                reference_mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
                reference_star_count=self.ui.spinBoxReferenceStarCount.value(),
                manual_reference_star_ids=tuple(self._manual_reference_star_ids),
            )
            image_path, json_path = save_reference_outputs(image, payload, self._next_reference_output_dir())
            self.render_now()
            self.ui.statusbar.showMessage(
                f"已导出参考图: {image_path}  参考星表: {json_path}  标注星数: {len(reference_stars)}"
            )
            QMessageBox.information(self, "参考图已导出", f"PNG：{image_path}\nJSON：{json_path}")
        except Exception as exc:  # noqa: BLE001 - 界面层需要把可恢复输入错误显示出来。
            self.ui.statusbar.showMessage(f"导出参考图失败: {exc}")
            QMessageBox.critical(self, "导出参考图失败", str(exc))
