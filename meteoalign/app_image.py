from __future__ import annotations
from .app_constants import *

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox, QProgressDialog

from .image_preview import IMAGE_FILE_FILTER
from .app_utils import _image_with_binary_mask
from .app_workers import ImagePreviewLoadWorker, SkyMaskLoadWorker
from .catalog import project_root
from .image_preview import ImagePreview, load_image_preview
from .reference import build_reference_payload, save_reference_outputs


class ImageMixin:
    """图像导入与蒙版管理 Mixin。"""

    ui: object
    _image_import_thread: QThread | None
    _image_import_worker: object | None
    _image_import_progress: QProgressDialog | None
    _mask_import_thread: QThread | None
    _mask_import_worker: object | None
    _mask_import_progress: QProgressDialog | None
    current_image_preview: ImagePreview | None
    current_sky_mask: np.ndarray | None
    current_sky_mask_path: Path | None
    current_sky_masked_image: QImage | None
    _preserve_sequence_on_next_image_load: bool
    _sky_alignment_transform: object | None
    _source_astrometric_model: object | None
    _reference_alignment_error_message: str
    _sky_alignment_error_message: str
    _source_model_error_message: str
    _mapping_validation_dialog: object | None
    real_image_item: object
    real_image_scene: object

    def _reset_imported_image_labels(self) -> None:
        self.ui.labelImportedImagePath.setProperty("fullPath", "")
        self._set_elided_label_text(self.ui.labelImportedImagePath, "未导入", "")
        self.ui.labelImportedImageSize.setText("-")

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
        if hasattr(self, "_invalidate_image_sequence_mask_cache"):
            self._invalidate_image_sequence_mask_cache()
        self._set_elided_label_text(self.ui.labelSkyMaskStatus, "未使用蒙版", "")
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

    def import_single_image(self) -> None:
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return
        default_dir = project_root() / "testimages"
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入单张图像",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        self.start_single_image_import(file_path)

    def start_single_image_import(self, file_path: str | Path, *, preserve_sequence_status: bool = False) -> None:
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return

        image_path = Path(file_path)
        self._preserve_sequence_on_next_image_load = bool(preserve_sequence_status)
        self._set_image_import_controls_enabled(False)
        self.ui.statusbar.showMessage(f"正在读取整张图像并量化为 8-bit: {image_path}")

        progress = QProgressDialog(self)
        progress.setWindowTitle("正在导入图像")
        progress.setLabelText(f"正在读取整张图像并量化为 8-bit 显示图...\n{image_path}")
        progress.setRange(0, 0)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        thread = QThread(self)
        worker = ImagePreviewLoadWorker(image_path, None)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_single_image_import_finished)
        worker.failed.connect(self._handle_single_image_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_single_image_import)

        self._image_import_thread = thread
        self._image_import_worker = worker
        self._image_import_progress = progress
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()
        thread.start()

    def load_single_image(self, file_path: str | Path) -> None:
        try:
            preview = load_image_preview(file_path, max_long_side_px=None)
            self._apply_loaded_image_preview(preview)
        except Exception as exc:  # noqa: BLE001 - 文件导入错误需要以对话框形式提示用户。
            self.ui.statusbar.showMessage(f"导入图像失败: {exc}")
            QMessageBox.critical(self, "导入图像失败", str(exc))

    def _reset_image_import_results(self, *, preserve_sequence_status: bool) -> None:
        if not preserve_sequence_status and hasattr(self, "_reset_image_sequence_status"):
            self._reset_image_sequence_status()
        self._reset_sky_mask_status()
        self._sky_alignment_transform = None
        self._source_astrometric_model = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""
        dialog = getattr(self, "_mapping_validation_dialog", None)
        if dialog is not None:
            try:
                dialog.close()
            except RuntimeError:
                pass
            self._mapping_validation_dialog = None

    def _set_image_import_controls_enabled(self, enabled: bool) -> None:
        self.ui.pushButtonImportSingleImage.setEnabled(enabled)
        self.ui.pushButtonImportImageSequence.setEnabled(enabled)
        self.ui.actionImportSingleImage.setEnabled(enabled)
        self.ui.actionImportImageSequence.setEnabled(enabled)
        if hasattr(self.ui, "pushButtonProcessImageSequence"):
            self._update_image_sequence_controls()

    def _set_json_import_controls_enabled(self, enabled: bool) -> None:
        self.ui.pushButtonImportReferenceJson.setEnabled(enabled)
        self.ui.pushButtonImportStarPairs.setEnabled(enabled)
        self.ui.pushButtonExportStarPairs.setEnabled(enabled)
        self.ui.pushButtonClearStarPairs.setEnabled(enabled)

    def _apply_loaded_image_preview(
        self,
        preview: ImagePreview,
        clear_existing_pairs: bool = True,
        *,
        switch_to_reference: bool = True,
    ) -> None:
        preserve_sequence_status = bool(getattr(self, "_preserve_sequence_on_next_image_load", False))
        self._preserve_sequence_on_next_image_load = False
        if clear_existing_pairs:
            self._clear_star_pair_positions_for_new_input("新的真实图像")
            self._reset_image_import_results(preserve_sequence_status=preserve_sequence_status)
        self.current_image_preview = preview
        if not clear_existing_pairs:
            self._clear_sky_mask_if_size_mismatch(preview.image.width(), preview.image.height())
        self._display_real_image_preview(preview)
        if switch_to_reference:
            self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        self._update_imported_image_labels(preview)
        self.ui.statusbar.showMessage(
            "已导入图像: {path}  原始: {width} x {height} px。右键配对表行选择“点选位置”。".format(
                path=Path(preview.path).expanduser().resolve(),
                width=preview.original_width,
                height=preview.original_height,
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

    def _handle_single_image_import_finished(self, preview: object) -> None:
        if self._image_import_progress is not None:
            self._image_import_progress.close()
        self._apply_loaded_image_preview(preview)  # type: ignore[arg-type]

    def _handle_single_image_import_failed(self, error_message: str) -> None:
        self._preserve_sequence_on_next_image_load = False
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
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入星空区域蒙版",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
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

        progress = QProgressDialog(self)
        progress.setWindowTitle("正在导入蒙版")
        progress.setLabelText(f"正在读取蒙版并生成缓存预览...\n{mask_path}")
        progress.setRange(0, 0)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        thread = QThread(self)
        worker = SkyMaskLoadWorker(
            mask_path,
            expected_size=(image.width(), image.height()),
            source_image=image,
            source_path=source_path,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_sky_mask_import_finished)
        worker.failed.connect(self._handle_sky_mask_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_sky_mask_import)

        self._mask_import_thread = thread
        self._mask_import_worker = worker
        self._mask_import_progress = progress
        thread.start()

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
