from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

from .image_preview import load_image_preview

class SequenceProcessingMixin:
    """图像序列批处理主流程编排。"""

    def process_image_sequence(self) -> None:
        if getattr(self, "_sequence_processing_active", False):
            QMessageBox.information(self, "正在处理序列", "图像序列仍在处理，请稍候。")
            return
        try:
            self._ensure_first_sequence_session_loaded_for_processing()
            self._ensure_sequence_ready_for_processing()
            if not self._confirm_overwrite_sequence_outputs():
                self.ui.statusbar.showMessage("已取消图像序列解析。")
                return
            first_item = self._current_sequence_first_item()
            first_output_paths = self._ensure_first_sequence_output_jsons(first_item)
            if first_output_paths:
                self._refresh_image_sequence_table()
            self._apply_sequence_observation_time(first_item, emit_signal=False)
            self.render_now()
            QApplication.processEvents()
            templates = self._sequence_base_templates()
            minimum_pair_count, desired_pair_count = self._sequence_pair_targets(templates)
            assert self.current_image_preview is not None
            target_size = (self.current_image_preview.image.width(), self.current_image_preview.image.height())
            fixed_model = self._fit_sequence_fixed_camera_model(templates, target_size)
        except Exception as exc:  # noqa: BLE001 - 批处理入口要把缺失条件直接反馈给用户。
            QMessageBox.warning(self, "无法处理图像序列", str(exc))
            self.ui.statusbar.showMessage(f"无法处理图像序列: {exc}")
            return

        items = list(getattr(self, "_image_sequence_items", []))
        process_items = items[1:]
        progress = QProgressDialog(self)
        progress.setWindowTitle("正在处理图像序列")
        progress.setRange(0, len(process_items))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()

        processed: list[tuple[Path, Path]] = []
        failures: list[str] = []
        previous_delta_seconds = 0.0
        self._sequence_processing_active = True
        self._set_image_import_controls_enabled(False)
        self._update_image_sequence_controls()
        try:
            for output_index, item in enumerate(process_items, start=1):
                if progress.wasCanceled():
                    failures.append("用户取消了后续处理。")
                    break
                frame_index = output_index + 1
                self._set_image_sequence_processing_index(frame_index - 1)
                progress.setValue(output_index - 1)
                progress.setLabelText(
                    "正在处理 {index}/{count}\n{path}".format(
                        index=frame_index,
                        count=len(items),
                        path=item.path,
                    )
                )
                QApplication.processEvents()

                try:
                    preview = load_image_preview(item.path, max_long_side_px=None)
                    if (preview.image.width(), preview.image.height()) != target_size:
                        raise ValueError(
                            "图像尺寸与第一张不一致：第一张 {base_w} x {base_h} px，当前 {w} x {h} px。".format(
                                base_w=target_size[0],
                                base_h=target_size[1],
                                w=preview.image.width(),
                                h=preview.image.height(),
                            )
                        )
                    stats = {
                        "failed_psf": 0,
                        "skipped_mask": 0,
                        "skipped_duplicate": 0,
                        "skipped_outside": 0,
                        "missing_target": 0,
                        "supplemental_matched": 0,
                    }
                    pairs = self._sequence_frame_matched_pairs(
                        item,
                        preview,
                        templates,
                        fixed_model,
                        target_size,
                        previous_delta_seconds,
                        desired_pair_count,
                        stats,
                    )
                    time_fit = self._sequence_time_fit_for_pairs(
                        item,
                        pairs,
                        fixed_model,
                        initial_delta_seconds=previous_delta_seconds,
                    )
                    pairs = self._apply_sequence_time_fit(
                        pairs,
                        time_fit,
                        require_accepted=True,
                    )
                    if len(pairs) < minimum_pair_count:
                        stats["second_pass_fill"] = 1
                        pairs = self._sequence_fill_frame_pairs_to_target(
                            item,
                            preview,
                            pairs,
                            fixed_model,
                            target_size,
                            float(time_fit.delta_t_seconds),
                            desired_pair_count,
                            stats,
                        )
                        time_fit = self._sequence_time_fit_for_pairs(
                            item,
                            pairs,
                            fixed_model,
                            initial_delta_seconds=float(time_fit.delta_t_seconds),
                        )
                        pairs = self._apply_sequence_time_fit(
                            pairs,
                            time_fit,
                            require_accepted=True,
                        )
                    if len(pairs) < minimum_pair_count:
                        raise ValueError(
                            "补充匹配后有效配对 {count} 个，低于序列目标下限 {target} 个；"
                            "请检查云层、遮挡、蒙版或增大自动匹配搜索半径。".format(
                                count=len(pairs),
                                target=minimum_pair_count,
                            )
                        )
                    output_paths = self._write_sequence_outputs(
                        item,
                        first_item,
                        preview,
                        pairs,
                        fixed_model,
                        time_fit,
                    )
                    processed.append(output_paths)
                    self._refresh_image_sequence_table()
                    self._set_image_sequence_processing_index(frame_index - 1)
                    previous_delta_seconds = float(time_fit.delta_t_seconds)
                except Exception as exc:  # noqa: BLE001 - 单帧失败不影响后续帧。
                    failures.append(f"{item.path.name}: {exc}")
                    self._set_image_sequence_processing_index(frame_index - 1)
                    continue
        finally:
            progress.setValue(len(process_items))
            progress.close()
            self._sequence_processing_active = False
            self._set_image_import_controls_enabled(True)
            self._refresh_image_sequence_table()
            self._update_image_sequence_controls()

        self.ui.statusbar.showMessage(
            f"序列处理完成：第一帧跳过，后续成功 {len(processed)} 张，失败 {len(failures)} 张。"
        )
        message = f"第一帧使用用户基准数据，未参与批处理。\n后续成功处理 {len(processed)} 张，失败 {len(failures)} 张。"
        if first_output_paths:
            message += "\n\n已补齐第一帧基准 JSON：\n" + "\n".join(str(path) for path in first_output_paths)
        if processed:
            first_starpair, first_model = processed[0]
            message += f"\n\n示例输出：\n{first_starpair}\n{first_model}"
        if failures:
            message += "\n\n失败明细：\n" + "\n".join(failures[:12])
            if len(failures) > 12:
                message += f"\n... 另有 {len(failures) - 12} 条"
            QMessageBox.warning(self, "图像序列解析完成", message)
        else:
            QMessageBox.information(self, "图像序列解析完成", message)
