from __future__ import annotations

import json
import math
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

from ..image_preview import load_image_preview
from ..image_sequence import ImageSequenceItem
from ..star_fitting import qimage_to_grayscale_array

class SequenceProcessingMixin:
    """图像序列批处理主流程编排。"""

    def process_image_sequence(self) -> None:
        """从第二帧开始重新处理整个序列。"""

        self._close_sequence_auxiliary_windows()
        self._run_image_sequence_processing(continue_only=False)

    def continue_image_sequence(self) -> None:
        """保留完整输出，从第一张未处理图像开始补齐剩余帧。"""

        self._close_sequence_auxiliary_windows()
        if not self._sequence_can_continue():
            QMessageBox.information(
                self,
                "没有可继续的序列",
                "继续处理需要第一帧基准 JSON，且序列中至少有一张后续图像尚未处理。",
            )
            self._update_image_sequence_controls()
            return
        self._run_image_sequence_processing(continue_only=True)

    def _sequence_output_resume_state(
        self,
        item: ImageSequenceItem,
        base_star_ids: frozenset[str],
    ) -> tuple[float | None, tuple[str, ...]]:
        """读取已有匹配 JSON 的时间偏移和补充星顺序。"""

        try:
            json_path = self._sequence_starpair_json_path(item.path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (AttributeError, OSError, json.JSONDecodeError, TypeError, ValueError):
            return None, ()
        if not isinstance(payload, dict):
            return None, ()

        delta_seconds: float | None = None
        timing = payload.get("sequence_timing")
        if isinstance(timing, dict):
            try:
                value = float(timing.get("delta_t_seconds"))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                delta_seconds = value

        supplemental_star_ids: list[str] = []
        seen_star_ids: set[str] = set()
        pairs = payload.get("pairs")
        if isinstance(pairs, list):
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                star_id = str(pair.get("star_id", "")).strip()
                if not star_id or star_id in base_star_ids or star_id in seen_star_ids:
                    continue
                seen_star_ids.add(star_id)
                supplemental_star_ids.append(star_id)
        return delta_seconds, tuple(supplemental_star_ids)

    @staticmethod
    def _sequence_supplemental_star_ids_for_pairs(
        pairs: list[object],
        base_star_ids: frozenset[str],
    ) -> tuple[str, ...]:
        """按当前帧匹配顺序提取不属于第一帧模板的补充星。"""

        supplemental_star_ids: list[str] = []
        seen_star_ids: set[str] = set()
        for pair in pairs:
            reference_star = getattr(pair, "reference_star", None)
            star_id = str(getattr(reference_star, "star_id", "")).strip()
            if not star_id or star_id in base_star_ids or star_id in seen_star_ids:
                continue
            seen_star_ids.add(star_id)
            supplemental_star_ids.append(star_id)
        return tuple(supplemental_star_ids)

    def _run_image_sequence_processing(self, *, continue_only: bool) -> None:
        if getattr(self, "_sequence_processing_active", False):
            QMessageBox.information(self, "正在处理序列", "图像序列仍在处理，请稍候。")
            return
        try:
            self._ensure_first_sequence_session_loaded_for_processing()
            self._ensure_sequence_ready_for_processing()
            first_unprocessed_index = self._sequence_first_unprocessed_index() if continue_only else 1
            if continue_only and first_unprocessed_index is None:
                self.ui.statusbar.showMessage("序列图像均已有完整输出，无需继续处理。")
                self._update_image_sequence_controls()
                return
            if not continue_only and not self._confirm_overwrite_sequence_outputs():
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
            minimum_pair_count, target_pair_count = self._sequence_pair_targets(templates)
            base_star_ids = frozenset(
                star_id
                for template in templates
                if (star_id := str(getattr(template, "star_id", "")).strip())
            )
            assert self.current_image_preview is not None
            target_size = (self.current_image_preview.image.width(), self.current_image_preview.image.height())
            fixed_model = self._fit_sequence_fixed_camera_model(templates, target_size)
        except Exception as exc:  # noqa: BLE001 - 批处理入口要把缺失条件直接反馈给用户。
            QMessageBox.warning(self, "无法处理图像序列", str(exc))
            self.ui.statusbar.showMessage(f"无法处理图像序列: {exc}")
            return

        items = list(getattr(self, "_image_sequence_items", []))
        start_index = int(first_unprocessed_index or 1)
        if continue_only:
            work_count = sum(
                not self._sequence_item_has_outputs(item)
                for item in items[start_index:]
            )
        else:
            work_count = max(len(items) - 1, 0)
        progress = QProgressDialog(self)
        progress.setWindowTitle("正在处理图像序列")
        progress.setRange(0, work_count)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()

        processed: list[tuple[Path, Path]] = []
        failures: list[str] = []
        previous_delta_seconds = 0.0
        preferred_supplemental_star_ids: tuple[str, ...] = ()
        for previous_item in items[1:start_index]:
            saved_delta_seconds, saved_supplemental_star_ids = self._sequence_output_resume_state(
                previous_item,
                base_star_ids,
            )
            if saved_delta_seconds is not None:
                previous_delta_seconds = saved_delta_seconds
            preferred_supplemental_star_ids = saved_supplemental_star_ids
        self._sequence_processing_active = True
        self._set_image_import_controls_enabled(False)
        self._update_image_sequence_controls()
        completed_work = 0
        retained_count = (
            sum(self._sequence_item_has_outputs(item) for item in items[1:])
            if continue_only
            else 0
        )
        try:
            for sequence_index, item in enumerate(items[start_index:], start=start_index):
                if continue_only and self._sequence_item_has_outputs(item):
                    saved_delta_seconds, saved_supplemental_star_ids = self._sequence_output_resume_state(
                        item,
                        base_star_ids,
                    )
                    if saved_delta_seconds is not None:
                        previous_delta_seconds = saved_delta_seconds
                    preferred_supplemental_star_ids = saved_supplemental_star_ids
                    continue
                if progress.wasCanceled():
                    failures.append("用户取消了后续处理。")
                    break
                frame_index = sequence_index + 1
                self._set_image_sequence_processing_index(sequence_index)
                progress.setValue(completed_work)
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
                    sequence_luminance = qimage_to_grayscale_array(preview.image)
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
                        sequence_luminance,
                        templates,
                        fixed_model,
                        target_size,
                        previous_delta_seconds,
                        target_pair_count,
                        stats,
                        preferred_supplemental_star_ids=preferred_supplemental_star_ids,
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
                            sequence_luminance,
                            pairs,
                            fixed_model,
                            target_size,
                            float(time_fit.delta_t_seconds),
                            target_pair_count,
                            stats,
                            preferred_supplemental_star_ids=preferred_supplemental_star_ids,
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
                            "补充匹配后有效匹配 {count} 个，低于序列目标下限 {target} 个；"
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
                    preferred_supplemental_star_ids = self._sequence_supplemental_star_ids_for_pairs(
                        pairs,
                        base_star_ids,
                    )
                    self._refresh_image_sequence_table()
                    self._set_image_sequence_processing_index(sequence_index)
                    previous_delta_seconds = float(time_fit.delta_t_seconds)
                except Exception as exc:  # noqa: BLE001 - 单帧失败不影响后续帧。
                    failures.append(f"{item.path.name}: {exc}")
                    self._set_image_sequence_processing_index(sequence_index)
                finally:
                    completed_work += 1
        finally:
            progress.setValue(completed_work)
            progress.close()
            self._sequence_processing_active = False
            self._set_image_import_controls_enabled(True)
            self._refresh_image_sequence_table()
            self._update_image_sequence_controls()

        action_name = "继续处理" if continue_only else "序列处理"
        self.ui.statusbar.showMessage(f"{action_name}完成：成功 {len(processed)} 张，失败 {len(failures)} 张。")
        message = f"第一帧使用用户基准数据，未参与批处理。\n成功处理 {len(processed)} 张，失败 {len(failures)} 张。"
        if continue_only and retained_count:
            message += f"\n保留已有完整输出 {retained_count} 张。"
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
