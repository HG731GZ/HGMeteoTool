from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QImage
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QTableWidgetItem

from .app_constants import RESIDUAL_WARNING_MIN_PX
from .app_utils import _image_with_binary_mask
from ..image_preview import ImagePreview, load_image_preview
from ..image_sequence import (
    ImageSequenceItem,
    RejectedSequenceImage,
    sequence_item_local_datetime,
    sequence_item_observation_time_utc,
    sequence_item_time_delta_seconds,
)
from ..sequence_constants import (
    IMAGE_SEQUENCE_DELTA_T_RMS_COLUMN,
    IMAGE_SEQUENCE_INDEX_COLUMN,
    IMAGE_SEQUENCE_INDEX_ROLE,
    IMAGE_SEQUENCE_MASK_CACHE_LIMIT,
    IMAGE_SEQUENCE_NAME_COLUMN,
    IMAGE_SEQUENCE_PATH_ROLE,
    IMAGE_SEQUENCE_POSE_RMS_COLUMN,
    IMAGE_SEQUENCE_PREVIEW_CACHE_LIMIT,
    IMAGE_SEQUENCE_REFIT_RMS_COLUMN,
    IMAGE_SEQUENCE_RMS_COLUMN,
    IMAGE_SEQUENCE_RMS_ROLE,
    IMAGE_SEQUENCE_SORTABLE_COLUMNS,
    IMAGE_SEQUENCE_SORT_KEY_INDEX,
    IMAGE_SEQUENCE_SORT_KEY_RMS,
)

class SequenceTablePreviewMixin:
    """图像序列表格、状态和预览显示。"""

    def _reset_image_sequence_status(self) -> None:
        self._image_sequence_items = []
        self._image_sequence_current_index = -1
        self._clear_image_sequence_preview_cache()
        self._set_sequence_status_label("未导入序列", "")
        self._refresh_image_sequence_table()
        self._reset_image_sequence_preview()
        self._update_image_sequence_controls()

    def _bounded_sequence_cache_set(self, cache: OrderedDict, key: object, value: object, limit: int) -> None:
        """写入小型 LRU 缓存，避免序列很长时占用过多内存。"""
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        while len(cache) > limit:
            cache.popitem(last=False)

    def _clear_image_sequence_preview_cache(self) -> None:
        cache = getattr(self, "_image_sequence_preview_cache", None)
        if cache is not None:
            cache.clear()
        self._invalidate_image_sequence_mask_cache()

    def _invalidate_image_sequence_mask_cache(self) -> None:
        scaled_cache = getattr(self, "_image_sequence_scaled_mask_cache", None)
        if scaled_cache is not None:
            scaled_cache.clear()
        masked_cache = getattr(self, "_image_sequence_masked_preview_cache", None)
        if masked_cache is not None:
            masked_cache.clear()

    def _sequence_status_labels(self) -> list[object]:
        labels = []
        for label_name in ("labelImageSequenceStatus", "labelImageSequenceSummary"):
            if hasattr(self.ui, label_name):
                labels.append(getattr(self.ui, label_name))
        return labels

    def _is_sequence_image_path(self, image_path: str | Path) -> bool:
        try:
            resolved_path = Path(image_path).expanduser().resolve()
        except OSError:
            resolved_path = Path(image_path).expanduser()
        for item in getattr(self, "_image_sequence_items", []):
            try:
                if item.path.expanduser().resolve() == resolved_path:
                    return True
            except OSError:
                if item.path.expanduser() == resolved_path:
                    return True
        return False

    def _sequence_mode_active(self) -> bool:
        return bool(getattr(self, "_image_sequence_items", []))

    def _sequence_can_process(self) -> bool:
        return (
            bool(getattr(self, "_image_sequence_items", []))
            and not bool(getattr(self, "_sequence_processing_active", False))
            and getattr(self, "_image_import_thread", None) is None
            and getattr(self, "_sequence_import_thread", None) is None
            and getattr(self, "_json_import_thread", None) is None
        )

    def _update_image_sequence_controls(self) -> None:
        if hasattr(self.ui, "pushButtonProcessImageSequence"):
            self.ui.pushButtonProcessImageSequence.setEnabled(self._sequence_can_process())
        self._update_image_sequence_mask_controls()
        self._update_sequence_refinement_controls()

    def _update_sequence_refinement_controls(self) -> None:
        """仅在整个序列已有成对输出时开放单帧结果修正。"""

        if not hasattr(self.ui, "comboBoxSequenceRefinementMode"):
            return
        ready_method = getattr(self, "_sequence_refinement_ready", None)
        ready = bool(ready_method()) if callable(ready_method) else False
        self.ui.comboBoxSequenceRefinementMode.setEnabled(ready)
        if hasattr(self.ui, "pushButtonRefineSequenceFrames"):
            self.ui.pushButtonRefineSequenceFrames.setEnabled(ready)

    def _update_image_sequence_mask_controls(self) -> None:
        if not hasattr(self.ui, "pushButtonImportImageSequenceSkyMask"):
            return
        sequence_ready = self._sequence_mode_active()
        controls_idle = (
            getattr(self, "_mask_import_thread", None) is None
            and getattr(self, "_image_import_thread", None) is None
            and getattr(self, "_sequence_import_thread", None) is None
            and getattr(self, "_json_import_thread", None) is None
            and not bool(getattr(self, "_sequence_processing_active", False))
        )
        has_mask = self.current_sky_mask is not None
        self.ui.pushButtonImportImageSequenceSkyMask.setEnabled(sequence_ready and controls_idle)
        self.ui.pushButtonClearImageSequenceSkyMask.setEnabled(sequence_ready and controls_idle and has_mask)
        self.ui.checkBoxShowImageSequenceMask.setEnabled(sequence_ready and has_mask)
        if (not sequence_ready or not has_mask) and self.ui.checkBoxShowImageSequenceMask.isChecked():
            was_blocked = self.ui.checkBoxShowImageSequenceMask.blockSignals(True)
            self.ui.checkBoxShowImageSequenceMask.setChecked(False)
            self.ui.checkBoxShowImageSequenceMask.blockSignals(was_blocked)

    def _configure_image_sequence_table_columns(self) -> None:
        if not hasattr(self.ui, "tableWidgetImageSequence"):
            return
        table = self.ui.tableWidgetImageSequence
        header = table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.Interactive)

        # 以较宽的文件名列作为初始布局；交互式表头允许用户后续拖动任意列边界。
        column_widths = {
            IMAGE_SEQUENCE_INDEX_COLUMN: 56,
            IMAGE_SEQUENCE_NAME_COLUMN: 300,
            IMAGE_SEQUENCE_DELTA_T_RMS_COLUMN: 84,
            IMAGE_SEQUENCE_POSE_RMS_COLUMN: 118,
            IMAGE_SEQUENCE_REFIT_RMS_COLUMN: 92,
        }
        for column, width in column_widths.items():
            table.setColumnWidth(column, width)
        self._update_image_sequence_sort_indicator()

    def _update_image_sequence_sort_indicator(self) -> None:
        if not hasattr(self.ui, "tableWidgetImageSequence"):
            return
        header = self.ui.tableWidgetImageSequence.horizontalHeader()
        column_by_sort_key = {
            IMAGE_SEQUENCE_SORT_KEY_INDEX: IMAGE_SEQUENCE_INDEX_COLUMN,
            IMAGE_SEQUENCE_SORT_KEY_RMS: IMAGE_SEQUENCE_RMS_COLUMN,
        }
        column = column_by_sort_key.get(self._image_sequence_sort_key or "")
        if column is None:
            header.setSortIndicatorShown(False)
            return
        sort_order = Qt.DescendingOrder if self._image_sequence_sort_descending else Qt.AscendingOrder
        header.setSortIndicator(column, sort_order)
        header.setSortIndicatorShown(True)

    def _handle_image_sequence_header_clicked(self, column: int) -> None:
        sort_key = IMAGE_SEQUENCE_SORTABLE_COLUMNS.get(column)
        if sort_key is None:
            return
        if self._image_sequence_sort_key == sort_key:
            self._image_sequence_sort_descending = not self._image_sequence_sort_descending
        else:
            self._image_sequence_sort_key = sort_key
            self._image_sequence_sort_descending = sort_key == IMAGE_SEQUENCE_SORT_KEY_RMS
        self._update_image_sequence_sort_indicator()
        self._refresh_image_sequence_table()

    def _handle_image_sequence_cell_clicked(self, row: int, _column: int) -> None:
        if not hasattr(self.ui, "tableWidgetImageSequence"):
            return
        index_item = self.ui.tableWidgetImageSequence.item(row, IMAGE_SEQUENCE_INDEX_COLUMN)
        if index_item is None:
            return
        try:
            sequence_index = int(index_item.data(IMAGE_SEQUENCE_INDEX_ROLE))
        except (TypeError, ValueError):
            return
        self._set_image_sequence_preview_index(sequence_index)

    def _format_sequence_time(self, item: ImageSequenceItem) -> str:
        local_dt = sequence_item_local_datetime(item, self.ui.doubleSpinBoxUtcOffset.value())
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")

    def _set_sequence_status_label(self, text: str, tooltip: str = "") -> None:
        for label in self._sequence_status_labels():
            self._set_elided_label_text(label, text, tooltip)

    def _read_only_sequence_table_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _sequence_starpair_rms_from_json(self, json_path: Path) -> tuple[float | None, int, str, bool]:
        if not json_path.exists():
            return None, 0, "未找到同名配对 JSON。", False
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - 这里只用于界面诊断，坏文件不阻断序列导入。
            return None, 0, f"已有配对 JSON：{json_path}\n无法读取：{exc}", True
        if not isinstance(payload, dict):
            return None, 0, f"已有配对 JSON：{json_path}\n根对象不是字典。", True

        pair_payloads = payload.get("pairs")
        if not isinstance(pair_payloads, list):
            return None, 0, f"已有配对 JSON：{json_path}\n没有 pairs 列表。", True
        residuals: list[float] = []
        for pair_payload in pair_payloads:
            if not isinstance(pair_payload, dict):
                continue
            try:
                residual = float(pair_payload["residual_px"])
            except (KeyError, TypeError, ValueError):
                try:
                    dx = float(pair_payload["residual_dx_px"])
                    dy = float(pair_payload["residual_dy_px"])
                except (KeyError, TypeError, ValueError):
                    continue
                residual = float(np.hypot(dx, dy))
            if math.isfinite(residual):
                residuals.append(residual)
        if not residuals:
            return None, len(pair_payloads), f"已有配对 JSON：{json_path}\n但没有可用残差字段。", True
        residual_array = np.asarray(residuals, dtype=np.float64)
        rms = float(np.sqrt(np.mean(residual_array * residual_array)))
        tooltip = "已有配对 JSON：{path}\n配对数：{count}\n残差 RMS：{rms:.2f} px".format(
            path=json_path,
            count=len(residuals),
            rms=rms,
        )
        return rms, len(residuals), tooltip, True

    def _sequence_refinement_rms_from_json(self, json_path: Path) -> tuple[float | None, float | None, str]:
        """从单帧模型 JSON 读取两种后处理精修的已保存 RMS。"""

        if not json_path.exists():
            return None, None, "未找到同名模型 JSON。"
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - 表格诊断不应阻断预览或序列处理。
            return None, None, f"模型 JSON 无法读取：{exc}"
        if not isinstance(payload, dict):
            return None, None, "模型 JSON 根对象不是字典。"
        refinement = payload.get("sequence_refinement")
        if not isinstance(refinement, dict):
            return None, None, "尚未执行单帧结果修正。"
        results = refinement.get("results")
        if not isinstance(results, dict):
            return None, None, "尚未保存单帧结果修正指标。"

        def result_rms(key: str) -> float | None:
            result = results.get(key)
            if not isinstance(result, dict):
                return None
            try:
                value = float(result.get("rms_px"))
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        pose_rms = result_rms("pose")
        refit_rms = result_rms("refit")
        lines = [f"模型 JSON：{json_path}"]
        if pose_rms is not None:
            lines.append(f"δt + pose RMS：{pose_rms:.2f} px")
        if refit_rms is not None:
            lines.append(f"重拟合 RMS：{refit_rms:.2f} px")
        if len(lines) == 1:
            lines.append("尚未执行单帧结果修正。")
        return pose_rms, refit_rms, "\n".join(lines)

    def _image_sequence_table_entries(
        self,
    ) -> list[tuple[int, ImageSequenceItem, float | None, float | None, float | None, int, str, bool]]:
        entries: list[tuple[int, ImageSequenceItem, float | None, float | None, float | None, int, str, bool]] = []
        for sequence_index, item in enumerate(getattr(self, "_image_sequence_items", [])):
            delta_t_rms, pair_count, tooltip, has_json = self._sequence_starpair_rms_from_json(
                self._sequence_starpair_json_path(item.path)
            )
            pose_rms, refit_rms, refinement_tooltip = self._sequence_refinement_rms_from_json(
                self._sequence_model_json_path(item.path)
            )
            entries.append(
                (
                    sequence_index,
                    item,
                    delta_t_rms,
                    pose_rms,
                    refit_rms,
                    pair_count,
                    f"{tooltip}\n{refinement_tooltip}",
                    has_json,
                )
            )
        if self._image_sequence_sort_key == IMAGE_SEQUENCE_SORT_KEY_RMS:
            return sorted(
                entries,
                key=lambda entry: (
                    entry[2] is None,
                    0.0 if entry[2] is None else (-entry[2] if self._image_sequence_sort_descending else entry[2]),
                    entry[0],
                ),
            )
        if self._image_sequence_sort_key == IMAGE_SEQUENCE_SORT_KEY_INDEX and self._image_sequence_sort_descending:
            return list(reversed(entries))
        return entries

    def _refresh_image_sequence_table(self) -> None:
        if not hasattr(self.ui, "tableWidgetImageSequence"):
            return
        table = self.ui.tableWidgetImageSequence
        entries = self._image_sequence_table_entries()
        signals_were_blocked = table.blockSignals(True)
        table.setRowCount(len(entries))
        for row, (sequence_index, item, delta_t_rms, pose_rms, refit_rms, _pair_count, tooltip, has_json) in enumerate(entries):
            display_index = sequence_index + 1
            index_item = self._read_only_sequence_table_item(str(display_index))
            name_item = self._read_only_sequence_table_item(item.path.name)
            delta_t_item = self._read_only_sequence_table_item(
                "—" if delta_t_rms is None else f"{delta_t_rms:.2f}"
            )
            pose_item = self._read_only_sequence_table_item("—" if pose_rms is None else f"{pose_rms:.2f}")
            refit_item = self._read_only_sequence_table_item("—" if refit_rms is None else f"{refit_rms:.2f}")

            for table_item in (index_item, name_item, delta_t_item, pose_item, refit_item):
                table_item.setData(IMAGE_SEQUENCE_PATH_ROLE, str(item.path))
                table_item.setData(IMAGE_SEQUENCE_INDEX_ROLE, sequence_index)
                table_item.setData(IMAGE_SEQUENCE_RMS_ROLE, delta_t_rms)
                table_item.setToolTip(tooltip)

            if delta_t_rms is not None and delta_t_rms >= RESIDUAL_WARNING_MIN_PX:
                background = QBrush(QColor(255, 210, 210))
                for table_item in (index_item, name_item, delta_t_item, pose_item, refit_item):
                    table_item.setBackground(background)
            elif has_json:
                background = QBrush(QColor(210, 244, 214))
                for table_item in (index_item, name_item, delta_t_item, pose_item, refit_item):
                    table_item.setBackground(background)

            table.setItem(row, IMAGE_SEQUENCE_INDEX_COLUMN, index_item)
            table.setItem(row, IMAGE_SEQUENCE_NAME_COLUMN, name_item)
            table.setItem(row, IMAGE_SEQUENCE_DELTA_T_RMS_COLUMN, delta_t_item)
            table.setItem(row, IMAGE_SEQUENCE_POSE_RMS_COLUMN, pose_item)
            table.setItem(row, IMAGE_SEQUENCE_REFIT_RMS_COLUMN, refit_item)
        table.blockSignals(signals_were_blocked)
        self._select_image_sequence_table_row()

    def _select_image_sequence_table_row(self, *, scroll_to_row: bool = False) -> None:
        self._select_image_sequence_table_index(
            int(getattr(self, "_image_sequence_current_index", -1)),
            scroll_to_row=scroll_to_row,
        )

    def _select_image_sequence_table_index(self, sequence_index: int, *, scroll_to_row: bool = False) -> None:
        if not hasattr(self.ui, "tableWidgetImageSequence"):
            return
        table = self.ui.tableWidgetImageSequence
        if sequence_index < 0:
            table.clearSelection()
            return
        for row in range(table.rowCount()):
            item = table.item(row, IMAGE_SEQUENCE_INDEX_COLUMN)
            if item is None:
                continue
            try:
                row_sequence_index = int(item.data(IMAGE_SEQUENCE_INDEX_ROLE))
            except (TypeError, ValueError):
                continue
            if row_sequence_index == sequence_index:
                table.selectRow(row)
                if scroll_to_row:
                    table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
                return

    def _set_image_sequence_processing_index(self, index: int) -> None:
        """处理序列时只联动左侧表格，不额外刷新右侧预览。"""
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            return
        sequence_index = max(0, min(int(index), len(items) - 1))
        self._select_image_sequence_table_index(sequence_index, scroll_to_row=True)

    def _reset_image_sequence_preview(self) -> None:
        if hasattr(self.ui, "labelImageSequencePreviewTitle"):
            self._set_elided_label_text(self.ui.labelImageSequencePreviewTitle, "未导入序列", "")
        if hasattr(self.ui, "labelImageSequenceExifInfo"):
            self.ui.labelImageSequenceExifInfo.setText("未导入序列")
            self.ui.labelImageSequenceExifInfo.setToolTip("")
        if hasattr(self.ui, "toolButtonImageSequencePrevious"):
            self.ui.toolButtonImageSequencePrevious.setEnabled(False)
        if hasattr(self.ui, "toolButtonImageSequenceNext"):
            self.ui.toolButtonImageSequenceNext.setEnabled(False)
        if hasattr(self, "image_sequence_item"):
            self.image_sequence_item.set_image(QImage())
            if hasattr(self, "image_sequence_scene"):
                self.image_sequence_scene.setSceneRect(self.image_sequence_item.boundingRect())

    def _sequence_exif_text(self, item: ImageSequenceItem, preview: ImagePreview | None) -> str:
        local_dt = sequence_item_local_datetime(item, self.ui.doubleSpinBoxUtcOffset.value())
        utc_dt = sequence_item_observation_time_utc(item, self.ui.doubleSpinBoxUtcOffset.value())
        offset_value = self.ui.doubleSpinBoxUtcOffset.value()
        offset_text = f"UTC{offset_value:+.1f}h"
        lines = [
            f"拍摄时间：{local_dt.strftime('%Y-%m-%d %H:%M:%S')} ({offset_text})",
            f"UTC时间：{utc_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"EXIF来源：{item.capture_time_source}",
        ]
        if preview is not None:
            lines.append(f"原始尺寸：{preview.original_width} x {preview.original_height} px")
        return "  |  ".join(lines)

    def _set_image_sequence_preview_index(self, index: int) -> None:
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            self._image_sequence_current_index = -1
            self._reset_image_sequence_preview()
            return
        self._image_sequence_current_index = max(0, min(int(index), len(items) - 1))
        self._update_image_sequence_preview()
        self._select_image_sequence_table_row()

    def show_previous_image_sequence_frame(self) -> None:
        self._set_image_sequence_preview_index(int(getattr(self, "_image_sequence_current_index", 0)) - 1)

    def show_next_image_sequence_frame(self) -> None:
        self._set_image_sequence_preview_index(int(getattr(self, "_image_sequence_current_index", -1)) + 1)

    def _sequence_preview_mask_visible(self) -> bool:
        return (
            self.current_sky_mask is not None
            and hasattr(self.ui, "checkBoxShowImageSequenceMask")
            and self.ui.checkBoxShowImageSequenceMask.isChecked()
        )

    def _sequence_preview_cache_key(self, item: ImageSequenceItem) -> str:
        try:
            return str(item.path.expanduser().resolve())
        except OSError:
            return str(item.path.expanduser())

    def _image_for_sequence_preview(self, item: ImageSequenceItem) -> ImagePreview:
        cache_key = self._sequence_preview_cache_key(item)
        cache = self._image_sequence_preview_cache
        cached_preview = cache.get(cache_key)
        if cached_preview is not None:
            cache.move_to_end(cache_key)
            return cached_preview
        preview = load_image_preview(item.path)
        self._bounded_sequence_cache_set(
            cache,
            cache_key,
            preview,
            IMAGE_SEQUENCE_PREVIEW_CACHE_LIMIT,
        )
        return preview

    def _scaled_sequence_mask_for_preview(self, width: int, height: int) -> np.ndarray:
        mask = self.current_sky_mask
        if mask is None:
            raise ValueError("当前没有可用蒙版。")
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape == (height, width):
            return mask_array

        cache_key = (id(mask), int(width), int(height))
        cached_mask = self._image_sequence_scaled_mask_cache.get(cache_key)
        if cached_mask is not None:
            self._image_sequence_scaled_mask_cache.move_to_end(cache_key)
            return cached_mask

        mask_height, mask_width = mask_array.shape
        if mask_height <= 0 or mask_width <= 0 or width <= 0 or height <= 0:
            raise ValueError("蒙版或预览图像尺寸无效。")
        y_indices = np.minimum(
            (np.arange(height, dtype=np.float64) * mask_height / float(height)).astype(np.int64),
            mask_height - 1,
        )
        x_indices = np.minimum(
            (np.arange(width, dtype=np.float64) * mask_width / float(width)).astype(np.int64),
            mask_width - 1,
        )
        scaled_mask = np.asarray(mask_array[np.ix_(y_indices, x_indices)], dtype=bool)
        self._bounded_sequence_cache_set(
            self._image_sequence_scaled_mask_cache,
            cache_key,
            scaled_mask,
            IMAGE_SEQUENCE_MASK_CACHE_LIMIT,
        )
        return scaled_mask

    def _sequence_preview_display_image(self, item: ImageSequenceItem, preview: ImagePreview) -> QImage:
        image = preview.image
        if not self._sequence_preview_mask_visible():
            return image
        cache_key = (
            self._sequence_preview_cache_key(item),
            id(self.current_sky_mask),
            int(image.width()),
            int(image.height()),
        )
        cached_image = self._image_sequence_masked_preview_cache.get(cache_key)
        if cached_image is not None:
            self._image_sequence_masked_preview_cache.move_to_end(cache_key)
            return cached_image
        try:
            display_image = _image_with_binary_mask(
                image,
                self._scaled_sequence_mask_for_preview(image.width(), image.height()),
            )
        except ValueError as exc:
            self.ui.statusbar.showMessage(f"序列蒙版尺寸与当前预览图像不一致，暂不显示蒙版: {exc}")
            return image
        self._bounded_sequence_cache_set(
            self._image_sequence_masked_preview_cache,
            cache_key,
            display_image,
            IMAGE_SEQUENCE_MASK_CACHE_LIMIT,
        )
        return display_image

    def _handle_image_sequence_mask_toggled(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_image_sequence_preview()

    def _update_image_sequence_preview(self) -> None:
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            self._reset_image_sequence_preview()
            return

        current_index = max(0, min(int(getattr(self, "_image_sequence_current_index", 0)), len(items) - 1))
        self._image_sequence_current_index = current_index
        item = items[current_index]
        title_text = f"{current_index + 1}/{len(items)}  {item.path.name}"
        if hasattr(self.ui, "labelImageSequencePreviewTitle"):
            self._set_elided_label_text(self.ui.labelImageSequencePreviewTitle, title_text, str(item.path))
        if hasattr(self.ui, "toolButtonImageSequencePrevious"):
            self.ui.toolButtonImageSequencePrevious.setEnabled(current_index > 0)
        if hasattr(self.ui, "toolButtonImageSequenceNext"):
            self.ui.toolButtonImageSequenceNext.setEnabled(current_index < len(items) - 1)

        preview: ImagePreview | None = None
        try:
            preview = self._image_for_sequence_preview(item)
            if hasattr(self, "image_sequence_item"):
                self.image_sequence_item.set_image(self._sequence_preview_display_image(item, preview))
                self.image_sequence_scene.setSceneRect(self.image_sequence_item.boundingRect())
                QTimer.singleShot(0, self.fit_image_sequence_preview)
        except Exception as exc:  # noqa: BLE001 - 序列页预览失败只影响当前缩略图。
            if hasattr(self, "image_sequence_item"):
                self.image_sequence_item.set_image(QImage())
                self.image_sequence_scene.setSceneRect(self.image_sequence_item.boundingRect())
            self.ui.statusbar.showMessage(f"序列预览读取失败: {item.path.name}: {exc}")

        if hasattr(self.ui, "labelImageSequenceExifInfo"):
            exif_text = self._sequence_exif_text(item, preview)
            self.ui.labelImageSequenceExifInfo.setText(exif_text)
            self.ui.labelImageSequenceExifInfo.setToolTip(str(item.path))

    def _handle_image_sequence_time_context_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if not getattr(self, "_image_sequence_items", []):
            return
        self._update_imported_sequence_status()
        self._update_image_sequence_preview()

    def _update_imported_sequence_status(self, rejected: list[RejectedSequenceImage] | None = None) -> None:
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            self._set_sequence_status_label("未导入序列", "")
            self._update_image_sequence_controls()
            return

        first_item = items[0]
        last_item = items[-1]
        span_seconds = sequence_item_time_delta_seconds(last_item, first_item)
        skipped_text = f"，跳过 {len(rejected)} 张" if rejected else ""
        label_text = "{count} 张，{first} -> {last}{skipped}".format(
            count=len(items),
            first=self._format_sequence_time(first_item),
            last=self._format_sequence_time(last_item),
            skipped=skipped_text,
        )
        tooltip_lines = [
            f"序列图像：{len(items)} 张",
            f"第一张：{items[0].path}",
            f"最后一张：{items[-1].path}",
            f"时间跨度：{span_seconds:.1f} 秒",
        ]
        if rejected:
            tooltip_lines.append(f"导入时跳过：{len(rejected)} 张")
        self._set_sequence_status_label(label_text, "\n".join(tooltip_lines))
        self._update_image_sequence_controls()

    def _sequence_item_to_qdatetime(self, item: ImageSequenceItem) -> QDateTime:
        local_dt = sequence_item_local_datetime(item, self.ui.doubleSpinBoxUtcOffset.value())
        return QDateTime.fromString(local_dt.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd HH:mm:ss")

    def _apply_sequence_observation_time(self, item: ImageSequenceItem, *, emit_signal: bool) -> None:
        qt_datetime = self._sequence_item_to_qdatetime(item)
        if not qt_datetime.isValid():
            raise ValueError("序列图像拍摄时间无法转换为界面时间。")
        was_blocked = self.ui.dateTimeEditObservation.blockSignals(not emit_signal)
        try:
            self.ui.dateTimeEditObservation.setDateTime(qt_datetime)
        finally:
            self.ui.dateTimeEditObservation.blockSignals(was_blocked)
        if emit_signal:
            self.schedule_render(delay_ms=0)
