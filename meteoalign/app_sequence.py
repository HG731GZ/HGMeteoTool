from __future__ import annotations
from .app_constants import *

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic

import numpy as np
from PyQt5.QtCore import QDateTime, Qt, QThread, QTimer
from PyQt5.QtGui import QBrush, QColor, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QProgressDialog,
    QTableWidgetItem,
)

from .alignment import (
    MIN_ALIGNMENT_PAIRS,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .app_auto_match import AUTO_MATCH_MIN_ALTITUDE_DEG
from .app_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
    AUTO_MATCH_MIN_AMPLITUDE,
    AUTO_MATCH_SEARCH_MAG_LIMIT,
)
from .app_utils import _image_with_binary_mask, _relative_image_path_for_session
from .app_workers import ImageSequenceCollectWorker
from .fixed_camera_model import (
    FixedCameraModel,
    FixedCameraTimeFitResult,
    estimate_frame_time_correction,
    fit_fixed_camera_model,
)
from .image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from .image_sequence import (
    ImageSequenceItem,
    RejectedSequenceImage,
    sequence_item_local_datetime,
    sequence_item_observation_time_utc,
    sequence_item_time_delta_seconds,
)
from .reference import build_reference_payload
from .simulator import (
    FISHEYE_EQUISOLID,
    ObserverSettings,
    ProjectedStarMap,
    ReferenceStar,
    camera_basis_from_view,
    local_vectors_from_altaz,
    project_horizontal_catalog,
)
from .sequence_geometry import frame_astrometric_model_from_fixed_camera
from .source_model import SourceAstrometricModel
from .star_fitting import FittedStarPosition, fit_star_position


SEQUENCE_MIN_PAIR_FRACTION = 0.80
SEQUENCE_FILL_GRID_COLUMNS = 5
SEQUENCE_FILL_GRID_ROWS = 4
SEQUENCE_SUPPLEMENTAL_FIT_WEIGHT = 0.5
SEQUENCE_SUPPLEMENTAL_PAIR_ORIGIN = "auto_match"
IMAGE_SEQUENCE_INDEX_COLUMN = 0
IMAGE_SEQUENCE_NAME_COLUMN = 1
IMAGE_SEQUENCE_RMS_COLUMN = 2
IMAGE_SEQUENCE_PATH_ROLE = Qt.UserRole + 20
IMAGE_SEQUENCE_INDEX_ROLE = Qt.UserRole + 21
IMAGE_SEQUENCE_RMS_ROLE = Qt.UserRole + 22
IMAGE_SEQUENCE_SORT_KEY_INDEX = "index"
IMAGE_SEQUENCE_SORT_KEY_RMS = "rms"
IMAGE_SEQUENCE_SORTABLE_COLUMNS = {
    IMAGE_SEQUENCE_INDEX_COLUMN: IMAGE_SEQUENCE_SORT_KEY_INDEX,
    IMAGE_SEQUENCE_RMS_COLUMN: IMAGE_SEQUENCE_SORT_KEY_RMS,
}
IMAGE_SEQUENCE_PREVIEW_CACHE_LIMIT = 12
IMAGE_SEQUENCE_MASK_CACHE_LIMIT = 24
IMAGE_SEQUENCE_IMPORT_PROGRESS_MIN_VISIBLE_MS = 500


@dataclass(frozen=True)
class _SequencePairTemplate:
    star_id: str
    reference_star: ReferenceStar
    fitted_position: FittedStarPosition
    fit_constraint_mode: str
    fit_weight: float
    pair_origin: str


@dataclass(frozen=True)
class _SequenceCandidate:
    reference_star: ReferenceStar
    predicted_x_px: float
    predicted_y_px: float


@dataclass(frozen=True)
class _SequenceFitPlan:
    candidate: _SequenceCandidate
    fit_weight: float
    pair_origin: str


@dataclass(frozen=True)
class _SequenceMatchedPair:
    reference_star: ReferenceStar
    fitted_position: FittedStarPosition
    fit_constraint_mode: str
    fit_weight: float
    pair_origin: str
    predicted_x_px: float | None = None
    predicted_y_px: float | None = None
    search_x_px: float | None = None
    search_y_px: float | None = None
    initial_predicted_x_px: float | None = None
    initial_predicted_y_px: float | None = None
    time_delta_seconds: float | None = None
    adaptive_offset_x_px: float = 0.0
    adaptive_offset_y_px: float = 0.0


class SequenceBatchMixin:
    """固定地平坐标视角的图像序列批处理 Mixin。"""

    ui: object
    _image_sequence_items: list[ImageSequenceItem]
    _image_sequence_current_index: int
    _image_sequence_sort_key: str | None
    _image_sequence_sort_descending: bool
    _sequence_processing_active: bool
    _preserve_sequence_on_next_image_load: bool
    _current_star_map: ProjectedStarMap | None
    _source_astrometric_model: SourceAstrometricModel | None
    current_image_preview: ImagePreview | None
    current_sky_mask: np.ndarray | None
    current_sky_mask_path: Path | None
    image_sequence_item: object
    image_sequence_scene: object
    _sequence_import_thread: QThread | None
    _sequence_import_worker: ImageSequenceCollectWorker | None
    _sequence_import_progress: QProgressDialog | None
    _sequence_import_progress_shown_at: float | None
    _image_sequence_preview_cache: OrderedDict[str, ImagePreview]
    _image_sequence_scaled_mask_cache: OrderedDict[tuple[int, int, int], np.ndarray]
    _image_sequence_masked_preview_cache: OrderedDict[tuple[str, int, int, int], QImage]

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
        header.setSectionResizeMode(IMAGE_SEQUENCE_INDEX_COLUMN, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(IMAGE_SEQUENCE_NAME_COLUMN, QHeaderView.Stretch)
        header.setSectionResizeMode(IMAGE_SEQUENCE_RMS_COLUMN, QHeaderView.ResizeToContents)
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

    def _image_sequence_table_entries(self) -> list[tuple[int, ImageSequenceItem, float | None, int, str, bool]]:
        entries: list[tuple[int, ImageSequenceItem, float | None, int, str, bool]] = []
        for sequence_index, item in enumerate(getattr(self, "_image_sequence_items", [])):
            rms, pair_count, tooltip, has_json = self._sequence_starpair_rms_from_json(
                self._sequence_starpair_json_path(item.path)
            )
            entries.append((sequence_index, item, rms, pair_count, tooltip, has_json))
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
        for row, (sequence_index, item, rms, _pair_count, tooltip, has_json) in enumerate(entries):
            display_index = sequence_index + 1
            index_item = self._read_only_sequence_table_item(str(display_index))
            name_item = self._read_only_sequence_table_item(item.path.name)
            rms_text = "—" if rms is None else f"{rms:.2f}"
            rms_item = self._read_only_sequence_table_item(rms_text)

            for table_item in (index_item, name_item, rms_item):
                table_item.setData(IMAGE_SEQUENCE_PATH_ROLE, str(item.path))
                table_item.setData(IMAGE_SEQUENCE_INDEX_ROLE, sequence_index)
                table_item.setData(IMAGE_SEQUENCE_RMS_ROLE, rms)
                table_item.setToolTip(tooltip)

            if rms is not None and rms >= RESIDUAL_WARNING_MIN_PX:
                background = QBrush(QColor(255, 210, 210))
                for table_item in (index_item, name_item, rms_item):
                    table_item.setBackground(background)
            elif has_json:
                background = QBrush(QColor(210, 244, 214))
                for table_item in (index_item, name_item, rms_item):
                    table_item.setBackground(background)

            table.setItem(row, IMAGE_SEQUENCE_INDEX_COLUMN, index_item)
            table.setItem(row, IMAGE_SEQUENCE_NAME_COLUMN, name_item)
            table.setItem(row, IMAGE_SEQUENCE_RMS_COLUMN, rms_item)
        table.blockSignals(signals_were_blocked)
        table.resizeColumnToContents(IMAGE_SEQUENCE_INDEX_COLUMN)
        table.resizeColumnToContents(IMAGE_SEQUENCE_RMS_COLUMN)
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
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "导入序列图像",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_paths:
            return

        self.start_image_sequence_import(file_paths)

    def start_image_sequence_import(self, file_paths: list[str] | tuple[str, ...]) -> None:
        """后台读取序列图像 EXIF 时间，并显示导入进度弹窗。"""
        if getattr(self, "_sequence_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入序列", "当前已有图像序列正在导入，请稍候。")
            return

        self._set_image_import_controls_enabled(False)
        self.ui.statusbar.showMessage(f"正在导入序列图像并读取 EXIF: {len(file_paths)} 张")

        progress = QProgressDialog(self)
        progress.setWindowTitle("正在导入序列图像")
        progress.setLabelText(
            "正在读取序列图像 EXIF 拍摄时间...\n已选择 {count} 张图像".format(count=len(file_paths))
        )
        progress.setRange(0, 0)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        progress.setValue(0)
        force_show = getattr(progress, "forceShow", None)
        if callable(force_show):
            force_show()
        if QApplication.platformName().lower() != "offscreen":
            progress.raise_()
            progress.activateWindow()
        progress.repaint()
        QApplication.processEvents()

        thread = QThread(self)
        worker = ImageSequenceCollectWorker(file_paths)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._queue_image_sequence_import_finished)
        worker.failed.connect(self._queue_image_sequence_import_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_image_sequence_import)

        self._sequence_import_thread = thread
        self._sequence_import_worker = worker
        self._sequence_import_progress = progress
        self._sequence_import_progress_shown_at = monotonic()
        QTimer.singleShot(50, thread.start)

    def _remaining_sequence_import_dialog_delay_ms(self) -> int:
        shown_at = getattr(self, "_sequence_import_progress_shown_at", None)
        if shown_at is None:
            return 0
        elapsed_ms = int((monotonic() - shown_at) * 1000)
        return max(0, IMAGE_SEQUENCE_IMPORT_PROGRESS_MIN_VISIBLE_MS - elapsed_ms)

    def _queue_image_sequence_import_finished(self, result: object) -> None:
        delay_ms = self._remaining_sequence_import_dialog_delay_ms()
        if delay_ms > 0:
            QTimer.singleShot(delay_ms, lambda result=result: self._handle_image_sequence_import_finished(result))
            return
        self._handle_image_sequence_import_finished(result)

    def _queue_image_sequence_import_failed(self, error_message: str) -> None:
        delay_ms = self._remaining_sequence_import_dialog_delay_ms()
        if delay_ms > 0:
            QTimer.singleShot(
                delay_ms,
                lambda error_message=error_message: self._handle_image_sequence_import_failed(error_message),
            )
            return
        self._handle_image_sequence_import_failed(error_message)

    def _quit_image_sequence_import_thread(self) -> None:
        thread = getattr(self, "_sequence_import_thread", None)
        if thread is not None and thread.isRunning():
            thread.quit()

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
        finally:
            if self._sequence_import_progress is not None:
                self._sequence_import_progress.close()
            self._quit_image_sequence_import_thread()

    def _handle_image_sequence_import_failed(self, error_message: str) -> None:
        try:
            if self._sequence_import_progress is not None:
                self._sequence_import_progress.close()
            self.ui.statusbar.showMessage(f"序列导入失败: {error_message}")
            QMessageBox.critical(self, "序列导入失败", error_message)
        finally:
            self._quit_image_sequence_import_thread()

    def _cleanup_image_sequence_import(self) -> None:
        if self._sequence_import_progress is not None:
            self._sequence_import_progress.close()
        self._sequence_import_thread = None
        self._sequence_import_worker = None
        self._sequence_import_progress = None
        self._sequence_import_progress_shown_at = None
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

    def _sequence_base_templates(self) -> list[_SequencePairTemplate]:
        templates: list[_SequencePairTemplate] = []
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            reference_star = self._reference_star_for_row(row)
            fitted_position = self._fitted_position_for_row(row)
            if reference_star is None or fitted_position is None:
                continue
            if not self._is_catalog_reference_star(reference_star):
                continue
            if not all(
                math.isfinite(value)
                for value in (
                    reference_star.ra_deg,
                    reference_star.dec_deg,
                    reference_star.sim_x,
                    reference_star.sim_y,
                    fitted_position.x,
                    fitted_position.y,
                )
            ):
                continue
            mode, fit_weight = self._star_pair_fit_constraint(row)
            pair_origin = "auto_match" if self._is_auto_match_row(row) else "manual"
            templates.append(
                _SequencePairTemplate(
                    star_id=reference_star.star_id.strip(),
                    reference_star=reference_star,
                    fitted_position=fitted_position,
                    fit_constraint_mode=mode,
                    fit_weight=float(fit_weight),
                    pair_origin=pair_origin,
                )
            )
        if len(templates) < MIN_ALIGNMENT_PAIRS:
            raise ValueError(f"第一张图只有 {len(templates)} 个有效恒星配对，至少需要 {MIN_ALIGNMENT_PAIRS} 个。")
        return templates

    def _sequence_pair_targets(self, templates: list[_SequencePairTemplate]) -> tuple[int, int]:
        base_count = len(templates)
        minimum_count = max(MIN_ALIGNMENT_PAIRS, int(math.ceil(base_count * SEQUENCE_MIN_PAIR_FRACTION)))
        desired_count = max(minimum_count, base_count)
        return minimum_count, desired_count

    def _sequence_source_size(self) -> tuple[int, int]:
        if self._current_star_map is None:
            self.render_now()
        if self._current_star_map is None:
            raise ValueError("当前参考星图尚未生成，无法推算序列理论位置。")
        return int(self._current_star_map.width), int(self._current_star_map.height)

    def _sequence_nominal_time_utc(self, item: ImageSequenceItem) -> datetime:
        return sequence_item_observation_time_utc(item, self.ui.doubleSpinBoxUtcOffset.value())

    def _sequence_time_with_delta(self, item: ImageSequenceItem, delta_seconds: float) -> datetime:
        return self._sequence_nominal_time_utc(item) + timedelta(seconds=float(delta_seconds))

    def _sequence_observer_for_item(self, item: ImageSequenceItem, delta_seconds: float = 0.0) -> ObserverSettings:
        return ObserverSettings(
            observation_time_utc=self._sequence_time_with_delta(item, delta_seconds),
            latitude_deg=self.ui.doubleSpinBoxLatitude.value(),
            longitude_deg=self.ui.doubleSpinBoxLongitude.value(),
            elevation_m=self.ui.doubleSpinBoxElevation.value(),
        )

    def _sequence_fixed_lens_model(self, target_size: tuple[int, int]) -> str:
        selected_model = self._alignment_model()
        if selected_model in SKY_KNOWN_PROJECTION_MODELS:
            return selected_model

        # 序列固定相机模型必须有物理基础投影；交互标定仍可继续使用普适锚点插值。
        camera = self._camera_settings_for_image_size(target_size[0], target_size[1])
        if camera.lens_model == FISHEYE_EQUISOLID:
            return SKY_MATCHING_MODEL_FISHEYE_EQUISOLID
        return SKY_MATCHING_MODEL_RECTILINEAR

    def _fit_sequence_fixed_camera_model(
        self,
        templates: list[_SequencePairTemplate],
        target_size: tuple[int, int],
    ) -> FixedCameraModel:
        local_vectors = local_vectors_from_altaz(
            np.asarray([template.reference_star.alt_deg for template in templates], dtype=np.float64),
            np.asarray([template.reference_star.az_deg for template in templates], dtype=np.float64),
        )
        pixel_points = np.asarray(
            [(template.fitted_position.x, template.fitted_position.y) for template in templates],
            dtype=np.float64,
        )
        point_weights = np.asarray([template.fit_weight for template in templates], dtype=np.float64)
        anchor_mask = np.asarray(
            [template.fit_constraint_mode != AUTO_MATCH_CONSTRAINT_SOFT for template in templates],
            dtype=bool,
        )
        initial_rotation_matrix = np.vstack(camera_basis_from_view(self._view_settings())).astype(np.float64)
        return fit_fixed_camera_model(
            enu_vectors=local_vectors,
            pixel_points=pixel_points,
            image_size=target_size,
            lens_model=self._sequence_fixed_lens_model(target_size),
            initial_rotation_matrix=initial_rotation_matrix,
            fisheye_fov_deg=None,
            point_weights=point_weights,
            residual_anchor_mask=anchor_mask,
        )

    def _sequence_projected_star_map(
        self,
        item: ImageSequenceItem,
        source_size: tuple[int, int],
        visible_mag_limit: float,
    ) -> ProjectedStarMap:
        observer = self._sequence_observer_for_item(item)
        camera = self._camera_settings_for_image_size(source_size[0], source_size[1])
        horizontal_catalog = self._get_horizontal_catalog(observer, visible_mag_limit)
        return project_horizontal_catalog(
            horizontal_catalog=horizontal_catalog,
            camera=camera,
            view=self._view_settings(),
            visible_mag_limit=visible_mag_limit,
        )

    def _sequence_candidate_stars(
        self,
        item: ImageSequenceItem,
        fixed_model: FixedCameraModel,
        target_size: tuple[int, int],
        initial_delta_seconds: float,
        visible_mag_limit: float,
    ) -> list[_SequenceCandidate]:
        observer = self._sequence_observer_for_item(item, initial_delta_seconds)
        horizontal_catalog = self._get_horizontal_catalog(observer, visible_mag_limit)
        if len(horizontal_catalog) <= 0:
            return []
        local_vectors = local_vectors_from_altaz(horizontal_catalog.alt_deg, horizontal_catalog.az_deg)
        predicted = fixed_model.project_enu_vectors(local_vectors)
        width_px, height_px = target_size
        finite = np.all(np.isfinite(predicted), axis=1)
        inside = (
            finite
            & (predicted[:, 0] >= 0.0)
            & (predicted[:, 0] < width_px)
            & (predicted[:, 1] >= 0.0)
            & (predicted[:, 1] < height_px)
            & (horizontal_catalog.alt_deg >= AUTO_MATCH_MIN_ALTITUDE_DEG)
        )
        if self.current_sky_mask is not None:
            mask_allowed = np.zeros(len(horizontal_catalog), dtype=bool)
            for index in np.where(inside)[0]:
                mask_allowed[index] = self._sky_mask_allows_point(
                    float(predicted[index, 0]),
                    float(predicted[index, 1]),
                )
            inside &= mask_allowed

        candidate_indices = np.where(inside)[0]
        if candidate_indices.size <= 0:
            return []
        candidate_indices = candidate_indices[np.argsort(horizontal_catalog.mag_v[candidate_indices], kind="stable")]

        candidates: list[_SequenceCandidate] = []
        seen_star_ids: set[str] = set()
        for star_index in candidate_indices:
            star_id = str(horizontal_catalog.star_ids[star_index]).strip()
            display_name = str(horizontal_catalog.display_names[star_index]).strip()
            common_name = str(horizontal_catalog.common_names[star_index]).strip()
            reference_star = ReferenceStar(
                index=0,
                star_id=star_id,
                name=common_name or display_name or star_id,
                display_name=display_name,
                common_name=common_name,
                ra_deg=float(horizontal_catalog.ra_deg[star_index]),
                dec_deg=float(horizontal_catalog.dec_deg[star_index]),
                mag_v=float(horizontal_catalog.mag_v[star_index]),
                sim_x=float(predicted[star_index, 0]),
                sim_y=float(predicted[star_index, 1]),
                alt_deg=float(horizontal_catalog.alt_deg[star_index]),
                az_deg=float(horizontal_catalog.az_deg[star_index]),
            )
            star_id = reference_star.star_id.strip()
            if not star_id or star_id in seen_star_ids:
                continue
            seen_star_ids.add(star_id)
            candidates.append(
                _SequenceCandidate(
                    reference_star=reference_star,
                    predicted_x_px=float(predicted[star_index, 0]),
                    predicted_y_px=float(predicted[star_index, 1]),
                )
            )
        return candidates

    def _sequence_templates_by_mode(
        self,
        templates: list[_SequencePairTemplate],
        mode: str,
    ) -> list[_SequencePairTemplate]:
        if mode == AUTO_MATCH_CONSTRAINT_SOFT:
            return [item for item in templates if item.fit_constraint_mode == AUTO_MATCH_CONSTRAINT_SOFT]
        return [item for item in templates if item.fit_constraint_mode != AUTO_MATCH_CONSTRAINT_SOFT]

    def _ordered_sequence_candidates_for_mode(
        self,
        candidates: list[_SequenceCandidate],
        templates: list[_SequencePairTemplate],
        mode: str,
        used_star_ids: set[str],
    ) -> list[tuple[_SequenceCandidate, _SequencePairTemplate]]:
        candidates_by_id = {candidate.reference_star.star_id.strip(): candidate for candidate in candidates}
        ordered: list[tuple[_SequenceCandidate, _SequencePairTemplate]] = []
        appended: set[str] = set()
        for template in self._sequence_templates_by_mode(templates, mode):
            star_id = template.star_id
            candidate = candidates_by_id.get(star_id)
            if candidate is not None and star_id not in used_star_ids and star_id not in appended:
                ordered.append((candidate, template))
                appended.add(star_id)
        return ordered

    def _sequence_position_is_duplicate(
        self,
        position: tuple[float, float],
        accepted_positions: list[tuple[float, float]],
    ) -> bool:
        for accepted_x, accepted_y in accepted_positions:
            if float(np.hypot(position[0] - accepted_x, position[1] - accepted_y)) < AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX:
                return True
        return False

    def _sequence_adaptive_search_offset(self, accepted_offsets: list[tuple[float, float]]) -> tuple[float, float]:
        if not accepted_offsets:
            return 0.0, 0.0
        offset_array = np.asarray(accepted_offsets, dtype=np.float64)
        if offset_array.ndim != 2 or offset_array.shape[1] != 2:
            return 0.0, 0.0
        finite = np.all(np.isfinite(offset_array), axis=1)
        if not np.any(finite):
            return 0.0, 0.0
        median_offset = np.median(offset_array[finite], axis=0)
        return float(median_offset[0]), float(median_offset[1])

    def _sequence_spatial_cell_index(
        self,
        x_px: float,
        y_px: float,
        target_size: tuple[int, int],
    ) -> int:
        width_px = max(float(target_size[0]), 1.0)
        height_px = max(float(target_size[1]), 1.0)
        column = int(np.clip(math.floor(float(x_px) / width_px * SEQUENCE_FILL_GRID_COLUMNS), 0, SEQUENCE_FILL_GRID_COLUMNS - 1))
        row = int(np.clip(math.floor(float(y_px) / height_px * SEQUENCE_FILL_GRID_ROWS), 0, SEQUENCE_FILL_GRID_ROWS - 1))
        return row * SEQUENCE_FILL_GRID_COLUMNS + column

    def _sequence_spatial_cell_counts(
        self,
        positions: list[tuple[float, float]],
        target_size: tuple[int, int],
    ) -> dict[int, int]:
        counts: dict[int, int] = {}
        for x_px, y_px in positions:
            if not math.isfinite(x_px) or not math.isfinite(y_px):
                continue
            cell_index = self._sequence_spatial_cell_index(x_px, y_px, target_size)
            counts[cell_index] = counts.get(cell_index, 0) + 1
        return counts

    def _sequence_order_supplemental_candidates(
        self,
        candidates: list[_SequenceCandidate],
        *,
        used_star_ids: set[str],
        attempted_star_ids: set[str],
        accepted_positions: list[tuple[float, float]],
        target_size: tuple[int, int],
    ) -> list[_SequenceCandidate]:
        counts = self._sequence_spatial_cell_counts(accepted_positions, target_size)
        grouped: dict[int, list[_SequenceCandidate]] = {}
        for candidate in candidates:
            star_id = candidate.reference_star.star_id.strip()
            if not star_id or star_id in used_star_ids or star_id in attempted_star_ids:
                continue
            predicted_position = (float(candidate.predicted_x_px), float(candidate.predicted_y_px))
            if (
                not math.isfinite(predicted_position[0])
                or not math.isfinite(predicted_position[1])
                or self._sequence_position_is_duplicate(predicted_position, accepted_positions)
            ):
                continue
            cell_index = self._sequence_spatial_cell_index(
                predicted_position[0],
                predicted_position[1],
                target_size,
            )
            grouped.setdefault(cell_index, []).append(candidate)

        for cell_candidates in grouped.values():
            cell_candidates.sort(
                key=lambda candidate: (
                    float(candidate.reference_star.mag_v),
                    float(candidate.predicted_x_px),
                    float(candidate.predicted_y_px),
                )
            )

        ordered: list[_SequenceCandidate] = []
        while grouped:
            active_cells = sorted(grouped, key=lambda cell_index: (counts.get(cell_index, 0), cell_index))
            for cell_index in active_cells:
                cell_candidates = grouped.get(cell_index)
                if not cell_candidates:
                    grouped.pop(cell_index, None)
                    continue
                ordered.append(cell_candidates.pop(0))
                counts[cell_index] = counts.get(cell_index, 0) + 1
                if not cell_candidates:
                    grouped.pop(cell_index, None)
        return ordered

    def _fit_sequence_candidate_plans(
        self,
        image,
        plans: list[_SequenceFitPlan],
        target_count: int,
        search_radius_px: int,
        used_star_ids: set[str],
        attempted_star_ids: set[str],
        accepted_positions: list[tuple[float, float]],
        accepted_offsets: list[tuple[float, float]],
        stats: dict[str, int],
    ) -> list[_SequenceMatchedPair]:
        matched: list[_SequenceMatchedPair] = []
        if target_count <= 0:
            return matched
        for plan in plans:
            if len(matched) >= target_count:
                break
            candidate = plan.candidate
            star_id = candidate.reference_star.star_id.strip()
            if not star_id or star_id in used_star_ids or star_id in attempted_star_ids:
                continue
            attempted_star_ids.add(star_id)
            offset_x, offset_y = self._sequence_adaptive_search_offset(accepted_offsets)
            search_x = candidate.predicted_x_px + offset_x
            search_y = candidate.predicted_y_px + offset_y
            if (
                not math.isfinite(search_x)
                or not math.isfinite(search_y)
                or search_x < 0.0
                or search_y < 0.0
                or search_x >= image.width()
                or search_y >= image.height()
            ):
                stats["skipped_outside"] += 1
                continue
            if not self._sky_mask_allows_point(search_x, search_y):
                stats["skipped_mask"] += 1
                continue
            try:
                fitted_position = fit_star_position(
                    image,
                    click_x=search_x,
                    click_y=search_y,
                    radius_px=search_radius_px,
                )
            except Exception:
                stats["failed_psf"] += 1
                continue

            distance_px = float(
                np.hypot(
                    fitted_position.x - search_x,
                    fitted_position.y - search_y,
                )
            )
            if distance_px > float(search_radius_px) or fitted_position.amplitude < AUTO_MATCH_MIN_AMPLITUDE:
                stats["failed_psf"] += 1
                continue
            if not self._sky_mask_allows_point(fitted_position.x, fitted_position.y):
                stats["skipped_mask"] += 1
                continue
            fitted_xy = (float(fitted_position.x), float(fitted_position.y))
            if self._sequence_position_is_duplicate(fitted_xy, accepted_positions):
                stats["skipped_duplicate"] += 1
                continue

            accepted_offsets.append(
                (
                    float(fitted_position.x - candidate.predicted_x_px),
                    float(fitted_position.y - candidate.predicted_y_px),
                )
            )
            matched.append(
                _SequenceMatchedPair(
                    reference_star=candidate.reference_star,
                    fitted_position=fitted_position,
                    fit_constraint_mode=AUTO_MATCH_CONSTRAINT_SOFT,
                    fit_weight=float(plan.fit_weight),
                    pair_origin=plan.pair_origin,
                    predicted_x_px=candidate.predicted_x_px,
                    predicted_y_px=candidate.predicted_y_px,
                    search_x_px=search_x,
                    search_y_px=search_y,
                    initial_predicted_x_px=candidate.predicted_x_px,
                    initial_predicted_y_px=candidate.predicted_y_px,
                    adaptive_offset_x_px=offset_x,
                    adaptive_offset_y_px=offset_y,
                )
            )
            used_star_ids.add(star_id)
            accepted_positions.append(fitted_xy)
        if len(matched) < target_count:
            stats["missing_target"] += target_count - len(matched)
        return matched

    def _fit_sequence_candidates_for_mode(
        self,
        image,
        candidates: list[_SequenceCandidate],
        templates: list[_SequencePairTemplate],
        mode: str,
        target_count: int,
        search_radius_px: int,
        used_star_ids: set[str],
        attempted_star_ids: set[str],
        accepted_positions: list[tuple[float, float]],
        accepted_offsets: list[tuple[float, float]],
        stats: dict[str, int],
    ) -> list[_SequenceMatchedPair]:
        mode_candidates = self._ordered_sequence_candidates_for_mode(candidates, templates, mode, used_star_ids)
        plans = [
            _SequenceFitPlan(
                candidate=candidate,
                fit_weight=float(template.fit_weight),
                pair_origin=template.pair_origin,
            )
            for candidate, template in mode_candidates
        ]
        return self._fit_sequence_candidate_plans(
            image,
            plans,
            target_count,
            search_radius_px,
            used_star_ids,
            attempted_star_ids,
            accepted_positions,
            accepted_offsets,
            stats,
        )

    def _fit_sequence_supplemental_candidates(
        self,
        image,
        candidates: list[_SequenceCandidate],
        target_size: tuple[int, int],
        target_total: int,
        search_radius_px: int,
        used_star_ids: set[str],
        attempted_star_ids: set[str],
        accepted_positions: list[tuple[float, float]],
        accepted_offsets: list[tuple[float, float]],
        stats: dict[str, int],
    ) -> list[_SequenceMatchedPair]:
        remaining_count = int(target_total) - len(accepted_positions)
        if remaining_count <= 0:
            return []
        ordered_candidates = self._sequence_order_supplemental_candidates(
            candidates,
            used_star_ids=used_star_ids,
            attempted_star_ids=attempted_star_ids,
            accepted_positions=accepted_positions,
            target_size=target_size,
        )
        plans = [
            _SequenceFitPlan(
                candidate=candidate,
                fit_weight=SEQUENCE_SUPPLEMENTAL_FIT_WEIGHT,
                pair_origin=SEQUENCE_SUPPLEMENTAL_PAIR_ORIGIN,
            )
            for candidate in ordered_candidates
        ]
        matched = self._fit_sequence_candidate_plans(
            image,
            plans,
            remaining_count,
            search_radius_px,
            used_star_ids,
            attempted_star_ids,
            accepted_positions,
            accepted_offsets,
            stats,
        )
        stats["supplemental_matched"] += len(matched)
        return matched

    def _first_frame_matched_pairs(self, templates: list[_SequencePairTemplate]) -> list[_SequenceMatchedPair]:
        pairs: list[_SequenceMatchedPair] = []
        for template in templates:
            pairs.append(
                _SequenceMatchedPair(
                    reference_star=template.reference_star,
                    fitted_position=template.fitted_position,
                    fit_constraint_mode=template.fit_constraint_mode,
                    fit_weight=template.fit_weight,
                    pair_origin=template.pair_origin,
                    predicted_x_px=template.fitted_position.x,
                    predicted_y_px=template.fitted_position.y,
                    search_x_px=template.fitted_position.x,
                    search_y_px=template.fitted_position.y,
                    initial_predicted_x_px=template.fitted_position.x,
                    initial_predicted_y_px=template.fitted_position.y,
                    time_delta_seconds=0.0,
                )
            )
        return pairs

    def _sequence_frame_matched_pairs(
        self,
        item: ImageSequenceItem,
        preview: ImagePreview,
        templates: list[_SequencePairTemplate],
        fixed_model: FixedCameraModel,
        target_size: tuple[int, int],
        initial_delta_seconds: float,
        desired_pair_count: int,
        stats: dict[str, int],
    ) -> list[_SequenceMatchedPair]:
        visible_mag_limit = max(self._reference_catalog_mag_limit(AUTO_MATCH_SEARCH_MAG_LIMIT), AUTO_MATCH_SEARCH_MAG_LIMIT)
        candidates = self._sequence_candidate_stars(
            item,
            fixed_model,
            target_size,
            initial_delta_seconds,
            visible_mag_limit,
        )
        anchor_templates = self._sequence_templates_by_mode(templates, "anchor")
        soft_templates = self._sequence_templates_by_mode(templates, AUTO_MATCH_CONSTRAINT_SOFT)
        used_star_ids: set[str] = set()
        attempted_star_ids: set[str] = set()
        accepted_positions: list[tuple[float, float]] = []
        accepted_offsets: list[tuple[float, float]] = []
        search_radius_px = int(self.ui.spinBoxAutoMatchRadius.value())

        anchor_pairs = self._fit_sequence_candidates_for_mode(
            preview.image,
            candidates,
            templates,
            "anchor",
            len(anchor_templates),
            search_radius_px,
            used_star_ids,
            attempted_star_ids,
            accepted_positions,
            accepted_offsets,
            stats,
        )
        soft_pairs = self._fit_sequence_candidates_for_mode(
            preview.image,
            candidates,
            templates,
            AUTO_MATCH_CONSTRAINT_SOFT,
            len(soft_templates),
            search_radius_px,
            used_star_ids,
            attempted_star_ids,
            accepted_positions,
            accepted_offsets,
            stats,
        )
        supplemental_pairs = self._fit_sequence_supplemental_candidates(
            preview.image,
            candidates,
            target_size,
            desired_pair_count,
            search_radius_px,
            used_star_ids,
            attempted_star_ids,
            accepted_positions,
            accepted_offsets,
            stats,
        )
        return [*anchor_pairs, *soft_pairs, *supplemental_pairs]

    def _sequence_fill_frame_pairs_to_target(
        self,
        item: ImageSequenceItem,
        preview: ImagePreview,
        pairs: list[_SequenceMatchedPair],
        fixed_model: FixedCameraModel,
        target_size: tuple[int, int],
        delta_seconds: float,
        desired_pair_count: int,
        stats: dict[str, int],
    ) -> list[_SequenceMatchedPair]:
        if len(pairs) >= desired_pair_count:
            return pairs
        visible_mag_limit = max(self._reference_catalog_mag_limit(AUTO_MATCH_SEARCH_MAG_LIMIT), AUTO_MATCH_SEARCH_MAG_LIMIT)
        candidates = self._sequence_candidate_stars(
            item,
            fixed_model,
            target_size,
            delta_seconds,
            visible_mag_limit,
        )
        used_star_ids = {pair.reference_star.star_id.strip() for pair in pairs if pair.reference_star.star_id.strip()}
        attempted_star_ids: set[str] = set()
        accepted_positions = [
            (float(pair.fitted_position.x), float(pair.fitted_position.y))
            for pair in pairs
        ]
        accepted_offsets = [
            (
                float(pair.fitted_position.x - pair.predicted_x_px),
                float(pair.fitted_position.y - pair.predicted_y_px),
            )
            for pair in pairs
            if pair.predicted_x_px is not None
            and pair.predicted_y_px is not None
            and math.isfinite(pair.predicted_x_px)
            and math.isfinite(pair.predicted_y_px)
        ]
        search_radius_px = int(self.ui.spinBoxAutoMatchRadius.value())
        extra_pairs = self._fit_sequence_supplemental_candidates(
            preview.image,
            candidates,
            target_size,
            desired_pair_count,
            search_radius_px,
            used_star_ids,
            attempted_star_ids,
            accepted_positions,
            accepted_offsets,
            stats,
        )
        return [*pairs, *extra_pairs]

    def _sequence_pair_fit_arrays(
        self,
        pairs: list[_SequenceMatchedPair],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pairs) < MIN_ALIGNMENT_PAIRS:
            raise ValueError(f"有效配对只有 {len(pairs)} 个，至少需要 {MIN_ALIGNMENT_PAIRS} 个。")
        ra_dec_points = np.asarray(
            [(pair.reference_star.ra_deg, pair.reference_star.dec_deg) for pair in pairs],
            dtype=np.float64,
        )
        pixel_points = np.asarray(
            [(pair.fitted_position.x, pair.fitted_position.y) for pair in pairs],
            dtype=np.float64,
        )
        point_weights = np.asarray([pair.fit_weight for pair in pairs], dtype=np.float64)
        return ra_dec_points, pixel_points, point_weights

    def _sequence_time_fit_for_pairs(
        self,
        item: ImageSequenceItem,
        pairs: list[_SequenceMatchedPair],
        fixed_model: FixedCameraModel,
        initial_delta_seconds: float,
        max_iterations: int | None = None,
    ) -> FixedCameraTimeFitResult:
        ra_dec_points, pixel_points, point_weights = self._sequence_pair_fit_arrays(pairs)
        return estimate_frame_time_correction(
            fixed_model=fixed_model,
            ra_dec_points=ra_dec_points,
            observed_pixels=pixel_points,
            nominal_time_utc=self._sequence_nominal_time_utc(item),
            latitude_deg=self.ui.doubleSpinBoxLatitude.value(),
            longitude_deg=self.ui.doubleSpinBoxLongitude.value(),
            elevation_m=self.ui.doubleSpinBoxElevation.value(),
            initial_delta_seconds=initial_delta_seconds,
            point_weights=point_weights,
            max_iterations=4 if max_iterations is None else int(max_iterations),
        )

    def _apply_sequence_time_fit(
        self,
        pairs: list[_SequenceMatchedPair],
        time_fit: FixedCameraTimeFitResult,
        *,
        require_accepted: bool,
    ) -> list[_SequenceMatchedPair]:
        if len(pairs) != int(time_fit.predicted_pixels.shape[0]):
            raise ValueError("时间修正结果与星点配对数量不一致。")
        updated_pairs: list[_SequenceMatchedPair] = []
        for index, pair in enumerate(pairs):
            if require_accepted and not bool(time_fit.accepted_mask[index]):
                continue
            predicted = time_fit.predicted_pixels[index]
            if not np.all(np.isfinite(predicted)):
                continue
            reference_star = replace(
                pair.reference_star,
                sim_x=float(predicted[0]),
                sim_y=float(predicted[1]),
                alt_deg=float(time_fit.alt_deg[index]),
                az_deg=float(time_fit.az_deg[index]),
            )
            updated_pairs.append(
                replace(
                    pair,
                    reference_star=reference_star,
                    predicted_x_px=float(predicted[0]),
                    predicted_y_px=float(predicted[1]),
                    initial_predicted_x_px=(
                        pair.initial_predicted_x_px
                        if pair.initial_predicted_x_px is not None
                        else pair.predicted_x_px
                    ),
                    initial_predicted_y_px=(
                        pair.initial_predicted_y_px
                        if pair.initial_predicted_y_px is not None
                        else pair.predicted_y_px
                    ),
                    time_delta_seconds=float(time_fit.delta_t_seconds),
                )
            )
        if len(updated_pairs) < MIN_ALIGNMENT_PAIRS:
            raise ValueError(f"时间修正后可靠配对只有 {len(updated_pairs)} 个，至少需要 {MIN_ALIGNMENT_PAIRS} 个。")
        return updated_pairs

    def _sequence_pair_records(
        self,
        pairs: list[_SequenceMatchedPair],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for output_index, pair in enumerate(pairs, start=1):
            reference_star = pair.reference_star
            fitted = pair.fitted_position
            predicted_x = pair.predicted_x_px if pair.predicted_x_px is not None else float("nan")
            predicted_y = pair.predicted_y_px if pair.predicted_y_px is not None else float("nan")
            residual_dx = float(predicted_x - fitted.x)
            residual_dy = float(predicted_y - fitted.y)
            record: dict[str, object] = {
                "reference_index": output_index,
                "star_id": reference_star.star_id,
                "name": reference_star.name,
                "display_name": reference_star.display_name,
                "common_name": reference_star.common_name,
                "ra_deg": float(reference_star.ra_deg),
                "dec_deg": float(reference_star.dec_deg),
                "mag_v": float(reference_star.mag_v),
                "image_x_px": float(fitted.x),
                "image_y_px": float(fitted.y),
                "sim_x": float(reference_star.sim_x),
                "sim_y": float(reference_star.sim_y),
                "fixed_model_x_px": float(predicted_x),
                "fixed_model_y_px": float(predicted_y),
                "alt_deg": float(reference_star.alt_deg),
                "az_deg": float(reference_star.az_deg),
                "object_type": "star",
                "pair_origin": pair.pair_origin,
                "fit_constraint_mode": pair.fit_constraint_mode,
                "fit_weight": float(pair.fit_weight),
                "amplitude": float(fitted.amplitude),
                "background": float(fitted.background),
                "sigma_x": float(fitted.sigma_x),
                "sigma_y": float(fitted.sigma_y),
                "residual_dx_px": residual_dx,
                "residual_dy_px": residual_dy,
                "residual_px": float(np.hypot(residual_dx, residual_dy)),
            }
            if pair.predicted_x_px is not None and pair.predicted_y_px is not None:
                record["theoretical_x_px"] = float(pair.predicted_x_px)
                record["theoretical_y_px"] = float(pair.predicted_y_px)
                record["psf_offset_from_theory_px"] = float(
                    np.hypot(fitted.x - pair.predicted_x_px, fitted.y - pair.predicted_y_px)
                )
                record["adaptive_offset_x_px"] = float(pair.adaptive_offset_x_px)
                record["adaptive_offset_y_px"] = float(pair.adaptive_offset_y_px)
            if pair.initial_predicted_x_px is not None and pair.initial_predicted_y_px is not None:
                record["initial_theoretical_x_px"] = float(pair.initial_predicted_x_px)
                record["initial_theoretical_y_px"] = float(pair.initial_predicted_y_px)
                record["psf_offset_from_initial_theory_px"] = float(
                    np.hypot(fitted.x - pair.initial_predicted_x_px, fitted.y - pair.initial_predicted_y_px)
                )
            if pair.time_delta_seconds is not None:
                record["delta_t_seconds"] = float(pair.time_delta_seconds)
            if pair.search_x_px is not None and pair.search_y_px is not None:
                record["search_x_px"] = float(pair.search_x_px)
                record["search_y_px"] = float(pair.search_y_px)
                record["psf_offset_from_search_px"] = float(
                    np.hypot(fitted.x - pair.search_x_px, fitted.y - pair.search_y_px)
                )
            records.append(record)
        return records

    def _sequence_image_base_payload(
        self,
        preview: ImagePreview,
        json_path: Path,
        item: ImageSequenceItem,
    ) -> dict[str, object]:
        image_path = Path(preview.path).expanduser().resolve()
        payload = {
            "path": str(image_path),
            "relative_path": _relative_image_path_for_session(image_path, json_path),
            "file_name": image_path.name,
            "original_width_px": int(preview.original_width),
            "original_height_px": int(preview.original_height),
        }
        payload.update(self._sequence_capture_payload(item))
        return payload

    def _sequence_real_image_payload(
        self,
        preview: ImagePreview,
        json_path: Path,
        item: ImageSequenceItem,
    ) -> dict[str, object]:
        payload = self._sequence_image_base_payload(preview, json_path, item)
        payload.update(
            {
                "display_width_px": int(preview.image.width()),
                "display_height_px": int(preview.image.height()),
            }
        )
        return payload

    def _sequence_source_image_payload(
        self,
        preview: ImagePreview,
        json_path: Path,
        item: ImageSequenceItem,
    ) -> dict[str, object]:
        payload = self._sequence_image_base_payload(preview, json_path, item)
        payload.update(
            {
                "model_width_px": int(preview.image.width()),
                "model_height_px": int(preview.image.height()),
            }
        )
        return payload

    def _sequence_capture_payload(self, item: ImageSequenceItem) -> dict[str, object]:
        payload: dict[str, object] = {
            "exif_capture_time": item.capture_datetime.isoformat(),
            "capture_time_source": item.capture_time_source,
        }
        if item.capture_datetime_utc is not None:
            payload["capture_time_utc"] = item.capture_datetime_utc.isoformat()
        return payload

    def _sequence_reference_payload(
        self,
        item: ImageSequenceItem,
        preview: ImagePreview,
        pairs: list[_SequenceMatchedPair],
    ) -> dict[str, object]:
        observer = self._sequence_observer_for_item(item)
        camera = self._camera_settings_for_image_size(preview.image.width(), preview.image.height())
        view = self._view_settings()
        visible_mag_limit = max(
            self._reference_catalog_mag_limit(AUTO_MATCH_SEARCH_MAG_LIMIT),
            AUTO_MATCH_SEARCH_MAG_LIMIT,
        )
        star_map = self._sequence_projected_star_map(
            item,
            (preview.image.width(), preview.image.height()),
            visible_mag_limit,
        )
        reference_stars = tuple(
            self._reference_star_with_index(pair.reference_star, index)
            for index, pair in enumerate(pairs, start=1)
        )
        payload = build_reference_payload(
            star_map=star_map,
            reference_stars=reference_stars,
            observer=observer,
            camera=camera,
            view=view,
            visible_mag_limit=visible_mag_limit,
            utc_offset_hours=self.ui.doubleSpinBoxUtcOffset.value(),
            reference_label_mode=self._reference_label_mode(),
            reference_mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            manual_reference_star_ids=tuple(pair.reference_star.star_id for pair in pairs),
        )
        observer_payload = payload.get("observer")
        if isinstance(observer_payload, dict):
            observer_payload.update(
                {
                    "observation_time_source": "real_image_exif",
                    "capture_time_source": item.capture_time_source,
                    "exif_capture_time": item.capture_datetime.isoformat(),
                }
            )
            if item.capture_datetime_utc is not None:
                observer_payload["capture_time_utc"] = item.capture_datetime_utc.isoformat()
        return payload

    def _sequence_matching_payload(self) -> dict[str, object]:
        return self._auto_match_settings_payload()

    def _sequence_timing_payload(
        self,
        item: ImageSequenceItem,
        first_item: ImageSequenceItem,
        time_fit: FixedCameraTimeFitResult,
    ) -> dict[str, object]:
        nominal_time_utc = self._sequence_nominal_time_utc(item)
        effective_time_utc = nominal_time_utc + timedelta(seconds=float(time_fit.delta_t_seconds))
        first_time_utc = self._sequence_nominal_time_utc(first_item)
        return {
            "time_model": "first_frame_relative_exif_plus_per_frame_delta_t",
            "first_frame_nominal_time_utc": first_time_utc.isoformat(),
            "frame_nominal_time_utc": nominal_time_utc.isoformat(),
            "frame_effective_time_utc": effective_time_utc.isoformat(),
            "exif_delta_from_first_seconds": float(sequence_item_time_delta_seconds(item, first_item)),
            "delta_t_seconds": float(time_fit.delta_t_seconds),
            "delta_t_reference": "relative_to_frame_nominal_time_not_chained",
            "delta_t_0_seconds": 0.0,
            "time_fit": time_fit.to_json_payload(),
        }

    def _sequence_model_payload(
        self,
        *,
        item: ImageSequenceItem,
        first_item: ImageSequenceItem,
        preview: ImagePreview,
        model_path: Path,
        fixed_model: FixedCameraModel,
        time_fit: FixedCameraTimeFitResult,
        records: list[dict[str, object]],
        reference_payload: dict[str, object],
        generated_at_utc: str,
    ) -> dict[str, object]:
        timing_payload = self._sequence_timing_payload(item, first_item, time_fit)
        observer = ObserverSettings(
            observation_time_utc=datetime.fromisoformat(
                str(timing_payload["frame_effective_time_utc"]).replace("Z", "+00:00")
            ).astimezone(timezone.utc),
            latitude_deg=float(self.ui.doubleSpinBoxLatitude.value()),
            longitude_deg=float(self.ui.doubleSpinBoxLongitude.value()),
            elevation_m=float(self.ui.doubleSpinBoxElevation.value()),
        )
        diagnostics = {
            "pair_count": len(records),
            "rms_px": float(time_fit.rms_px),
            "median_residual_px": float(time_fit.median_residual_px),
            "max_residual_px": float(time_fit.max_residual_px),
            **time_fit.to_json_payload(),
        }
        frame_model = frame_astrometric_model_from_fixed_camera(
            fixed_camera_model=fixed_model,
            observer=observer,
            fit_metadata={
                "model_type": "sequence_frame_astrometric_model",
                "source_sequence_model": "fixed_camera_enu_sequence_geometry",
                "control_point_count": len(records),
                "sequence_timing": timing_payload,
                "scene_observer_hint": {
                    "observation_time_utc": observer.observation_time_utc.isoformat(),
                    "latitude_deg": observer.latitude_deg,
                    "longitude_deg": observer.longitude_deg,
                    "elevation_m": observer.elevation_m,
                    "utc_offset_hours": float(self.ui.doubleSpinBoxUtcOffset.value()),
                    **timing_payload,
                },
                "scene_observer_hint_role": "metadata_only_not_required_for_pixel_icrs_model",
            },
            diagnostics=diagnostics,
        )
        return frame_model.to_json_payload(
            source_image=self._sequence_source_image_payload(preview, model_path, item),
            mask=self._sky_mask_payload(model_path),
            matching=self._sequence_matching_payload(),
            fit_pairs=records,
            reference_payload=reference_payload,
            generated_at_utc=generated_at_utc,
        )

    def _sequence_starpair_json_path(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_starpairs.json")

    def _sequence_model_json_path(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_model.json")

    def _write_sequence_outputs(
        self,
        item: ImageSequenceItem,
        first_item: ImageSequenceItem,
        preview: ImagePreview,
        pairs: list[_SequenceMatchedPair],
        fixed_model: FixedCameraModel,
        time_fit: FixedCameraTimeFitResult,
    ) -> tuple[Path, Path]:
        image_path = Path(preview.path).expanduser().resolve()
        starpair_path = self._sequence_starpair_json_path(image_path)
        model_path = self._sequence_model_json_path(image_path)
        records = self._sequence_pair_records(pairs)
        reference_payload = self._sequence_reference_payload(item, preview, pairs)
        generated_at_utc = datetime.now(timezone.utc).isoformat()
        timing_payload = self._sequence_timing_payload(item, first_item, time_fit)
        starpair_payload = {
            "format": STAR_PAIR_SESSION_FORMAT,
            "version": STAR_PAIR_SESSION_VERSION,
            "generated_at_utc": generated_at_utc,
            "real_image": self._sequence_real_image_payload(preview, starpair_path, item),
            "reference_payload": reference_payload,
            "sky_alignment_model": self._alignment_model(),
            "image_model": "fixed_camera_model",
            "sequence_timing": timing_payload,
            "pair_count": len(records),
            "pairs": records,
            "mask": self._sky_mask_payload(starpair_path),
            "matching": self._sequence_matching_payload(),
        }
        source_payload = self._sequence_model_payload(
            item=item,
            first_item=first_item,
            preview=preview,
            model_path=model_path,
            fixed_model=fixed_model,
            time_fit=time_fit,
            records=records,
            reference_payload=reference_payload,
            generated_at_utc=generated_at_utc,
        )

        starpair_path.write_text(json.dumps(starpair_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        model_path.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return starpair_path, model_path

    def _current_preview_is_sequence_first_item(self, first_item: ImageSequenceItem) -> bool:
        if self.current_image_preview is None:
            return False
        try:
            return Path(self.current_image_preview.path).expanduser().resolve() == first_item.path.expanduser().resolve()
        except OSError:
            return False

    def _load_first_sequence_image_without_tab_switch(self, first_item: ImageSequenceItem) -> None:
        preview = load_image_preview(first_item.path, max_long_side_px=None)
        self._preserve_sequence_on_next_image_load = True
        self._apply_loaded_image_preview(
            preview,
            clear_existing_pairs=True,
            switch_to_reference=False,
        )

    def _ensure_first_sequence_image_ready_for_mask(self) -> ImageSequenceItem:
        first_item = self._current_sequence_first_item()
        if self._current_preview_is_sequence_first_item(first_item):
            return first_item
        if self._sequence_starpair_json_path(first_item.path).exists():
            self._load_first_sequence_session_for_reference_page(raise_on_error=True)
        else:
            current_tab = self.ui.tabWidgetMain.currentWidget()
            self._load_first_sequence_image_without_tab_switch(first_item)
            if current_tab is not None:
                self.ui.tabWidgetMain.setCurrentWidget(current_tab)
        if not self._current_preview_is_sequence_first_item(first_item):
            raise ValueError("序列第一帧图像尚未载入，无法导入序列蒙版。")
        return first_item

    def import_image_sequence_sky_mask(self) -> None:
        if not self._sequence_mode_active():
            QMessageBox.information(self, "尚未导入序列", "请先导入图像序列，再导入序列蒙版。")
            return
        if getattr(self, "_mask_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入蒙版", "当前已有蒙版正在导入，请稍候。")
            return
        if getattr(self, "_image_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return
        try:
            first_item = self._ensure_first_sequence_image_ready_for_mask()
        except Exception as exc:  # noqa: BLE001 - 蒙版入口需要把基准图未就绪原因直接反馈给用户。
            QMessageBox.warning(self, "无法导入序列蒙版", str(exc))
            return

        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入序列蒙版",
            str(first_item.path.parent),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        self.start_sky_mask_import(file_path)

    def clear_image_sequence_sky_mask(self) -> None:
        if not self._sequence_mode_active():
            self.ui.statusbar.showMessage("当前没有正在处理的图像序列。")
            return
        self.clear_sky_mask()
        self._update_image_sequence_controls()
        self._update_image_sequence_preview()
        self.ui.statusbar.showMessage("已清除序列蒙版，后续序列自动匹配将使用整张图像。")

    def _load_first_sequence_session_for_reference_page(self, *, raise_on_error: bool) -> bool:
        first_item = self._current_sequence_first_item()
        json_path = self._sequence_starpair_json_path(first_item.path)
        if not json_path.exists():
            return False
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            preview = load_image_preview(first_item.path, max_long_side_px=None)
            current_tab = self.ui.tabWidgetMain.currentWidget()
            self._clear_star_pair_positions_for_new_input("第一帧配对 JSON")
            self._apply_star_pair_session_payload(
                payload,
                json_path,
                preview=preview,
                switch_to_reference=False,
            )
            if current_tab is not None:
                self.ui.tabWidgetMain.setCurrentWidget(current_tab)
            return True
        except Exception as exc:  # noqa: BLE001 - 序列导入时尽量保留序列页，错误转为界面提示。
            message = f"无法自动载入第一帧配对 JSON：{json_path}\n{exc}"
            if raise_on_error:
                raise ValueError(message) from exc
            try:
                self._load_first_sequence_image_without_tab_switch(first_item)
            except Exception as image_exc:  # noqa: BLE001 - 兜底载图失败同样反馈给用户。
                message += f"\n\n第一帧图像也无法载入：{image_exc}"
            QMessageBox.warning(self, "第一帧配对 JSON 载入失败", message)
            self.ui.statusbar.showMessage(f"第一帧配对 JSON 载入失败: {json_path}")
            return False

    def _ensure_first_sequence_session_loaded_for_processing(self) -> None:
        first_item = self._current_sequence_first_item()
        if self._current_preview_is_sequence_first_item(first_item) and self._star_pair_position_count() >= MIN_ALIGNMENT_PAIRS:
            return

        self._load_first_sequence_session_for_reference_page(raise_on_error=True)

    def _ensure_first_sequence_output_jsons(self, first_item: ImageSequenceItem) -> list[Path]:
        """确保第一帧有用户基准 JSON；已有文件绝不覆盖。"""
        starpair_path = self._sequence_starpair_json_path(first_item.path)
        model_path = self._sequence_model_json_path(first_item.path)
        created_paths: list[Path] = []
        if not starpair_path.exists():
            starpair_payload = self._build_star_pair_session_payload(starpair_path)
            starpair_path.write_text(json.dumps(starpair_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            created_paths.append(starpair_path)
        if not model_path.exists():
            model_payload = self._build_source_model_payload(model_path)
            model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            created_paths.append(model_path)
        return created_paths

    def _confirm_overwrite_sequence_outputs(self) -> bool:
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            return False
        existing_paths: list[Path] = []
        for item in items[1:]:
            for output_path in (
                self._sequence_starpair_json_path(item.path),
                self._sequence_model_json_path(item.path),
            ):
                if output_path.exists():
                    existing_paths.append(output_path)

        if existing_paths:
            sample_lines = "\n".join(str(path) for path in existing_paths[:6])
            if len(existing_paths) > 6:
                sample_lines += f"\n... 另有 {len(existing_paths) - 6} 个文件"
            message = (
                "开始处理后会覆盖第二帧及之后已有的配对 JSON 与模型 JSON。\n"
                "第一帧已有 JSON 会保留，不参与覆盖。\n\n"
                f"已发现 {len(existing_paths)} 个已有输出文件：\n{sample_lines}\n\n是否继续？"
            )
        else:
            message = (
                "开始处理后会为第二帧及之后写入同名配对 JSON 与模型 JSON。\n"
                "第一帧只作为用户基准，已有 JSON 会保留，缺失 JSON 会先自动补齐。是否继续？"
            )
        reply = QMessageBox.question(
            self,
            "确认处理图像序列",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def process_image_sequence(self) -> None:
        if getattr(self, "_sequence_processing_active", False):
            QMessageBox.information(self, "正在处理序列", "图像序列仍在处理，请稍候。")
            return
        try:
            self._ensure_first_sequence_session_loaded_for_processing()
            self._ensure_sequence_ready_for_processing()
            if not self._confirm_overwrite_sequence_outputs():
                self.ui.statusbar.showMessage("已取消图像序列处理。")
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
            QMessageBox.warning(self, "图像序列处理完成", message)
        else:
            QMessageBox.information(self, "图像序列处理完成", message)
