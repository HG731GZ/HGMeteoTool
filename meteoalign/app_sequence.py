from __future__ import annotations
from .app_constants import *

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

from .alignment import MIN_ALIGNMENT_PAIRS, ReferenceAlignmentTransform, fit_reference_alignment
from .app_auto_match import AUTO_MATCH_MIN_ALTITUDE_DEG
from .app_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
    AUTO_MATCH_MIN_AMPLITUDE,
    AUTO_MATCH_SEARCH_MAG_LIMIT,
)
from .app_utils import _relative_image_path_for_session
from .coordinates import radec_to_unit_vectors
from .image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from .image_sequence import (
    ImageSequenceItem,
    RejectedSequenceImage,
    collect_image_sequence,
    sequence_item_local_datetime,
    sequence_item_observation_time_utc,
    sequence_item_time_delta_seconds,
)
from .reference import build_reference_payload
from .simulator import (
    ObserverSettings,
    ProjectedStarMap,
    ReferenceStar,
    camera_basis_from_view,
    local_vectors_from_altaz,
    project_horizontal_catalog,
)
from .source_model import SourceAstrometricModel, fit_source_astrometric_model
from .star_fitting import FittedStarPosition, fit_star_position


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
    adaptive_offset_x_px: float = 0.0
    adaptive_offset_y_px: float = 0.0


class SequenceBatchMixin:
    """固定地平坐标视角的图像序列批处理 Mixin。"""

    ui: object
    _image_sequence_items: list[ImageSequenceItem]
    _sequence_processing_active: bool
    _current_star_map: ProjectedStarMap | None
    _source_astrometric_model: SourceAstrometricModel | None
    current_image_preview: ImagePreview | None
    current_sky_mask: np.ndarray | None
    current_sky_mask_path: Path | None

    def _reset_image_sequence_status(self) -> None:
        self._image_sequence_items = []
        if hasattr(self.ui, "labelImageSequenceStatus"):
            self._set_elided_label_text(self.ui.labelImageSequenceStatus, "未导入序列", "")
        self._update_image_sequence_controls()

    def _sequence_can_process(self) -> bool:
        return bool(getattr(self, "_image_sequence_items", [])) and not bool(
            getattr(self, "_sequence_processing_active", False)
        ) and getattr(self, "_image_import_thread", None) is None and self.current_image_preview is not None

    def _update_image_sequence_controls(self) -> None:
        if hasattr(self.ui, "pushButtonProcessImageSequence"):
            self.ui.pushButtonProcessImageSequence.setEnabled(self._sequence_can_process())

    def _format_sequence_time(self, item: ImageSequenceItem) -> str:
        local_dt = sequence_item_local_datetime(item, self.ui.doubleSpinBoxUtcOffset.value())
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")

    def _set_sequence_status_label(self, text: str, tooltip: str = "") -> None:
        if hasattr(self.ui, "labelImageSequenceStatus"):
            self._set_elided_label_text(self.ui.labelImageSequenceStatus, text, tooltip)

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

        items, rejected = collect_image_sequence(file_paths)
        if not items:
            self._reset_image_sequence_status()
            message = "所选图像都没有读到 EXIF 拍摄时间，已全部跳过。"
            if rejected:
                message += "\n\n" + self._rejected_sequence_summary(rejected)
            QMessageBox.warning(self, "未导入序列图像", message)
            self.ui.statusbar.showMessage("序列导入失败：全部图像缺少可用 EXIF 拍摄时间。")
            return

        self._image_sequence_items = items
        self._update_imported_sequence_status(rejected)
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
        self.ui.statusbar.showMessage(
            f"已导入序列 {len(items)} 张，第一张将作为匹配基准: {items[0].path}"
        )
        self.start_single_image_import(items[0].path)

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

    def _sequence_source_size(self) -> tuple[int, int]:
        if self._current_star_map is None:
            self.render_now()
        if self._current_star_map is None:
            raise ValueError("当前参考星图尚未生成，无法推算序列理论位置。")
        return int(self._current_star_map.width), int(self._current_star_map.height)

    def _fit_sequence_sim_to_real_transform(
        self,
        templates: list[_SequencePairTemplate],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> ReferenceAlignmentTransform:
        source_points = np.asarray(
            [(item.reference_star.sim_x, item.reference_star.sim_y) for item in templates],
            dtype=np.float64,
        )
        target_points = np.asarray(
            [(item.fitted_position.x, item.fitted_position.y) for item in templates],
            dtype=np.float64,
        )
        return fit_reference_alignment(
            source_points=source_points,
            target_points=target_points,
            source_size=source_size,
            target_size=target_size,
        )

    def _fit_sequence_pairs_sim_to_real_transform(
        self,
        pairs: list[_SequenceMatchedPair],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> ReferenceAlignmentTransform:
        source_points = np.asarray(
            [(item.reference_star.sim_x, item.reference_star.sim_y) for item in pairs],
            dtype=np.float64,
        )
        target_points = np.asarray(
            [(item.fitted_position.x, item.fitted_position.y) for item in pairs],
            dtype=np.float64,
        )
        return fit_reference_alignment(
            source_points=source_points,
            target_points=target_points,
            source_size=source_size,
            target_size=target_size,
        )

    def _sequence_projected_star_map(
        self,
        item: ImageSequenceItem,
        source_size: tuple[int, int],
        visible_mag_limit: float,
    ) -> ProjectedStarMap:
        observer = ObserverSettings(
            observation_time_utc=sequence_item_observation_time_utc(item, self.ui.doubleSpinBoxUtcOffset.value()),
            latitude_deg=self.ui.doubleSpinBoxLatitude.value(),
            longitude_deg=self.ui.doubleSpinBoxLongitude.value(),
            elevation_m=self.ui.doubleSpinBoxElevation.value(),
        )
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
        star_map: ProjectedStarMap,
        sim_to_real: ReferenceAlignmentTransform,
        target_size: tuple[int, int],
    ) -> list[_SequenceCandidate]:
        if len(star_map) <= 0:
            return []
        sim_points = np.column_stack((star_map.x_px, star_map.y_px))
        predicted = sim_to_real.transform_points(sim_points)
        width_px, height_px = target_size
        finite = np.all(np.isfinite(predicted), axis=1)
        inside = (
            finite
            & (predicted[:, 0] >= 0.0)
            & (predicted[:, 0] < width_px)
            & (predicted[:, 1] >= 0.0)
            & (predicted[:, 1] < height_px)
            & (star_map.alt_deg >= AUTO_MATCH_MIN_ALTITUDE_DEG)
        )
        if self.current_sky_mask is not None:
            mask_allowed = np.zeros(len(star_map), dtype=bool)
            for index in np.where(inside)[0]:
                mask_allowed[index] = self._sky_mask_allows_point(
                    float(predicted[index, 0]),
                    float(predicted[index, 1]),
                )
            inside &= mask_allowed

        candidate_indices = np.where(inside)[0]
        if candidate_indices.size <= 0:
            return []
        candidate_indices = candidate_indices[np.argsort(star_map.mag_v[candidate_indices], kind="stable")]

        candidates: list[_SequenceCandidate] = []
        seen_star_ids: set[str] = set()
        for star_index in candidate_indices:
            reference_star = self._reference_star_from_star_map_index(star_map, int(star_index), output_index=0)
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
    ) -> list[_SequenceCandidate]:
        candidates_by_id = {candidate.reference_star.star_id.strip(): candidate for candidate in candidates}
        ordered: list[_SequenceCandidate] = []
        appended: set[str] = set()
        for template in self._sequence_templates_by_mode(templates, mode):
            star_id = template.star_id
            candidate = candidates_by_id.get(star_id)
            if candidate is not None and star_id not in used_star_ids and star_id not in appended:
                ordered.append(candidate)
                appended.add(star_id)

        for candidate in candidates:
            star_id = candidate.reference_star.star_id.strip()
            if star_id and star_id not in used_star_ids and star_id not in appended:
                ordered.append(candidate)
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

    def _fit_sequence_candidates_for_mode(
        self,
        image,
        candidates: list[_SequenceCandidate],
        templates: list[_SequencePairTemplate],
        mode: str,
        target_count: int,
        search_radius_px: int,
        used_star_ids: set[str],
        accepted_positions: list[tuple[float, float]],
        accepted_offsets: list[tuple[float, float]],
        stats: dict[str, int],
    ) -> list[_SequenceMatchedPair]:
        matched: list[_SequenceMatchedPair] = []
        if target_count <= 0:
            return matched
        mode_candidates = self._ordered_sequence_candidates_for_mode(candidates, templates, mode, used_star_ids)
        fit_weight = self._auto_match_soft_weight() if mode == AUTO_MATCH_CONSTRAINT_SOFT else 1.0
        pair_origin = "auto_match"
        attempted_star_ids: set[str] = set()
        for candidate in mode_candidates:
            if len(matched) >= target_count:
                break
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
                    fit_constraint_mode=mode,
                    fit_weight=fit_weight,
                    pair_origin=pair_origin,
                    predicted_x_px=candidate.predicted_x_px,
                    predicted_y_px=candidate.predicted_y_px,
                    search_x_px=search_x,
                    search_y_px=search_y,
                    adaptive_offset_x_px=offset_x,
                    adaptive_offset_y_px=offset_y,
                )
            )
            used_star_ids.add(star_id)
            accepted_positions.append(fitted_xy)
        if len(matched) < target_count:
            stats["missing_target"] += target_count - len(matched)
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
                )
            )
        return pairs

    def _sequence_frame_matched_pairs(
        self,
        item: ImageSequenceItem,
        preview: ImagePreview,
        templates: list[_SequencePairTemplate],
        sim_to_real: ReferenceAlignmentTransform,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        stats: dict[str, int],
    ) -> list[_SequenceMatchedPair]:
        visible_mag_limit = max(self._reference_catalog_mag_limit(AUTO_MATCH_SEARCH_MAG_LIMIT), AUTO_MATCH_SEARCH_MAG_LIMIT)
        star_map = self._sequence_projected_star_map(item, source_size, visible_mag_limit)
        candidates = self._sequence_candidate_stars(star_map, sim_to_real, target_size)
        anchor_templates = self._sequence_templates_by_mode(templates, "anchor")
        soft_templates = self._sequence_templates_by_mode(templates, AUTO_MATCH_CONSTRAINT_SOFT)
        used_star_ids: set[str] = set()
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
            accepted_positions,
            accepted_offsets,
            stats,
        )
        return [*anchor_pairs, *soft_pairs]

    def _fit_sequence_source_model(
        self,
        pairs: list[_SequenceMatchedPair],
        image_size: tuple[int, int],
    ) -> SourceAstrometricModel:
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
        anchor_mask = np.asarray(
            [pair.fit_constraint_mode != AUTO_MATCH_CONSTRAINT_SOFT for pair in pairs],
            dtype=bool,
        )
        initial_rotation_matrix = self._sequence_initial_projection_rotation_matrix(pairs)
        return fit_source_astrometric_model(
            ra_dec_points=ra_dec_points,
            pixel_points=pixel_points,
            image_size=image_size,
            matching_model=self._alignment_model(),
            fisheye_fov_deg=None,
            initial_rotation_matrix=initial_rotation_matrix,
            point_weights=point_weights,
            residual_anchor_mask=anchor_mask,
        )

    def _sequence_initial_projection_rotation_matrix(
        self,
        pairs: list[_SequenceMatchedPair],
    ) -> np.ndarray | None:
        reference_stars = [
            pair.reference_star
            for pair in pairs
            if all(
                math.isfinite(value)
                for value in (
                    pair.reference_star.ra_deg,
                    pair.reference_star.dec_deg,
                    pair.reference_star.alt_deg,
                    pair.reference_star.az_deg,
                )
            )
        ]
        if len(reference_stars) < 3:
            return None

        world_vectors = radec_to_unit_vectors(
            np.asarray([star.ra_deg for star in reference_stars], dtype=np.float64),
            np.asarray([star.dec_deg for star in reference_stars], dtype=np.float64),
        )
        local_vectors = local_vectors_from_altaz(
            np.asarray([star.alt_deg for star in reference_stars], dtype=np.float64),
            np.asarray([star.az_deg for star in reference_stars], dtype=np.float64),
        )
        finite = np.all(np.isfinite(world_vectors), axis=1) & np.all(np.isfinite(local_vectors), axis=1)
        if np.count_nonzero(finite) < 3:
            return None

        try:
            local_from_world_transposed, _residuals, _rank, _singular_values = np.linalg.lstsq(
                world_vectors[finite],
                local_vectors[finite],
                rcond=None,
            )
            local_from_world = local_from_world_transposed.T
            u_matrix, _values, vt_matrix = np.linalg.svd(local_from_world)
        except np.linalg.LinAlgError:
            return None

        local_from_world = u_matrix @ vt_matrix
        if np.linalg.det(local_from_world) < 0.0:
            u_matrix[:, -1] *= -1.0
            local_from_world = u_matrix @ vt_matrix

        camera_from_local = np.vstack(camera_basis_from_view(self._view_settings())).astype(np.float64)
        rotation_matrix = camera_from_local @ local_from_world
        if rotation_matrix.shape != (3, 3) or not np.all(np.isfinite(rotation_matrix)):
            return None
        return rotation_matrix.astype(np.float64)

    def _sequence_pair_records(
        self,
        pairs: list[_SequenceMatchedPair],
        model: SourceAstrometricModel,
    ) -> list[dict[str, object]]:
        ra_dec_points = np.asarray(
            [(pair.reference_star.ra_deg, pair.reference_star.dec_deg) for pair in pairs],
            dtype=np.float64,
        )
        predicted_pixels = model.direction_to_pixel_points(ra_dec_points)
        records: list[dict[str, object]] = []
        for output_index, pair in enumerate(pairs, start=1):
            reference_star = pair.reference_star
            fitted = pair.fitted_position
            predicted_model = predicted_pixels[output_index - 1]
            residual_dx = float(predicted_model[0] - fitted.x)
            residual_dy = float(predicted_model[1] - fitted.y)
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
        observer = ObserverSettings(
            observation_time_utc=sequence_item_observation_time_utc(item, self.ui.doubleSpinBoxUtcOffset.value()),
            latitude_deg=self.ui.doubleSpinBoxLatitude.value(),
            longitude_deg=self.ui.doubleSpinBoxLongitude.value(),
            elevation_m=self.ui.doubleSpinBoxElevation.value(),
        )
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

    def _sequence_starpair_json_path(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_starpairs.json")

    def _sequence_model_json_path(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_model.json")

    def _write_sequence_outputs(
        self,
        item: ImageSequenceItem,
        preview: ImagePreview,
        pairs: list[_SequenceMatchedPair],
        model: SourceAstrometricModel,
    ) -> tuple[Path, Path]:
        image_path = Path(preview.path).expanduser().resolve()
        starpair_path = self._sequence_starpair_json_path(image_path)
        model_path = self._sequence_model_json_path(image_path)
        records = self._sequence_pair_records(pairs, model)
        reference_payload = self._sequence_reference_payload(item, preview, pairs)
        generated_at_utc = datetime.now(timezone.utc).isoformat()
        starpair_payload = {
            "format": STAR_PAIR_SESSION_FORMAT,
            "version": STAR_PAIR_SESSION_VERSION,
            "generated_at_utc": generated_at_utc,
            "real_image": self._sequence_real_image_payload(preview, starpair_path, item),
            "reference_payload": reference_payload,
            "sky_alignment_model": self._alignment_model(),
            "pair_count": len(records),
            "pairs": records,
            "mask": self._sky_mask_payload(starpair_path),
            "matching": self._sequence_matching_payload(),
        }
        source_payload = model.to_json_payload(
            source_image=self._sequence_source_image_payload(preview, model_path, item),
            fit_pairs=records,
            mask=self._sky_mask_payload(model_path),
            matching=self._sequence_matching_payload(),
            reference_payload=reference_payload,
        )

        starpair_path.write_text(json.dumps(starpair_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        model_path.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return starpair_path, model_path

    def process_image_sequence(self) -> None:
        if getattr(self, "_sequence_processing_active", False):
            QMessageBox.information(self, "正在处理序列", "图像序列仍在处理，请稍候。")
            return
        try:
            self._ensure_sequence_ready_for_processing()
            first_item = self._current_sequence_first_item()
            self._apply_sequence_observation_time(first_item, emit_signal=False)
            self.render_now()
            QApplication.processEvents()
            templates = self._sequence_base_templates()
            source_size = self._sequence_source_size()
            assert self.current_image_preview is not None
            target_size = (self.current_image_preview.image.width(), self.current_image_preview.image.height())
            sim_to_real = self._fit_sequence_sim_to_real_transform(templates, source_size, target_size)
        except Exception as exc:  # noqa: BLE001 - 批处理入口要把缺失条件直接反馈给用户。
            QMessageBox.warning(self, "无法处理图像序列", str(exc))
            self.ui.statusbar.showMessage(f"无法处理图像序列: {exc}")
            return

        items = list(getattr(self, "_image_sequence_items", []))
        progress = QProgressDialog(self)
        progress.setWindowTitle("正在处理图像序列")
        progress.setRange(0, len(items))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()

        processed: list[tuple[Path, Path]] = []
        failures: list[str] = []
        self._sequence_processing_active = True
        self._set_image_import_controls_enabled(False)
        self._update_image_sequence_controls()
        try:
            for frame_index, item in enumerate(items, start=1):
                if progress.wasCanceled():
                    failures.append("用户取消了后续处理。")
                    break
                progress.setValue(frame_index - 1)
                progress.setLabelText(
                    "正在处理 {index}/{count}\n{path}".format(
                        index=frame_index,
                        count=len(items),
                        path=item.path,
                    )
                )
                QApplication.processEvents()

                try:
                    if frame_index == 1:
                        assert self.current_image_preview is not None
                        preview = self.current_image_preview
                        pairs = self._first_frame_matched_pairs(templates)
                    else:
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
                        }
                        pairs = self._sequence_frame_matched_pairs(
                            item,
                            preview,
                            templates,
                            sim_to_real,
                            source_size,
                            target_size,
                            stats,
                        )
                    model = self._fit_sequence_source_model(pairs, target_size)
                    output_paths = self._write_sequence_outputs(
                        item,
                        preview,
                        pairs,
                        model,
                    )
                    processed.append(output_paths)
                    sim_to_real = self._fit_sequence_pairs_sim_to_real_transform(
                        pairs,
                        source_size,
                        target_size,
                    )
                except Exception as exc:  # noqa: BLE001 - 单帧失败不影响后续帧。
                    failures.append(f"{item.path.name}: {exc}")
                    continue
        finally:
            progress.setValue(len(items))
            progress.close()
            self._sequence_processing_active = False
            self._set_image_import_controls_enabled(True)
            self._update_image_sequence_controls()

        self.ui.statusbar.showMessage(
            f"序列处理完成：成功 {len(processed)} 张，失败 {len(failures)} 张。"
        )
        message = f"成功处理 {len(processed)} 张，失败 {len(failures)} 张。"
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
