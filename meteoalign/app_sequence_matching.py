from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from .alignment.constants import (
    MIN_ALIGNMENT_PAIRS,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .matching_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
    AUTO_MATCH_MIN_ALTITUDE_DEG,
    AUTO_MATCH_MIN_AMPLITUDE,
    AUTO_MATCH_SEARCH_MAG_LIMIT,
)
from .fixed_camera_model import (
    FixedCameraModel,
    FixedCameraTimeFitResult,
    estimate_frame_time_correction,
    fit_fixed_camera_model,
)
from .image_preview import ImagePreview
from .image_sequence import ImageSequenceItem, sequence_item_observation_time_utc
from .sequence_constants import (
    SEQUENCE_FILL_GRID_COLUMNS,
    SEQUENCE_FILL_GRID_ROWS,
    SEQUENCE_MIN_PAIR_FRACTION,
    SEQUENCE_SUPPLEMENTAL_FIT_WEIGHT,
    SEQUENCE_SUPPLEMENTAL_PAIR_ORIGIN,
)
from .sequence_types import (
    _SequenceCandidate,
    _SequenceFitPlan,
    _SequenceMatchedPair,
    _SequencePairTemplate,
)
from .simulator import (
    FISHEYE_EQUISOLID,
    ObserverSettings,
    ProjectedStarMap,
    ReferenceStar,
    camera_basis_from_view,
    local_vectors_from_altaz,
    project_horizontal_catalog,
)
from .star_fitting import fit_star_position

class SequenceMatchingMixin:
    """序列批处理中的星点模板、候选匹配和固定相机求解。"""

    def _sequence_base_templates(self) -> list[_SequencePairTemplate]:
        templates: list[_SequencePairTemplate] = []
        for record in self._star_pair_store.snapshot():
            if not record.is_valid_for_fit():
                continue
            reference_star = record.reference_star
            fitted_position = record.fitted_position
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
            templates.append(
                _SequencePairTemplate(
                    star_id=record.star_id,
                    reference_star=reference_star,
                    fitted_position=fitted_position,
                    fit_constraint_mode=record.fit_constraint_mode,
                    fit_weight=float(record.fit_weight),
                    pair_origin=record.pair_origin,
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
            star_color_mag_limit=self.ui_config.star_color_mag_limit,
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
