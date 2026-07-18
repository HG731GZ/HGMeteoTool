from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication, QGraphicsView, QMessageBox, QProgressDialog

from ..alignment.constants import MIN_ALIGNMENT_PAIRS, MIN_PRELIMINARY_ALIGNMENT_PAIRS
from ..alignment.models import SkyAlignmentTransform
from ..auto_match_quality import (
    AUTO_MATCH_ASSIGNMENT_COST_KEY,
    AUTO_MATCH_ASSIGNMENT_ROW_MARGIN_KEY,
    AUTO_MATCH_ASSIGNMENT_SCORE_KEY,
    AUTO_MATCH_ASSIGNMENT_SOURCE_MARGIN_KEY,
    AUTO_MATCH_GEOMETRY_SCORE_KEY,
    AUTO_MATCH_OBSERVED_FLUX_KEY,
    AUTO_MATCH_PSF_BRIGHTNESS_KEY,
    AUTO_MATCH_PHOTOMETRIC_RESIDUAL_MAG_KEY,
    AUTO_MATCH_PHOTOMETRIC_SCATTER_MAG_KEY,
    AUTO_MATCH_PHOTOMETRIC_SCORE_KEY,
    AUTO_MATCH_PHOTOMETRIC_ZERO_POINT_MAG_KEY,
    AUTO_MATCH_POSITION_SCORE_KEY,
    AUTO_MATCH_PREDICTION_OFFSET_PX_KEY,
    AUTO_MATCH_PREDICTION_OFFSET_RATIO_KEY,
    AUTO_MATCH_QUALITY_SCORE_KEY,
    AutoMatchPhotometricModel,
    AutoMatchPhotometrySample,
    auto_match_position_quality,
    combine_auto_match_quality,
    evaluate_auto_match_photometry,
    fit_auto_match_photometric_model,
    psf_brightness_proxy,
)
from ..matching_constants import (
    AUTO_MATCH_ANNOTATION_LIMIT,
    AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
    AUTO_MATCH_FILL_GRID_COLUMNS,
    AUTO_MATCH_FILL_GRID_ROWS,
    AUTO_MATCH_MIN_ALTITUDE_DEG,
    AUTO_MATCH_MIN_AMPLITUDE,
    AUTO_MATCH_SEARCH_MAG_LIMIT,
    MIN_PSF_RADIUS_PX,
    REFERENCE_STAR_PICK_SCREEN_RADIUS_PX,
)
from ..simulator import ProjectedStarMap, ReferenceStar
from ..psf.matching import (
    SourceAssignmentDiagnostic,
    assign_predicted_sources,
    merge_source_candidates,
)
from ..psf.models import StarSourceCandidate
from ..star_fitting import FittedStarPosition, detect_star_candidates, fit_star_position


@dataclass(frozen=True)
class _AutoExpansionFit:
    """尚未写入匹配表的自动扩展拟合结果。"""

    row: int
    reference_star: ReferenceStar
    fitted_position: FittedStarPosition
    assigned_source: StarSourceCandidate
    predicted_x: float
    predicted_y: float
    assignment_diagnostic: SourceAssignmentDiagnostic


class AutoMatchMixin:
    """自动匹配 Mixin：参考星图点选、单星自动匹配、场星批量匹配。"""

    ui: object
    _sky_alignment_transform: SkyAlignmentTransform | None
    _current_star_map: ProjectedStarMap | None
    _current_reference_star_map: ProjectedStarMap | None
    _current_reference_stars: tuple
    _auto_match_reference_star_ids: list
    _auto_match_group_order: list
    _auto_match_group_by_star_id: dict
    _auto_match_group_expanded_by_id: dict
    _auto_match_next_group_index: int
    _excluded_reference_star_ids: list
    _mask_excluded_reference_star_ids: set
    _manual_reference_star_ids: list
    _active_star_pair_row: int | None
    _star_pick_circle_diameter_px: int
    current_image_preview: object | None
    current_sky_mask: np.ndarray | None
    reference_star_map_item: object
    ui_config: object
    _sky_alignment_error_message: object
    _reference_alignment_error_message: object
    _build_aligned_reference_star_map: object  # method
    _reference_selection_star_map: object  # method
    _focus_star_pair_image_point: object  # method
    _enter_star_pick_mode: object  # method
    _star_pair_position_count: object  # method

    def _current_psf_image(self) -> object:
        """按精度选项返回 8-bit 显示图或原始位深亮度图。"""

        preview = self.current_image_preview
        if preview is None:
            raise ValueError("请先导入真实图像。")
        ui_config = getattr(self, "ui_config", None)
        if bool(getattr(ui_config, "use_8bit_psf_precision", True)):
            return preview.image
        native_luminance = getattr(preview, "native_luminance", None)
        return native_luminance if native_luminance is not None else preview.image

    def _scene_radius_from_screen_radius(
        self,
        view: QGraphicsView,
        viewport_pos: QPoint,
        screen_radius_px: int,
    ) -> float:
        scene_center = view.mapToScene(viewport_pos)
        scene_edge = view.mapToScene(viewport_pos + QPoint(max(1, screen_radius_px), 0))
        return max(1.0, float(np.hypot(scene_edge.x() - scene_center.x(), scene_edge.y() - scene_center.y())))

    def _reference_pick_star_positions(self) -> list[tuple[str, str, float, float, float, float]]:
        star_map = self._reference_selection_star_map()
        if star_map is None:
            return []

        transform = self._sky_alignment_transform if self.current_image_preview is not None else None
        positions: list[tuple[str, str, float, float, float, float]] = []
        if len(star_map) > 0:
            if transform is None:
                star_points = np.column_stack((star_map.x_px, star_map.y_px))
            else:
                star_points = transform.transform_radec_points(np.column_stack((star_map.ra_deg, star_map.dec_deg)))
            for star_index in range(len(star_map)):
                x_value = float(star_points[star_index, 0])
                y_value = float(star_points[star_index, 1])
                if not np.isfinite(x_value) or not np.isfinite(y_value):
                    continue
                reference_star = self._reference_star_from_star_map_index(star_map, star_index, output_index=0)
                positions.append(
                    (
                        reference_star.star_id,
                        reference_star.name,
                        x_value,
                        y_value,
                        float(star_map.mag_v[star_index]),
                        max(1.0, float(star_map.radius_px[star_index]) * self._current_reference_element_scale()),
                    )
                )

        return positions

    def _current_reference_element_scale(self) -> float:
        if self._sky_alignment_transform is None or self.current_image_preview is None:
            base_scale = 1.0
        else:
            image = self.current_image_preview.image
            base_scale = self._aligned_star_element_scale((image.width(), image.height()))
        return base_scale * self.reference_star_map_item.star_radius_zoom_scale()

    def _nearest_reference_pick_star(
        self,
        viewport_pos: QPoint,
    ) -> tuple[str, str, float] | None:
        positions = self._reference_pick_star_positions()
        if not positions:
            return None

        scene_pos = self.ui.referenceImageView.mapToScene(viewport_pos)
        click_x = float(scene_pos.x())
        click_y = float(scene_pos.y())
        scene_radius = self._scene_radius_from_screen_radius(
            self.ui.referenceImageView,
            viewport_pos,
            REFERENCE_STAR_PICK_SCREEN_RADIUS_PX,
        )
        mag_values = np.asarray([position[4] for position in positions], dtype=np.float64)
        brightest_mag = float(np.nanmin(mag_values))
        faintest_mag = float(np.nanmax(mag_values))
        mag_span = max(faintest_mag - brightest_mag, 1e-6)

        candidates: list[tuple[float, float, float, str, str]] = []
        for star_id, name, x_value, y_value, mag_v, radius_px in positions:
            distance = float(np.hypot(x_value - click_x, y_value - click_y))
            search_radius = scene_radius + max(radius_px * 1.5, 2.0)
            if distance > search_radius:
                continue
            brightness_rank = (mag_v - brightest_mag) / mag_span
            score = distance / max(search_radius, 1.0) + brightness_rank * 0.22
            candidates.append((score, distance, mag_v, star_id, name))

        if not candidates:
            return None
        _score, distance, _mag_v, star_id, name = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return star_id, name, distance

    def _handle_reference_map_click(self, viewport_pos: QPoint) -> None:
        picked = self._nearest_reference_pick_star(viewport_pos)
        if picked is None:
            self.ui.statusbar.showMessage("参考星图点击位置附近没有可用亮星，请稍微靠近星点再试。")
            return

        star_id, star_name, _distance_px = picked
        if star_id in self._excluded_reference_star_ids:
            self._excluded_reference_star_ids.remove(star_id)
        existing_row = self._select_star_pair_row_by_id(star_id)
        if existing_row is not None:
            self._show_star_pair_assistant()
            self._enter_star_pick_mode(existing_row)
            return

        if star_id not in self._manual_reference_star_ids:
            self._manual_reference_star_ids.append(star_id)
        self._refresh_reference_stars_from_current_map()
        row = self._select_star_pair_row_by_id(star_id)
        if row is None:
            self.ui.statusbar.showMessage(f"未能添加参考星 {star_name or star_id}，请检查当前星等上限和视野。")
            self._show_star_pair_assistant()
            return
        self._show_star_pair_assistant()
        self._enter_star_pick_mode(row)

    def _handle_real_image_pick_click(self, viewport_pos: QPoint) -> None:
        if self._active_star_pair_row is None or self.current_image_preview is None:
            return

        image = self.current_image_preview.image
        scene_pos = self.ui.realImageView.mapToScene(viewport_pos)
        image_x = float(scene_pos.x())
        image_y = float(scene_pos.y())
        if not (0.0 <= image_x < image.width() and 0.0 <= image_y < image.height()):
            self.ui.statusbar.showMessage("点击位置不在真实图像范围内，请重新点选。")
            return
        if not self._sky_mask_allows_point(image_x, image_y):
            self.ui.statusbar.showMessage(
                "手动匹配失败：点击位置位于天空蒙版外，已拒绝把地景纹理作为星点。"
            )
            return

        search_radius_px = self._star_pick_search_radius_px(viewport_pos)
        max_fit_radius_px = self._star_pick_psf_radius_px(viewport_pos)
        try:
            fitted_position = fit_star_position(
                AutoMatchMixin._current_psf_image(self),
                click_x=image_x,
                click_y=image_y,
                radius_px=search_radius_px,
                max_fit_radius_px=max_fit_radius_px,
                selection_mode="manual",
                fit_error_limit=self.ui_config.star_pick_psf_fit_error_limit,
                saturated_fit_error_limit=(
                    self.ui_config.star_pick_saturated_psf_fit_error_limit
                ),
            )
        except Exception as exc:  # noqa: BLE001 - 交互式点选需要把拟合失败原因直接反馈给用户。
            self.ui.statusbar.showMessage(f"手动匹配失败：PSF 拟合失败：{exc}")
            return
        if not self._sky_mask_allows_point(fitted_position.x, fitted_position.y):
            self.ui.statusbar.showMessage(
                "手动匹配失败：检测到的 PSF 中心位于天空蒙版外，结果已拒绝。"
            )
            return

        row = self._active_star_pair_row
        star_name = self._star_pair_name(row)
        self._set_star_pair_position(row, fitted_position)
        self._add_or_update_star_pair_annotation(row, fitted_position)
        self._leave_star_pick_mode()
        self.ui.statusbar.showMessage(
            "已记录 {name} 的图像坐标: x={x:.2f}, y={y:.2f}；搜索半径 {search_radius} px，"
            "PSF FWHM=({fwhm_x:.2f}, {fwhm_y:.2f}) px，质量={quality:.2f}{saturated}。"
            "右键匹配表行可继续点选。".format(
                name=star_name,
                x=fitted_position.x,
                y=fitted_position.y,
                search_radius=search_radius_px,
                fwhm_x=fitted_position.fwhm_x,
                fwhm_y=fitted_position.fwhm_y,
                quality=fitted_position.quality_score,
                saturated="，饱和兼容拟合" if fitted_position.saturated else "",
            )
        )

    def _auto_pair_search_radius_px(self, transform: SkyAlignmentTransform) -> int:
        if self.current_image_preview is None:
            return self.ui_config.star_pick_psf_max_radius_px
        image = self.current_image_preview.image
        min_dimension = min(image.width(), image.height())
        radius = max(
            MIN_PSF_RADIUS_PX,
            int(
                round(
                    transform.rms_px * self.ui_config.auto_pair_search_rms_multiplier
                    + self.ui_config.auto_pair_search_base_radius_px
                )
            ),
        )
        return min(
            radius,
            self.ui_config.auto_pair_search_max_radius_px,
            max(MIN_PSF_RADIUS_PX, min_dimension // 4),
        )

    def _report_auto_pair_failure(self, message: str) -> bool:
        """在状态栏静默报告单星自动匹配失败。"""

        self.ui.statusbar.showMessage(f"自动匹配失败：{message}")
        return False

    def _auto_pair_star(self, row: int) -> bool:
        if self.current_image_preview is None:
            return self._report_auto_pair_failure("请先导入真实图像，再自动匹配星点。")

        transform = self._sky_alignment_transform
        if transform is None:
            self._update_reference_alignment_transform()
            transform = self._sky_alignment_transform
        if transform is None:
            return self._report_auto_pair_failure(
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or f"至少需要 {MIN_PRELIMINARY_ALIGNMENT_PAIRS} 对已配准参考星。",
            )

        reference_star = self._reference_star_for_row(row)
        if reference_star is None:
            return self._report_auto_pair_failure("当前行没有对应的参考星。")

        predicted_x, predicted_y = transform.transform_radec(reference_star.ra_deg, reference_star.dec_deg)
        image = self.current_image_preview.image
        if not (0.0 <= predicted_x < image.width() and 0.0 <= predicted_y < image.height()):
            return self._report_auto_pair_failure(
                f"{self._star_pair_label(row)} 的预测位置在真实图像外。"
            )

        search_radius_px = self._auto_pair_search_radius_px(transform)
        self._focus_star_pair_image_point(row, predicted_x, predicted_y, search_radius_px)
        self.ui.statusbar.showMessage(
            f"已跳转到 {self._star_pair_label(row)} 的自动匹配预测位置，正在搜索真实星点..."
        )
        QApplication.processEvents()

        try:
            fitted_position = fit_star_position(
                AutoMatchMixin._current_psf_image(self),
                click_x=predicted_x,
                click_y=predicted_y,
                radius_px=search_radius_px,
                max_fit_radius_px=self.ui_config.star_pick_psf_max_radius_px,
                reject_ambiguous=True,
                selection_mode="predicted",
                fit_error_limit=self.ui_config.star_pick_psf_fit_error_limit,
                saturated_fit_error_limit=(
                    self.ui_config.star_pick_saturated_psf_fit_error_limit
                ),
            )
        except Exception as exc:  # noqa: BLE001 - 自动匹配要把失败原因反馈给用户。
            return self._report_auto_pair_failure(str(exc))

        distance_px = ((fitted_position.x - predicted_x) ** 2 + (fitted_position.y - predicted_y) ** 2) ** 0.5
        self._set_star_pair_position(row, fitted_position)
        self._add_or_update_star_pair_annotation(
            row,
            fitted_position,
            preserve_focus_annotation=True,
        )
        self.ui.tableWidgetStarPairs.selectRow(row)
        self.ui.statusbar.showMessage(
            "{label} 自动匹配完成: x={x:.2f}, y={y:.2f}；预测偏差 {distance:.2f} px，搜索半径 {radius} px。".format(
                label=self._star_pair_label(row),
                x=fitted_position.x,
                y=fitted_position.y,
                distance=distance_px,
                radius=search_radius_px,
            )
        )
        return True

    def _auto_match_required_mag_limit(self) -> float:
        return max(
            self.ui.doubleSpinBoxMagLimit.value(),
            AUTO_MATCH_SEARCH_MAG_LIMIT,
        )

    def _ensure_current_star_map_for_auto_match(self, mag_limit: float) -> None:
        if self.ui.doubleSpinBoxMagLimit.value() + 1e-6 < mag_limit:
            was_blocked = self.ui.doubleSpinBoxMagLimit.blockSignals(True)
            self.ui.doubleSpinBoxMagLimit.setValue(min(mag_limit, self.ui.doubleSpinBoxMagLimit.maximum()))
            self.ui.doubleSpinBoxMagLimit.blockSignals(was_blocked)
            self.render_now()
        elif self._current_star_map is None:
            self.render_now()

    def _auto_match_reference_star_map(
        self,
        transform: SkyAlignmentTransform,
        mag_limit: float,
    ) -> ProjectedStarMap | None:
        if self.current_image_preview is None:
            return None
        image = self.current_image_preview.image
        star_map = self._build_aligned_reference_star_map(
            transform,
            (image.width(), image.height()),
            visible_mag_limit=mag_limit,
        )
        self._current_reference_star_map = star_map
        return star_map

    def _auto_match_candidate_stars(
        self,
        transform: SkyAlignmentTransform,
    ) -> tuple[list[ReferenceStar], dict[str, tuple[float, float]], int]:
        if self.current_image_preview is None:
            return [], {}, 0

        mag_limit = self._auto_match_required_mag_limit()
        star_map = self._auto_match_reference_star_map(transform, mag_limit)
        if star_map is None or len(star_map) <= 0:
            return [], {}, 0

        image = self.current_image_preview.image
        ra_dec_points = np.column_stack((star_map.ra_deg, star_map.dec_deg))
        predicted = transform.transform_radec_points(ra_dec_points)
        finite = np.all(np.isfinite(predicted), axis=1)
        inside = (
            finite
            & (predicted[:, 0] >= 0.0)
            & (predicted[:, 0] < image.width())
            & (predicted[:, 1] >= 0.0)
            & (predicted[:, 1] < image.height())
            & (star_map.alt_deg >= AUTO_MATCH_MIN_ALTITUDE_DEG)
        )

        mask_prefiltered_star_ids: set[str] = set()
        if self.current_sky_mask is not None:
            mask_allowed = np.zeros(len(star_map), dtype=bool)
            for index in np.where(inside)[0]:
                point_allowed = self._sky_mask_allows_point(float(predicted[index, 0]), float(predicted[index, 1]))
                mask_allowed[index] = point_allowed
                if not point_allowed:
                    star_id = str(star_map.star_ids[index]).strip()
                    if star_id:
                        mask_prefiltered_star_ids.add(star_id)
                        self._mask_excluded_reference_star_ids.add(star_id)
            inside &= mask_allowed

        candidate_indices = np.where(inside)[0]
        if candidate_indices.size <= 0:
            return [], {}, len(mask_prefiltered_star_ids)

        order = np.argsort(star_map.mag_v[candidate_indices], kind="stable")
        candidate_indices = candidate_indices[order]
        blocked_star_ids = self._auto_match_blocked_reference_star_ids()
        new_star_limit = max(1, int(self.ui.spinBoxAutoMatchCount.value()))

        candidates: list[ReferenceStar] = []
        predicted_by_id: dict[str, tuple[float, float]] = {}
        seen_star_ids: set[str] = set()
        for star_index in candidate_indices:
            reference_star = self._reference_star_from_star_map_index(star_map, int(star_index), output_index=0)
            star_id = reference_star.star_id.strip()
            if not star_id or star_id in seen_star_ids or star_id in blocked_star_ids:
                continue
            seen_star_ids.add(star_id)
            candidates.append(reference_star)
            predicted_by_id[star_id] = (float(predicted[star_index, 0]), float(predicted[star_index, 1]))

        candidates = self._auto_match_order_spatial_candidates(
            candidates,
            predicted_by_id=predicted_by_id,
            accepted_positions=self._existing_matched_positions(),
            target_size=(image.width(), image.height()),
        )[:new_star_limit]
        selected_predictions = {
            reference_star.star_id.strip(): predicted_by_id[reference_star.star_id.strip()]
            for reference_star in candidates
        }
        return candidates, selected_predictions, len(mask_prefiltered_star_ids)

    def _auto_match_spatial_cell_index(
        self,
        x_px: float,
        y_px: float,
        target_size: tuple[int, int],
    ) -> int:
        """返回图像坐标所在的自动匹配均衡网格编号。"""

        width_px = max(float(target_size[0]), 1.0)
        height_px = max(float(target_size[1]), 1.0)
        column = int(
            np.clip(
                math.floor(float(x_px) / width_px * AUTO_MATCH_FILL_GRID_COLUMNS),
                0,
                AUTO_MATCH_FILL_GRID_COLUMNS - 1,
            )
        )
        row = int(
            np.clip(
                math.floor(float(y_px) / height_px * AUTO_MATCH_FILL_GRID_ROWS),
                0,
                AUTO_MATCH_FILL_GRID_ROWS - 1,
            )
        )
        return row * AUTO_MATCH_FILL_GRID_COLUMNS + column

    def _auto_match_spatial_cell_counts(
        self,
        positions: list[tuple[float, float]],
        target_size: tuple[int, int],
    ) -> dict[int, int]:
        """统计已有匹配点在各均衡网格中的数量。"""

        counts: dict[int, int] = {}
        for x_px, y_px in positions:
            if not math.isfinite(x_px) or not math.isfinite(y_px):
                continue
            cell_index = self._auto_match_spatial_cell_index(x_px, y_px, target_size)
            counts[cell_index] = counts.get(cell_index, 0) + 1
        return counts

    def _auto_match_order_spatial_candidates(
        self,
        candidates: list[ReferenceStar],
        *,
        predicted_by_id: dict[str, tuple[float, float]],
        accepted_positions: list[tuple[float, float]],
        target_size: tuple[int, int],
    ) -> list[ReferenceStar]:
        """按网格稀疏程度轮询候选，同一网格内优先匹配亮星。"""

        counts = self._auto_match_spatial_cell_counts(accepted_positions, target_size)
        grouped: dict[int, list[ReferenceStar]] = {}
        for candidate in candidates:
            star_id = candidate.star_id.strip()
            predicted_position = predicted_by_id.get(star_id)
            if not star_id or predicted_position is None:
                continue
            predicted_x, predicted_y = predicted_position
            if not math.isfinite(predicted_x) or not math.isfinite(predicted_y):
                continue
            cell_index = self._auto_match_spatial_cell_index(predicted_x, predicted_y, target_size)
            grouped.setdefault(cell_index, []).append(candidate)

        for cell_candidates in grouped.values():
            cell_candidates.sort(
                key=lambda candidate: (
                    float(candidate.mag_v),
                    float(predicted_by_id[candidate.star_id.strip()][0]),
                    float(predicted_by_id[candidate.star_id.strip()][1]),
                )
            )

        ordered: list[ReferenceStar] = []
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

    def _auto_match_blocked_reference_star_ids(self) -> set[str]:
        """返回当前不能再次进入自动匹配候选池的星号。"""

        existing_star_ids = {
            self._star_pair_star_id(row)
            for row in range(self.ui.tableWidgetStarPairs.rowCount())
            if self._star_pair_star_id(row)
        }
        return (
            existing_star_ids
            | set(self._auto_match_reference_star_ids)
            | set(self._mask_excluded_reference_star_ids)
        )

    def _ensure_auto_match_candidates_visible(self, candidates: list[ReferenceStar], group_id: str) -> set[str]:
        visible_star_ids = {
            self._star_pair_star_id(row)
            for row in range(self.ui.tableWidgetStarPairs.rowCount())
            if self._star_pair_star_id(row)
        }
        candidate_auto_star_ids: set[str] = set()
        added_any = False
        self._ensure_auto_match_group(group_id, expanded=True)
        constraint_mode = self._auto_match_constraint_mode()
        fit_weight = self._auto_match_soft_weight()
        for reference_star in candidates:
            star_id = reference_star.star_id.strip()
            if (
                not star_id
                or star_id in visible_star_ids
                or star_id in self._auto_match_reference_star_ids
                or star_id in self._mask_excluded_reference_star_ids
            ):
                continue
            if star_id in self._excluded_reference_star_ids:
                self._excluded_reference_star_ids.remove(star_id)
            self._auto_match_reference_star_ids.append(star_id)
            self._auto_match_group_by_star_id[star_id] = group_id

            store = getattr(self, "_star_pair_store", None)
            if store is not None and star_id in store:
                store.set_constraint(star_id, constraint_mode, fit_weight)

            candidate_auto_star_ids.add(star_id)
            visible_star_ids.add(star_id)
            added_any = True
        if added_any:
            self._refresh_reference_stars_from_current_map()
        return candidate_auto_star_ids

    def _remove_unmatched_auto_match_candidates(self, candidate_star_ids: set[str]) -> int:
        if not candidate_star_ids:
            return 0
        matched_positions = self._star_pair_position_texts_from_store()
        remove_star_ids = {
            star_id
            for star_id in candidate_star_ids
            if star_id in self._auto_match_reference_star_ids and star_id not in matched_positions
        }
        if not remove_star_ids:
            return 0

        self._auto_match_reference_star_ids = [
            star_id for star_id in self._auto_match_reference_star_ids if star_id not in remove_star_ids
        ]
        for star_id in remove_star_ids:
            self._auto_match_group_by_star_id.pop(star_id, None)
            if star_id not in self._excluded_reference_star_ids:
                self._excluded_reference_star_ids.append(star_id)
        self._normalize_auto_match_groups()
        self._refresh_reference_stars_from_current_map()
        return len(remove_star_ids)

    def _existing_matched_positions(self) -> list[tuple[float, float]]:
        return [record.position for record in self._star_pair_store.snapshot()]

    def _existing_auto_match_photometry_samples(self) -> list[AutoMatchPhotometrySample]:
        """读取既有自动扩展测光样本；手动匹配不进入亮度标定。"""

        store = getattr(self, "_star_pair_store", None)
        if store is None:
            return []
        samples: list[AutoMatchPhotometrySample] = []
        for record in store.auto_match_records():
            if record.psf is None:
                continue
            flux = psf_brightness_proxy(
                record.psf.amplitude,
                record.psf.sigma_x,
                record.psf.sigma_y,
            )
            sample = AutoMatchPhotometrySample(
                star_id=record.star_id,
                catalog_mag=float(record.reference_star.mag_v),
                flux=flux,
                x_px=float(record.image_x_px),
                y_px=float(record.image_y_px),
            )
            if all(
                math.isfinite(value)
                for value in (sample.catalog_mag, sample.flux, sample.x_px, sample.y_px)
            ) and sample.flux > 0.0:
                samples.append(sample)
        return samples

    def _auto_expansion_quality_fields(
        self,
        provisional: _AutoExpansionFit,
        *,
        search_radius_px: float,
        photometric_model: AutoMatchPhotometricModel | None,
    ) -> tuple[dict[str, object], bool]:
        """计算自动扩展综合质量字段，并返回是否应按亮度异常拒绝。"""

        fitted = provisional.fitted_position
        source = provisional.assigned_source
        diagnostic = provisional.assignment_diagnostic
        prediction_offset_px = float(
            np.hypot(fitted.x - provisional.predicted_x, fitted.y - provisional.predicted_y)
        )
        position_score = auto_match_position_quality(prediction_offset_px, search_radius_px)
        sample = AutoMatchPhotometrySample(
            star_id=provisional.reference_star.star_id.strip(),
            catalog_mag=float(provisional.reference_star.mag_v),
            flux=psf_brightness_proxy(fitted.amplitude, fitted.sigma_x, fitted.sigma_y),
            x_px=float(fitted.x),
            y_px=float(fitted.y),
        )
        photometric_evaluation = None
        if photometric_model is not None:
            photometric_evaluation = evaluate_auto_match_photometry(
                sample,
                photometric_model,
                saturated=bool(fitted.saturated or source.saturated),
            )
        photometric_score = (
            None
            if photometric_evaluation is None
            else photometric_evaluation.quality_score
        )
        quality_score, geometry_score = combine_auto_match_quality(
            position_score,
            diagnostic.confidence_score,
            photometric_score,
        )
        radius = max(float(search_radius_px), 1.0)
        fields: dict[str, object] = {
            AUTO_MATCH_QUALITY_SCORE_KEY: quality_score,
            AUTO_MATCH_GEOMETRY_SCORE_KEY: geometry_score,
            AUTO_MATCH_POSITION_SCORE_KEY: position_score,
            AUTO_MATCH_ASSIGNMENT_SCORE_KEY: float(diagnostic.confidence_score),
            AUTO_MATCH_PREDICTION_OFFSET_PX_KEY: prediction_offset_px,
            AUTO_MATCH_PREDICTION_OFFSET_RATIO_KEY: prediction_offset_px / radius,
            AUTO_MATCH_ASSIGNMENT_COST_KEY: float(diagnostic.assigned_cost),
            AUTO_MATCH_OBSERVED_FLUX_KEY: float(source.flux),
            AUTO_MATCH_PSF_BRIGHTNESS_KEY: float(sample.flux),
        }
        if math.isfinite(float(diagnostic.row_margin)):
            fields[AUTO_MATCH_ASSIGNMENT_ROW_MARGIN_KEY] = float(diagnostic.row_margin)
        if math.isfinite(float(diagnostic.source_margin)):
            fields[AUTO_MATCH_ASSIGNMENT_SOURCE_MARGIN_KEY] = float(diagnostic.source_margin)
        if photometric_evaluation is not None:
            fields[AUTO_MATCH_PHOTOMETRIC_RESIDUAL_MAG_KEY] = float(
                photometric_evaluation.residual_mag
            )
            if photometric_evaluation.quality_score is not None:
                fields[AUTO_MATCH_PHOTOMETRIC_SCORE_KEY] = float(
                    photometric_evaluation.quality_score
                )
        if photometric_model is not None:
            fields[AUTO_MATCH_PHOTOMETRIC_ZERO_POINT_MAG_KEY] = float(
                photometric_model.zero_point_at(sample.catalog_mag, fitted.x, fitted.y)
            )
            fields[AUTO_MATCH_PHOTOMETRIC_SCATTER_MAG_KEY] = float(
                photometric_model.scatter_mag
            )
        reject = bool(photometric_evaluation is not None and photometric_evaluation.reject)
        return fields, reject

    def _position_is_duplicate(
        self,
        position: tuple[float, float],
        accepted_positions: list[tuple[float, float]],
        minimum_distance_px: float = AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
    ) -> bool:
        for accepted_x, accepted_y in accepted_positions:
            if float(np.hypot(position[0] - accepted_x, position[1] - accepted_y)) < float(minimum_distance_px):
                return True
        return False

    def auto_match_field_stars(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再自动扩展匹配。")
            return
        matched_count = int(self._star_pair_position_count())
        if matched_count < MIN_ALIGNMENT_PAIRS:
            QMessageBox.information(
                self,
                "无法自动扩展匹配",
                f"当前只有 {matched_count} 对星点；自动扩展匹配至少需要 {MIN_ALIGNMENT_PAIRS} 对。",
            )
            return

        transform = self._sky_alignment_transform
        if transform is None:
            self._update_reference_alignment_transform()
            transform = self._sky_alignment_transform
        if transform is None:
            QMessageBox.information(
                self,
                "无法自动扩展匹配",
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or f"至少需要 {MIN_ALIGNMENT_PAIRS} 对已配准参考星。",
            )
            return

        candidates, predicted_by_id, mask_prefiltered_count = self._auto_match_candidate_stars(transform)
        if not candidates:
            if self.current_sky_mask is None:
                mask_status = "蒙版未启用"
            else:
                mask_status = f"蒙版预筛 {mask_prefiltered_count} 个视场星点"
            self.ui.statusbar.showMessage(f"自动扩展匹配：没有可新增星；{mask_status}。")
            QMessageBox.information(
                self,
                "没有可新增星",
                f"当前视场、数量设置和蒙版下没有新的可匹配参考星。\n{mask_status}。",
            )
            return

        auto_group_id = self._create_auto_match_group()
        candidate_star_ids = self._ensure_auto_match_candidates_visible(candidates, auto_group_id)
        image = self.current_image_preview.image
        search_radius_px = self.ui.spinBoxAutoMatchRadius.value()
        annotate_matches = len(candidates) <= AUTO_MATCH_ANNOTATION_LIMIT
        accepted_positions = self._existing_matched_positions()
        matched_count = 0
        skipped_existing = 0
        skipped_mask = 0
        skipped_duplicate = 0
        skipped_ambiguous = 0
        skipped_photometric = 0
        failed_count = 0
        canceled = False
        provisional_matches: list[_AutoExpansionFit] = []
        progress = QProgressDialog(self)
        progress.setWindowTitle("正在自动扩展匹配")
        progress.setLabelText(f"正在检测 {len(candidates)} 颗新增星附近的图像星源...")
        progress.setRange(0, len(candidates) * 2)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()

        detected_sources = []
        for candidate_index, reference_star in enumerate(candidates, start=1):
            if progress.wasCanceled():
                canceled = True
                break
            if candidate_index == 1 or candidate_index % 10 == 0:
                progress.setValue(candidate_index - 1)
                QApplication.processEvents()
            star_id = reference_star.star_id.strip()
            predicted_position = predicted_by_id.get(star_id)
            if predicted_position is None:
                continue
            predicted_x, predicted_y = predicted_position
            if not self._sky_mask_allows_point(predicted_x, predicted_y):
                self._mask_excluded_reference_star_ids.add(star_id)
                skipped_mask += 1
                continue
            try:
                nearby_sources = detect_star_candidates(
                    AutoMatchMixin._current_psf_image(self),
                    click_x=predicted_x,
                    click_y=predicted_y,
                    radius_px=search_radius_px,
                )
            except Exception:
                continue
            detected_sources.extend(
                source for source in nearby_sources if self._sky_mask_allows_point(source.x, source.y)
            )

        unique_sources = merge_source_candidates(detected_sources)
        assignment = assign_predicted_sources(
            {
                star_id: position
                for star_id, position in predicted_by_id.items()
                if self._sky_mask_allows_point(position[0], position[1])
            },
            unique_sources,
            search_radius_px=float(search_radius_px),
            strict_mutual=True,
        )
        skipped_ambiguous = len(assignment.ambiguous_star_ids)
        progress.setLabelText(f"正在对 {len(assignment.assignments)} 个一对一星源做 PSF 拟合...")
        QApplication.processEvents()

        table = self.ui.tableWidgetStarPairs
        signals_were_blocked = table.blockSignals(True)
        try:
            for candidate_index, reference_star in enumerate(candidates, start=1):
                if progress.wasCanceled():
                    canceled = True
                    break
                if candidate_index == 1 or candidate_index % 10 == 0:
                    progress.setValue(len(candidates) + candidate_index - 1)
                    QApplication.processEvents()

                star_id = reference_star.star_id.strip()
                row = self._row_for_star_id(star_id)
                if row is None:
                    failed_count += 1
                    continue
                if self._star_pair_position_text(row):
                    skipped_existing += 1
                    continue

                predicted_position = predicted_by_id.get(star_id)
                assigned_source = assignment.assignments.get(star_id)
                if predicted_position is None or assigned_source is None:
                    if star_id not in assignment.ambiguous_star_ids and star_id not in self._mask_excluded_reference_star_ids:
                        failed_count += 1
                    continue
                predicted_x, predicted_y = predicted_position
                if not self._sky_mask_allows_point(predicted_x, predicted_y):
                    self._mask_excluded_reference_star_ids.add(star_id)
                    skipped_mask += 1
                    continue

                try:
                    source_fit_search_radius = max(
                        MIN_PSF_RADIUS_PX,
                        min(
                            self.ui_config.star_pick_psf_max_radius_px,
                            int(math.ceil(assigned_source.major_axis * 2.8)),
                        ),
                    )
                    fitted_position = fit_star_position(
                        AutoMatchMixin._current_psf_image(self),
                        click_x=assigned_source.x,
                        click_y=assigned_source.y,
                        radius_px=source_fit_search_radius,
                        max_fit_radius_px=self.ui_config.star_pick_psf_max_radius_px,
                        selection_mode="manual",
                        fit_error_limit=self.ui_config.star_pick_psf_fit_error_limit,
                        saturated_fit_error_limit=(
                            self.ui_config.star_pick_saturated_psf_fit_error_limit
                        ),
                    )
                except Exception:
                    failed_count += 1
                    continue

                distance_px = float(np.hypot(fitted_position.x - predicted_x, fitted_position.y - predicted_y))
                if distance_px > float(search_radius_px) or fitted_position.amplitude < AUTO_MATCH_MIN_AMPLITUDE:
                    failed_count += 1
                    continue
                if not self._sky_mask_allows_point(fitted_position.x, fitted_position.y):
                    self._mask_excluded_reference_star_ids.add(star_id)
                    skipped_mask += 1
                    continue
                diagnostic = assignment.diagnostics.get(star_id)
                if diagnostic is None:
                    failed_count += 1
                    continue
                provisional_matches.append(
                    _AutoExpansionFit(
                        row=row,
                        reference_star=reference_star,
                        fitted_position=fitted_position,
                        assigned_source=assigned_source,
                        predicted_x=float(predicted_x),
                        predicted_y=float(predicted_y),
                        assignment_diagnostic=diagnostic,
                    )
                )

            image_size = (image.width(), image.height())
            photometry_samples = self._existing_auto_match_photometry_samples()
            photometry_samples.extend(
                AutoMatchPhotometrySample(
                    star_id=provisional.reference_star.star_id.strip(),
                    catalog_mag=float(provisional.reference_star.mag_v),
                    flux=psf_brightness_proxy(
                        provisional.fitted_position.amplitude,
                        provisional.fitted_position.sigma_x,
                        provisional.fitted_position.sigma_y,
                    ),
                    x_px=float(provisional.fitted_position.x),
                    y_px=float(provisional.fitted_position.y),
                )
                for provisional in provisional_matches
            )
            photometric_model = fit_auto_match_photometric_model(
                photometry_samples,
                image_size,
            )

            for provisional in provisional_matches:
                quality_fields, photometric_rejected = self._auto_expansion_quality_fields(
                    provisional,
                    search_radius_px=float(search_radius_px),
                    photometric_model=photometric_model,
                )
                if photometric_rejected:
                    skipped_photometric += 1
                    continue
                fitted = provisional.fitted_position
                fitted_xy = (float(fitted.x), float(fitted.y))
                duplicate_radius = max(
                    AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
                    min(24.0, max(fitted.fwhm_x, fitted.fwhm_y) * 0.65),
                )
                if self._position_is_duplicate(fitted_xy, accepted_positions, duplicate_radius):
                    skipped_duplicate += 1
                    continue
                self._set_star_pair_position(
                    provisional.row,
                    fitted,
                    update_alignment=False,
                )
                star_id = provisional.reference_star.star_id.strip()
                self._star_pair_store.update_extra_fields(star_id, quality_fields)
                self._refresh_star_pair_quality_cell(provisional.row)
                if annotate_matches:
                    self._add_or_update_star_pair_annotation(
                        provisional.row,
                        fitted,
                    )
                accepted_positions.append(fitted_xy)
                matched_count += 1
        finally:
            table.blockSignals(signals_were_blocked)
            progress.setValue(len(candidates) * 2)
            progress.close()

        self._remove_unmatched_auto_match_candidates(candidate_star_ids)
        self._refresh_star_pair_table_styles()
        self._update_reference_alignment_transform()
        status_prefix = "自动扩展匹配已取消" if canceled else "自动扩展匹配完成"
        if self.current_sky_mask is None:
            mask_status = "蒙版未启用"
        else:
            mask_status = f"蒙版预筛 {mask_prefiltered_count} 个视场星点，拟合后落入蒙版 {skipped_mask}"
        self.ui.statusbar.showMessage(
            "{status_prefix}：{group_name} 本次新增 {candidate_count}，匹配成功 {matched_count}，已有 {skipped_existing}，"
            "{mask_status}，歧义跳过 {skipped_ambiguous}，亮度异常 {skipped_photometric}，"
            "重复跳过 {skipped_duplicate}，失败 {failed_count}。".format(
                status_prefix=status_prefix,
                group_name=self._auto_match_group_label(auto_group_id),
                candidate_count=len(candidates),
                matched_count=matched_count,
                skipped_existing=skipped_existing,
                mask_status=mask_status,
                skipped_ambiguous=skipped_ambiguous,
                skipped_photometric=skipped_photometric,
                skipped_duplicate=skipped_duplicate,
                failed_count=failed_count,
            )
        )
        if matched_count <= 0 and not canceled:
            QMessageBox.information(
                self,
                "自动扩展匹配完成",
                "没有新增匹配。可以检查蒙版、搜索半径和新增数量，或先增加几颗手动匹配星提高初始配准精度。",
            )
