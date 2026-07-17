from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication, QGraphicsView, QMessageBox, QProgressDialog

from ..alignment.constants import MIN_ALIGNMENT_PAIRS, MIN_PRELIMINARY_ALIGNMENT_PAIRS
from ..alignment.models import SkyAlignmentTransform
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
from ..psf.matching import assign_predicted_sources, merge_source_candidates
from ..star_fitting import FittedStarPosition, detect_star_candidates, fit_star_position


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
    _star_pair_position_count: object  # method

    def _show_matching_failure_dialog(
        self,
        title: str,
        message: str,
        *,
        warning: bool,
    ) -> None:
        """显示匹配失败提示，并在弹窗关闭后恢复星点匹配助手层级。"""

        parent_method = getattr(self, "_star_pair_assistant_message_parent", None)
        message_parent = parent_method() if callable(parent_method) else self
        try:
            if warning:
                QMessageBox.warning(message_parent, title, message)
            else:
                QMessageBox.information(message_parent, title, message)
        finally:
            reactivate_method = getattr(self, "_reactivate_star_pair_assistant", None)
            if callable(reactivate_method):
                reactivate_method()

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

        star_id, star_name, distance_px = picked
        if star_id in self._excluded_reference_star_ids:
            self._excluded_reference_star_ids.remove(star_id)
        existing_row = self._select_star_pair_row_by_id(star_id)
        if existing_row is not None:
            self.ui.statusbar.showMessage(
                f"已选中参考星 {self._star_pair_label(existing_row)}；可在真实图像中点选对应星点。"
            )
            self._show_star_pair_assistant()
            return

        if star_id not in self._manual_reference_star_ids:
            self._manual_reference_star_ids.append(star_id)
        self._refresh_reference_stars_from_current_map()
        row = self._select_star_pair_row_by_id(star_id)
        if row is None:
            self.ui.statusbar.showMessage(f"未能添加参考星 {star_name or star_id}，请检查当前星等上限和视野。")
            self._show_star_pair_assistant()
            return
        self.ui.statusbar.showMessage(
            f"已添加待匹配参考星 {self._star_pair_label(row)}；点击偏差约 {distance_px:.1f} px。"
        )
        self._show_star_pair_assistant()

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
            message = "点击位置位于天空蒙版外，已拒绝把地景纹理作为星点。"
            self.ui.statusbar.showMessage(message)
            self._show_matching_failure_dialog("星点已被蒙版拒绝", message, warning=True)
            return

        search_radius_px = self._star_pick_search_radius_px(viewport_pos)
        max_fit_radius_px = self._star_pick_psf_radius_px(viewport_pos)
        try:
            fitted_position = fit_star_position(
                image,
                click_x=image_x,
                click_y=image_y,
                radius_px=search_radius_px,
                max_fit_radius_px=max_fit_radius_px,
                selection_mode="manual",
            )
        except Exception as exc:  # noqa: BLE001 - 交互式点选需要把拟合失败原因直接反馈给用户。
            self.ui.statusbar.showMessage(f"PSF 拟合失败: {exc}")
            self._show_matching_failure_dialog("PSF 拟合失败", str(exc), warning=True)
            return
        if not self._sky_mask_allows_point(fitted_position.x, fitted_position.y):
            message = "检测到的 PSF 中心位于天空蒙版外，结果已拒绝。"
            self.ui.statusbar.showMessage(message)
            self._show_matching_failure_dialog("PSF 已被蒙版拒绝", message, warning=True)
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

    def _report_auto_pair_failure(
        self,
        message: str,
        *,
        silent_failure: bool,
        dialog_title: str = "自动匹配失败",
        warning: bool = False,
    ) -> bool:
        """统一报告单星自动匹配失败；双击联动模式只写状态栏。"""

        self.ui.statusbar.showMessage(f"自动匹配失败：{message}")
        if not silent_failure:
            self._show_matching_failure_dialog(dialog_title, message, warning=warning)
        return False

    def _auto_pair_star(self, row: int, *, silent_failure: bool = False) -> bool:
        if self.current_image_preview is None:
            return self._report_auto_pair_failure(
                "请先导入真实图像，再自动匹配星点。",
                silent_failure=silent_failure,
                dialog_title="尚未导入图像",
            )

        transform = self._sky_alignment_transform
        if transform is None:
            self._update_reference_alignment_transform()
            transform = self._sky_alignment_transform
        if transform is None:
            return self._report_auto_pair_failure(
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or f"至少需要 {MIN_PRELIMINARY_ALIGNMENT_PAIRS} 对已配准参考星。",
                silent_failure=silent_failure,
                dialog_title="无法自动匹配",
            )

        reference_star = self._reference_star_for_row(row)
        if reference_star is None:
            return self._report_auto_pair_failure(
                "当前行没有对应的参考星。",
                silent_failure=silent_failure,
                dialog_title="无法自动匹配",
                warning=True,
            )

        predicted_x, predicted_y = transform.transform_radec(reference_star.ra_deg, reference_star.dec_deg)
        image = self.current_image_preview.image
        if not (0.0 <= predicted_x < image.width() and 0.0 <= predicted_y < image.height()):
            self.ui.statusbar.showMessage(
                f"{self._star_pair_label(row)} 的预测位置在真实图像外，无法自动匹配。"
            )
            return False

        search_radius_px = self._auto_pair_search_radius_px(transform)
        self._focus_star_pair_image_point(row, predicted_x, predicted_y, search_radius_px)
        self.ui.statusbar.showMessage(
            f"已跳转到 {self._star_pair_label(row)} 的自动匹配预测位置，正在搜索真实星点..."
        )
        QApplication.processEvents()

        try:
            fitted_position = fit_star_position(
                image,
                click_x=predicted_x,
                click_y=predicted_y,
                radius_px=search_radius_px,
                max_fit_radius_px=self.ui_config.star_pick_psf_max_radius_px,
                reject_ambiguous=True,
                selection_mode="predicted",
            )
        except Exception as exc:  # noqa: BLE001 - 自动匹配要把失败原因反馈给用户。
            return self._report_auto_pair_failure(
                str(exc),
                silent_failure=silent_failure,
                warning=True,
            )

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
        failed_count = 0
        canceled = False
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
                    image,
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
                        image,
                        click_x=assigned_source.x,
                        click_y=assigned_source.y,
                        radius_px=source_fit_search_radius,
                        max_fit_radius_px=self.ui_config.star_pick_psf_max_radius_px,
                        selection_mode="manual",
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
                fitted_xy = (float(fitted_position.x), float(fitted_position.y))
                duplicate_radius = max(
                    AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
                    min(24.0, max(fitted_position.fwhm_x, fitted_position.fwhm_y) * 0.65),
                )
                if self._position_is_duplicate(fitted_xy, accepted_positions, duplicate_radius):
                    skipped_duplicate += 1
                    continue

                self._set_star_pair_position(row, fitted_position, update_alignment=False)
                if annotate_matches:
                    self._add_or_update_star_pair_annotation(row, fitted_position)
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
            "{mask_status}，歧义跳过 {skipped_ambiguous}，重复跳过 {skipped_duplicate}，失败 {failed_count}。".format(
                status_prefix=status_prefix,
                group_name=self._auto_match_group_label(auto_group_id),
                candidate_count=len(candidates),
                matched_count=matched_count,
                skipped_existing=skipped_existing,
                mask_status=mask_status,
                skipped_ambiguous=skipped_ambiguous,
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
