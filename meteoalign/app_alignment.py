from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QGraphicsView, QMessageBox

from .alignment.constants import MIN_ALIGNMENT_PAIRS, SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION
from .alignment.fitting import fit_sky_alignment
from .alignment.models import SkyAlignmentTransform
from .camera_calibration import CameraCalibrationProfile
from .coordinates import radec_to_unit_vectors
from .frame_astrometry import FrameAstrometricModel
from .simulator import (
    ReferenceStar, camera_basis_from_view, local_vectors_from_altaz,
)

from .app_constants import (
    STAR_PAIR_RESIDUAL_COLUMN, RESIDUAL_WARNING_MIN_PX,
    RESIDUAL_SEVERE_MIN_PX, RESIDUAL_SEVERE_RMS_SCALE,
    SKY_ALIGNMENT_MODELS,
    SKY_ALIGNMENT_MODEL_ALIASES, SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION as _ANCHOR,
    AUTO_MATCH_CONSTRAINT_SOFT, AUTO_MATCH_CONSTRAINT_MODES,
    SOURCE_MODEL_JSON_FILTER,
)
from .source_model import (
    FixedProfilePoseSourceModel,
    SourceAstrometricModel,
    fit_source_astrometric_model,
    fit_source_astrometric_model_with_fixed_profile,
)


PROFILE_SOLVE_NOT_USED = "profile_not_used"
PROFILE_SOLVE_IMPORTED_PROFILE_POSE_ONLY = "imported_profile_pose_only"
PROFILE_SOLVE_IMPORTED_PROFILE_POSE_LOCAL_RESIDUAL = "imported_profile_pose_local_residual"
PROFILE_SOLVE_MODES = (
    PROFILE_SOLVE_NOT_USED,
    PROFILE_SOLVE_IMPORTED_PROFILE_POSE_ONLY,
    PROFILE_SOLVE_IMPORTED_PROFILE_POSE_LOCAL_RESIDUAL,
)


class AlignmentMixin:
    """配准管理 Mixin：天球配准变换、残差计算、参考星图联动。"""

    ui: object
    _sky_alignment_transform: SkyAlignmentTransform | None
    _source_astrometric_model: SourceAstrometricModel | FixedProfilePoseSourceModel | None
    _imported_camera_calibration_profile: CameraCalibrationProfile | None
    _imported_camera_calibration_profile_path: Path | None
    _reference_alignment_error_message: str
    _sky_alignment_error_message: str
    _source_model_error_message: str
    _suspend_alignment_updates: bool
    _current_reference_stars: tuple
    _current_reference_star_map: object | None
    _last_reference_render_size: object
    _simulator_controls_locked: bool
    current_image_preview: object | None
    reference_star_map_item: object
    real_reference_overlay_item: object
    reference_scene: object
    real_image_scene: object
    _aligned_star_element_scale: object  # method
    _clear_focused_star_annotations: object  # method
    _update_star_pair_annotation_visibility: object  # method
    _update_live_star_map_zoom_scale: object  # method
    _star_pair_star_id: object  # method
    _parse_star_pair_position_text: object  # method
    _star_pair_fit_constraint: object  # method
    _star_pair_position_count: object  # method
    _alignment_residual_distances: object  # method
    _view_settings: object  # method
    _build_aligned_reference_star_map: object  # method
    _update_lens_model_controls: object  # method
    _update_reference_label_controls: object  # method

    def _alignment_model(self) -> str:
        index = self.ui.comboBoxSkyAlignmentModel.currentIndex()
        if index < 0 or index >= len(SKY_ALIGNMENT_MODELS):
            return SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION
        return SKY_ALIGNMENT_MODELS[index]

    def _auto_match_constraint_mode(self) -> str:
        index = self.ui.comboBoxAutoMatchConstraintMode.currentIndex()
        if index < 0 or index >= len(AUTO_MATCH_CONSTRAINT_MODES):
            return AUTO_MATCH_CONSTRAINT_SOFT
        return AUTO_MATCH_CONSTRAINT_MODES[index]

    def _auto_match_soft_weight(self) -> float:
        if self._auto_match_constraint_mode() != AUTO_MATCH_CONSTRAINT_SOFT:
            return 1.0
        return max(0.01, min(1.0, float(self.ui.doubleSpinBoxAutoMatchSoftWeight.value())))

    def _set_alignment_model(self, model: object) -> None:
        model_text = str(model or "").strip()
        model_text = SKY_ALIGNMENT_MODEL_ALIASES.get(model_text, model_text)
        if model_text not in SKY_ALIGNMENT_MODELS:
            return
        self.ui.comboBoxSkyAlignmentModel.setCurrentIndex(SKY_ALIGNMENT_MODELS.index(model_text))

    def _handle_alignment_model_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_reference_alignment_transform()

    def _profile_reuse_enabled(self) -> bool:
        return (
            self._profile_reuse_solve_mode() != PROFILE_SOLVE_NOT_USED
            and getattr(self, "_imported_camera_calibration_profile", None) is not None
        )

    def _profile_reuse_solve_mode(self) -> str:
        if not hasattr(self.ui, "comboBoxProfileSolveMode"):
            return PROFILE_SOLVE_NOT_USED
        index = self.ui.comboBoxProfileSolveMode.currentIndex()
        if index < 0 or index >= len(PROFILE_SOLVE_MODES):
            return PROFILE_SOLVE_NOT_USED
        return PROFILE_SOLVE_MODES[index]

    def _imported_profile_label_text(self) -> tuple[str, str]:
        profile = getattr(self, "_imported_camera_calibration_profile", None)
        profile_path = getattr(self, "_imported_camera_calibration_profile_path", None)
        if profile is None:
            return "未导入", ""
        path_text = "" if profile_path is None else str(profile_path)
        projection_text = str(profile.base_projection_type)
        return (
            f"{projection_text}  {profile.image_width_px} x {profile.image_height_px}",
            path_text,
        )

    def _update_camera_profile_controls(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self.ui, "pushButtonImportCameraProfile"):
            return
        profile = getattr(self, "_imported_camera_calibration_profile", None)
        has_profile = profile is not None
        if hasattr(self.ui, "labelImportedCameraProfile"):
            label_text, tooltip = self._imported_profile_label_text()
            self._set_elided_label_text(self.ui.labelImportedCameraProfile, label_text, tooltip)
        if hasattr(self.ui, "pushButtonClearCameraProfile"):
            self.ui.pushButtonClearCameraProfile.setEnabled(has_profile)
        if hasattr(self.ui, "comboBoxProfileSolveMode"):
            self.ui.comboBoxProfileSolveMode.setEnabled(has_profile)

    def _load_camera_profile_from_json_payload(self, payload: object) -> CameraCalibrationProfile:
        if not isinstance(payload, dict):
            raise ValueError("Profile JSON 根对象必须是对象。")
        if str(payload.get("schema", "")) == "hgmeteo_source_astrometric_model":
            frame_model = FrameAstrometricModel.from_json_payload(payload)
            return frame_model.camera_calibration_profile
        if isinstance(payload.get("camera_calibration_profile"), dict):
            return CameraCalibrationProfile.from_json_payload(payload["camera_calibration_profile"])
        return CameraCalibrationProfile.from_json_payload(payload)

    def import_camera_calibration_profile(self) -> None:
        default_dir = Path.cwd()
        profile_path = getattr(self, "_imported_camera_calibration_profile_path", None)
        if profile_path is not None:
            default_dir = profile_path.parent
        elif self.current_image_preview is not None:
            default_dir = Path(self.current_image_preview.path).expanduser().resolve().parent
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "从模型 JSON 导入 Camera Profile",
            str(default_dir),
            SOURCE_MODEL_JSON_FILTER,
        )
        if not file_path:
            return
        try:
            json_path = Path(file_path).expanduser().resolve()
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            profile = self._load_camera_profile_from_json_payload(payload)
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON/Profile 错误直接反馈给用户。
            QMessageBox.critical(self, "导入 Camera Profile 失败", str(exc))
            self.ui.statusbar.showMessage(f"导入 Camera Profile 失败: {exc}")
            return

        self._imported_camera_calibration_profile = profile
        self._imported_camera_calibration_profile_path = json_path
        if hasattr(self.ui, "comboBoxProfileSolveMode"):
            self.ui.comboBoxProfileSolveMode.setCurrentIndex(1)
        self._update_camera_profile_controls()
        self._update_reference_alignment_transform()
        self.ui.statusbar.showMessage(f"已导入 Camera Profile: {json_path}")

    def clear_camera_calibration_profile(self) -> None:
        self._imported_camera_calibration_profile = None
        self._imported_camera_calibration_profile_path = None
        self._update_camera_profile_controls()
        self._update_reference_alignment_transform()
        self.ui.statusbar.showMessage("已清除导入的 Camera Profile。")

    def _handle_profile_reuse_options_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_camera_profile_controls()
        self._update_reference_alignment_transform()

    def _update_auto_match_controls(self, *unused) -> None:  # type: ignore[no-untyped-def]
        soft_mode = self._auto_match_constraint_mode() == AUTO_MATCH_CONSTRAINT_SOFT
        self.ui.labelAutoMatchSoftWeight.setEnabled(soft_mode)
        self.ui.doubleSpinBoxAutoMatchSoftWeight.setEnabled(soft_mode)

    def _simulator_lock_widgets(self) -> tuple[object, ...]:
        widget_names = (
            "doubleSpinBoxSensorWidth",
            "doubleSpinBoxSensorHeight",
            "pushButtonSwapOrientation",
            "spinBoxImageWidth",
            "spinBoxImageHeight",
            "doubleSpinBoxFocalLength",
            "comboBoxLensModel",
            "doubleSpinBoxFisheyeFov",
            "doubleSpinBoxMagLimit",
            "doubleSpinBoxAz",
            "doubleSpinBoxAlt",
            "doubleSpinBoxRoll",
            "pushButtonImportReferenceJson",
        )
        label_names = (
            "labelSensorWidth",
            "labelSensorHeight",
            "labelImageWidth",
            "labelImageHeight",
            "labelFocalLength",
            "labelLensModel",
            "labelFisheyeFov",
            "labelMagLimit",
            "labelAz",
            "labelAlt",
            "labelRoll",
        )
        return tuple(
            getattr(self.ui, name)
            for name in (*widget_names, *label_names)
            if hasattr(self.ui, name)
        )

    def _update_simulator_controls_lock(self, matched_count: int) -> None:
        locked = int(matched_count) >= MIN_ALIGNMENT_PAIRS
        if bool(getattr(self, "_simulator_controls_locked", False)) == locked:
            return

        self._simulator_controls_locked = locked
        for widget in self._simulator_lock_widgets():
            widget.setEnabled(not locked)
        if locked:
            self.drag_start = None
            self.last_drag_pos = None
            self.ui.starMapView.viewport().unsetCursor()
            return

        self._update_lens_model_controls()
        self._update_reference_label_controls()

    def _reference_star_lookup(self) -> dict[str, ReferenceStar]:
        return {star.star_id.strip(): star for star in self._current_reference_stars if star.star_id.strip()}

    def _reference_star_for_row(self, row: int) -> ReferenceStar | None:
        star_id = self._star_pair_star_id(row)
        if not star_id:
            return None
        return self._reference_star_lookup().get(star_id)

    def _matched_sky_alignment_points(self) -> tuple[np.ndarray, np.ndarray]:
        sky_points, target_points, _weights, _anchor_mask = self._matched_sky_alignment_data()
        return sky_points, target_points

    def _matched_sky_alignment_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """收集用于天球配准的配对数据，只从 StarPairStore 读取。"""
        records = self._star_pair_store.valid_fit_records()

        sky_points: list[tuple[float, float]] = []
        target_points: list[tuple[float, float]] = []
        point_weights: list[float] = []
        anchor_flags: list[bool] = []
        for record in records:
            reference_star = record.reference_star
            sky_points.append((reference_star.ra_deg, reference_star.dec_deg))
            target_points.append(record.position)
            point_weights.append(float(record.fit_weight))
            anchor_flags.append(record.fit_constraint_mode != AUTO_MATCH_CONSTRAINT_SOFT)
        return (
            np.asarray(sky_points, dtype=np.float64),
            np.asarray(target_points, dtype=np.float64),
            np.asarray(point_weights, dtype=np.float64),
            np.asarray(anchor_flags, dtype=bool),
        )

    def _initial_projection_rotation_matrix(self) -> np.ndarray | None:
        reference_stars = [
            star
            for star in self._current_reference_stars
            if all(
                math.isfinite(value)
                for value in (star.ra_deg, star.dec_deg, star.alt_deg, star.az_deg)
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

    def _star_pair_alignment_residual(self, row: int) -> tuple[float, float, float] | None:
        transform = self._sky_alignment_transform
        if transform is None:
            return None

        reference_star = self._reference_star_for_row(row)
        target_position = self._parse_star_pair_position_text(row)
        if reference_star is None or target_position is None:
            return None

        predicted_x, predicted_y = transform.transform_radec(reference_star.ra_deg, reference_star.dec_deg)
        if not all(math.isfinite(value) for value in (predicted_x, predicted_y)):
            return None
        dx = predicted_x - target_position[0]
        dy = predicted_y - target_position[1]
        distance = float(np.hypot(dx, dy))
        return float(dx), float(dy), distance

    def _alignment_residual_distances(self) -> np.ndarray:
        distances: list[float] = []
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            residual = self._star_pair_alignment_residual(row)
            if residual is not None:
                distances.append(residual[2])
        return np.asarray(distances, dtype=np.float64)

    def _residual_warning_thresholds(self) -> tuple[float, float]:
        transform = self._sky_alignment_transform
        if transform is None:
            return RESIDUAL_WARNING_MIN_PX, RESIDUAL_SEVERE_MIN_PX
        warning = max(RESIDUAL_WARNING_MIN_PX, float(transform.rms_px))
        severe = max(RESIDUAL_SEVERE_MIN_PX, float(transform.rms_px) * RESIDUAL_SEVERE_RMS_SCALE)
        return warning, severe

    def _ensure_star_pair_residual_item(self, row: int, column: int) -> QTableWidgetItem:
        table = self.ui.tableWidgetStarPairs
        item = table.item(row, column)
        if item is None:
            item = self._read_only_table_item("")
            table.setItem(row, column, item)
        else:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _update_star_pair_residual_columns(self) -> None:
        table = self.ui.tableWidgetStarPairs
        signals_were_blocked = table.blockSignals(True)
        for row in range(table.rowCount()):
            residual_item = self._ensure_star_pair_residual_item(row, STAR_PAIR_RESIDUAL_COLUMN)

            residual = self._star_pair_alignment_residual(row)
            star_id = self._star_pair_star_id(row)
            if residual is None:
                residual_item.setText("")
                residual_item.setToolTip("")
                if star_id:
                    self._star_pair_store.set_residual(star_id, None, None, None)
                continue

            dx, dy, distance = residual
            residual_item.setText(f"{distance:.2f}")
            residual_item.setToolTip("残差为天球 RA/Dec 模型预测位置与真实图像记录位置之间的像素距离。")
            if star_id:
                self._star_pair_store.set_residual(star_id, dx, dy, distance)
        table.blockSignals(signals_were_blocked)

        self._apply_star_pair_table_column_widths()
        self._update_auto_match_group_row_text()
        self._refresh_star_pair_table_styles()

    def _update_reference_alignment_transform(self) -> None:
        self._sky_alignment_transform = None
        self._source_astrometric_model = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""
        if self._suspend_alignment_updates:
            return
        sky_points, sky_target_points, fit_weights, anchor_mask = self._matched_sky_alignment_data()
        self._update_simulator_controls_lock(int(sky_points.shape[0]))
        if self.current_image_preview is None:
            self._reference_alignment_error_message = "导入真实图像后可计算实时星空叠加。"
            self._sky_alignment_error_message = "导入真实图像后可计算天球残差。"
            self._source_model_error_message = "导入真实图像后可生成 xy→RA/Dec 映射。"
            self._update_star_pair_residual_columns()
            self._update_reference_alignment_display()
            return

        if sky_points.shape[0] < MIN_ALIGNMENT_PAIRS:
            self._reference_alignment_error_message = (
                f"已配对 {sky_points.shape[0]} 颗星；至少 {MIN_ALIGNMENT_PAIRS} 颗后可实时叠加星空。"
            )
            self._sky_alignment_error_message = (
                f"已配对 {sky_points.shape[0]} 颗星；至少 {MIN_ALIGNMENT_PAIRS} 颗后可计算天球残差。"
            )
            self._source_model_error_message = (
                f"已配对 {sky_points.shape[0]} 颗星；至少 {MIN_ALIGNMENT_PAIRS} 颗后可生成 xy→RA/Dec 映射。"
            )
            self._update_star_pair_residual_columns()
            self._update_reference_alignment_display()
            return

        try:
            image = self.current_image_preview.image
            initial_rotation_matrix = self._initial_projection_rotation_matrix()
            # 源图基础投影由真实图像配对拟合，避免复用星空模拟页的镜头投影或鱼眼视场。
            source_projection_fov_deg = None
            self._sky_alignment_transform = fit_sky_alignment(
                ra_dec_points=sky_points,
                target_points=sky_target_points,
                matching_model=self._alignment_model(),
                image_size=(image.width(), image.height()),
                fisheye_fov_deg=source_projection_fov_deg,
                initial_rotation_matrix=initial_rotation_matrix,
                point_weights=fit_weights,
                residual_anchor_mask=anchor_mask,
            )
        except Exception as exc:  # noqa: BLE001 - 天球残差失败需要直接反馈给交互界面。
            self._sky_alignment_error_message = str(exc)
        if self._sky_alignment_transform is not None and self.current_image_preview is not None:
            try:
                image = self.current_image_preview.image
                initial_rotation_matrix = self._initial_projection_rotation_matrix()
                imported_profile = getattr(self, "_imported_camera_calibration_profile", None)
                if self._profile_reuse_enabled():
                    if imported_profile is None:
                        raise ValueError("已选择使用 Profile，但尚未导入 Camera Profile。")
                    self._source_astrometric_model = fit_source_astrometric_model_with_fixed_profile(
                        ra_dec_points=sky_points,
                        pixel_points=sky_target_points,
                        image_size=(image.width(), image.height()),
                        camera_calibration_profile=imported_profile,
                        initial_rotation_matrix=initial_rotation_matrix,
                        point_weights=fit_weights,
                        residual_anchor_mask=anchor_mask,
                        profile_source_path=str(getattr(self, "_imported_camera_calibration_profile_path", "") or ""),
                        solve_mode=self._profile_reuse_solve_mode(),
                    )
                else:
                    self._source_astrometric_model = fit_source_astrometric_model(
                        ra_dec_points=sky_points,
                        pixel_points=sky_target_points,
                        image_size=(image.width(), image.height()),
                        matching_model=self._alignment_model(),
                        fisheye_fov_deg=source_projection_fov_deg,
                        initial_rotation_matrix=initial_rotation_matrix,
                        point_weights=fit_weights,
                        residual_anchor_mask=anchor_mask,
                    )
            except Exception as exc:  # noqa: BLE001 - 源图模型错误要保留给导出按钮和状态栏。
                self._source_model_error_message = str(exc)
        self._update_star_pair_residual_columns()
        self._update_reference_alignment_display()

    def _update_reference_overlay_opacity_label(self) -> None:
        opacity = self.ui.doubleSpinBoxReferenceOverlayOpacity.value()
        self.ui.doubleSpinBoxReferenceOverlayOpacity.setToolTip(f"实时星空叠加透明度：{opacity:.1f}%")

    def _reference_overlay_opacity(self) -> float:
        return max(0.0, min(1.0, self.ui.doubleSpinBoxReferenceOverlayOpacity.value() / 100.0))

    def _show_reference_annotations(self) -> bool:
        return self.ui.checkBoxHideReferenceAnnotations.isChecked()

    def _show_real_image_annotations(self) -> bool:
        return self.ui.checkBoxHideRealImageAnnotations.isChecked()

    def _set_alignment_status_text(self, text: str, tooltip: str | None = None) -> None:
        self._set_elided_label_text(
            self.ui.labelAlignmentTransformStatus,
            text.strip(),
            (tooltip or text).strip(),
        )

    def _handle_reference_overlay_opacity_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_reference_overlay_opacity_label()
        self.real_reference_overlay_item.setOpacity(self._reference_overlay_opacity())
        self._update_reference_alignment_controls()

    def _handle_show_reference_annotations_toggled(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._clear_focused_star_annotations()
        self._update_reference_alignment_display()

    def _handle_show_real_image_annotations_toggled(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._clear_focused_star_annotations()
        self._update_star_pair_annotation_visibility()

    def _handle_reference_real_sync_toggled(self, checked: bool) -> None:
        if checked and self._can_sync_reference_real_views():
            self._sync_reference_real_view_from(self.ui.realImageView, force=True)

    def _reference_alignment_scene_rect(self) -> QRectF:
        if self.current_image_preview is None:
            return QRectF()
        return QRectF(
            0.0,
            0.0,
            float(self.current_image_preview.image.width()),
            float(self.current_image_preview.image.height()),
        )

    def _update_reference_alignment_controls(self) -> None:
        has_alignment = self._sky_alignment_transform is not None and self.current_image_preview is not None
        has_source_model = self._source_astrometric_model is not None and self.current_image_preview is not None
        has_export_model = has_source_model
        self.ui.checkBoxOverlayReferenceMap.setEnabled(has_alignment)
        self.ui.labelReferenceOverlayOpacityTitle.setEnabled(True)
        self.ui.doubleSpinBoxReferenceOverlayOpacity.setEnabled(True)
        self.ui.checkBoxSyncReferenceAndRealView.setEnabled(has_alignment)
        mask_controls_enabled = self._mask_import_thread is None
        sequence_mode = bool(hasattr(self, "_sequence_mode_active") and self._sequence_mode_active())
        self.ui.pushButtonImportSkyMask.setEnabled(
            mask_controls_enabled and not sequence_mode and self.current_image_preview is not None
        )
        self.ui.pushButtonClearSkyMask.setEnabled(
            mask_controls_enabled and not sequence_mode and self.current_sky_mask is not None
        )
        self.ui.checkBoxShowSkyMask.setEnabled(mask_controls_enabled and self.current_sky_mask is not None)
        if self.current_sky_mask is None and self.ui.checkBoxShowSkyMask.isChecked():
            was_blocked = self.ui.checkBoxShowSkyMask.blockSignals(True)
            self.ui.checkBoxShowSkyMask.setChecked(False)
            self.ui.checkBoxShowSkyMask.blockSignals(was_blocked)
        self.ui.pushButtonAutoMatchFieldStars.setEnabled(has_alignment)
        self.ui.pushButtonExportSourceModel.setEnabled(has_export_model)
        self._update_camera_profile_controls()
        if not has_alignment and self.ui.checkBoxSyncReferenceAndRealView.isChecked():
            self.ui.checkBoxSyncReferenceAndRealView.blockSignals(True)
            self.ui.checkBoxSyncReferenceAndRealView.setChecked(False)
            self.ui.checkBoxSyncReferenceAndRealView.blockSignals(False)

        sky_transform = self._sky_alignment_transform
        if sky_transform is not None:
            source_model_text = "，模型可导出"
            distances = self._alignment_residual_distances()
            compact_summary = ""
            residual_summary = "暂无逐星残差"
            if distances.size > 0:
                median_distance = float(np.median(distances))
                max_distance = float(np.max(distances))
                compact_summary = f"，中位 {median_distance:.2f}，最大 {max_distance:.2f}"
                residual_summary = f"中位 {median_distance:.2f} px，最大 {max_distance:.2f} px"
            projection_rms = getattr(sky_transform, "projection_rms_px", None)
            projection_summary = ""
            projection_tooltip = ""
            if projection_rms is not None and math.isfinite(float(projection_rms)):
                projection_summary = f"，投影 {float(projection_rms):.2f}px"
                projection_tooltip = f"\n已知投影原始 RMS：{float(projection_rms):.2f} px。"
            soft_count = int(getattr(sky_transform, "residual_soft_constraint_count", 0) or 0)
            soft_summary = f"，软约束 {soft_count}" if soft_count > 0 else ""
            soft_tooltip = ""
            if soft_count > 0:
                soft_tooltip = (
                    "\n残差修正包含 {count} 个软约束点，权重范围 {min_weight:.2f}-{max_weight:.2f}。"
                ).format(
                    count=soft_count,
                    min_weight=float(getattr(sky_transform, "residual_soft_weight_min", 1.0)),
                    max_weight=float(getattr(sky_transform, "residual_soft_weight_max", 1.0)),
                )
            display_text = (
                "配准 {count} 对，{model} RMS {rms:.2f}px{summary}{projection}{soft}{source_model}".format(
                    count=sky_transform.pair_count,
                    model=sky_transform.display_name,
                    rms=sky_transform.rms_px,
                    summary=compact_summary,
                    projection=projection_summary,
                    soft=soft_summary,
                    source_model=source_model_text,
                )
            )
            tooltip = (
                "实时星空叠加：已用 {count} 对星建立 RA/Dec {model}，RMS {rms:.2f} px。\n"
                "天球残差：{residual_summary}。{projection_tooltip}{soft_tooltip}\n"
                "源图映射：{source_model_summary}".format(
                    count=sky_transform.pair_count,
                    model=sky_transform.display_name,
                    rms=sky_transform.rms_px,
                    residual_summary=residual_summary,
                    projection_tooltip=projection_tooltip,
                    soft_tooltip=soft_tooltip,
                    source_model_summary=(
                        "已可导出 xy→RA/Dec JSON，并可在全景构图中验证。"
                        if self._source_astrometric_model is not None
                        else self._source_model_error_message or "尚未就绪。"
                    ),
                )
            )
            self._set_alignment_status_text(display_text, tooltip)
        else:
            status_text = (
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or self._source_model_error_message
                or f"至少配对 {MIN_ALIGNMENT_PAIRS} 颗星后可自动配准。"
            )
            self._set_alignment_status_text(status_text)

    def _update_reference_alignment_display(self, *unused) -> None:  # type: ignore[no-untyped-def]
        star_map = self._current_star_map
        transform = self._sky_alignment_transform
        has_alignment = transform is not None and self.current_image_preview is not None
        self._update_reference_alignment_controls()

        if star_map is None:
            self.reference_star_map_item.clear()
            self.real_reference_overlay_item.clear()
            self.real_reference_overlay_item.setVisible(False)
            self._last_reference_render_size = None
            return

        display_star_map = star_map
        if has_alignment:
            assert transform is not None
            scene_rect = self._reference_alignment_scene_rect()
            target_size = (int(scene_rect.width()), int(scene_rect.height()))
            display_key: tuple[object, ...] = ("aligned", target_size[0], target_size[1])
            element_scale = self._aligned_star_element_scale(target_size)
            number_reference_stars = self._show_reference_annotations()
            display_star_map = self._build_aligned_reference_star_map(transform, target_size)
            self._current_reference_star_map = display_star_map
            self.reference_star_map_item.set_star_map(
                display_star_map,
                reference_stars=self._current_reference_stars,
                sky_transform=transform,
                target_size=target_size,
                element_scale=element_scale,
                draw_common_names=False,
                number_reference_stars=number_reference_stars,
            )
            if not scene_rect.isEmpty():
                self.reference_scene.setSceneRect(scene_rect)
        else:
            target_size = None
            display_key = ("native", star_map.width, star_map.height)
            number_reference_stars = self._show_reference_annotations()
            self._current_reference_star_map = star_map
            self.reference_star_map_item.set_star_map(
                star_map,
                reference_stars=self._current_reference_stars,
                sky_transform=None,
                target_size=None,
                element_scale=1.0,
                draw_common_names=False,
                number_reference_stars=number_reference_stars,
            )
            self.reference_scene.setSceneRect(0.0, 0.0, float(star_map.width), float(star_map.height))
        self._fit_reference_map_if_display_changed(display_key)

        overlay_visible = (
            has_alignment
            and self.current_image_preview is not None
            and self.ui.checkBoxOverlayReferenceMap.isChecked()
        )
        if overlay_visible:
            assert transform is not None
            assert target_size is not None
            self.real_reference_overlay_item.set_star_map(
                display_star_map,
                reference_stars=self._current_reference_stars,
                sky_transform=transform,
                target_size=target_size,
                element_scale=self._aligned_star_element_scale(target_size),
                draw_common_names=False,
                number_reference_stars=self._show_reference_annotations(),
            )
            self.real_reference_overlay_item.setOpacity(self._reference_overlay_opacity())
            self.real_reference_overlay_item.setVisible(True)
        else:
            self.real_reference_overlay_item.setVisible(False)

        self._update_live_star_map_zoom_scale(self.ui.referenceImageView)
        self._update_live_star_map_zoom_scale(self.ui.realImageView)
        if self.ui.checkBoxSyncReferenceAndRealView.isChecked() and self._can_sync_reference_real_views():
            self._sync_reference_real_view_from(self.ui.realImageView, force=True)

    def _can_sync_reference_real_views(self) -> bool:
        return (
            self.ui.checkBoxSyncReferenceAndRealView.isChecked()
            and self._sky_alignment_transform is not None
            and self.current_image_preview is not None
        )

    def _sync_reference_real_view_from(
        self,
        source_view: QGraphicsView,
        force: bool = False,
        source_center=None,
    ) -> None:
        if self._syncing_reference_real_views:
            return
        if not force and not self._can_sync_reference_real_views():
            return
        if source_view not in (self.ui.referenceImageView, self.ui.realImageView):
            return

        target_view = self.ui.realImageView if source_view is self.ui.referenceImageView else self.ui.referenceImageView
        self._syncing_reference_real_views = True
        try:
            target_view.setTransform(source_view.transform())
            center = source_center
            if center is None:
                center = source_view.mapToScene(source_view.viewport().rect().center())
            target_view.centerOn(center)
            if target_view is self.ui.realImageView:
                self._cap_graphics_view_to_max_scale(target_view)
            self._update_live_star_map_zoom_scale(source_view)
            self._update_live_star_map_zoom_scale(target_view)
        finally:
            self._syncing_reference_real_views = False
