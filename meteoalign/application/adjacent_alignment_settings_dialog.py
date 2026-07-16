"""参考图像粗略取景超参数设置窗口。"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QDoubleSpinBox, QMessageBox, QSpinBox

from ..adjacent_alignment import (
    ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
    ADJACENT_ALIGNMENT_MODE_STARS,
    adjacent_alignment_mode_display_name,
)
from ..config import load_adjacent_alignment_config
from ..preference_manager import DEFAULT_PREFERENCE_VALUES, update_preference_values
from ..ui.ui_adjacent_alignment_settings_dialog import Ui_AdjacentAlignmentSettingsDialog


_COMMON_SETTING_CONTROLS = {
    "adjacent_alignment_max_correspondences": (
        "spinBoxAdjacentAlignmentMaxCorrespondencesStar",
        "spinBoxAdjacentAlignmentMaxCorrespondencesLandscape",
    ),
}

_STAR_SETTING_CONTROLS = {
    "adjacent_star_background_bw_px": "spinBoxAdjacentStarBackgroundBwPx",
    "adjacent_star_background_bh_px": "spinBoxAdjacentStarBackgroundBhPx",
    "adjacent_star_background_fw_px": "spinBoxAdjacentStarBackgroundFwPx",
    "adjacent_star_background_fh_px": "spinBoxAdjacentStarBackgroundFhPx",
    "adjacent_star_detection_sigma": "doubleSpinBoxAdjacentStarDetectionSigma",
    "adjacent_star_detection_min_area_px": "spinBoxAdjacentStarDetectionMinAreaPx",
    "adjacent_star_deblend_nthresh": "spinBoxAdjacentStarDeblendNthresh",
    "adjacent_star_deblend_cont": "doubleSpinBoxAdjacentStarDeblendCont",
    "adjacent_star_detection_edge_margin_px": "spinBoxAdjacentStarDetectionEdgeMarginPx",
    "adjacent_star_min_major_axis_px": "doubleSpinBoxAdjacentStarMinMajorAxisPx",
    "adjacent_star_max_major_axis_px": "doubleSpinBoxAdjacentStarMaxMajorAxisPx",
    "adjacent_star_min_minor_axis_px": "doubleSpinBoxAdjacentStarMinMinorAxisPx",
    "adjacent_star_max_axis_ratio": "doubleSpinBoxAdjacentStarMaxAxisRatio",
    "adjacent_star_max_detected_stars": "spinBoxAdjacentStarMaxDetectedStars",
    "adjacent_star_max_alignment_stars": "spinBoxAdjacentStarMaxAlignmentStars",
    "adjacent_star_min_triangle_side_deg": "doubleSpinBoxAdjacentStarMinTriangleSideDeg",
    "adjacent_star_triangle_match_tolerance_deg": "doubleSpinBoxAdjacentStarTriangleMatchToleranceDeg",
    "adjacent_star_rotation_inlier_tolerance_deg": "doubleSpinBoxAdjacentStarRotationInlierToleranceDeg",
    "adjacent_star_max_triangle_hypotheses": "spinBoxAdjacentStarMaxTriangleHypotheses",
    "adjacent_star_min_initial_rotation_inliers": "spinBoxAdjacentStarMinInitialRotationInliers",
    "adjacent_star_initial_match_distance_px": "doubleSpinBoxAdjacentStarInitialMatchDistancePx",
    "adjacent_star_homography_match_distance_px": "doubleSpinBoxAdjacentStarHomographyMatchDistancePx",
    "adjacent_star_final_match_distance_px": "doubleSpinBoxAdjacentStarFinalMatchDistancePx",
    "adjacent_star_homography_ransac_max_iterations": "spinBoxAdjacentStarHomographyRansacMaxIterations",
    "adjacent_star_homography_ransac_confidence": "doubleSpinBoxAdjacentStarHomographyRansacConfidence",
    "adjacent_star_min_match_count": "spinBoxAdjacentStarMinMatchCount",
    "adjacent_star_focal_scale_min_ratio": "doubleSpinBoxAdjacentStarFocalScaleMinRatio",
}

_LANDSCAPE_SETTING_CONTROLS = {
    "adjacent_landscape_normalization_low_percentile": "doubleSpinBoxAdjacentLandscapeNormalizationLowPercentile",
    "adjacent_landscape_normalization_high_percentile": "doubleSpinBoxAdjacentLandscapeNormalizationHighPercentile",
    "adjacent_landscape_clahe_clip_limit": "doubleSpinBoxAdjacentLandscapeClaheClipLimit",
    "adjacent_landscape_clahe_grid_size": "spinBoxAdjacentLandscapeClaheGridSize",
    "adjacent_landscape_sift_max_features": "spinBoxAdjacentLandscapeSiftMaxFeatures",
    "adjacent_landscape_sift_contrast_threshold": "doubleSpinBoxAdjacentLandscapeSiftContrastThreshold",
    "adjacent_landscape_sift_edge_threshold": "doubleSpinBoxAdjacentLandscapeSiftEdgeThreshold",
    "adjacent_landscape_sift_sigma": "doubleSpinBoxAdjacentLandscapeSiftSigma",
    "adjacent_landscape_flann_trees": "spinBoxAdjacentLandscapeFlannTrees",
    "adjacent_landscape_flann_checks": "spinBoxAdjacentLandscapeFlannChecks",
    "adjacent_landscape_ratio_test_threshold": "doubleSpinBoxAdjacentLandscapeRatioTestThreshold",
    "adjacent_landscape_ransac_reprojection_threshold_px": "doubleSpinBoxAdjacentLandscapeRansacReprojectionThresholdPx",
    "adjacent_landscape_ransac_max_iterations": "spinBoxAdjacentLandscapeRansacMaxIterations",
    "adjacent_landscape_ransac_confidence": "doubleSpinBoxAdjacentLandscapeRansacConfidence",
    "adjacent_landscape_min_inlier_matches": "spinBoxAdjacentLandscapeMinInlierMatches",
}


class AdjacentAlignmentSettingsDialog(QDialog):
    """编辑当前参考图像对齐模式对应的可持久化超参数。"""

    def __init__(
        self,
        mode: str,
        parent: object | None = None,
        *,
        preference_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in {ADJACENT_ALIGNMENT_MODE_STARS, ADJACENT_ALIGNMENT_MODE_LANDSCAPE}:
            raise ValueError(f"不支持的参考图像对齐模式：{mode}")
        self._mode = mode
        self._preference_path = preference_path
        self.ui = Ui_AdjacentAlignmentSettingsDialog()
        self.ui.setupUi(self)
        self._configure_mode()
        self._load_effective_values()
        self._connect_inputs()

    def _configure_mode(self) -> None:
        """显示当前模式唯一相关的一页参数，避免误改另一种模式。"""

        display_name = adjacent_alignment_mode_display_name(self._mode)
        self.setWindowTitle(f"{display_name}参数设置")
        self.ui.labelCurrentMode.setText(
            f"当前工作模式：{display_name}。保存后将写入 preference.json，并立即用于后续粗略取景计算。"
        )
        page = (
            self.ui.pageStarSettings
            if self._mode == ADJACENT_ALIGNMENT_MODE_STARS
            else self.ui.pageLandscapeSettings
        )
        self.ui.stackedWidgetSettings.setCurrentWidget(page)

    def _connect_inputs(self) -> None:
        """连接参数约束与底部操作按钮。"""

        self.ui.doubleSpinBoxAdjacentStarMinMajorAxisPx.valueChanged.connect(
            self._synchronize_linked_ranges
        )
        self.ui.doubleSpinBoxAdjacentLandscapeNormalizationLowPercentile.valueChanged.connect(
            self._synchronize_linked_ranges
        )
        self.ui.spinBoxAdjacentStarMaxAlignmentStars.valueChanged.connect(
            self._synchronize_linked_ranges
        )
        self.ui.pushButtonRestoreDefaults.clicked.connect(self._restore_defaults)
        self.ui.pushButtonSave.clicked.connect(self._save_settings)
        self.ui.pushButtonCancel.clicked.connect(self.reject)

    def _active_setting_controls(self) -> dict[str, str | tuple[str, str]]:
        """返回当前页面可调整的配置字段及其控件名称。"""

        controls: dict[str, str | tuple[str, str]] = dict(_COMMON_SETTING_CONTROLS)
        if self._mode == ADJACENT_ALIGNMENT_MODE_STARS:
            controls.update(_STAR_SETTING_CONTROLS)
        else:
            controls.update(_LANDSCAPE_SETTING_CONTROLS)
        return controls

    def _control_for_key(self, key: str) -> QSpinBox | QDoubleSpinBox:
        """返回当前页面中对应配置字段的数值控件。"""

        control_name = self._active_setting_controls()[key]
        if isinstance(control_name, tuple):
            control_name = (
                control_name[0]
                if self._mode == ADJACENT_ALIGNMENT_MODE_STARS
                else control_name[1]
            )
        control = getattr(self.ui, control_name)
        if not isinstance(control, (QSpinBox, QDoubleSpinBox)):
            raise TypeError(f"配置控件类型不正确：{control_name}")
        return control

    def _load_effective_values(self) -> None:
        """读取当前实际生效的参数值，而非未校正的原始文本。"""

        config = load_adjacent_alignment_config(self._preference_path)
        values: dict[str, object] = {
            "adjacent_alignment_max_correspondences": config.max_correspondences,
        }
        values.update(
            {
                f"adjacent_star_{field.name}": getattr(config.stars, field.name)
                for field in fields(config.stars)
            }
        )
        values.update(
            {
                f"adjacent_landscape_{field.name}": getattr(config.landscape, field.name)
                for field in fields(config.landscape)
            }
        )
        self._set_values(values)

    def _set_values(self, values: dict[str, object]) -> None:
        """把配置值写入当前页面，并同步关联控件的边界。"""

        for key in self._active_setting_controls():
            value = values[key]
            self._control_for_key(key).setValue(value)
        self._synchronize_linked_ranges()

    def _synchronize_linked_ranges(self, *unused: object) -> None:
        """维持配置加载器同样要求的参数大小关系。"""

        min_major_axis = self.ui.doubleSpinBoxAdjacentStarMinMajorAxisPx.value()
        max_major_axis_control = self.ui.doubleSpinBoxAdjacentStarMaxMajorAxisPx
        max_major_axis_control.setMinimum(min_major_axis)
        if max_major_axis_control.value() < min_major_axis:
            max_major_axis_control.setValue(min_major_axis)

        low_percentile = self.ui.doubleSpinBoxAdjacentLandscapeNormalizationLowPercentile.value()
        high_percentile_control = self.ui.doubleSpinBoxAdjacentLandscapeNormalizationHighPercentile
        high_percentile_control.setMinimum(min(100.0, low_percentile + 0.01))
        if high_percentile_control.value() < high_percentile_control.minimum():
            high_percentile_control.setValue(high_percentile_control.minimum())

        max_alignment_stars = self.ui.spinBoxAdjacentStarMaxAlignmentStars.value()
        min_rotation_inliers_control = self.ui.spinBoxAdjacentStarMinInitialRotationInliers
        min_rotation_inliers_control.setMaximum(max_alignment_stars)
        if min_rotation_inliers_control.value() > max_alignment_stars:
            min_rotation_inliers_control.setValue(max_alignment_stars)

    def _restore_defaults(self) -> None:
        """只将当前模式页面的参数恢复为 preference_manager 中的默认设置。"""

        self._set_values(DEFAULT_PREFERENCE_VALUES)

    def _save_settings(self) -> None:
        """保存当前模式的参数并关闭窗口；失败时保留编辑内容。"""

        values: dict[str, object] = {}
        for key in self._active_setting_controls():
            control = self._control_for_key(key)
            values[key] = int(control.value()) if isinstance(control, QSpinBox) else float(control.value())
        if not update_preference_values(values, path=self._preference_path):
            QMessageBox.critical(self, "保存设置失败", "无法写入 preference.json，请检查配置目录的写入权限。")
            return
        self.accept()
