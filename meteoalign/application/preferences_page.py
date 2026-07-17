from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog, QMessageBox, QWidget

from ..config import StarMapUiConfig, load_star_map_ui_config
from ..meteor_showers import METEOR_SHOWER_BY_ID
from ..preference_manager import update_preference_values
from ..ui.ui_preferences_page import Ui_PreferencesPage
from .meteor_shower_selection_dialog import MeteorShowerSelectionDialog


# 本页只维护普通软件参数；界面字号、粗略取景、MetDet 和最近目录不在这里出现。
EDITABLE_PREFERENCE_KEYS = frozenset(
    {
        "direction_label_font_size_pt",
        "star_name_font_size_pt",
        "constellation_name_font_size_pt",
        "reference_label_font_size_pt",
        "show_constellation_names",
        "show_constellation_lines",
        "base_star_marker_radius_px",
        "star_marker_size_multiplier",
        "star_color_mag_limit",
        "aligned_reference_scale_multiplier",
        "auto_sync_simulator_time_from_exif",
        "constellation_line_width_px",
        "constellation_line_color_hex",
        "constellation_line_opacity",
        "star_pick_circle_default_diameter_px",
        "star_pick_circle_min_diameter_px",
        "star_pick_circle_max_diameter_px",
        "star_pick_psf_max_radius_px",
        "star_pick_psf_fit_error_limit",
        "star_pick_saturated_psf_fit_error_limit",
        "star_pair_psf_outer_diameter_multiplier",
        "auto_pair_search_rms_multiplier",
        "auto_pair_search_base_radius_px",
        "auto_pair_search_max_radius_px",
        "double_click_focus_auto_pair_enabled",
        "star_pair_assistant_always_on_top",
        "default_latitude_deg",
        "default_longitude_deg",
        "default_elevation_m",
        "auto_match_default_new_count",
        "auto_match_default_constraint_mode",
        "auto_match_default_soft_weight",
        "auto_match_default_search_radius_px",
        "sequence_psf_search_radius_px",
        "wheel_zoom_enabled",
        "touchpad_pinch_zoom_enabled",
        "mosaic_texture_scale_percent",
        "mosaic_texture_max_long_side_px",
        "mosaic_grid_precision_default",
        "mosaic_render_fps_limit",
        "mosaic_font_size_multiplier",
        "mosaic_star_marker_size_multiplier",
        "mosaic_export_block_rows",
        "mosaic_map_tile_size_px",
        "mosaic_export_tiff_lzw_compression",
        "show_meteor_showers",
        "meteor_radiant_only",
        "meteor_radiant_label_font_size_pt",
        "meteor_count_multiplier",
        "meteor_max_length_deg",
        "meteor_min_length_deg",
        "meteor_opacity",
        "meteor_thickness_ratio",
        "meteor_random_seed",
        "selected_meteor_shower_ids",
    }
)

# 这些参数只定义后续任务或下次启动的初始值，不参与当前会话热更新。
DEFAULT_ONLY_PREFERENCE_KEYS = frozenset(
    {
        "star_pick_circle_default_diameter_px",
        "star_pair_assistant_always_on_top",
        "default_latitude_deg",
        "default_longitude_deg",
        "default_elevation_m",
        "auto_match_default_new_count",
        "auto_match_default_constraint_mode",
        "auto_match_default_soft_weight",
        "auto_match_default_search_radius_px",
        "sequence_psf_search_radius_px",
        "mosaic_grid_precision_default",
    }
)


class PreferencesPage(QWidget):
    """读取、编辑、应用并保存普通软件参数的标签页。"""

    preferences_applied = pyqtSignal(object)
    preferences_saved = pyqtSignal(object)
    close_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None, *, preference_path: Path | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_PreferencesPage()
        self.ui.setupUi(self)
        self._preference_path = preference_path
        self._populating_controls = False
        self._selected_meteor_shower_ids: tuple[str, ...] = ()
        self.ui.pushButtonChooseConstellationColor.clicked.connect(self._choose_constellation_color)
        self.ui.pushButtonSelectMeteorShowers.clicked.connect(self._choose_meteor_showers)
        self.ui.pushButtonReadPreferences.clicked.connect(self.read_preferences)
        self.ui.pushButtonSavePreferences.clicked.connect(self.save_preferences)
        self.ui.pushButtonClosePreferences.clicked.connect(self.close_requested.emit)
        self.ui.comboBoxAutoMatchDefaultConstraintMode.currentIndexChanged.connect(
            self._update_auto_match_soft_weight_state
        )
        self.ui.checkBoxShowConstellationLines.toggled.connect(self._update_constellation_line_controls_state)
        self.ui.lineEditConstellationLineColor.textChanged.connect(self._update_color_preview)
        self._persisted_config = load_star_map_ui_config(self._preference_path)
        self._runtime_config = self._persisted_config
        self._populate_controls(self._persisted_config)
        self._connect_immediate_apply_controls()

    def read_preferences(self) -> None:
        """从磁盘重读配置，并把其中非默认参数应用到当前会话。"""

        config = load_star_map_ui_config(self._preference_path)
        self._persisted_config = config
        self._populate_controls(config)
        self.apply_preferences()

    def reload_preferences(self) -> None:
        """兼容既有调用：重新读取磁盘配置。"""

        self.read_preferences()

    def _populate_controls(self, config: StarMapUiConfig) -> None:
        """把经过边界校验的配置值显示到对应控件。"""

        was_populating = self._populating_controls
        self._populating_controls = True
        ui = self.ui
        try:
            ui.spinBoxDirectionLabelFontSize.setValue(config.direction_label_font_size_pt)
            ui.spinBoxStarNameFontSize.setValue(config.star_name_font_size_pt)
            ui.spinBoxConstellationNameFontSize.setValue(config.constellation_name_font_size_pt)
            ui.spinBoxReferenceLabelFontSize.setValue(config.reference_label_font_size_pt)
            ui.checkBoxShowConstellationNames.setChecked(config.show_constellation_names)
            ui.checkBoxShowConstellationLines.setChecked(config.show_constellation_lines)
            ui.doubleSpinBoxBaseStarMarkerRadius.setValue(config.base_star_marker_radius_px)
            ui.doubleSpinBoxStarMarkerSizeMultiplier.setValue(config.star_marker_size_multiplier)
            ui.doubleSpinBoxStarColorMagLimit.setValue(config.star_color_mag_limit)
            ui.doubleSpinBoxAlignedReferenceScale.setValue(config.aligned_reference_scale_multiplier)
            ui.checkBoxAutoSyncSimulatorTimeFromExif.setChecked(
                config.auto_sync_simulator_time_from_exif
            )
            ui.doubleSpinBoxConstellationLineWidth.setValue(config.constellation_line_width_px)
            ui.lineEditConstellationLineColor.setText(config.constellation_line_color_hex)
            ui.doubleSpinBoxConstellationLineOpacity.setValue(config.constellation_line_opacity)
            ui.spinBoxStarPickDefaultDiameter.setValue(config.star_pick_circle_default_diameter_px)
            ui.spinBoxStarPickMinDiameter.setValue(config.star_pick_circle_min_diameter_px)
            ui.spinBoxStarPickMaxDiameter.setValue(config.star_pick_circle_max_diameter_px)
            ui.spinBoxStarPickPsfMaxRadius.setValue(config.star_pick_psf_max_radius_px)
            ui.doubleSpinBoxStarPickPsfFitErrorLimit.setValue(
                config.star_pick_psf_fit_error_limit
            )
            ui.doubleSpinBoxStarPickSaturatedPsfFitErrorLimit.setValue(
                config.star_pick_saturated_psf_fit_error_limit
            )
            ui.doubleSpinBoxStarPairPsfOuterDiameterMultiplier.setValue(
                config.star_pair_psf_outer_diameter_multiplier
            )
            ui.doubleSpinBoxAutoPairSearchRmsMultiplier.setValue(config.auto_pair_search_rms_multiplier)
            ui.spinBoxAutoPairSearchBaseRadius.setValue(config.auto_pair_search_base_radius_px)
            ui.spinBoxAutoPairSearchMaxRadius.setValue(config.auto_pair_search_max_radius_px)
            ui.checkBoxDoubleClickFocusAutoPairEnabled.setChecked(
                config.double_click_focus_auto_pair_enabled
            )
            ui.checkBoxStarPairAssistantAlwaysOnTopDefault.setChecked(
                config.star_pair_assistant_always_on_top
            )
            ui.doubleSpinBoxDefaultLatitude.setValue(config.default_latitude_deg)
            ui.doubleSpinBoxDefaultLongitude.setValue(config.default_longitude_deg)
            ui.doubleSpinBoxDefaultElevation.setValue(config.default_elevation_m)
            ui.spinBoxAutoMatchDefaultNewCount.setValue(config.auto_match_default_new_count)
            self._set_combo_data(ui.comboBoxAutoMatchDefaultConstraintMode, config.auto_match_default_constraint_mode)
            ui.doubleSpinBoxAutoMatchDefaultSoftWeight.setValue(config.auto_match_default_soft_weight)
            ui.spinBoxAutoMatchDefaultSearchRadius.setValue(config.auto_match_default_search_radius_px)
            ui.spinBoxSequencePsfSearchRadiusDefault.setValue(config.sequence_psf_search_radius_px)
            ui.checkBoxWheelZoomEnabled.setChecked(config.wheel_zoom_enabled)
            ui.checkBoxTouchpadPinchZoomEnabled.setChecked(config.touchpad_pinch_zoom_enabled)
            ui.doubleSpinBoxMosaicTextureScalePercent.setValue(config.mosaic_texture_scale_percent)
            ui.spinBoxMosaicTextureMaxLongSide.setValue(config.mosaic_texture_max_long_side_px)
            ui.spinBoxMosaicGridPrecisionDefault.setValue(config.mosaic_grid_precision_default)
            ui.spinBoxMosaicRenderFpsLimit.setValue(config.mosaic_render_fps_limit)
            ui.doubleSpinBoxMosaicFontSizeMultiplier.setValue(config.mosaic_font_size_multiplier)
            ui.doubleSpinBoxMosaicStarMarkerSizeMultiplier.setValue(config.mosaic_star_marker_size_multiplier)
            ui.spinBoxMosaicExportBlockRows.setValue(config.mosaic_export_block_rows)
            ui.spinBoxMosaicMapTileSize.setValue(config.mosaic_map_tile_size_px)
            ui.checkBoxMosaicTiffLzwCompression.setChecked(config.mosaic_export_tiff_lzw_compression)
            ui.checkBoxShowMeteorShowers.setChecked(config.show_meteor_showers)
            ui.checkBoxMeteorRadiantOnly.setChecked(config.meteor_radiant_only)
            ui.spinBoxMeteorRadiantLabelFontSize.setValue(config.meteor_radiant_label_font_size_pt)
            ui.doubleSpinBoxMeteorCountMultiplier.setValue(config.meteor_count_multiplier)
            ui.doubleSpinBoxMeteorMaxLength.setValue(config.meteor_max_length_deg)
            ui.doubleSpinBoxMeteorMinLength.setValue(config.meteor_min_length_deg)
            ui.doubleSpinBoxMeteorOpacity.setValue(config.meteor_opacity)
            ui.doubleSpinBoxMeteorThicknessRatio.setValue(config.meteor_thickness_ratio)
            ui.spinBoxMeteorRandomSeed.setValue(config.meteor_random_seed)
            self._selected_meteor_shower_ids = config.selected_meteor_shower_ids
            self._update_selected_meteor_shower_label()
            self._update_auto_match_soft_weight_state()
            self._update_constellation_line_controls_state()
            self._update_meteor_controls_state()
        finally:
            self._populating_controls = was_populating

    def _connect_immediate_apply_controls(self) -> None:
        """让所有非默认参数在控件变化后立即应用。"""

        ui = self.ui
        value_controls = (
            ui.spinBoxDirectionLabelFontSize,
            ui.spinBoxStarNameFontSize,
            ui.spinBoxConstellationNameFontSize,
            ui.spinBoxReferenceLabelFontSize,
            ui.doubleSpinBoxBaseStarMarkerRadius,
            ui.doubleSpinBoxStarMarkerSizeMultiplier,
            ui.doubleSpinBoxStarColorMagLimit,
            ui.doubleSpinBoxAlignedReferenceScale,
            ui.doubleSpinBoxConstellationLineWidth,
            ui.doubleSpinBoxConstellationLineOpacity,
            ui.spinBoxStarPickMinDiameter,
            ui.spinBoxStarPickMaxDiameter,
            ui.spinBoxStarPickPsfMaxRadius,
            ui.doubleSpinBoxStarPickPsfFitErrorLimit,
            ui.doubleSpinBoxStarPickSaturatedPsfFitErrorLimit,
            ui.doubleSpinBoxStarPairPsfOuterDiameterMultiplier,
            ui.doubleSpinBoxAutoPairSearchRmsMultiplier,
            ui.spinBoxAutoPairSearchBaseRadius,
            ui.spinBoxAutoPairSearchMaxRadius,
            ui.doubleSpinBoxMosaicTextureScalePercent,
            ui.spinBoxMosaicTextureMaxLongSide,
            ui.spinBoxMosaicRenderFpsLimit,
            ui.doubleSpinBoxMosaicFontSizeMultiplier,
            ui.doubleSpinBoxMosaicStarMarkerSizeMultiplier,
            ui.spinBoxMosaicExportBlockRows,
            ui.spinBoxMosaicMapTileSize,
            ui.spinBoxMeteorRadiantLabelFontSize,
            ui.doubleSpinBoxMeteorCountMultiplier,
            ui.doubleSpinBoxMeteorMaxLength,
            ui.doubleSpinBoxMeteorMinLength,
            ui.doubleSpinBoxMeteorOpacity,
            ui.doubleSpinBoxMeteorThicknessRatio,
            ui.spinBoxMeteorRandomSeed,
        )
        for control in value_controls:
            control.valueChanged.connect(self._apply_immediate_preferences)
        for control in (
            ui.checkBoxShowConstellationNames,
            ui.checkBoxShowConstellationLines,
            ui.checkBoxAutoSyncSimulatorTimeFromExif,
            ui.checkBoxWheelZoomEnabled,
            ui.checkBoxTouchpadPinchZoomEnabled,
            ui.checkBoxDoubleClickFocusAutoPairEnabled,
            ui.checkBoxMosaicTiffLzwCompression,
            ui.checkBoxShowMeteorShowers,
            ui.checkBoxMeteorRadiantOnly,
        ):
            control.toggled.connect(self._apply_immediate_preferences)
        ui.checkBoxShowMeteorShowers.toggled.connect(self._update_meteor_controls_state)
        ui.checkBoxMeteorRadiantOnly.toggled.connect(self._update_meteor_controls_state)
        ui.lineEditConstellationLineColor.textChanged.connect(self._apply_immediate_preferences)

    def _apply_immediate_preferences(self, *unused: object) -> None:
        """静默应用有效的非默认参数，输入暂时无效时等待下一次变化。"""

        if self._populating_controls:
            return
        self.apply_preferences(show_errors=False)

    @staticmethod
    def _set_combo_data(combo_box, value: str) -> None:  # type: ignore[no-untyped-def]
        """按配置代码选择约束模式。"""

        combo_box.setCurrentIndex(0 if value == "anchor" else 1)

    def _update_auto_match_soft_weight_state(self, *unused) -> None:  # type: ignore[no-untyped-def]
        """只有软约束模式需要编辑权重。"""

        is_soft = self.ui.comboBoxAutoMatchDefaultConstraintMode.currentIndex() == 1
        self.ui.labelAutoMatchSoftWeight.setEnabled(is_soft)
        self.ui.doubleSpinBoxAutoMatchDefaultSoftWeight.setEnabled(is_soft)

    def _update_constellation_line_controls_state(self, *unused) -> None:  # type: ignore[no-untyped-def]
        """隐藏星座连线时禁用仅影响连线样式的控件。"""

        enabled = self.ui.checkBoxShowConstellationLines.isChecked()
        for widget in (
            self.ui.labelConstellationLineWidth,
            self.ui.doubleSpinBoxConstellationLineWidth,
            self.ui.labelConstellationLineColor,
            self.ui.widgetConstellationColor,
            self.ui.labelConstellationOpacity,
            self.ui.doubleSpinBoxConstellationLineOpacity,
        ):
            widget.setEnabled(enabled)

    def _choose_constellation_color(self) -> None:
        """打开系统颜色选择器并写回标准十六进制颜色。"""

        initial = QColor(self.ui.lineEditConstellationLineColor.text().strip())
        if not initial.isValid():
            initial = QColor("#E6E6E6")
        selected = QColorDialog.getColor(initial, self, "选择星座连线颜色")
        if selected.isValid():
            self.ui.lineEditConstellationLineColor.setText(selected.name().upper())

    def _choose_meteor_showers(self) -> None:
        """打开全年流星雨多选窗口，并立即应用选择结果。"""

        dialog = MeteorShowerSelectionDialog(
            self,
            selected_ids=self._selected_meteor_shower_ids,
        )
        if dialog.exec_() != dialog.Accepted:
            return
        self._selected_meteor_shower_ids = dialog.selected_ids()
        self._update_selected_meteor_shower_label()
        self._apply_immediate_preferences()

    def _update_selected_meteor_shower_label(self) -> None:
        """用简短名称显示当前选择，完整信息留在选择窗口中。"""

        names = [
            METEOR_SHOWER_BY_ID[shower_id].chinese_name
            for shower_id in self._selected_meteor_shower_ids
            if shower_id in METEOR_SHOWER_BY_ID
        ]
        self.ui.labelSelectedMeteorShowers.setText(
            f"已选 {len(names)} 个：{'、'.join(names)}" if names else "尚未选择流星雨"
        )

    def _update_meteor_controls_state(self, *unused) -> None:  # type: ignore[no-untyped-def]
        """按总开关与辐射点模式启用当前有效的流星雨参数。"""

        enabled = self.ui.checkBoxShowMeteorShowers.isChecked()
        radiant_only = self.ui.checkBoxMeteorRadiantOnly.isChecked()
        for widget in (
            self.ui.labelMeteorOpacity,
            self.ui.doubleSpinBoxMeteorOpacity,
            self.ui.checkBoxMeteorRadiantOnly,
            self.ui.pushButtonSelectMeteorShowers,
            self.ui.labelSelectedMeteorShowers,
        ):
            widget.setEnabled(enabled)
        for widget in (
            self.ui.labelMeteorCountMultiplier,
            self.ui.doubleSpinBoxMeteorCountMultiplier,
            self.ui.labelMeteorMinLength,
            self.ui.doubleSpinBoxMeteorMinLength,
            self.ui.labelMeteorMaxLength,
            self.ui.doubleSpinBoxMeteorMaxLength,
            self.ui.labelMeteorThicknessRatio,
            self.ui.doubleSpinBoxMeteorThicknessRatio,
            self.ui.labelMeteorRandomSeed,
            self.ui.spinBoxMeteorRandomSeed,
        ):
            widget.setEnabled(enabled and not radiant_only)
        for widget in (
            self.ui.labelMeteorRadiantLabelFontSize,
            self.ui.spinBoxMeteorRadiantLabelFontSize,
        ):
            widget.setEnabled(enabled and radiant_only)

    def _update_color_preview(self, color_text: str) -> None:
        """在颜色输入框左侧显示当前有效颜色。"""

        normalized = color_text.strip().upper()
        if re.fullmatch(r"#[0-9A-F]{6}", normalized):
            self.ui.lineEditConstellationLineColor.setStyleSheet(
                f"QLineEdit {{ border-left: 18px solid {normalized}; }}"
            )
        else:
            self.ui.lineEditConstellationLineColor.setStyleSheet("")

    def _control_values(self) -> dict[str, object]:
        """收集本页所有可编辑项，键集合必须与公开清单完全一致。"""

        ui = self.ui
        return {
            "direction_label_font_size_pt": ui.spinBoxDirectionLabelFontSize.value(),
            "star_name_font_size_pt": ui.spinBoxStarNameFontSize.value(),
            "constellation_name_font_size_pt": ui.spinBoxConstellationNameFontSize.value(),
            "reference_label_font_size_pt": ui.spinBoxReferenceLabelFontSize.value(),
            "show_constellation_names": ui.checkBoxShowConstellationNames.isChecked(),
            "show_constellation_lines": ui.checkBoxShowConstellationLines.isChecked(),
            "base_star_marker_radius_px": ui.doubleSpinBoxBaseStarMarkerRadius.value(),
            "star_marker_size_multiplier": ui.doubleSpinBoxStarMarkerSizeMultiplier.value(),
            "star_color_mag_limit": ui.doubleSpinBoxStarColorMagLimit.value(),
            "aligned_reference_scale_multiplier": ui.doubleSpinBoxAlignedReferenceScale.value(),
            "auto_sync_simulator_time_from_exif": (
                ui.checkBoxAutoSyncSimulatorTimeFromExif.isChecked()
            ),
            "constellation_line_width_px": ui.doubleSpinBoxConstellationLineWidth.value(),
            "constellation_line_color_hex": ui.lineEditConstellationLineColor.text().strip().upper(),
            "constellation_line_opacity": ui.doubleSpinBoxConstellationLineOpacity.value(),
            "star_pick_circle_default_diameter_px": ui.spinBoxStarPickDefaultDiameter.value(),
            "star_pick_circle_min_diameter_px": ui.spinBoxStarPickMinDiameter.value(),
            "star_pick_circle_max_diameter_px": ui.spinBoxStarPickMaxDiameter.value(),
            "star_pick_psf_max_radius_px": ui.spinBoxStarPickPsfMaxRadius.value(),
            "star_pick_psf_fit_error_limit": ui.doubleSpinBoxStarPickPsfFitErrorLimit.value(),
            "star_pick_saturated_psf_fit_error_limit": (
                ui.doubleSpinBoxStarPickSaturatedPsfFitErrorLimit.value()
            ),
            "star_pair_psf_outer_diameter_multiplier": (
                ui.doubleSpinBoxStarPairPsfOuterDiameterMultiplier.value()
            ),
            "auto_pair_search_rms_multiplier": ui.doubleSpinBoxAutoPairSearchRmsMultiplier.value(),
            "auto_pair_search_base_radius_px": ui.spinBoxAutoPairSearchBaseRadius.value(),
            "auto_pair_search_max_radius_px": ui.spinBoxAutoPairSearchMaxRadius.value(),
            "double_click_focus_auto_pair_enabled": (
                ui.checkBoxDoubleClickFocusAutoPairEnabled.isChecked()
            ),
            "star_pair_assistant_always_on_top": (
                ui.checkBoxStarPairAssistantAlwaysOnTopDefault.isChecked()
            ),
            "default_latitude_deg": ui.doubleSpinBoxDefaultLatitude.value(),
            "default_longitude_deg": ui.doubleSpinBoxDefaultLongitude.value(),
            "default_elevation_m": ui.doubleSpinBoxDefaultElevation.value(),
            "auto_match_default_new_count": ui.spinBoxAutoMatchDefaultNewCount.value(),
            "auto_match_default_constraint_mode": (
                "anchor" if ui.comboBoxAutoMatchDefaultConstraintMode.currentIndex() == 0 else "soft"
            ),
            "auto_match_default_soft_weight": ui.doubleSpinBoxAutoMatchDefaultSoftWeight.value(),
            "auto_match_default_search_radius_px": ui.spinBoxAutoMatchDefaultSearchRadius.value(),
            "sequence_psf_search_radius_px": ui.spinBoxSequencePsfSearchRadiusDefault.value(),
            "wheel_zoom_enabled": ui.checkBoxWheelZoomEnabled.isChecked(),
            "touchpad_pinch_zoom_enabled": ui.checkBoxTouchpadPinchZoomEnabled.isChecked(),
            "mosaic_texture_scale_percent": ui.doubleSpinBoxMosaicTextureScalePercent.value(),
            "mosaic_texture_max_long_side_px": ui.spinBoxMosaicTextureMaxLongSide.value(),
            "mosaic_grid_precision_default": ui.spinBoxMosaicGridPrecisionDefault.value(),
            "mosaic_render_fps_limit": ui.spinBoxMosaicRenderFpsLimit.value(),
            "mosaic_font_size_multiplier": ui.doubleSpinBoxMosaicFontSizeMultiplier.value(),
            "mosaic_star_marker_size_multiplier": ui.doubleSpinBoxMosaicStarMarkerSizeMultiplier.value(),
            "mosaic_export_block_rows": ui.spinBoxMosaicExportBlockRows.value(),
            "mosaic_map_tile_size_px": ui.spinBoxMosaicMapTileSize.value(),
            "mosaic_export_tiff_lzw_compression": ui.checkBoxMosaicTiffLzwCompression.isChecked(),
            "show_meteor_showers": ui.checkBoxShowMeteorShowers.isChecked(),
            "meteor_radiant_only": ui.checkBoxMeteorRadiantOnly.isChecked(),
            "meteor_radiant_label_font_size_pt": ui.spinBoxMeteorRadiantLabelFontSize.value(),
            "meteor_count_multiplier": ui.doubleSpinBoxMeteorCountMultiplier.value(),
            "meteor_max_length_deg": ui.doubleSpinBoxMeteorMaxLength.value(),
            "meteor_min_length_deg": ui.doubleSpinBoxMeteorMinLength.value(),
            "meteor_opacity": ui.doubleSpinBoxMeteorOpacity.value(),
            "meteor_thickness_ratio": ui.doubleSpinBoxMeteorThicknessRatio.value(),
            "meteor_random_seed": ui.spinBoxMeteorRandomSeed.value(),
            "selected_meteor_shower_ids": tuple(self._selected_meteor_shower_ids),
        }

    def _validation_error(
        self,
        values: dict[str, object],
        *,
        validate_default_value: bool = True,
    ) -> str | None:
        """校验跨控件约束以及无法由数值控件表达的颜色格式。"""

        color_text = str(values["constellation_line_color_hex"])
        if re.fullmatch(r"#[0-9A-F]{6}", color_text) is None:
            return "星座连线颜色必须使用 #RRGGBB 格式，例如 #E6E6E6。"
        minimum = int(values["star_pick_circle_min_diameter_px"])
        maximum = int(values["star_pick_circle_max_diameter_px"])
        if minimum > maximum:
            return "星点点选圆圈的最小直径不能大于最大直径。"
        default = int(values["star_pick_circle_default_diameter_px"])
        if validate_default_value and not minimum <= default <= maximum:
            return "星点点选圆圈的默认直径必须位于最小值和最大值之间。"
        if float(values["meteor_min_length_deg"]) > float(values["meteor_max_length_deg"]):
            return "流星最短长度不能大于最长长度。"
        return None

    def _validated_control_values(
        self,
        *,
        validate_default_value: bool,
        show_errors: bool = True,
    ) -> dict[str, object] | None:
        """收集并校验控件值，校验失败时直接提示用户。"""

        values = self._control_values()
        if set(values) != set(EDITABLE_PREFERENCE_KEYS):
            raise RuntimeError("软件参数页面的控件映射与可编辑键清单不一致。")
        error_message = self._validation_error(values, validate_default_value=validate_default_value)
        if error_message and show_errors:
            QMessageBox.warning(self, "参数无效", error_message)
        if error_message:
            return None
        return values

    def apply_preferences(self, *, show_errors: bool = True) -> None:
        """只把可热更新参数应用到当前会话，不写入配置文件。"""

        values = self._validated_control_values(
            validate_default_value=False,
            show_errors=show_errors,
        )
        if values is None:
            return

        immediate_values = {
            key: value
            for key, value in values.items()
            if key not in DEFAULT_ONLY_PREFERENCE_KEYS
        }
        minimum = int(values["star_pick_circle_min_diameter_px"])
        maximum = int(values["star_pick_circle_max_diameter_px"])
        runtime_default_diameter = min(
            max(self._runtime_config.star_pick_circle_default_diameter_px, minimum),
            maximum,
        )
        config = replace(
            self._runtime_config,
            **immediate_values,
            star_pick_circle_default_diameter_px=runtime_default_diameter,
        )
        self._runtime_config = config
        self.preferences_applied.emit(config)

    def save_preferences(self) -> None:
        """保存本页参数并重新读取规范值，但不改变当前会话配置。"""

        values = self._validated_control_values(validate_default_value=True)
        if values is None:
            return
        if not update_preference_values(values, path=self._preference_path):
            QMessageBox.critical(self, "保存参数失败", "无法写入 preference.json，请检查配置目录写入权限。")
            return

        config = load_star_map_ui_config(self._preference_path)
        self._persisted_config = config
        self._populate_controls(config)
        self.preferences_saved.emit(config)


__all__ = ["DEFAULT_ONLY_PREFERENCE_KEYS", "EDITABLE_PREFERENCE_KEYS", "PreferencesPage"]
