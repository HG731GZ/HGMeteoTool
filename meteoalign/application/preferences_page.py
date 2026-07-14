from __future__ import annotations

import re
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog, QMessageBox, QWidget

from ..config import StarMapUiConfig, load_star_map_ui_config
from ..preference_manager import update_preference_values
from ..ui.ui_preferences_page import Ui_PreferencesPage


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
        "constellation_line_width_px",
        "constellation_line_color_hex",
        "constellation_line_opacity",
        "star_pick_circle_default_diameter_px",
        "star_pick_circle_min_diameter_px",
        "star_pick_circle_max_diameter_px",
        "star_pick_psf_radius_scale",
        "star_pick_psf_max_radius_px",
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
    }
)


class PreferencesPage(QWidget):
    """读取、编辑并保存普通软件参数的标签页。"""

    preferences_saved = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None, *, preference_path: Path | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_PreferencesPage()
        self.ui.setupUi(self)
        self._preference_path = preference_path
        self.ui.pushButtonChooseConstellationColor.clicked.connect(self._choose_constellation_color)
        self.ui.pushButtonSavePreferences.clicked.connect(self.save_preferences)
        self.ui.pushButtonCancelPreferences.clicked.connect(self.reload_preferences)
        self.ui.comboBoxAutoMatchDefaultConstraintMode.currentIndexChanged.connect(
            self._update_auto_match_soft_weight_state
        )
        self.ui.checkBoxShowConstellationLines.toggled.connect(self._update_constellation_line_controls_state)
        self.ui.lineEditConstellationLineColor.textChanged.connect(self._update_color_preview)
        self.reload_preferences()

    def reload_preferences(self) -> None:
        """丢弃未保存改动，并从磁盘重新加载所有控件。"""

        config = load_star_map_ui_config(self._preference_path)
        self._populate_controls(config)

    def _populate_controls(self, config: StarMapUiConfig) -> None:
        """把经过边界校验的配置值显示到对应控件。"""

        ui = self.ui
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
        ui.doubleSpinBoxConstellationLineWidth.setValue(config.constellation_line_width_px)
        ui.lineEditConstellationLineColor.setText(config.constellation_line_color_hex)
        ui.doubleSpinBoxConstellationLineOpacity.setValue(config.constellation_line_opacity)
        ui.spinBoxStarPickDefaultDiameter.setValue(config.star_pick_circle_default_diameter_px)
        ui.spinBoxStarPickMinDiameter.setValue(config.star_pick_circle_min_diameter_px)
        ui.spinBoxStarPickMaxDiameter.setValue(config.star_pick_circle_max_diameter_px)
        ui.doubleSpinBoxStarPickPsfRadiusScale.setValue(config.star_pick_psf_radius_scale)
        ui.spinBoxStarPickPsfMaxRadius.setValue(config.star_pick_psf_max_radius_px)
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
        self._update_auto_match_soft_weight_state()
        self._update_constellation_line_controls_state()

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
            "constellation_line_width_px": ui.doubleSpinBoxConstellationLineWidth.value(),
            "constellation_line_color_hex": ui.lineEditConstellationLineColor.text().strip().upper(),
            "constellation_line_opacity": ui.doubleSpinBoxConstellationLineOpacity.value(),
            "star_pick_circle_default_diameter_px": ui.spinBoxStarPickDefaultDiameter.value(),
            "star_pick_circle_min_diameter_px": ui.spinBoxStarPickMinDiameter.value(),
            "star_pick_circle_max_diameter_px": ui.spinBoxStarPickMaxDiameter.value(),
            "star_pick_psf_radius_scale": ui.doubleSpinBoxStarPickPsfRadiusScale.value(),
            "star_pick_psf_max_radius_px": ui.spinBoxStarPickPsfMaxRadius.value(),
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
        }

    def _validation_error(self, values: dict[str, object]) -> str | None:
        """校验跨控件约束以及无法由数值控件表达的颜色格式。"""

        color_text = str(values["constellation_line_color_hex"])
        if re.fullmatch(r"#[0-9A-F]{6}", color_text) is None:
            return "星座连线颜色必须使用 #RRGGBB 格式，例如 #E6E6E6。"
        minimum = int(values["star_pick_circle_min_diameter_px"])
        default = int(values["star_pick_circle_default_diameter_px"])
        maximum = int(values["star_pick_circle_max_diameter_px"])
        if minimum > maximum:
            return "星点点选圆圈的最小直径不能大于最大直径。"
        if not minimum <= default <= maximum:
            return "星点点选圆圈的默认直径必须位于最小值和最大值之间。"
        return None

    def save_preferences(self) -> None:
        """保存本页参数，重新读取规范值并通知主窗口热更新。"""

        values = self._control_values()
        if set(values) != set(EDITABLE_PREFERENCE_KEYS):
            raise RuntimeError("软件参数页面的控件映射与可编辑键清单不一致。")
        error_message = self._validation_error(values)
        if error_message:
            QMessageBox.warning(self, "参数无效", error_message)
            return
        if not update_preference_values(values, path=self._preference_path):
            QMessageBox.critical(self, "保存参数失败", "无法写入 preference.json，请检查配置目录写入权限。")
            return

        config = load_star_map_ui_config(self._preference_path)
        self._populate_controls(config)
        self.preferences_saved.emit(config)


__all__ = ["EDITABLE_PREFERENCE_KEYS", "PreferencesPage"]
