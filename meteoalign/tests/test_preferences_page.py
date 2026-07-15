"""软件选项弹窗的配置范围、即时应用、读写与关闭行为测试。"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QGroupBox
import pytest

from meteoalign.application.preferences_page import (
    DEFAULT_ONLY_PREFERENCE_KEYS,
    EDITABLE_PREFERENCE_KEYS,
    PreferencesPage,
)
from meteoalign.application.preferences_dialog import PreferencesDialog, PreferencesLauncher
from meteoalign.application.main_window import MainWindow
from meteoalign.config import StarMapUiConfig
from meteoalign.preference_manager import (
    DEFAULT_PREFERENCE_VALUES,
    LAST_IMPORT_DIRECTORY_KEY,
    strip_json_comments,
)
from meteoalign.renderer import StarMapRenderer


def _read_jsonc(path) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return json.loads(strip_json_comments(path.read_text(encoding="utf-8")))


def test_editable_keys_include_all_general_preferences_and_exclude_dedicated_settings() -> None:
    """普通参数页必须完整覆盖范围，同时避开题目明确排除的专用配置。"""

    excluded = {
        "controls_font_size_pt",
        "status_bar_font_size_pt",
        LAST_IMPORT_DIRECTORY_KEY,
    }
    excluded.update(key for key in DEFAULT_PREFERENCE_VALUES if key.startswith("adjacent_"))
    excluded.update(key for key in DEFAULT_PREFERENCE_VALUES if key.startswith("meteor_detection_"))

    assert EDITABLE_PREFERENCE_KEYS == set(DEFAULT_PREFERENCE_VALUES) - excluded


def test_page_groups_controls_and_saves_without_touching_excluded_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """保存只更新本页参数，粗略取景、MetDet、字体和最近目录必须保留。"""

    app = QApplication.instance() or QApplication([])
    preference_path = tmp_path / "preference.json"
    preference_path.write_text(
        json.dumps(
            {
                "controls_font_size_pt": 19,
                "adjacent_alignment_max_correspondences": 88,
                "meteor_detection_provider": "cpu",
                "auto_match_default_search_radius_px": 65,
                "sequence_psf_search_radius_px": 55,
                LAST_IMPORT_DIRECTORY_KEY: "/keep/me",
                "star_name_font_size_pt": 14,
            }
        ),
        encoding="utf-8",
    )
    page = PreferencesPage(preference_path=preference_path)
    emitted = []
    applied = []
    page.preferences_saved.connect(emitted.append)
    page.preferences_applied.connect(applied.append)

    assert len(page.findChildren(QGroupBox)) >= 6
    assert page.ui.pushButtonReadPreferences.text() == "读取配置"
    assert page.ui.pushButtonSavePreferences.text() == "保存配置"
    assert page.ui.pushButtonClosePreferences.text() == "关闭"
    page.ui.spinBoxStarNameFontSize.setValue(20)
    assert applied and applied[-1].star_name_font_size_pt == 20
    applied_count_before_default_change = len(applied)
    page.ui.doubleSpinBoxDefaultLatitude.setValue(35.5)
    assert len(applied) == applied_count_before_default_change
    applied_count_before_save = len(applied)
    page.save_preferences()

    written = _read_jsonc(preference_path)
    assert written["star_name_font_size_pt"] == 20
    assert written["default_latitude_deg"] == 35.5
    assert written["controls_font_size_pt"] == 19
    assert written["adjacent_alignment_max_correspondences"] == 88
    assert written["meteor_detection_provider"] == "cpu"
    assert written["auto_match_default_search_radius_px"] == 65
    assert written["sequence_psf_search_radius_px"] == 55
    assert written[LAST_IMPORT_DIRECTORY_KEY] == "/keep/me"
    assert emitted and emitted[-1].star_name_font_size_pt == 20
    assert len(applied) == applied_count_before_save

    page.ui.spinBoxStarNameFontSize.setValue(9)
    page.read_preferences()
    assert page.ui.spinBoxStarNameFontSize.value() == 20
    assert applied[-1].star_name_font_size_pt == 20
    assert applied[-1].default_latitude_deg == 40.0
    page.close()
    app.processEvents()


def test_non_default_changes_apply_immediately_without_writing_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """非默认控件变化应立即发出内存配置，默认控件和值文件保持不变。"""

    app = QApplication.instance() or QApplication([])
    preference_path = tmp_path / "preference.json"
    preference_path.write_text(
        json.dumps(
            {
                "star_name_font_size_pt": 14,
                "default_latitude_deg": 40.0,
                "auto_match_default_search_radius_px": 65,
            }
        ),
        encoding="utf-8",
    )
    page = PreferencesPage(preference_path=preference_path)
    before_apply = preference_path.read_bytes()
    applied = []
    saved = []
    page.preferences_applied.connect(applied.append)
    page.preferences_saved.connect(saved.append)

    page.ui.spinBoxStarNameFontSize.setValue(20)
    assert len(applied) == 1
    applied_count_before_defaults = len(applied)
    page.ui.doubleSpinBoxDefaultLatitude.setValue(35.5)
    page.ui.spinBoxAutoMatchDefaultSearchRadius.setValue(80)

    assert preference_path.read_bytes() == before_apply
    assert len(applied) == applied_count_before_defaults
    assert not saved
    assert applied[0].star_name_font_size_pt == 20
    assert applied[0].default_latitude_deg == 40.0
    assert applied[0].auto_match_default_search_radius_px == 65
    page.close()
    app.processEvents()


def test_radiant_only_controls_switch_state_and_are_saved(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """辐射点模式应禁用轨迹参数、启用标注字号，并把两项写入 JSON。"""

    app = QApplication.instance() or QApplication([])
    preference_path = tmp_path / "preference.json"
    page = PreferencesPage(preference_path=preference_path)

    assert not page.ui.checkBoxShowMeteorShowers.isChecked()
    assert not page.ui.checkBoxMeteorRadiantOnly.isEnabled()
    assert not page.ui.spinBoxMeteorRadiantLabelFontSize.isEnabled()
    page.ui.checkBoxShowMeteorShowers.setChecked(True)
    assert page.ui.checkBoxMeteorRadiantOnly.isEnabled()
    assert not page.ui.spinBoxMeteorRadiantLabelFontSize.isEnabled()
    page.ui.checkBoxMeteorRadiantOnly.setChecked(True)
    assert page.ui.spinBoxMeteorRadiantLabelFontSize.isEnabled()
    assert not page.ui.doubleSpinBoxMeteorCountMultiplier.isEnabled()
    assert page.ui.doubleSpinBoxMeteorOpacity.isEnabled()
    page.ui.spinBoxMeteorRadiantLabelFontSize.setValue(18)
    page.save_preferences()

    written = _read_jsonc(preference_path)
    assert written["meteor_radiant_only"] is True
    assert written["meteor_radiant_label_font_size_pt"] == 18
    page.close()
    app.processEvents()


def test_preferences_dialog_is_non_modal_and_launcher_uses_text_button(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """软件选项应使用单独非模态弹窗和跨平台稳定的中文文字入口。"""

    app = QApplication.instance() or QApplication([])
    dialog = PreferencesDialog(preference_path=tmp_path / "preference.json")
    launcher = PreferencesLauncher()

    assert not dialog.isModal()
    assert dialog.windowTitle() == "软件选项"
    assert dialog.width() == 860
    assert dialog.height() == 700
    assert dialog.minimumWidth() == 760
    assert launcher.ui.pushButtonOpenPreferences.text() == "选项"
    assert launcher.ui.pushButtonOpenPreferences.accessibleName() == "打开软件选项"

    dialog.show()
    app.processEvents()
    assert dialog.preferences_page.ui.scrollAreaPreferences.horizontalScrollBar().maximum() == 0
    dialog.preferences_page.ui.pushButtonClosePreferences.click()
    app.processEvents()
    assert not dialog.isVisible()
    launcher.close()


def test_default_only_preference_keys_cover_all_default_groups() -> None:
    """即时应用不得把任何默认类参数带入当前会话。"""

    assert DEFAULT_ONLY_PREFERENCE_KEYS == {
        "star_pick_circle_default_diameter_px",
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


def test_star_marker_multiplier_changes_only_computed_star_radius() -> None:
    """星点倍率应进入恒星最终半径计算，并保持基础半径的相对比例。"""

    renderer = StarMapRenderer(StarMapUiConfig(star_marker_size_multiplier=1.75))

    assert renderer.star_marker_radius(2.0, 1.5, 0.5) == 2.625
    assert renderer.star_marker_radius(4.0, 1.5, 0.5) == 5.25


def test_base_star_marker_radius_replaces_the_simulator_minimum_radius() -> None:
    """基础星点大小应替换星等公式中的最暗星半径，再应用总倍率。"""

    renderer = StarMapRenderer(
        StarMapUiConfig(
            base_star_marker_radius_px=1.2,
            star_marker_size_multiplier=2.0,
        )
    )

    assert renderer.star_marker_radius(0.8, 1.0, 1.0) == pytest.approx(2.4)
    assert renderer.star_marker_radius(5.6, 1.0, 1.0) == pytest.approx(12.0)


def test_hot_apply_does_not_replace_current_values_with_new_defaults() -> None:
    """热更新默认参数时不得写入当前观测位置或当前自动匹配控件。"""

    class RejectingValueControl:
        def setValue(self, _value) -> None:  # type: ignore[no-untyped-def]
            raise AssertionError("热更新不应覆盖当前控件值")

        def setCurrentIndex(self, _index) -> None:  # type: ignore[no-untyped-def]
            raise AssertionError("热更新不应覆盖当前控件值")

    class StatusBarStub:
        def showMessage(self, *_args) -> None:  # type: ignore[no-untyped-def]
            return

    host = SimpleNamespace(
        ui=SimpleNamespace(
            statusbar=StatusBarStub(),
            doubleSpinBoxLatitude=RejectingValueControl(),
            doubleSpinBoxLongitude=RejectingValueControl(),
            doubleSpinBoxElevation=RejectingValueControl(),
            spinBoxAutoMatchCount=RejectingValueControl(),
            comboBoxAutoMatchConstraintMode=RejectingValueControl(),
            doubleSpinBoxAutoMatchSoftWeight=RejectingValueControl(),
            spinBoxAutoMatchRadius=RejectingValueControl(),
        ),
        renderer=SimpleNamespace(ui_config=None),
        _apply_ui_font_config=lambda _config: None,
    )
    config = StarMapUiConfig(
        default_latitude_deg=-33.0,
        default_longitude_deg=151.0,
        default_elevation_m=850.0,
        auto_match_default_new_count=999,
    )

    MainWindow._apply_preferences(host, config)

    assert host.ui_config is config
    assert host.renderer.ui_config is config
