"""参考图像粗略取景参数窗口测试。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from meteoalign.adjacent_alignment import (
    ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
    ADJACENT_ALIGNMENT_MODE_STARS,
)
from meteoalign.application.adjacent_alignment_settings_dialog import AdjacentAlignmentSettingsDialog
from meteoalign.config import load_adjacent_alignment_config
from meteoalign.preference_manager import DEFAULT_PREFERENCE_VALUES


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_star_settings_dialog_restores_defaults_and_saves_current_mode(tmp_path: Path) -> None:
    """星点模式窗口仅编辑星点参数，恢复默认值来自 preference_manager。"""

    app = _application()
    preference_path = tmp_path / "preference.json"
    dialog = AdjacentAlignmentSettingsDialog(
        ADJACENT_ALIGNMENT_MODE_STARS,
        preference_path=preference_path,
    )
    assert app is not None

    assert dialog.ui.stackedWidgetSettings.currentWidget() is dialog.ui.pageStarSettings
    assert dialog.ui.pushButtonRestoreDefaults.text() == "回到默认"
    dialog.ui.doubleSpinBoxAdjacentStarDetectionSigma.setValue(2.75)
    dialog._restore_defaults()
    assert dialog.ui.doubleSpinBoxAdjacentStarDetectionSigma.value() == DEFAULT_PREFERENCE_VALUES[
        "adjacent_star_detection_sigma"
    ]

    dialog.ui.doubleSpinBoxAdjacentStarDetectionSigma.setValue(2.75)
    dialog.ui.spinBoxAdjacentAlignmentMaxCorrespondencesStar.setValue(72)
    dialog._save_settings()

    config = load_adjacent_alignment_config(preference_path)
    assert config.stars.detection_sigma == 2.75
    assert config.max_correspondences == 72
    assert dialog.result() == dialog.Accepted


def test_landscape_settings_dialog_preserves_star_mode_settings(tmp_path: Path) -> None:
    """地景模式窗口保存时不应覆盖星点模式已保存的参数。"""

    app = _application()
    preference_path = tmp_path / "preference.json"
    star_dialog = AdjacentAlignmentSettingsDialog(
        ADJACENT_ALIGNMENT_MODE_STARS,
        preference_path=preference_path,
    )
    assert app is not None
    star_dialog.ui.doubleSpinBoxAdjacentStarDetectionSigma.setValue(3.25)
    star_dialog._save_settings()

    landscape_dialog = AdjacentAlignmentSettingsDialog(
        ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
        preference_path=preference_path,
    )
    assert landscape_dialog.ui.stackedWidgetSettings.currentWidget() is landscape_dialog.ui.pageLandscapeSettings
    landscape_dialog.ui.doubleSpinBoxAdjacentLandscapeRatioTestThreshold.setValue(0.66)
    landscape_dialog._save_settings()

    config = load_adjacent_alignment_config(preference_path)
    assert config.stars.detection_sigma == 3.25
    assert config.landscape.ratio_test_threshold == 0.66
