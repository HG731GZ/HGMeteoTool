from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMainWindow

from meteoalign.application.app_reference_json_io import ReferenceJsonIOMixin
from meteoalign.simulator import ObserverSettings
from meteoalign.ui.ui_main_window import Ui_MainWindow


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    app = QApplication.instance() or QApplication([])
    _QT_APP = app
    return app


class _ReferencePayloadHarness(ReferenceJsonIOMixin):
    """只保留参考 payload 应用所需的界面与回调。"""

    def __init__(self) -> None:
        _qapp()
        self.window = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.window)
        self._syncing_camera_dimensions = False
        self.render_count = 0

    def _reference_star_lookup_from_records(self, *_args, **_kwargs) -> dict:
        return {}

    def _observer_settings(self) -> ObserverSettings:
        return ObserverSettings(
            observation_time_utc=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
            latitude_deg=40.0,
            longitude_deg=116.0,
            elevation_m=50.0,
        )

    def _update_reference_label_controls(self) -> None:
        return

    def _update_lens_model_controls(self) -> None:
        return

    def render_now(self) -> None:
        self.render_count += 1


def _reference_payload(reference_star_count: int = 40) -> dict[str, object]:
    return {
        "format": "meteoalign_phase1_reference",
        "observer": {
            "observation_time_utc": "2026-07-16T12:00:00+00:00",
            "utc_offset_hours": 8.0,
            "latitude_deg": 40.0,
            "longitude_deg": 116.0,
            "elevation_m": 50.0,
        },
        "camera": {
            "sensor_width_mm": 36.0,
            "sensor_height_mm": 24.0,
            "image_width_px": 1920,
            "image_height_px": 1280,
            "focal_length_mm": 24.0,
            "lens_model": "rectilinear",
            "fisheye_fov_deg": 180.0,
        },
        "view": {
            "center_az_deg": 15.0,
            "center_alt_deg": 35.0,
            "roll_deg": 2.0,
        },
        "render": {
            "visible_mag_limit": 6.5,
            "reference_label_mode": "fixed_count",
            "reference_star_count": reference_star_count,
            "reference_mag_limit": 5.0,
        },
        "manual_reference_star_ids": [],
        "stars": [],
    }


def test_pair_session_reference_payload_preserves_current_star_count() -> None:
    """旧匹配 JSON 中错误写入的 40 不得污染当前的标注星数。"""

    harness = _ReferencePayloadHarness()
    harness.ui.comboBoxReferenceLabelMode.setCurrentIndex(1)
    harness.ui.spinBoxReferenceStarCount.setValue(12)
    harness.ui.doubleSpinBoxReferenceMagLimit.setValue(3.2)

    harness._apply_reference_payload(
        _reference_payload(40),
        Path("legacy_starpairs.json"),
        preserve_reference_star_count=True,
    )

    assert harness.ui.comboBoxReferenceLabelMode.currentIndex() == 0
    assert harness.ui.spinBoxReferenceStarCount.value() == 12
    assert harness.ui.doubleSpinBoxReferenceMagLimit.value() == 5.0
    assert harness.render_count == 1
    harness.window.close()


def test_explicit_reference_json_still_restores_its_list_settings() -> None:
    """显式导入参考星图 JSON 时仍应完整恢复其中的星空模拟设置。"""

    harness = _ReferencePayloadHarness()
    harness.ui.comboBoxReferenceLabelMode.setCurrentIndex(1)
    harness.ui.spinBoxReferenceStarCount.setValue(12)
    harness.ui.doubleSpinBoxReferenceMagLimit.setValue(3.2)

    harness._apply_reference_payload(_reference_payload(40), Path("reference.json"))

    assert harness.ui.comboBoxReferenceLabelMode.currentIndex() == 0
    assert harness.ui.spinBoxReferenceStarCount.value() == 40
    assert harness.ui.doubleSpinBoxReferenceMagLimit.value() == 5.0
    assert harness.render_count == 1
    harness.window.close()
