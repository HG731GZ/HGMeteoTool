from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QApplication, QMainWindow

from meteoalign.application.app_reference_json_io import ReferenceJsonIOMixin
from meteoalign.application.app_star_pair_reference_payload import StarPairReferencePayloadMixin
from meteoalign.simulator import ObserverSettings
from meteoalign.ui.ui_main_window import Ui_MainWindow


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    app = QApplication.instance() or QApplication([])
    _QT_APP = app
    return app


class _ReferencePayloadHarness(ReferenceJsonIOMixin, StarPairReferencePayloadMixin):
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


def test_old_pair_session_without_simulator_time_preserves_current_time() -> None:
    """关闭 EXIF 同步时，旧匹配会话没有模拟时间字段就保持当前设置。"""

    harness = _ReferencePayloadHarness()
    original_time = QDateTime.fromString("2026-01-02 03:04:05", "yyyy-MM-dd HH:mm:ss")
    harness.ui.dateTimeEditObservation.setDateTime(original_time)
    harness.ui.doubleSpinBoxUtcOffset.setValue(9.0)

    harness._apply_reference_payload(
        _reference_payload(),
        Path("auto_starpairs.json"),
        preserve_reference_star_count=True,
        restore_observation_time=False,
    )

    assert harness.ui.dateTimeEditObservation.dateTime() == original_time
    assert harness.ui.doubleSpinBoxUtcOffset.value() == 9.0
    assert harness.ui.doubleSpinBoxAz.value() == 15.0
    assert harness.render_count == 1
    harness.window.close()


def test_pair_session_restores_recorded_simulator_time_when_exif_sync_is_off() -> None:
    """关闭 EXIF 同步时，新匹配会话应优先恢复独立保存的模拟时间。"""

    harness = _ReferencePayloadHarness()
    harness.ui.dateTimeEditObservation.setDateTime(
        QDateTime.fromString("2026-01-02 03:04:05", "yyyy-MM-dd HH:mm:ss")
    )
    harness.ui.doubleSpinBoxUtcOffset.setValue(9.0)
    simulator_time = {
        "observation_time_utc": "2025-12-14T19:15:45+00:00",
        "utc_offset_hours": 8.0,
    }

    harness._apply_reference_payload(
        _reference_payload(),
        Path("new_starpairs.json"),
        preserve_reference_star_count=True,
        restore_observation_time=False,
        simulator_time_payload=simulator_time,
    )

    expected_local_time = QDateTime.fromString("2025-12-15 03:15:45", "yyyy-MM-dd HH:mm:ss")
    assert harness.ui.dateTimeEditObservation.dateTime() == expected_local_time
    assert harness.ui.doubleSpinBoxUtcOffset.value() == 8.0
    harness.window.close()


def test_export_overwrites_existing_simulator_time_with_current_ui_value() -> None:
    """重复导出时必须用当前模拟时间覆盖 payload 中的旧记录。"""

    harness = _ReferencePayloadHarness()
    harness.ui.doubleSpinBoxUtcOffset.setValue(8.0)
    current_observer = ObserverSettings(
        observation_time_utc=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        latitude_deg=40.0,
        longitude_deg=116.0,
        elevation_m=50.0,
    )
    harness._observer_settings = lambda: current_observer
    payload: dict[str, object] = {
        "simulator_time": {
            "observation_time_utc": "2000-01-01T00:00:00+00:00",
        }
    }

    result = harness._with_current_simulator_time(payload)

    assert result is payload
    assert payload["simulator_time"] == {
        "observation_time_utc": "2026-01-02T03:04:05+00:00",
        "observation_time_local": "2026-01-02T11:04:05+08:00",
        "utc_offset_hours": 8.0,
        "source": "star_simulator_ui",
    }
