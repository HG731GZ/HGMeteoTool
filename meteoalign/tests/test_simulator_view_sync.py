from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from meteoalign.alignment.models import PreliminarySkyAlignmentTransform
from meteoalign.application.app_alignment import AlignmentMixin
from meteoalign.domain.settings import ObserverSettings, ViewSettings
from meteoalign.frame_astrometry import FramePose
from meteoalign.projection.camera_models import camera_basis_from_view
from meteoalign.sequence_geometry import icrs_to_enu_rotation_matrix
from meteoalign.simulator_view_sync import (
    preliminary_alignment_rotation_matrix,
    view_center_from_icrs_direction,
    view_settings_from_icrs_to_camera,
)


def _observer() -> ObserverSettings:
    return ObserverSettings(
        observation_time_utc=datetime(2025, 8, 12, 16, 30, tzinfo=timezone.utc),
        latitude_deg=31.2,
        longitude_deg=121.5,
        elevation_m=12.0,
    )


def _icrs_to_camera(view: ViewSettings, observer: ObserverSettings) -> np.ndarray:
    camera_from_enu = np.vstack(camera_basis_from_view(view)).astype(np.float64)
    return FramePose(camera_from_enu @ icrs_to_enu_rotation_matrix(observer)).icrs_to_camera


def _angle_delta_deg(left: float, right: float) -> float:
    return (float(left) - float(right) + 180.0) % 360.0 - 180.0


def test_frame_pose_round_trips_to_simulator_view_center() -> None:
    """正式匹配或粗略取景的姿态应恢复为相同的模拟取景中心。"""

    observer = _observer()
    expected = ViewSettings(center_az_deg=217.3, center_alt_deg=42.6, roll_deg=-31.4)

    center_az_deg, center_alt_deg = view_center_from_icrs_direction(
        _icrs_to_camera(expected, observer)[2],
        observer,
    )

    assert abs(_angle_delta_deg(center_az_deg, expected.center_az_deg)) < 0.01
    assert abs(center_alt_deg - expected.center_alt_deg) < 0.01

    restored = view_settings_from_icrs_to_camera(_icrs_to_camera(expected, observer), observer)

    assert abs(_angle_delta_deg(restored.roll_deg, expected.roll_deg)) < 0.01


def test_preliminary_alignment_recovers_image_center() -> None:
    """两点或三点预配准也应能同步图像中心和画面上方向。"""

    observer = _observer()
    expected = ViewSettings(center_az_deg=132.0, center_alt_deg=28.0, roll_deg=17.0)
    expected_rotation = _icrs_to_camera(expected, observer)
    right, up, forward = expected_rotation
    image_size = (1200, 800)
    scale_px_per_deg = 18.0
    transform = PreliminarySkyAlignmentTransform(
        pair_count=2,
        center_vector=forward,
        east_vector=right,
        north_vector=up,
        linear_matrix=np.asarray(
            ((scale_px_per_deg, 0.0), (0.0, -scale_px_per_deg)),
            dtype=np.float64,
        ),
        offset_px=np.asarray((image_size[0] * 0.5, image_size[1] * 0.5), dtype=np.float64),
        rms_px=0.0,
        orientation=-1,
    )

    rotation_matrix = preliminary_alignment_rotation_matrix(transform, image_size)
    center_az_deg, center_alt_deg = view_center_from_icrs_direction(rotation_matrix[2], observer)
    restored = view_settings_from_icrs_to_camera(rotation_matrix, observer)

    assert abs(_angle_delta_deg(center_az_deg, expected.center_az_deg)) < 0.02
    assert abs(center_alt_deg - expected.center_alt_deg) < 0.02
    assert abs(_angle_delta_deg(restored.roll_deg, expected.roll_deg)) < 0.02


class _SpinBox:
    """模拟已禁用但仍允许程序写值的 Qt 数值控件。"""

    def __init__(self, value: float) -> None:
        self._value = float(value)
        self.enabled = False
        self._signals_blocked = False

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self._value = round(float(value), 2)

    def decimals(self) -> int:
        return 2

    def blockSignals(self, blocked: bool) -> bool:  # noqa: N802 - 保持 Qt 接口名称。
        previous = self._signals_blocked
        self._signals_blocked = bool(blocked)
        return previous


def test_known_projection_syncs_solved_center_and_roll() -> None:
    """TAN、ARC 和 ZEA 应同步中心与画面旋转，且只调度一次渲染。"""

    observer = _observer()
    expected = ViewSettings(center_az_deg=246.4, center_alt_deg=36.2, roll_deg=-12.8)
    frame_model = SimpleNamespace(
        frame_pose=FramePose(_icrs_to_camera(expected, observer)),
    )
    harness = AlignmentMixin()
    harness.ui = SimpleNamespace(
        doubleSpinBoxAz=_SpinBox(0.0),
        doubleSpinBoxAlt=_SpinBox(0.0),
        doubleSpinBoxRoll=_SpinBox(48.5),
    )
    harness._simulator_controls_locked = True
    harness._source_astrometric_model = SimpleNamespace(
        to_frame_astrometric_model=lambda: frame_model,
    )
    harness._sky_alignment_transform = None
    harness._alignment_model = lambda: "fisheye_equisolid"
    harness._observer_settings = lambda: observer
    scheduled_delays: list[int] = []
    harness.schedule_render = lambda *, delay_ms: scheduled_delays.append(delay_ms)

    changed = harness._sync_simulator_view_from_alignment()

    assert changed
    assert harness.ui.doubleSpinBoxAz.value() == 246.4
    assert harness.ui.doubleSpinBoxAlt.value() == 36.2
    assert harness.ui.doubleSpinBoxRoll.value() == -12.8
    assert scheduled_delays == [0]


def test_other_projection_only_syncs_solved_center() -> None:
    """MER、CAR 和锚点插值等模型只同步中心，不覆盖模拟器 Roll。"""

    observer = _observer()
    expected = ViewSettings(center_az_deg=168.2, center_alt_deg=51.3, roll_deg=-27.0)
    harness = AlignmentMixin()
    harness.ui = SimpleNamespace(
        doubleSpinBoxAz=_SpinBox(0.0),
        doubleSpinBoxAlt=_SpinBox(0.0),
        doubleSpinBoxRoll=_SpinBox(63.0),
    )
    harness._source_astrometric_model = SimpleNamespace(
        to_frame_astrometric_model=lambda: SimpleNamespace(
            frame_pose=FramePose(_icrs_to_camera(expected, observer)),
        ),
    )
    harness._sky_alignment_transform = None
    harness._alignment_model = lambda: "mercator"
    harness._observer_settings = lambda: observer
    harness.schedule_render = lambda *, delay_ms: None

    changed = harness._sync_simulator_view_from_alignment()

    assert changed
    assert harness.ui.doubleSpinBoxAz.value() == 168.2
    assert harness.ui.doubleSpinBoxAlt.value() == 51.3
    assert harness.ui.doubleSpinBoxRoll.value() == 63.0
