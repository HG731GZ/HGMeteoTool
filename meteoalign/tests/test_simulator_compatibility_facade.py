"""simulator 设置与投影数学兼容 facade 的回归测试。"""

from __future__ import annotations

import numpy as np

import meteoalign.simulator as simulator
from meteoalign.domain.settings import CameraSettings, ObserverSettings, ViewSettings
from meteoalign.projection.camera_models import (
    RECTILINEAR_LENS_MODEL,
    _project_altaz_points,
    camera_basis_from_view,
    horizontal_fov_deg,
    image_points_to_local_vectors,
    local_vectors_from_altaz,
    local_vectors_to_altaz,
    vertical_fov_deg,
)


def test_simulator_reexports_settings_and_projection_math() -> None:
    """旧 simulator 路径必须引用新模块的同一类型和函数对象。"""

    assert simulator.CameraSettings is CameraSettings
    assert simulator.ObserverSettings is ObserverSettings
    assert simulator.ViewSettings is ViewSettings
    assert simulator.camera_basis_from_view is camera_basis_from_view
    assert simulator._project_altaz_points is _project_altaz_points
    assert simulator.image_points_to_local_vectors is image_points_to_local_vectors


def test_simulator_projection_facade_keeps_fixed_round_trip_result() -> None:
    """固定投影样例在新模块和旧路径下保持相同数值结果。"""

    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=960,
        image_height_px=640,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
    )
    view = ViewSettings(center_az_deg=180.0, center_alt_deg=42.0, roll_deg=8.0)
    altitude_deg = np.asarray([22.0, 35.0, 48.0, 61.0], dtype=np.float64)
    azimuth_deg = np.asarray([142.0, 168.0, 190.0, 215.0], dtype=np.float64)

    basis = simulator.camera_basis_from_view(view)
    x_px, y_px, valid = simulator._project_altaz_points(altitude_deg, azimuth_deg, camera, basis)
    vectors, inverse_valid = simulator.image_points_to_local_vectors(x_px, y_px, camera, basis)
    recovered_altitude_deg, recovered_azimuth_deg, recovered_valid = simulator.local_vectors_to_altaz(vectors)

    expected_vectors = local_vectors_from_altaz(altitude_deg, azimuth_deg)
    recovered_vectors = local_vectors_from_altaz(recovered_altitude_deg, recovered_azimuth_deg)
    assert np.all(valid & inverse_valid & recovered_valid)
    assert np.max(np.linalg.norm(recovered_vectors - expected_vectors, axis=1)) < 1e-8
    assert simulator.horizontal_fov_deg(camera) == horizontal_fov_deg(camera)
    assert simulator.vertical_fov_deg(camera) == vertical_fov_deg(camera)
