"""固定小图的拼图重投影 map 契约测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from meteoalign.coordinates import unit_vectors_to_radec
from meteoalign.mosaic.export.geometry import MosaicExportGeometry
from meteoalign.mosaic.export.remap_builder import build_reprojection_map
from meteoalign.simulator import (
    CameraSettings,
    ObserverSettings,
    RECTILINEAR_LENS_MODEL,
    ViewSettings,
    _project_altaz_points,
    camera_basis_from_view,
    compute_altaz_from_radec,
)


class _MatchingProjectionSourceModel:
    """与目标相机完全相同的最小源图模型，用于固定 map 契约。"""

    def __init__(self, camera: CameraSettings, view: ViewSettings, observer: ObserverSettings) -> None:
        self.image_width_px = int(camera.image_width_px)
        self.image_height_px = int(camera.image_height_px)
        self._camera = camera
        self._observer = observer
        self._basis = camera_basis_from_view(view)

    def icrs_vectors_to_pixel_points(self, vectors: np.ndarray) -> np.ndarray:
        """把 ICRS 向量投影回与目标完全相同的源图像素。"""

        radec = unit_vectors_to_radec(np.asarray(vectors, dtype=np.float64))
        altitude_deg, azimuth_deg = compute_altaz_from_radec(radec[:, 0], radec[:, 1], self._observer)
        x_px, y_px, valid = _project_altaz_points(
            altitude_deg,
            azimuth_deg,
            camera=self._camera,
            basis=self._basis,
        )
        pixels = np.column_stack((x_px, y_px)).astype(np.float64)
        pixels[~valid] = np.nan
        return pixels


def test_fixed_small_map_contract_preserves_crop_coordinates_across_blocks() -> None:
    """相同投影下，裁剪后目标像素应映射到裁剪前的同名源像素。"""

    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=7,
        image_height_px=5,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=90.0,
    )
    observer = ObserverSettings(
        observation_time_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        latitude_deg=25.0,
        longitude_deg=102.0,
        elevation_m=0.0,
    )
    view = ViewSettings(center_az_deg=180.0, center_alt_deg=45.0, roll_deg=0.0)
    geometry = MosaicExportGeometry(
        boundary_width_px=7,
        boundary_height_px=5,
        crop_left_px=1,
        crop_top_px=1,
        output_width_px=5,
        output_height_px=3,
    )

    reprojection_map = build_reprojection_map(
        source_model=_MatchingProjectionSourceModel(camera, view, observer),
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=2,
    )
    map_x = reprojection_map.map_x
    map_y = reprojection_map.map_y

    expected_map_x = np.tile(np.arange(1.0, 6.0, dtype=np.float32), (3, 1))
    expected_map_y = np.tile(np.arange(1.0, 4.0, dtype=np.float32)[:, None], (1, 5))
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32
    assert reprojection_map.valid_mask.shape == (3, 5)
    assert reprojection_map.valid_mask.all()
    assert reprojection_map.metadata["block_rows"] == 2
    # 坐标转换涉及天文坐标往返，保留跨平台数值容差而固定几何语义。
    np.testing.assert_allclose(map_x, expected_map_x, rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(map_y, expected_map_y, rtol=0.0, atol=5e-4)
