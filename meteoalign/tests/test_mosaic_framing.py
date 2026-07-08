from __future__ import annotations

import math

import numpy as np

from meteoalign.alignment.constants import (
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from meteoalign.coordinates import sky_plane_to_radec
from meteoalign.mosaic_framing import (
    estimate_mosaic_optimal_resolution,
    source_center_angular_resolution_rad_per_px,
    target_output_dimensions_for_resolution,
)


class _LinearTangentSourceModel:
    def __init__(self, width_px: int, height_px: int, x_rad_per_px: float, y_rad_per_px: float) -> None:
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.x_rad_per_px = float(x_rad_per_px)
        self.y_rad_per_px = float(y_rad_per_px)
        self.center = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        self.east = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        self.north = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    def pixel_to_sky_points(self, pixel_points: np.ndarray) -> np.ndarray:
        points = np.asarray(pixel_points, dtype=np.float64)
        center_x = (self.width_px - 1) * 0.5
        center_y = (self.height_px - 1) * 0.5
        plane_rad = np.column_stack(
            (
                (points[:, 0] - center_x) * self.x_rad_per_px,
                (points[:, 1] - center_y) * self.y_rad_per_px,
            )
        )
        return sky_plane_to_radec(np.rad2deg(plane_rad), self.center, self.east, self.north)


def test_source_center_resolution_uses_geometric_jacobian_scale() -> None:
    model = _LinearTangentSourceModel(401, 301, 0.001, 0.002)

    resolution, jacobian, center_x, center_y = source_center_angular_resolution_rad_per_px(
        model,
        image_width_px=model.width_px,
        image_height_px=model.height_px,
    )

    assert math.isclose(resolution, math.sqrt(0.001 * 0.002), rel_tol=1e-5)
    assert np.allclose(jacobian, [[0.001, 0.0], [0.0, 0.002]], atol=1e-8)
    assert center_x == 200.0
    assert center_y == 150.0


def test_rectilinear_optimal_size_matches_center_angular_resolution() -> None:
    model = _LinearTangentSourceModel(401, 301, 0.001, 0.001)

    estimate = estimate_mosaic_optimal_resolution(
        model,
        source_image_width_px=model.width_px,
        source_image_height_px=model.height_px,
        projection_model=SKY_MATCHING_MODEL_RECTILINEAR,
        fov_deg=90.0,
        viewport_width_px=800,
        viewport_height_px=400,
    )

    assert estimate.boundary_width_px == 2000
    assert estimate.boundary_height_px == 1000
    assert math.isclose(estimate.target_center_px_per_rad, 1000.0, rel_tol=1e-6)


def test_cylindrical_and_fisheye_dimensions_follow_projection_boundary() -> None:
    cylindrical_width, cylindrical_height, _scale = target_output_dimensions_for_resolution(
        angular_resolution_rad_per_px=0.001,
        projection_model=SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
        fov_deg=180.0,
        height_over_width=0.5,
    )
    fisheye_width, fisheye_height, _scale = target_output_dimensions_for_resolution(
        angular_resolution_rad_per_px=0.001,
        projection_model=SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
        fov_deg=180.0,
        height_over_width=0.5,
    )

    assert cylindrical_width == math.ceil(math.pi * 1000.0)
    assert cylindrical_height == math.ceil(math.pi * 1000.0 * 0.5)
    assert fisheye_height == math.ceil(math.pi * 1000.0 + 1.0)
    assert fisheye_width == math.ceil(fisheye_height / 0.5)
