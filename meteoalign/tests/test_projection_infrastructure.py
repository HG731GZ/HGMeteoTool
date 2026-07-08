from __future__ import annotations

import numpy as np

from meteoalign.geometry2d import cell_crosses_angle_break, expand_polygon_radially
from meteoalign.mapping_validation import MappingValidationDialog
from meteoalign.projection_grid import build_pixel_grid, grid_shape_for_long_side, project_altaz_grid_to_screen
from meteoalign.simulator import CameraSettings, RECTILINEAR_LENS_MODEL, ViewSettings
from meteoalign.view_gestures import ViewZoomPolicy, fov_after_zoom, roll_after_drag, sky_center_after_drag, wheel_zoom_factor


def test_expand_polygon_radially_moves_vertices_outward() -> None:
    square = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )

    expanded = expand_polygon_radially(square, 1.0)

    assert expanded.shape == square.shape
    assert np.allclose(np.mean(expanded, axis=0), [1.0, 1.0])
    assert np.all(np.linalg.norm(expanded - [1.0, 1.0], axis=1) > np.linalg.norm(square - [1.0, 1.0], axis=1))


def test_cell_crosses_angle_break_detects_wrapped_quad() -> None:
    normal = np.deg2rad(np.asarray([[10.0, 12.0], [11.0, 13.0]], dtype=np.float64))
    wrapped = np.deg2rad(np.asarray([[170.0, -170.0], [171.0, -171.0]], dtype=np.float64))

    assert not cell_crosses_angle_break(normal, 0, 0)
    assert cell_crosses_angle_break(wrapped, 0, 0)


def test_pixel_grid_preserves_aspect_ratio_and_image_edges() -> None:
    rows, columns = grid_shape_for_long_side(400, 200, 20, min_minor_cells=3)
    grid = build_pixel_grid(400, 200, rows, columns)

    assert (rows, columns) == (10, 20)
    assert grid.point_count == 200
    assert grid.x_px[0, 0] == 0.0
    assert grid.y_px[0, 0] == 0.0
    assert grid.x_px[-1, -1] == 399.0
    assert grid.y_px[-1, -1] == 199.0


def test_project_altaz_grid_to_screen_marks_center_visible() -> None:
    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=600,
        image_height_px=400,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=180.0,
    )
    view = ViewSettings(center_az_deg=180.0, center_alt_deg=30.0, roll_deg=0.0)
    alt = np.asarray([[30.0]], dtype=np.float64)
    az = np.asarray([[180.0]], dtype=np.float64)

    screen = project_altaz_grid_to_screen(alt, az, camera=camera, view=view)

    assert bool(screen.valid[0, 0])
    assert abs(float(screen.x_px[0, 0]) - (camera.image_width_px - 1) * 0.5) < 1.0
    assert abs(float(screen.y_px[0, 0]) - (camera.image_height_px - 1) * 0.5) < 1.0


def test_view_gesture_math_is_directionally_consistent() -> None:
    assert wheel_zoom_factor(120, 1.25) == 1.25
    assert wheel_zoom_factor(-120, 1.25) == 0.8
    assert fov_after_zoom(100.0, 2.0, ViewZoomPolicy(min_fov=20.0, max_fov=120.0)) == 50.0

    az, alt = sky_center_after_drag(
        center_az_deg=100.0,
        center_alt_deg=20.0,
        dx_px=10,
        dy_px=-5,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=30.0,
        viewport_width_px=600,
        viewport_height_px=300,
    )
    assert az == 99.0
    assert alt == 19.5
    assert roll_after_drag(179.0, 8, drag_sign=1.0) == -179.0


def test_mapping_validation_camera_for_render_updates_base_camera() -> None:
    dialog = MappingValidationDialog.__new__(MappingValidationDialog)
    dialog.base_camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=100,
        image_height_px=100,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=120.0,
    )
    dialog.focal_length_mm = 35.0
    dialog.fisheye_fov_deg = 180.0

    camera = MappingValidationDialog._camera_for_render(dialog, 640, 480)

    assert camera.image_width_px == 640
    assert camera.image_height_px == 480
    assert camera.focal_length_mm == 35.0
    assert camera.fisheye_fov_deg == 180.0
