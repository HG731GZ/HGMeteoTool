from __future__ import annotations

import numpy as np

from meteoalign.simulator import (
    CYLINDRICAL_EQUIDISTANT_LENS_MODEL,
    FISHEYE_EQUIDISTANT,
    MERCATOR_LENS_MODEL,
    RECTILINEAR_LENS_MODEL,
    CameraSettings,
    HorizontalMilkyWayCatalog,
    HorizontalMilkyWayPolygon,
    HorizontalMilkyWayRing,
    HorizontalStarCatalog,
    ViewSettings,
    _project_milky_way_polygons,
    _project_altaz_points,
    camera_basis_from_view,
    image_points_to_local_vectors,
    local_vectors_from_altaz,
    local_vectors_to_altaz,
    project_horizontal_catalog,
)


def _synthetic_horizontal_catalog() -> HorizontalStarCatalog:
    return HorizontalStarCatalog(
        source_name="synthetic",
        star_ids=np.asarray(["center", "east", "west"], dtype=object),
        display_names=np.asarray(["center", "east", "west"], dtype=object),
        ra_deg=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        dec_deg=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        mag_v=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        color_index_bv=np.asarray([0.4, 0.5, 0.6], dtype=np.float64),
        spectral_type=np.asarray(["G", "G", "G"], dtype=object),
        common_names=np.asarray(["", "", ""], dtype=object),
        alt_deg=np.asarray([30.0, 35.0, 25.0], dtype=np.float64),
        az_deg=np.asarray([180.0, 165.0, 195.0], dtype=np.float64),
    )


def test_cylindrical_target_projection_modes_render_visible_catalog() -> None:
    catalog = _synthetic_horizontal_catalog()
    view = ViewSettings(center_az_deg=180.0, center_alt_deg=30.0, roll_deg=0.0)

    for lens_model in (MERCATOR_LENS_MODEL, CYLINDRICAL_EQUIDISTANT_LENS_MODEL):
        camera = CameraSettings(
            sensor_width_mm=36.0,
            sensor_height_mm=18.0,
            image_width_px=800,
            image_height_px=400,
            focal_length_mm=24.0,
            lens_model=lens_model,
            fisheye_fov_deg=180.0,
        )

        star_map = project_horizontal_catalog(catalog, camera, view, visible_mag_limit=6.5)

        assert len(star_map) == len(catalog)
        assert np.all(np.isfinite(star_map.x_px))
        assert np.all(np.isfinite(star_map.y_px))
        assert np.all((star_map.x_px >= 0.0) & (star_map.x_px <= camera.image_width_px - 1))
        assert np.all((star_map.y_px >= 0.0) & (star_map.y_px <= camera.image_height_px - 1))
        assert star_map.grid_lines


def _ring_at_camera_angle(
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    theta_deg: float,
) -> HorizontalMilkyWayRing:
    polar_angle = np.deg2rad(np.linspace(0.0, 360.0, 361, dtype=np.float64))
    theta = np.deg2rad(theta_deg)
    camera_vectors = np.column_stack(
        (
            np.sin(theta) * np.cos(polar_angle),
            np.sin(theta) * np.sin(polar_angle),
            np.full_like(polar_angle, np.cos(theta)),
        )
    )
    right, up, forward = basis
    local_vectors = (
        camera_vectors[:, 0, None] * right
        + camera_vectors[:, 1, None] * up
        + camera_vectors[:, 2, None] * forward
    )
    alt_deg, az_deg, valid = local_vectors_to_altaz(local_vectors)
    assert np.all(valid)
    return HorizontalMilkyWayRing(alt_deg=alt_deg, az_deg=az_deg)


def test_cylindrical_milky_way_ring_crossing_projection_seam_is_skipped() -> None:
    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=18.0,
        image_width_px=800,
        image_height_px=400,
        focal_length_mm=24.0,
        lens_model=CYLINDRICAL_EQUIDISTANT_LENS_MODEL,
        fisheye_fov_deg=360.0,
    )
    basis = camera_basis_from_view(ViewSettings(center_az_deg=0.0, center_alt_deg=0.0, roll_deg=0.0))
    seam_crossing_ring = HorizontalMilkyWayRing(
        alt_deg=np.asarray([-5.0, 5.0, 5.0, -5.0, -5.0], dtype=np.float64),
        az_deg=np.asarray([170.0, 170.0, 190.0, 190.0, 170.0], dtype=np.float64),
    )
    normal_ring = HorizontalMilkyWayRing(
        alt_deg=np.asarray([-5.0, 5.0, 5.0, -5.0, -5.0], dtype=np.float64),
        az_deg=np.asarray([350.0, 350.0, 10.0, 10.0, 350.0], dtype=np.float64),
    )

    seam_catalog = HorizontalMilkyWayCatalog(
        source_name="synthetic",
        polygons=(HorizontalMilkyWayPolygon(rings=(seam_crossing_ring,)),),
    )
    normal_catalog = HorizontalMilkyWayCatalog(
        source_name="synthetic",
        polygons=(HorizontalMilkyWayPolygon(rings=(normal_ring,)),),
    )

    assert _project_milky_way_polygons(seam_catalog, camera=camera, basis=basis) == ()
    assert _project_milky_way_polygons(normal_catalog, camera=camera, basis=basis)


def test_cylindrical_milky_way_drops_global_layer_when_one_boundary_crosses_seam() -> None:
    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=18.0,
        image_width_px=800,
        image_height_px=400,
        focal_length_mm=24.0,
        lens_model=MERCATOR_LENS_MODEL,
        fisheye_fov_deg=240.0,
    )
    basis = camera_basis_from_view(ViewSettings(center_az_deg=0.0, center_alt_deg=0.0, roll_deg=0.0))
    front_boundary = _ring_at_camera_angle(basis, 80.0)
    back_boundary = _ring_at_camera_angle(basis, 100.0)
    catalog = HorizontalMilkyWayCatalog(
        source_name="synthetic",
        polygons=(HorizontalMilkyWayPolygon(rings=(front_boundary, back_boundary)),),
    )

    projected = _project_milky_way_polygons(catalog, camera=camera, basis=basis)

    assert projected == ()


def test_fisheye_milky_way_drops_global_layer_when_one_boundary_is_outside_view() -> None:
    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=18.0,
        image_width_px=800,
        image_height_px=400,
        focal_length_mm=24.0,
        lens_model=FISHEYE_EQUIDISTANT,
        fisheye_fov_deg=120.0,
    )
    basis = camera_basis_from_view(ViewSettings(center_az_deg=0.0, center_alt_deg=0.0, roll_deg=0.0))
    inner_boundary = _ring_at_camera_angle(basis, 50.0)
    outside_boundary = _ring_at_camera_angle(basis, 80.0)
    catalog = HorizontalMilkyWayCatalog(
        source_name="synthetic",
        polygons=(HorizontalMilkyWayPolygon(rings=(inner_boundary, outside_boundary)),),
    )

    projected = _project_milky_way_polygons(catalog, camera=camera, basis=basis)

    assert projected == ()


def test_target_projection_pixel_inverse_round_trips_altaz() -> None:
    alt_deg = np.asarray([22.0, 35.0, 48.0, 61.0], dtype=np.float64)
    az_deg = np.asarray([142.0, 168.0, 190.0, 215.0], dtype=np.float64)
    view = ViewSettings(center_az_deg=180.0, center_alt_deg=42.0, roll_deg=8.0)
    basis = camera_basis_from_view(view)

    for lens_model in (
        RECTILINEAR_LENS_MODEL,
        FISHEYE_EQUIDISTANT,
        MERCATOR_LENS_MODEL,
        CYLINDRICAL_EQUIDISTANT_LENS_MODEL,
    ):
        camera = CameraSettings(
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            image_width_px=960,
            image_height_px=640,
            focal_length_mm=24.0,
            lens_model=lens_model,
            fisheye_fov_deg=180.0,
        )
        x_px, y_px, valid = _project_altaz_points(
            alt_deg,
            az_deg,
            camera=camera,
            basis=basis,
        )
        assert np.all(valid)

        vectors, inverse_valid = image_points_to_local_vectors(x_px, y_px, camera=camera, basis=basis)
        recovered_alt, recovered_az, altaz_valid = local_vectors_to_altaz(vectors)
        expected_vectors = local_vectors_from_altaz(alt_deg, az_deg)
        recovered_vectors = local_vectors_from_altaz(recovered_alt, recovered_az)

        assert np.all(inverse_valid & altaz_valid)
        assert np.max(np.linalg.norm(recovered_vectors - expected_vectors, axis=1)) < 1e-8
