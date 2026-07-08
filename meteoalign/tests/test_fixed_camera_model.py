from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from meteoalign.alignment import SKY_MATCHING_MODEL_RECTILINEAR, _project_unit_vectors_with_known_projection
from meteoalign.fixed_camera_model import FixedCameraModel, estimate_frame_time_correction, fit_fixed_camera_model
from meteoalign.sequence_geometry import SequenceGeometryModel, frame_astrometric_model_from_fixed_camera
from meteoalign.simulator import ObserverSettings, ViewSettings, camera_basis_from_view, local_vectors_from_altaz


def _radec_from_altaz(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    observer: ObserverSettings,
) -> np.ndarray:
    location = EarthLocation.from_geodetic(
        lon=observer.longitude_deg * u.deg,
        lat=observer.latitude_deg * u.deg,
        height=observer.elevation_m * u.m,
    )
    altaz = SkyCoord(
        alt=alt_deg * u.deg,
        az=az_deg * u.deg,
        frame=AltAz(obstime=Time(observer.observation_time_utc), location=location),
    )
    icrs = altaz.icrs
    return np.column_stack((icrs.ra.degree, icrs.dec.degree)).astype(np.float64)


def _synthetic_fixed_camera_fixture():
    observer = ObserverSettings(
        observation_time_utc=datetime(2025, 12, 14, 19, 15, 45, tzinfo=timezone.utc),
        latitude_deg=40.0,
        longitude_deg=116.0,
        elevation_m=200.0,
    )
    alt_deg = np.asarray([28.0, 34.0, 41.0, 48.0, 55.0, 62.0, 36.0, 57.0], dtype=np.float64)
    az_deg = np.asarray([178.0, 194.0, 208.0, 224.0, 238.0, 204.0, 252.0, 170.0], dtype=np.float64)
    radec = _radec_from_altaz(alt_deg, az_deg, observer)
    enu_vectors = local_vectors_from_altaz(alt_deg, az_deg)
    rotation = np.vstack(
        camera_basis_from_view(
            ViewSettings(
                center_az_deg=210.0,
                center_alt_deg=45.0,
                roll_deg=12.0,
            )
        )
    )
    pixels, valid = _project_unit_vectors_with_known_projection(
        vectors=enu_vectors,
        rotation_matrix=rotation,
        center_x_px=500.0,
        center_y_px=380.0,
        scale_px=1250.0,
        lens_model=SKY_MATCHING_MODEL_RECTILINEAR,
        strict_visibility=True,
    )
    assert np.all(valid)
    fixed_model = fit_fixed_camera_model(
        enu_vectors=enu_vectors,
        pixel_points=pixels,
        image_size=(1000, 800),
        lens_model=SKY_MATCHING_MODEL_RECTILINEAR,
        initial_rotation_matrix=rotation,
    )
    return fixed_model, observer, radec, pixels


def test_fixed_camera_model_projects_radec_through_local_enu() -> None:
    fixed_model, observer, radec, pixels = _synthetic_fixed_camera_fixture()

    predicted, alt_deg, az_deg = fixed_model.project_radec_points(radec, observer)

    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 1e-5
    assert np.all(np.isfinite(alt_deg))
    assert np.all(np.isfinite(az_deg))


def test_fixed_camera_model_inverse_pixels_to_altaz_matches_forward_projection() -> None:
    fixed_model, observer, radec, pixels = _synthetic_fixed_camera_fixture()
    _predicted, expected_alt_deg, expected_az_deg = fixed_model.project_radec_points(radec, observer)

    actual_alt_deg, actual_az_deg, valid = fixed_model.pixel_to_altaz_points(pixels)

    assert np.all(valid)
    expected_vectors = local_vectors_from_altaz(expected_alt_deg, expected_az_deg)
    actual_vectors = local_vectors_from_altaz(actual_alt_deg, actual_az_deg)
    assert np.max(np.linalg.norm(actual_vectors - expected_vectors, axis=1)) < 1e-4


def test_frame_time_correction_recovers_independent_delta_t() -> None:
    fixed_model, observer, radec, _pixels = _synthetic_fixed_camera_fixture()
    true_delta_seconds = 35.0
    observed_pixels, _alt_deg, _az_deg = fixed_model.project_radec_at_time(
        radec,
        observation_time_utc=observer.observation_time_utc + timedelta(seconds=true_delta_seconds),
        latitude_deg=observer.latitude_deg,
        longitude_deg=observer.longitude_deg,
        elevation_m=observer.elevation_m,
    )

    result = estimate_frame_time_correction(
        fixed_model=fixed_model,
        ra_dec_points=radec,
        observed_pixels=observed_pixels,
        nominal_time_utc=observer.observation_time_utc,
        latitude_deg=observer.latitude_deg,
        longitude_deg=observer.longitude_deg,
        elevation_m=observer.elevation_m,
        initial_delta_seconds=0.0,
    )

    assert abs(result.delta_t_seconds - true_delta_seconds) < 0.5
    assert result.accepted_count == radec.shape[0]
    assert result.rms_px < 0.05


def test_fixed_camera_payload_contains_static_model_sections() -> None:
    fixed_model, _observer, _radec, _pixels = _synthetic_fixed_camera_fixture()

    payload = fixed_model.to_json_payload()

    assert payload["kind"] == "fixed_camera_enu_model"
    assert payload["camera_intrinsics"]["base_projection"] == SKY_MATCHING_MODEL_RECTILINEAR
    assert "rotation_matrix_enu_to_camera" in payload["fixed_camera_pose"]
    assert payload["static_residual_distortion"]["kind"] == "thin_plate_spline"


def test_fixed_camera_model_restores_from_json_payload() -> None:
    fixed_model, _observer, _radec, pixels = _synthetic_fixed_camera_fixture()
    payload = fixed_model.to_json_payload()

    restored = FixedCameraModel.from_json_payload(payload)
    expected_alt_deg, expected_az_deg, expected_valid = fixed_model.pixel_to_altaz_points(pixels)
    actual_alt_deg, actual_az_deg, actual_valid = restored.pixel_to_altaz_points(pixels)

    assert restored.image_width_px == fixed_model.image_width_px
    assert restored.image_height_px == fixed_model.image_height_px
    assert restored.lens_model == fixed_model.lens_model
    assert np.all(actual_valid == expected_valid)
    expected_vectors = local_vectors_from_altaz(expected_alt_deg, expected_az_deg)
    actual_vectors = local_vectors_from_altaz(actual_alt_deg, actual_az_deg)
    assert np.max(np.linalg.norm(actual_vectors - expected_vectors, axis=1)) < 1e-8


def test_fixed_camera_model_exports_frame_astrometric_model() -> None:
    fixed_model, observer, radec, pixels = _synthetic_fixed_camera_fixture()

    frame_model = frame_astrometric_model_from_fixed_camera(
        fixed_camera_model=fixed_model,
        observer=observer,
        fit_metadata={"model_type": "sequence_frame_astrometric_model"},
        diagnostics={"pair_count": int(radec.shape[0])},
    )
    predicted = frame_model.sky_to_pixel_points(radec)
    payload = frame_model.to_json_payload()

    assert payload["schema"] == "hgmeteo_source_astrometric_model"
    assert payload["direction_frame"] == "ICRS"
    assert payload["fit_metadata"]["model_type"] == "sequence_frame_astrometric_model"
    assert payload["camera_calibration_profile"]["base_projection"]["type"] == SKY_MATCHING_MODEL_RECTILINEAR
    assert "fixed_camera_model" not in payload
    assert "dynamic_sky_conversion" not in payload
    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 0.2


def test_sequence_geometry_exports_each_frame_as_icrs_model_at_frame_time() -> None:
    fixed_model, observer, radec, _pixels = _synthetic_fixed_camera_fixture()
    sequence_model = SequenceGeometryModel(solver_observer=observer, fixed_camera_model=fixed_model)
    frame_time = observer.observation_time_utc + timedelta(seconds=83.0)

    sequence_pixels, _alt_deg, _az_deg = sequence_model.project_radec_at_time(radec, frame_time)
    frame_model = sequence_model.frame_astrometric_model(
        frame_time,
        fit_metadata={"model_type": "sequence_frame_astrometric_model"},
    )
    frame_pixels = frame_model.sky_to_pixel_points(radec)
    payload = frame_model.to_json_payload()

    assert payload["schema"] == "hgmeteo_source_astrometric_model"
    assert payload["direction_frame"] == "ICRS"
    assert payload["fit_metadata"]["model_type"] == "sequence_frame_astrometric_model"
    assert payload["camera_calibration_profile"]["coverage"]["image_width_px"] == fixed_model.image_width_px
    assert np.max(np.linalg.norm(frame_pixels - sequence_pixels, axis=1)) < 0.2
