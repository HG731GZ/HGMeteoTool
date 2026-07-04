from __future__ import annotations

import numpy as np

from meteoalign.coordinates import normalize_vector, radec_to_unit_vectors, sky_plane_to_radec
from meteoalign.source_model import SOURCE_MODEL_FORMAT, fit_source_astrometric_model


def _local_basis(ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = radec_to_unit_vectors(np.asarray([ra_deg]), np.asarray([dec_deg]))[0]
    celestial_north = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    east = normalize_vector(np.cross(celestial_north, center))
    north = normalize_vector(np.cross(center, east))
    return center, east, north


def _angular_error_arcsec(first_radec: np.ndarray, second_radec: np.ndarray) -> np.ndarray:
    first_vec = radec_to_unit_vectors(first_radec[:, 0], first_radec[:, 1])
    second_vec = radec_to_unit_vectors(second_radec[:, 0], second_radec[:, 1])
    dot = np.sum(first_vec * second_vec, axis=1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))) * 3600.0


def test_source_astrometric_model_round_trip_and_wcs_payload() -> None:
    center, east, north = _local_basis(120.0, 30.0)
    sky_plane = np.asarray(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    radec = sky_plane_to_radec(sky_plane, center, east, north)
    pixels = np.column_stack(
        (
            500.0 + 18.0 * sky_plane[:, 0] + 2.0 * sky_plane[:, 1],
            420.0 - 3.0 * sky_plane[:, 0] + 20.0 * sky_plane[:, 1],
        )
    )

    model = fit_source_astrometric_model(radec, pixels, image_size=(1000, 800))

    predicted_pixels = model.direction_to_pixel_points(radec)
    assert np.allclose(predicted_pixels, pixels, atol=1e-8)

    recovered_radec = model.pixel_to_radec_points(pixels)
    assert float(np.max(_angular_error_arcsec(radec, recovered_radec))) < 1e-2

    payload = model.to_json_payload()
    assert payload["format"] == SOURCE_MODEL_FORMAT
    assert payload["fits_wcs_compat"]["header_cards"]["CTYPE1"] == "RA---TAN"
    assert payload["fits_wcs_compat"]["header_cards"]["CTYPE2"] == "DEC--TAN"
