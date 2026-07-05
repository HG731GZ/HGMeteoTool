from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meteoalign.alignment import (
    RESIDUAL_CORRECTION_TPS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_RECTILINEAR,
    _project_unit_vectors_with_known_projection,
    fit_sky_alignment,
)
from meteoalign.coordinates import normalize_vector, radec_to_unit_vectors, sky_plane_to_radec
from meteoalign.simulator import ViewSettings, camera_basis_from_view, local_vectors_from_altaz
from meteoalign.source_model import fit_source_astrometric_model


KNOWN_PROJECTION_MODELS = (
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
)


def _projection_fixture(lens_model: str) -> tuple[np.ndarray, np.ndarray]:
    center = radec_to_unit_vectors(np.asarray([120.0]), np.asarray([30.0]))[0]
    east = normalize_vector(np.cross(np.asarray([0.0, 0.0, 1.0]), center))
    north = normalize_vector(np.cross(center, east))
    sky_plane = np.asarray(
        [
            [-4.0, -3.0],
            [-2.0, 2.0],
            [0.0, 0.0],
            [3.0, -1.0],
            [4.0, 2.0],
            [-5.0, 1.0],
            [2.0, 4.0],
            [1.0, -4.0],
        ],
        dtype=np.float64,
    )
    radec = sky_plane_to_radec(sky_plane, center, east, north)
    vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])

    roll_rad = np.deg2rad(18.0)
    right = normalize_vector(east * np.cos(roll_rad) + north * np.sin(roll_rad))
    up = normalize_vector(-east * np.sin(roll_rad) + north * np.cos(roll_rad))
    rotation_matrix = np.vstack((right, up, center))
    pixels, valid = _project_unit_vectors_with_known_projection(
        vectors=vectors,
        rotation_matrix=rotation_matrix,
        center_x_px=500.0,
        center_y_px=400.0,
        scale_px=900.0,
        lens_model=lens_model,
        strict_visibility=True,
    )
    assert np.all(valid)
    return radec, pixels


@pytest.mark.parametrize("lens_model", KNOWN_PROJECTION_MODELS)
def test_known_projection_alignment_round_trip(lens_model: str) -> None:
    radec, pixels = _projection_fixture(lens_model)

    transform = fit_sky_alignment(
        radec,
        pixels,
        matching_model=lens_model,
        image_size=(1000, 800),
    )

    predicted = transform.transform_radec_points(radec)
    assert transform.rms_px < 1e-6
    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 1e-5


def test_universal_alignment_uses_anchor_interpolation() -> None:
    radec, pixels = _projection_fixture(SKY_MATCHING_MODEL_RECTILINEAR)
    warped_pixels = pixels + np.column_stack(
        (
            0.015 * (pixels[:, 0] - 500.0) * (pixels[:, 1] - 400.0) / 100.0,
            0.01 * (pixels[:, 0] - 500.0) ** 2 / 100.0,
        )
    )

    transform = fit_sky_alignment(
        radec,
        warped_pixels,
        matching_model=SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
        image_size=(1000, 800),
    )

    predicted = transform.transform_radec_points(radec)
    assert transform.display_name == "普适锚点插值"
    assert transform.rms_px < 1e-6
    assert np.max(np.linalg.norm(predicted - warped_pixels, axis=1)) < 1e-5


def test_source_model_exports_known_projection_payload() -> None:
    radec, pixels = _projection_fixture(SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT)

    model = fit_source_astrometric_model(
        radec,
        pixels,
        image_size=(1000, 800),
        matching_model=SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    )

    payload = model.to_json_payload()
    assert payload["model_type"] == "known_projection_with_residual_interpolation"
    assert payload["sky_to_pixel"]["kind"] == "known_projection_with_residual_anchor_interpolation"
    assert "degree" not in payload["sky_to_pixel"]
    assert "coeff_x" not in payload["sky_to_pixel"]
    assert payload["known_projection"]["lens_model"] == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
    assert payload["known_projection"]["residual_correction"]["kind"] == RESIDUAL_CORRECTION_TPS
    assert "degree" not in payload["known_projection"]["residual_correction"]
    assert "coeff_x_px" not in payload["known_projection"]["residual_correction"]
    assert payload["diagnostics"]["rms_px"] < 1e-6


def _initial_rotation_from_reference_payload(reference_payload: dict[str, object]) -> np.ndarray:
    reference_stars = reference_payload["stars"]
    world_vectors = radec_to_unit_vectors(
        np.asarray([star["ra_deg"] for star in reference_stars], dtype=np.float64),
        np.asarray([star["dec_deg"] for star in reference_stars], dtype=np.float64),
    )
    local_vectors = local_vectors_from_altaz(
        np.asarray([star["alt_deg"] for star in reference_stars], dtype=np.float64),
        np.asarray([star["az_deg"] for star in reference_stars], dtype=np.float64),
    )
    local_from_world_transposed, _residuals, _rank, _singular_values = np.linalg.lstsq(
        world_vectors,
        local_vectors,
        rcond=None,
    )
    local_from_world = local_from_world_transposed.T
    u_matrix, _values, vt_matrix = np.linalg.svd(local_from_world)
    local_from_world = u_matrix @ vt_matrix
    if np.linalg.det(local_from_world) < 0.0:
        u_matrix[:, -1] *= -1.0
        local_from_world = u_matrix @ vt_matrix

    view = reference_payload["view"]
    camera_from_local = np.vstack(
        camera_basis_from_view(
            ViewSettings(
                center_az_deg=float(view["center_az_deg"]),
                center_alt_deg=float(view["center_alt_deg"]),
                roll_deg=float(view["roll_deg"]),
            )
        )
    )
    initial_rotation = camera_from_local @ local_from_world
    return initial_rotation


def _fit_real_pair_payload(payload: dict[str, object], matching_model: str):
    reference_payload = payload["reference_payload"]
    radec = np.asarray([(pair["ra_deg"], pair["dec_deg"]) for pair in payload["pairs"]], dtype=np.float64)
    pixels = np.asarray(
        [(pair["image_x_px"], pair["image_y_px"]) for pair in payload["pairs"]],
        dtype=np.float64,
    )
    image_size = (
        int(payload["real_image"]["display_width_px"]),
        int(payload["real_image"]["display_height_px"]),
    )

    transform = fit_sky_alignment(
        radec,
        pixels,
        matching_model=matching_model,
        image_size=image_size,
        fisheye_fov_deg=float(reference_payload["camera"]["fisheye_fov_deg"]),
        initial_rotation_matrix=_initial_rotation_from_reference_payload(reference_payload),
    )
    return transform, radec, pixels


def test_real_star_pair_json_fisheye_alignment_stays_well_conditioned() -> None:
    json_path = Path("outputs/star_pairs_20260704_171042.json")
    if not json_path.exists():
        pytest.skip("缺少实际星点配对 JSON，跳过真实数据回归测试。")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    transform, radec, pixels = _fit_real_pair_payload(payload, SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT)

    predicted = transform.transform_radec_points(radec)
    assert transform.residual_kind == RESIDUAL_CORRECTION_TPS
    assert transform.rms_px < 1e-6
    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 1e-6
    assert np.nanmin(predicted[:, 0]) > 2500.0
    assert np.nanmax(predicted[:, 0]) < 13000.0
    assert np.nanmin(predicted[:, 1]) > 2000.0
    assert np.nanmax(predicted[:, 1]) < 12500.0


def test_single_capture_equisolid_json_anchors_matched_stars() -> None:
    json_path = Path("outputs/star_pairs_20260704_214537.json")
    if not json_path.exists():
        pytest.skip("缺少单次成像鱼眼配对 JSON，跳过锚点插值回归测试。")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    transform, radec, pixels = _fit_real_pair_payload(payload, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID)

    predicted = transform.transform_radec_points(radec)
    assert transform.residual_kind == RESIDUAL_CORRECTION_TPS
    assert transform.projection_rms_px > 10.0
    assert transform.rms_px < 1e-6
    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 1e-6
