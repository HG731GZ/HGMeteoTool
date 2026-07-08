from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meteoalign.alignment import (
    RESIDUAL_CORRECTION_TPS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_RECTILINEAR,
    _project_unit_vectors_with_known_projection,
    fit_sky_alignment,
)
from meteoalign.coordinates import normalize_vector, radec_to_unit_vectors, sky_plane_to_radec
from meteoalign.frame_astrometry import FrameAstrometricModel
from meteoalign.simulator import ViewSettings, camera_basis_from_view, local_vectors_from_altaz
from meteoalign.camera_calibration import CameraCalibrationProfile
from meteoalign.source_model import fit_source_astrometric_model, fit_source_astrometric_model_with_fixed_profile


KNOWN_PROJECTION_MODELS = (
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
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


def test_known_projection_soft_constraint_does_not_force_bad_auto_match() -> None:
    radec, pixels = _projection_fixture(SKY_MATCHING_MODEL_RECTILINEAR)
    target_pixels = pixels.copy()
    target_pixels[-1] += np.asarray([90.0, -70.0], dtype=np.float64)
    point_weights = np.ones(target_pixels.shape[0], dtype=np.float64)
    point_weights[-1] = 0.2
    anchor_mask = np.ones(target_pixels.shape[0], dtype=bool)
    anchor_mask[-1] = False

    transform = fit_sky_alignment(
        radec,
        target_pixels,
        matching_model=SKY_MATCHING_MODEL_RECTILINEAR,
        image_size=(1000, 800),
        point_weights=point_weights,
        residual_anchor_mask=anchor_mask,
    )

    predicted = transform.transform_radec_points(radec)
    hard_anchor_error = np.linalg.norm(predicted[:-1] - target_pixels[:-1], axis=1)
    soft_error = float(np.linalg.norm(predicted[-1] - target_pixels[-1]))
    assert transform.residual_soft_constraint_count == 1
    assert np.max(hard_anchor_error) < 1e-5
    assert soft_error > 1.0


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
    assert payload["fit_metadata"]["model_type"] == "known_projection_with_residual_interpolation"
    assert payload["frame_pose"]["type"] == "rotation_matrix"
    profile = payload["camera_calibration_profile"]
    assert profile["base_projection"]["type"] == SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
    assert profile["base_projection"]["parameters"]["projection_code"] == "ARC"
    assert profile["base_projection"]["parameters"]["display_name"] == "等距鱼眼(ARC)"
    assert profile["base_projection"]["parameters"]["fov_deg"] is None
    assert profile["global_distortion"]["type"] == "tps_residual_field"
    assert profile["global_distortion"]["parameters"]["kind"] == RESIDUAL_CORRECTION_TPS
    assert "degree" not in profile["global_distortion"]["parameters"]
    assert "coeff_x_px" not in profile["global_distortion"]["parameters"]
    assert payload["frame_local_residual"]["enabled"] is False
    assert payload["diagnostics"]["rms_px"] < 1e-6
    assert payload["diagnostics"]["round_trip_rms_px"] < 1e-6

    restored = FrameAstrometricModel.from_json_payload(payload)
    restored_pixels = restored.sky_to_pixel_points(radec)
    assert np.max(np.linalg.norm(restored_pixels - pixels, axis=1)) < 1e-5
    restored_radec = restored.pixel_to_sky_points(pixels)
    expected_vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])
    actual_vectors = radec_to_unit_vectors(restored_radec[:, 0], restored_radec[:, 1])
    assert np.max(np.linalg.norm(actual_vectors - expected_vectors, axis=1)) < 1e-6


def test_fixed_profile_pose_solver_reuses_embedded_profile_without_refitting_intrinsics() -> None:
    radec, pixels = _projection_fixture(SKY_MATCHING_MODEL_RECTILINEAR)
    transform = fit_sky_alignment(
        radec,
        pixels,
        matching_model=SKY_MATCHING_MODEL_RECTILINEAR,
        image_size=(1000, 800),
    )
    profile = CameraCalibrationProfile.from_projection_transform(transform)

    imported_profile_model = fit_source_astrometric_model_with_fixed_profile(
        radec,
        pixels,
        image_size=(1000, 800),
        camera_calibration_profile=profile,
        initial_rotation_matrix=transform.rotation_matrix,
        profile_source_path="synthetic_model.json",
    )
    frame_model = imported_profile_model.to_frame_astrometric_model()
    predicted = frame_model.sky_to_pixel_points(radec)

    assert frame_model.camera_calibration_profile.to_json_payload() == profile.to_json_payload()
    assert frame_model.fit_metadata["camera_profile_reuse"]["profile_frozen"] is True
    assert frame_model.fit_metadata["camera_profile_reuse"]["automatic_equipment_check"] is False
    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 1e-5


def test_fixed_profile_pose_local_residual_round_trips_from_json() -> None:
    radec, pixels = _projection_fixture(SKY_MATCHING_MODEL_RECTILINEAR)
    transform = fit_sky_alignment(
        radec,
        pixels,
        matching_model=SKY_MATCHING_MODEL_RECTILINEAR,
        image_size=(1000, 800),
    )
    profile = CameraCalibrationProfile.from_projection_transform(transform)
    warped_pixels = pixels + np.column_stack(
        (
            0.02 * (pixels[:, 0] - 500.0),
            -0.015 * (pixels[:, 1] - 400.0),
        )
    )

    imported_profile_model = fit_source_astrometric_model_with_fixed_profile(
        radec,
        warped_pixels,
        image_size=(1000, 800),
        camera_calibration_profile=profile,
        initial_rotation_matrix=transform.rotation_matrix,
        solve_mode="imported_profile_pose_local_residual",
    )
    frame_model = imported_profile_model.to_frame_astrometric_model()
    restored = FrameAstrometricModel.from_json_payload(frame_model.to_json_payload())
    predicted = restored.sky_to_pixel_points(radec)
    restored_radec = restored.pixel_to_sky_points(warped_pixels)

    assert restored.frame_local_residual.enabled is True
    assert restored.frame_local_residual.to_json_payload()["parameters"]["extrapolation_policy"] == (
        "identity_outside_anchor_bbox_padding"
    )
    assert np.max(np.linalg.norm(predicted - warped_pixels, axis=1)) < 1e-5
    expected_vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])
    actual_vectors = radec_to_unit_vectors(restored_radec[:, 0], restored_radec[:, 1])
    assert np.max(np.linalg.norm(actual_vectors - expected_vectors, axis=1)) < 1e-6


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
        fisheye_fov_deg=None,
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


def test_real_star_pair_json_rectilinear_alignment_uses_reference_pose() -> None:
    json_path = Path("outputs/star_pairs_20260705_210257.json")
    if not json_path.exists():
        pytest.skip("缺少 14mm 普通广角配对 JSON，跳过真实数据回归测试。")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    transform, radec, pixels = _fit_real_pair_payload(payload, SKY_MATCHING_MODEL_RECTILINEAR)

    raw_projected = transform.raw_project_radec_points(radec)
    predicted = transform.transform_radec_points(radec)
    assert transform.residual_kind == RESIDUAL_CORRECTION_TPS
    assert transform.projection_rms_px < 20.0
    assert transform.scale_px > 1000.0
    assert np.nanmin(raw_projected[:, 0]) > 500.0
    assert np.nanmax(raw_projected[:, 0]) < 5600.0
    assert np.nanmin(raw_projected[:, 1]) > 300.0
    assert np.nanmax(raw_projected[:, 1]) < 3200.0
    assert transform.rms_px < 1e-6
    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 1e-6


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
