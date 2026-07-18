from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meteoalign.alignment import fit_sky_alignment
from meteoalign.alignment.models import SkyAlignmentTransform
from meteoalign.application.app_alignment import AlignmentMixin
from meteoalign.application.app_star_pair_session import StarPairSessionMixin
from meteoalign.application.app_constants import STAR_PAIR_SESSION_VERSION
from meteoalign.frame_astrometry import FrameAstrometricModel, FramePose


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "testimages" / "28mm测试" / "残差异常"
FIXTURE_IMAGE = FIXTURE_DIR / "IMG_0084.TIF"
FIXTURE_STARPAIRS = FIXTURE_DIR / "IMG_0084_starpairs.json"
FIXTURE_MODEL = FIXTURE_DIR / "IMG_0084_model.json"


def _fixture_payloads() -> tuple[dict[str, object], FrameAstrometricModel]:
    starpairs_payload = json.loads(FIXTURE_STARPAIRS.read_text(encoding="utf-8"))
    model_payload = json.loads(FIXTURE_MODEL.read_text(encoding="utf-8"))
    return starpairs_payload, FrameAstrometricModel.from_json_payload(model_payload)


def _fit_fixture_with_pose(
    payload: dict[str, object],
    rotation: np.ndarray,
) -> SkyAlignmentTransform:
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    pair_records = [pair for pair in pairs if isinstance(pair, dict)]
    radec = np.asarray(
        [(float(pair["ra_deg"]), float(pair["dec_deg"])) for pair in pair_records],
        dtype=np.float64,
    )
    pixels = np.asarray(
        [(float(pair["image_x_px"]), float(pair["image_y_px"])) for pair in pair_records],
        dtype=np.float64,
    )
    weights = np.asarray([float(pair.get("fit_weight", 1.0)) for pair in pair_records], dtype=np.float64)
    anchors = np.asarray(
        [str(pair.get("fit_constraint_mode", "anchor")) != "soft" for pair in pair_records],
        dtype=bool,
    )
    real_image = payload["real_image"]
    assert isinstance(real_image, dict)
    return fit_sky_alignment(
        ra_dec_points=radec,
        target_points=pixels,
        matching_model=str(payload["sky_alignment_model"]),
        image_size=(int(real_image["original_width_px"]), int(real_image["original_height_px"])),
        initial_rotation_matrix=rotation,
        point_weights=weights,
        residual_anchor_mask=anchors,
    )


def test_old_starpairs_falls_back_to_sibling_model_pose_and_restores_low_rms() -> None:
    """旧匹配 JSON 没有姿态时，应使用同名 model 的姿态避免误入高残差局部解。"""

    payload, model = _fixture_payloads()
    harness = StarPairSessionMixin()

    restored_rotation = harness._restored_frame_pose_rotation(
        payload,
        FIXTURE_IMAGE,
        (model.image_width_px, model.image_height_px),
    )

    assert restored_rotation is not None
    assert np.allclose(restored_rotation, model.frame_pose.icrs_to_camera)
    transform = _fit_fixture_with_pose(payload, restored_rotation)
    assert transform.rms_px < 1.0
    assert abs(transform.rms_px - 0.827354) < 1e-3


def test_starpairs_pose_takes_priority_over_sibling_model_pose() -> None:
    """starpairs 已有姿态时，不得再被旁边的 model 覆盖。"""

    payload, model = _fixture_payloads()
    session_pose = FramePose(np.diag([1.0, -1.0, -1.0]))
    payload["frame_pose"] = session_pose.to_json_payload()
    harness = StarPairSessionMixin()

    restored_rotation = harness._restored_frame_pose_rotation(
        payload,
        FIXTURE_IMAGE,
        (model.image_width_px, model.image_height_px),
    )

    assert restored_rotation is not None
    assert np.allclose(restored_rotation, session_pose.icrs_to_camera)
    assert not np.allclose(restored_rotation, model.frame_pose.icrs_to_camera)


def test_exported_starpairs_uses_current_solved_source_model_pose() -> None:
    """导出匹配 JSON 时应从当前已求解源图模型写入 frame_pose。"""

    _payload, model = _fixture_payloads()
    harness = StarPairSessionMixin()
    harness._source_astrometric_model = SimpleNamespace(to_frame_astrometric_model=lambda: model)
    harness._sky_alignment_transform = None

    pose_payload = harness._current_solved_frame_pose_payload()

    assert pose_payload is not None
    restored_pose = FramePose.from_json_payload(pose_payload)
    assert np.allclose(restored_pose.icrs_to_camera, model.frame_pose.icrs_to_camera)


def test_built_starpairs_payload_contains_current_solved_pose(tmp_path: Path) -> None:
    """完整的匹配 JSON 导出路径应把已求解姿态放在顶层 frame_pose。"""

    _payload, model = _fixture_payloads()
    image_path = tmp_path / "frame.tif"
    harness = StarPairSessionMixin()
    harness.current_image_preview = SimpleNamespace(
        path=image_path,
        original_width=model.image_width_px,
        original_height=model.image_height_px,
        image=SimpleNamespace(width=lambda: model.image_width_px, height=lambda: model.image_height_px),
    )
    harness._source_astrometric_model = SimpleNamespace(to_frame_astrometric_model=lambda: model)
    harness._sky_alignment_transform = None
    harness._auto_match_reference_star_ids = set()
    harness._star_pair_records = lambda: []
    harness._single_image_fixed_camera_export_bundle = lambda: (_ for _ in ()).throw(ValueError("not ready"))
    harness._build_reference_payload_for_records = lambda _records: {}
    harness._current_real_image_capture_payload = lambda: {}
    harness._with_current_simulator_time = lambda payload: payload
    harness._alignment_model = lambda: "rectilinear"
    harness._auto_match_groups_payload = lambda: []
    harness._auto_match_constraints_payload = lambda: {}
    harness._sky_mask_payload = lambda _json_path: None
    harness._auto_match_settings_payload = lambda: {}

    exported = harness._build_star_pair_session_payload(tmp_path / "frame_starpairs.json")

    assert exported["version"] == STAR_PAIR_SESSION_VERSION == 2
    restored_pose = FramePose.from_json_payload(exported["frame_pose"])
    assert np.allclose(restored_pose.icrs_to_camera, model.frame_pose.icrs_to_camera)


def test_manual_alignment_prefers_restored_starpairs_pose_as_fit_seed() -> None:
    """交互页重算 RMS 时必须把恢复的姿态传给投影拟合。"""

    _payload, model = _fixture_payloads()
    harness = AlignmentMixin()
    harness._restored_alignment_initial_rotation_matrix = model.frame_pose.icrs_to_camera

    initial_rotation = harness._manual_projection_initial_rotation_matrix()

    assert initial_rotation is not None
    assert np.allclose(initial_rotation, model.frame_pose.icrs_to_camera)
    assert initial_rotation is not harness._restored_alignment_initial_rotation_matrix


def test_fixed_profile_collects_all_available_pose_candidates() -> None:
    """固定 Profile 应同时保留恢复姿态、粗取景和模拟页姿态供求解器竞争。"""

    harness = AlignmentMixin()
    restored = np.eye(3, dtype=np.float64)
    rough = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    simulated = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), dtype=np.float64)
    harness._restored_alignment_initial_rotation_matrix = restored
    harness._rough_framing_initial_rotation_matrix = lambda: rough.copy()
    harness._initial_projection_rotation_matrix = lambda: simulated.copy()

    candidates = harness._fixed_profile_pose_initial_rotation_matrices()

    assert len(candidates) == 3
    assert np.allclose(candidates[0], restored)
    assert np.allclose(candidates[1], rough)
    assert np.allclose(candidates[2], simulated)
