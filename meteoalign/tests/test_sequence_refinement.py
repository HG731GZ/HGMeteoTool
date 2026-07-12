"""图像序列单帧结果修正测试。"""

from __future__ import annotations

import json
from pathlib import Path

from meteoalign.application.app_sequence_refinement import (
    SEQUENCE_REFINEMENT_MODE_POSE,
    SEQUENCE_REFINEMENT_MODE_REFIT,
    SequenceRefinementMixin,
)


def _sample_sequence_payloads(tmp_path: Path) -> tuple[Path, Path]:
    """从 14mm 样本提取少量配对，构造快速且真实的精修输入。"""

    source_dir = Path(__file__).resolve().parents[2] / "testimages" / "14mm"
    starpair_payload = json.loads(
        (source_dir / "A7M3_1214_DSC04176_starpairs.json").read_text(encoding="utf-8")
    )
    starpair_payload["pairs"] = starpair_payload["pairs"][:24]
    starpair_payload["pair_count"] = 24
    starpair_path = tmp_path / "frame_starpairs.json"
    starpair_path.write_text(json.dumps(starpair_payload, ensure_ascii=False), encoding="utf-8")

    model_path = tmp_path / "frame_model.json"
    model_path.write_text(
        (source_dir / "A7M3_1214_DSC04176_model.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return starpair_path, model_path


def test_sequence_refinement_preserves_both_mode_metrics(tmp_path: Path) -> None:
    """两种精修应可先后执行，并保留冻结 Profile 与各自 RMS。"""

    starpair_path, model_path = _sample_sequence_payloads(tmp_path)
    harness = SequenceRefinementMixin()

    pose_rms = harness._refine_sequence_frame(
        starpair_path=starpair_path,
        model_path=model_path,
        mode=SEQUENCE_REFINEMENT_MODE_POSE,
    )
    refit_rms = harness._refine_sequence_frame(
        starpair_path=starpair_path,
        model_path=model_path,
        mode=SEQUENCE_REFINEMENT_MODE_REFIT,
    )
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    refinement = payload["sequence_refinement"]

    assert pose_rms >= 0.0
    assert refit_rms >= 0.0
    assert refinement["base_sequence_camera_calibration_profile"]
    assert refinement["results"][SEQUENCE_REFINEMENT_MODE_POSE]["rms_px"] == pose_rms
    assert refinement["results"][SEQUENCE_REFINEMENT_MODE_REFIT]["rms_px"] == refit_rms
