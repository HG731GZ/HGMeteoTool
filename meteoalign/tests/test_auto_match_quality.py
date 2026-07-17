"""自动扩展匹配几何、分配和亮度质量测试。"""

from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from meteoalign.auto_match_quality import (
    AutoMatchPhotometrySample,
    auto_match_position_quality,
    combine_auto_match_quality,
    evaluate_auto_match_photometry,
    fit_auto_match_photometric_model,
    psf_brightness_proxy,
)
from meteoalign.application.app_star_pair_table_groups import StarPairTableGroupsMixin
from meteoalign.psf.matching import assign_predicted_sources
from meteoalign.psf.models import StarSourceCandidate
from meteoalign.simulator import ReferenceStar
from meteoalign.star_pair_model import PAIR_ORIGIN_AUTO_MATCH, PAIR_ORIGIN_MANUAL, StarPairRecord


def _candidate(x: float, y: float, label: int) -> StarSourceCandidate:
    return StarSourceCandidate(
        x=x,
        y=y,
        major_axis=1.8,
        minor_axis=1.6,
        theta_rad=0.0,
        flux=500.0,
        peak=80.0,
        snr=20.0,
        npix=14,
        label=label,
        quality_score=20.0,
    )


def _flux_for_instrumental_mag(instrumental_mag: float) -> float:
    return 10.0 ** (-0.4 * instrumental_mag)


def _reference_star(star_id: str) -> ReferenceStar:
    return ReferenceStar(
        index=1,
        star_id=star_id,
        name=star_id,
        display_name=star_id,
        common_name="",
        ra_deg=10.0,
        dec_deg=20.0,
        mag_v=2.0,
        sim_x=100.0,
        sim_y=200.0,
        alt_deg=40.0,
        az_deg=180.0,
    )


def test_strict_auto_expansion_rejects_globally_displaced_second_choice() -> None:
    """全局一对一冲突不得把参考星推给它明显更差的第二候选。"""

    predicted = {"A": (2.0, 0.0), "B": (0.0, 0.0)}
    candidates = [_candidate(0.0, 0.0, 1), _candidate(6.0, 0.0, 2)]

    legacy = assign_predicted_sources(predicted, candidates, search_radius_px=10.0)
    strict = assign_predicted_sources(
        predicted,
        candidates,
        search_radius_px=10.0,
        strict_mutual=True,
    )

    # 默认路径保持旧行为，确保序列处理不受自动扩展专用约束影响。
    assert legacy.assignments["A"].x == pytest.approx(6.0)
    assert legacy.assignments["B"].x == pytest.approx(0.0)
    assert "A" not in legacy.ambiguous_star_ids

    assert "A" not in strict.assignments
    assert "A" in strict.ambiguous_star_ids
    assert strict.assignments["B"].x == pytest.approx(0.0)
    assert not strict.diagnostics["A"].mutual_best
    assert strict.diagnostics["B"].mutual_best


def test_photometric_model_rejects_bright_catalog_star_matched_to_faint_source() -> None:
    """亮参考星若匹配到明显更暗的图像源，应被测光一致性拒绝。"""

    samples = [
        AutoMatchPhotometrySample(
            star_id=f"S{index}",
            catalog_mag=float(index),
            flux=_flux_for_instrumental_mag(float(index) - 10.0),
            x_px=100.0 + index * 80.0,
            y_px=200.0 + index * 40.0,
        )
        for index in range(6)
    ]
    wrong = AutoMatchPhotometrySample(
        star_id="wrong",
        catalog_mag=0.5,
        # 该流量相当于同一曝光下约 4.5 等的暗星。
        flux=_flux_for_instrumental_mag(4.5 - 10.0),
        x_px=700.0,
        y_px=500.0,
    )
    model = fit_auto_match_photometric_model([*samples, wrong], (1000, 800))

    assert model is not None
    evaluation = evaluate_auto_match_photometry(wrong, model, saturated=False)
    assert evaluation is not None
    assert evaluation.residual_mag > 3.0
    assert evaluation.quality_score is not None
    assert evaluation.quality_score < 0.05
    assert evaluation.reject


def test_psf_brightness_proxy_uses_saturated_star_size() -> None:
    """峰值相同时，更大的饱和星 PSF 应得到更高的亮度代理。"""

    compact = psf_brightness_proxy(200.0, 2.0, 2.0)
    expanded = psf_brightness_proxy(200.0, 4.0, 4.0)

    assert expanded == pytest.approx(compact * 4.0)


def test_saturated_psf_brightness_supports_two_sided_outlier_check() -> None:
    """饱和星也应产生亮度分，并拒绝与星表亮度明显不符的 PSF。"""

    samples = [
        AutoMatchPhotometrySample(
            star_id=f"S{index}",
            catalog_mag=float(index),
            # 饱和响应会被压缩，所以仪器星等斜率不强制为 1。
            flux=_flux_for_instrumental_mag(0.78 * float(index) - 13.7),
            x_px=100.0 + index * 80.0,
            y_px=200.0,
        )
        for index in range(8)
    ]
    model = fit_auto_match_photometric_model(samples, (1000, 800))
    assert model is not None

    correct = AutoMatchPhotometrySample(
        star_id="correct-saturated",
        catalog_mag=2.5,
        flux=_flux_for_instrumental_mag(0.78 * 2.5 - 13.7 + 0.4),
        x_px=500.0,
        y_px=400.0,
    )
    wrong = AutoMatchPhotometrySample(
        star_id="wrong-saturated",
        catalog_mag=0.5,
        # 很亮的星表目标却套到了约 4 等星大小的 PSF 上。
        flux=_flux_for_instrumental_mag(0.78 * 4.0 - 13.7),
        x_px=500.0,
        y_px=400.0,
    )
    correct_evaluation = evaluate_auto_match_photometry(correct, model, saturated=True)
    wrong_evaluation = evaluate_auto_match_photometry(wrong, model, saturated=True)

    assert correct_evaluation is not None
    assert not correct_evaluation.reject
    assert correct_evaluation.quality_score is not None
    assert correct_evaluation.quality_score > 0.80
    assert wrong_evaluation is not None
    assert wrong_evaluation.reject
    assert wrong_evaluation.quality_score is not None
    assert wrong_evaluation.quality_score < 0.01


def test_14mm_saturated_stars_have_stable_photometric_quality() -> None:
    """真实星野中的柳宿六和天囷一不应因饱和而失去亮度检查或被降成低分。"""

    model_path = (
        Path(__file__).resolve().parents[2]
        / "testimages"
        / "14mm"
        / "A7M3_1214_DSC04149_model.json"
    )
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    samples: list[AutoMatchPhotometrySample] = []
    named_samples: dict[str, AutoMatchPhotometrySample] = {}
    named_pairs: dict[str, dict[str, object]] = {}
    for pair in payload["fit_pairs"]:
        if pair.get("pair_origin") != "auto_match":
            continue
        sample = AutoMatchPhotometrySample(
            star_id=str(pair["star_id"]),
            catalog_mag=float(pair["mag_v"]),
            flux=psf_brightness_proxy(
                pair["amplitude"],
                pair["sigma_x"],
                pair["sigma_y"],
            ),
            x_px=float(pair["image_x_px"]),
            y_px=float(pair["image_y_px"]),
        )
        samples.append(sample)
        named_samples[str(pair["name"])] = sample
        named_pairs[str(pair["name"])] = pair

    geometry = payload["image_geometry"]
    photometric_model = fit_auto_match_photometric_model(
        samples,
        (int(geometry["width_px"]), int(geometry["height_px"])),
    )

    assert photometric_model is not None
    assert photometric_model.sample_count >= 90
    for name in ("柳宿六", "天囷一"):
        evaluation = evaluate_auto_match_photometry(
            named_samples[name],
            photometric_model,
            saturated=True,
        )
        assert evaluation is not None
        assert not evaluation.reject
        assert evaluation.quality_score is not None
        assert evaluation.quality_score > 0.90
        pair = named_pairs[name]
        position_score = auto_match_position_quality(
            float(pair["auto_match_prediction_offset_px"]),
            float(payload["matching"]["search_radius_px"]),
        )
        quality_score, _geometry_score = combine_auto_match_quality(
            position_score,
            float(pair["auto_match_assignment_score"]),
            evaluation.quality_score,
        )
        assert quality_score > 0.75


def test_combined_quality_uses_photometry_only_when_available() -> None:
    """测光样本不足时综合质量应退化为纯几何质量。"""

    without_photometry, geometry = combine_auto_match_quality(0.8, 0.6, None)
    with_photometry, geometry_again = combine_auto_match_quality(0.8, 0.6, 0.0)

    assert geometry == pytest.approx(0.67)
    assert without_photometry == pytest.approx(geometry)
    assert geometry_again == pytest.approx(geometry)
    assert with_photometry == pytest.approx(geometry * 0.60)
    assert math.isfinite(with_photometry)


def test_position_quality_uses_wider_relative_scale() -> None:
    """预测偏差约为搜索半径六成时，不应把准确的宽角边缘星压成极低分。"""

    assert auto_match_position_quality(6.0, 10.0) == pytest.approx(math.exp(-0.5))


def test_quality_column_ignores_manual_pair_even_if_extra_field_exists() -> None:
    """新指标列必须只显示自动扩展匹配，手动匹配始终留空。"""

    records = [
        StarPairRecord(
            reference_star=_reference_star("manual"),
            image_x_px=10.0,
            image_y_px=20.0,
            pair_origin=PAIR_ORIGIN_MANUAL,
            extra_fields={"auto_match_quality_score": 0.99},
        ),
        StarPairRecord(
            reference_star=_reference_star("auto"),
            image_x_px=30.0,
            image_y_px=40.0,
            pair_origin=PAIR_ORIGIN_AUTO_MATCH,
            extra_fields={"auto_match_quality_score": 0.72},
        ),
    ]
    host = SimpleNamespace(_star_pair_record_for_row=lambda row: records[row])

    assert StarPairTableGroupsMixin._star_pair_quality_score(host, 0) is None
    assert StarPairTableGroupsMixin._star_pair_quality_score(host, 1) == pytest.approx(0.72)
