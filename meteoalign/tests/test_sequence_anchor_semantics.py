from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from meteoalign.alignment import SKY_MATCHING_MODEL_RECTILINEAR
from meteoalign.application import app_sequence_matching
from meteoalign.application.app_constants import AUTO_MATCH_CONSTRAINT_SOFT
from meteoalign.application.app_sequence import SequenceBatchMixin, _SequenceCandidate, _SequencePairTemplate
from meteoalign.auto_match_quality import AUTO_MATCH_QUALITY_SCORE_KEY
from meteoalign.simulator import (
    ObserverSettings,
    ReferenceStar,
    ViewSettings,
    compute_altaz_from_radec,
    local_vectors_from_altaz,
)
from meteoalign.star_fitting import FittedStarPosition
from meteoalign.star_pair_model import PAIR_ORIGIN_AUTO_MATCH, PsfFit, StarPairRecord
from meteoalign.star_pair_store import StarPairStore


def _star(star_id: str, mag_v: float = 1.0) -> ReferenceStar:
    return ReferenceStar(
        index=0,
        star_id=star_id,
        name=star_id,
        display_name=star_id,
        common_name="",
        ra_deg=10.0,
        dec_deg=20.0,
        mag_v=mag_v,
        sim_x=100.0,
        sim_y=120.0,
        alt_deg=40.0,
        az_deg=180.0,
    )


def _template(star_id: str, mode: str = "anchor") -> _SequencePairTemplate:
    return _SequencePairTemplate(
        star_id=star_id,
        reference_star=_star(star_id),
        fitted_position=FittedStarPosition(
            x=100.0,
            y=120.0,
            amplitude=10.0,
            background=1.0,
            sigma_x=1.5,
            sigma_y=1.5,
        ),
        fit_constraint_mode=mode,
        fit_weight=1.0,
        pair_origin="manual",
    )


def _candidate(star_id: str, x_px: float = 100.0, y_px: float = 120.0, mag_v: float = 1.0) -> _SequenceCandidate:
    return _SequenceCandidate(
        reference_star=_star(star_id, mag_v),
        predicted_x_px=x_px,
        predicted_y_px=y_px,
    )


def test_sequence_candidates_do_not_replace_missing_anchor_identity() -> None:
    harness = SequenceBatchMixin()
    templates = [_template("HR1"), _template("HR2")]
    candidates = [_candidate("HR2"), _candidate("HR3")]

    ordered = harness._ordered_sequence_candidates_for_mode(
        candidates,
        templates,
        "anchor",
        used_star_ids=set(),
    )

    assert [template.star_id for _candidate_item, template in ordered] == ["HR2"]


def test_sequence_candidates_do_not_replace_missing_soft_identity() -> None:
    harness = SequenceBatchMixin()
    templates = [_template("HR1", AUTO_MATCH_CONSTRAINT_SOFT), _template("HR2", AUTO_MATCH_CONSTRAINT_SOFT)]
    candidates = [_candidate("HR2"), _candidate("HR3")]

    ordered = harness._ordered_sequence_candidates_for_mode(
        candidates,
        templates,
        AUTO_MATCH_CONSTRAINT_SOFT,
        used_star_ids=set(),
    )

    assert [template.star_id for _candidate_item, template in ordered] == ["HR2"]


def test_sequence_pair_target_keeps_eighty_percent_floor() -> None:
    harness = SequenceBatchMixin()
    templates = [_template(f"HR{index}") for index in range(100)]

    minimum_count, target_count = harness._sequence_pair_targets(templates)

    assert minimum_count == 80
    assert target_count == 86


def test_first_frame_export_preserves_auto_match_quality_only_for_auto_pairs() -> None:
    """固定相机首帧重建 fit_pairs 时不得丢失自动匹配综合质量。"""

    store = StarPairStore()
    for index in range(4):
        fitted = FittedStarPosition(
            x=100.0 + index,
            y=120.0 + index,
            amplitude=10.0,
            background=1.0,
            sigma_x=1.5,
            sigma_y=1.5,
            quality_score=0.80 + index * 0.01,
        )
        is_auto_match = index == 3
        store.add(
            StarPairRecord(
                reference_star=_star(f"HR{index}"),
                image_x_px=fitted.x,
                image_y_px=fitted.y,
                psf=PsfFit.from_fitted_position(fitted),
                pair_origin=PAIR_ORIGIN_AUTO_MATCH if is_auto_match else "manual",
                extra_fields=(
                    {AUTO_MATCH_QUALITY_SCORE_KEY: 0.73}
                    if is_auto_match
                    else {}
                ),
            )
        )
    harness = SequenceBatchMixin()
    harness._star_pair_store = store

    templates = harness._sequence_base_templates()
    pairs = harness._first_frame_matched_pairs(templates)
    payloads = harness._sequence_pair_records(pairs)
    payload_by_id = {str(payload["star_id"]): payload for payload in payloads}

    assert payload_by_id["HR3"][AUTO_MATCH_QUALITY_SCORE_KEY] == 0.73
    assert AUTO_MATCH_QUALITY_SCORE_KEY not in payload_by_id["HR0"]
    assert payload_by_id["HR0"]["quality_score"] == 0.80


def test_fixed_camera_fit_recomputes_altaz_for_current_observation_time(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """固定相机拟合不能复用记录中可能过期的地平坐标。"""

    observer = ObserverSettings(
        observation_time_utc=datetime(2017, 12, 31, 16, 18, 33, tzinfo=timezone.utc),
        latitude_deg=40.0,
        longitude_deg=116.0,
        elevation_m=200.0,
    )
    templates = [
        replace(
            _template(f"HR{index}"),
            reference_star=replace(
                _star(f"HR{index}"),
                ra_deg=10.0 + index * 12.0,
                dec_deg=20.0 + index * 3.0,
                alt_deg=-70.0 + index,
                az_deg=270.0 + index,
            ),
        )
        for index in range(4)
    ]
    captured: dict[str, object] = {}
    expected_model = object()

    def _capture_fixed_camera_fit(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(app_sequence_matching, "fit_fixed_camera_model", _capture_fixed_camera_fit)
    harness = SequenceBatchMixin()
    harness._sequence_fixed_lens_model = lambda _target_size: SKY_MATCHING_MODEL_RECTILINEAR
    harness._view_settings = lambda: ViewSettings(center_az_deg=35.0, center_alt_deg=25.0)

    result = harness._fit_sequence_fixed_camera_model(templates, (6000, 4000), observer)

    ra_deg = np.asarray([template.reference_star.ra_deg for template in templates], dtype=np.float64)
    dec_deg = np.asarray([template.reference_star.dec_deg for template in templates], dtype=np.float64)
    expected_alt_deg, expected_az_deg = compute_altaz_from_radec(ra_deg, dec_deg, observer)
    expected_vectors = local_vectors_from_altaz(expected_alt_deg, expected_az_deg)
    stale_vectors = local_vectors_from_altaz(
        np.asarray([template.reference_star.alt_deg for template in templates], dtype=np.float64),
        np.asarray([template.reference_star.az_deg for template in templates], dtype=np.float64),
    )

    assert result is expected_model
    assert np.allclose(captured["enu_vectors"], expected_vectors)
    assert not np.allclose(captured["enu_vectors"], stale_vectors)


def test_sequence_supplemental_candidates_prefer_sparse_cells_over_brightness() -> None:
    harness = SequenceBatchMixin()
    accepted_positions = [(850.0, 100.0), (870.0, 120.0), (890.0, 140.0)]
    candidates = [
        _candidate("bright-crowded", 860.0, 180.0, mag_v=0.1),
        _candidate("faint-empty", 120.0, 120.0, mag_v=4.0),
        _candidate("mid-empty", 520.0, 620.0, mag_v=2.0),
    ]

    ordered = harness._sequence_order_supplemental_candidates(
        candidates,
        used_star_ids=set(),
        attempted_star_ids=set(),
        accepted_positions=accepted_positions,
        target_size=(1000, 800),
    )

    assert [candidate.reference_star.star_id for candidate in ordered[:2]] == ["faint-empty", "mid-empty"]


def test_sequence_supplemental_candidates_prioritize_previous_frame_ids() -> None:
    """上一帧成功的补充星应排在普通空间均衡候选之前。"""

    harness = SequenceBatchMixin()
    candidates = [
        _candidate("ordinary", 120.0, 120.0, mag_v=1.0),
        _candidate("previous", 860.0, 180.0, mag_v=4.0),
    ]

    ordered = harness._sequence_order_supplemental_candidates(
        candidates,
        used_star_ids=set(),
        attempted_star_ids=set(),
        accepted_positions=[(850.0, 100.0)],
        target_size=(1000, 800),
        preferred_star_ids=("previous",),
    )

    assert [candidate.reference_star.star_id for candidate in ordered] == ["previous", "ordinary"]


def test_sequence_supplemental_candidates_use_batches_and_stop_early() -> None:
    """补星应逐批尝试，并在达到目标后不再扫描剩余候选。"""

    harness = SequenceBatchMixin()
    candidates = [
        _candidate(f"HR{index}", float(index % 20) * 40.0 + 10.0, float(index // 20) * 40.0 + 10.0)
        for index in range(130)
    ]
    calls: list[tuple[int, int, bool]] = []

    def _fake_fit(
        _luminance,
        plans,
        target_count,
        _search_radius_px,
        _used_star_ids,
        _attempted_star_ids,
        _accepted_positions,
        _accepted_offsets,
        _stats,
        *,
        record_missing_target=True,
    ):
        calls.append((len(plans), target_count, record_missing_target))
        if len(calls) == 1:
            return []
        return [object()] * target_count

    harness._fit_sequence_candidate_plans = _fake_fit
    stats = {"missing_target": 0, "supplemental_matched": 0}

    matched = harness._fit_sequence_supplemental_candidates(
        np.zeros((800, 1000), dtype=np.uint8),
        candidates,
        (1000, 800),
        3,
        20,
        set(),
        set(),
        [],
        [],
        stats,
    )

    assert len(matched) == 3
    assert calls == [(64, 3, False), (64, 3, False)]
    assert stats["supplemental_batches"] == 2
    assert stats["supplemental_candidates_attempted"] == 128
    assert stats["missing_target"] == 0
