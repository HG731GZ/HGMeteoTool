from __future__ import annotations

from meteoalign.app_constants import AUTO_MATCH_CONSTRAINT_SOFT
from meteoalign.app_sequence import SequenceBatchMixin, _SequenceCandidate, _SequencePairTemplate
from meteoalign.simulator import ReferenceStar
from meteoalign.star_fitting import FittedStarPosition


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

    minimum_count, desired_count = harness._sequence_pair_targets(templates)

    assert minimum_count == 80
    assert desired_count == 100


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
