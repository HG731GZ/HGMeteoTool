from __future__ import annotations

from dataclasses import dataclass

from .simulator import ReferenceStar
from .star_fitting import FittedStarPosition


@dataclass(frozen=True)
class _SequencePairTemplate:
    star_id: str
    reference_star: ReferenceStar
    fitted_position: FittedStarPosition
    fit_constraint_mode: str
    fit_weight: float
    pair_origin: str
    auto_match_quality_score: float | None = None


@dataclass(frozen=True)
class _SequenceCandidate:
    reference_star: ReferenceStar
    predicted_x_px: float
    predicted_y_px: float


@dataclass(frozen=True)
class _SequenceFitPlan:
    candidate: _SequenceCandidate
    fit_weight: float
    pair_origin: str


@dataclass(frozen=True)
class _SequenceMatchedPair:
    reference_star: ReferenceStar
    fitted_position: FittedStarPosition
    fit_constraint_mode: str
    fit_weight: float
    pair_origin: str
    auto_match_quality_score: float | None = None
    predicted_x_px: float | None = None
    predicted_y_px: float | None = None
    search_x_px: float | None = None
    search_y_px: float | None = None
    initial_predicted_x_px: float | None = None
    initial_predicted_y_px: float | None = None
    time_delta_seconds: float | None = None
    adaptive_offset_x_px: float = 0.0
    adaptive_offset_y_px: float = 0.0
