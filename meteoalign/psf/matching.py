"""星表预测位置与图像检测星源的一对一分配。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import StarSourceCandidate


@dataclass(frozen=True)
class SourceAssignmentResult:
    """一对一分配结果和被判为歧义的星号。"""

    assignments: dict[str, StarSourceCandidate]
    ambiguous_star_ids: frozenset[str]
    diagnostics: dict[str, "SourceAssignmentDiagnostic"] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceAssignmentDiagnostic:
    """一条全局分配边的几何代价、竞争余量和可信度。"""

    distance_px: float
    normalized_distance: float
    assigned_cost: float
    row_margin: float
    source_margin: float
    mutual_best: bool
    confidence_score: float


def _margin_quality(margin: float, ambiguity_margin: float) -> float:
    if np.isposinf(margin):
        return 1.0
    scale = max(float(ambiguity_margin) * 3.0, 1e-6)
    return float(np.clip(float(margin) / scale, 0.0, 1.0))


def merge_source_candidates(
    candidates: list[StarSourceCandidate],
    *,
    merge_radius_px: float = 1.5,
) -> list[StarSourceCandidate]:
    """合并重叠搜索窗口重复检测到的同一星源。"""

    merged: list[StarSourceCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.quality_score, reverse=True):
        duplicate = False
        for accepted in merged:
            if float(np.hypot(candidate.x - accepted.x, candidate.y - accepted.y)) <= merge_radius_px:
                duplicate = True
                break
        if not duplicate:
            merged.append(candidate)
    return merged


def assign_predicted_sources(
    predicted_by_id: dict[str, tuple[float, float]],
    candidates: list[StarSourceCandidate],
    *,
    search_radius_px: float,
    ambiguity_margin: float = 0.10,
    strict_mutual: bool = False,
) -> SourceAssignmentResult:
    """用带未匹配虚拟列的匈牙利算法完成全局一对一分配。

    ``strict_mutual`` 仅供自动扩展匹配启用；序列处理和旧调用保持原有行为。
    """

    star_ids = [star_id for star_id in predicted_by_id if star_id]
    if not star_ids or not candidates:
        return SourceAssignmentResult({}, frozenset(), {})
    radius = max(float(search_radius_px), 1.0)
    row_count = len(star_ids)
    source_count = len(candidates)
    unmatched_cost = 1.08
    invalid_cost = 1e6
    costs = np.full((row_count, source_count + row_count), invalid_cost, dtype=np.float64)
    real_costs = np.full((row_count, source_count), invalid_cost, dtype=np.float64)

    for row, star_id in enumerate(star_ids):
        predicted_x, predicted_y = predicted_by_id[star_id]
        for column, candidate in enumerate(candidates):
            distance = float(np.hypot(candidate.x - predicted_x, candidate.y - predicted_y))
            if distance > radius:
                continue
            axis_ratio = candidate.major_axis / max(candidate.minor_axis, 1e-6)
            shape_penalty = max(axis_ratio - 1.5, 0.0) * 0.035
            snr_penalty = max(5.0 - candidate.snr, 0.0) / 5.0 * 0.20
            blend_penalty = 0.035 if candidate.blended else 0.0
            cost = distance / radius + shape_penalty + snr_penalty + blend_penalty
            costs[row, column] = cost
            real_costs[row, column] = cost
        costs[row, source_count + row] = unmatched_cost

    assigned_rows, assigned_columns = linear_sum_assignment(costs)
    assignments: dict[str, StarSourceCandidate] = {}
    ambiguous_star_ids: set[str] = set()
    diagnostics: dict[str, SourceAssignmentDiagnostic] = {}
    for row, column in zip(assigned_rows, assigned_columns):
        star_id = star_ids[int(row)]
        if column >= source_count or costs[row, column] >= unmatched_cost:
            continue
        finite_columns = np.flatnonzero(real_costs[row] < invalid_cost)
        finite_real = real_costs[row, finite_columns]
        sorted_real = np.sort(finite_real)
        assigned_cost = float(real_costs[row, column])
        other_row_costs = real_costs[row, finite_columns[finite_columns != int(column)]]
        row_margin = (
            float(np.min(other_row_costs) - assigned_cost)
            if other_row_costs.size > 0
            else float("inf")
        )
        finite_rows = np.flatnonzero(real_costs[:, column] < invalid_cost)
        other_source_costs = real_costs[finite_rows[finite_rows != int(row)], column]
        source_margin = (
            float(np.min(other_source_costs) - assigned_cost)
            if other_source_costs.size > 0
            else float("inf")
        )
        row_best_column = int(finite_columns[int(np.argmin(finite_real))])
        source_best_row = int(finite_rows[int(np.argmin(real_costs[finite_rows, column]))])
        mutual_best = row_best_column == int(column) and source_best_row == int(row)
        cost_quality = float(np.clip(1.0 - assigned_cost / unmatched_cost, 0.0, 1.0))
        confidence_score = (
            0.50 * cost_quality
            + 0.25 * _margin_quality(row_margin, ambiguity_margin)
            + 0.25 * _margin_quality(source_margin, ambiguity_margin)
        )
        predicted_x, predicted_y = predicted_by_id[star_id]
        candidate = candidates[int(column)]
        distance_px = float(np.hypot(candidate.x - predicted_x, candidate.y - predicted_y))
        diagnostics[star_id] = SourceAssignmentDiagnostic(
            distance_px=distance_px,
            normalized_distance=distance_px / radius,
            assigned_cost=assigned_cost,
            row_margin=row_margin,
            source_margin=source_margin,
            mutual_best=mutual_best,
            confidence_score=float(np.clip(confidence_score, 0.0, 1.0)),
        )

        row_is_ambiguous = sorted_real.size >= 2 and float(sorted_real[1] - sorted_real[0]) < ambiguity_margin
        strict_is_ambiguous = strict_mutual and (
            not mutual_best
            or row_margin < ambiguity_margin
            or source_margin < ambiguity_margin
        )
        if row_is_ambiguous or strict_is_ambiguous:
            ambiguous_star_ids.add(star_id)
            if strict_mutual and source_margin < ambiguity_margin:
                competing_rows = finite_rows[
                    real_costs[finite_rows, column] <= assigned_cost + float(ambiguity_margin)
                ]
                ambiguous_star_ids.update(star_ids[int(competing_row)] for competing_row in competing_rows)
            continue
        assignments[star_id] = candidate

    if ambiguous_star_ids:
        assignments = {
            star_id: candidate
            for star_id, candidate in assignments.items()
            if star_id not in ambiguous_star_ids
        }
    return SourceAssignmentResult(assignments, frozenset(ambiguous_star_ids), diagnostics)


def source_separation_radius(candidate: StarSourceCandidate, minimum_px: float = 4.0) -> float:
    """根据检测星源尺度给出去重半径。"""

    estimated_fwhm = max(candidate.major_axis, candidate.minor_axis) * 2.0
    return max(float(minimum_px), min(24.0, estimated_fwhm * 0.75))


__all__ = [
    "SourceAssignmentResult",
    "SourceAssignmentDiagnostic",
    "assign_predicted_sources",
    "merge_source_candidates",
    "source_separation_radius",
]
