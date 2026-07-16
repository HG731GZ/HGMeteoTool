"""星表预测位置与图像检测星源的一对一分配。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import StarSourceCandidate


@dataclass(frozen=True)
class SourceAssignmentResult:
    """一对一分配结果和被判为歧义的星号。"""

    assignments: dict[str, StarSourceCandidate]
    ambiguous_star_ids: frozenset[str]


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
) -> SourceAssignmentResult:
    """用带未匹配虚拟列的匈牙利算法完成全局一对一分配。"""

    star_ids = [star_id for star_id in predicted_by_id if star_id]
    if not star_ids or not candidates:
        return SourceAssignmentResult({}, frozenset())
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
    for row, column in zip(assigned_rows, assigned_columns):
        star_id = star_ids[int(row)]
        if column >= source_count or costs[row, column] >= unmatched_cost:
            continue
        finite_real = np.sort(real_costs[row][real_costs[row] < invalid_cost])
        if finite_real.size >= 2 and float(finite_real[1] - finite_real[0]) < ambiguity_margin:
            ambiguous_star_ids.add(star_id)
            continue
        assignments[star_id] = candidates[int(column)]

    return SourceAssignmentResult(assignments, frozenset(ambiguous_star_ids))


def source_separation_radius(candidate: StarSourceCandidate, minimum_px: float = 4.0) -> float:
    """根据检测星源尺度给出去重半径。"""

    estimated_fwhm = max(candidate.major_axis, candidate.minor_axis) * 2.0
    return max(float(minimum_px), min(24.0, estimated_fwhm * 0.75))


__all__ = [
    "SourceAssignmentResult",
    "assign_predicted_sources",
    "merge_source_candidates",
    "source_separation_radius",
]
