from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .bspline import (
    mean_field_weights,
    rasterize_correction,
    smoothness_regularization,
    tensor_product_basis,
)
from .types import (
    CancelCallback,
    DiagnosticsResult,
    ObservationSet,
    PhotometricSolution,
    ProgressCallback,
    SolverConfig,
)


def _normalized_points(points_xy: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).copy()
    points[:, 0] /= max(1, width_px - 1)
    points[:, 1] /= max(1, height_px - 1)
    return np.clip(points, 0.0, 1.0)


def build_observation_matrix(
    observations: ObservationSet,
    config: SolverConfig,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    coefficient_count = config.grid_columns * config.grid_rows
    point_i = _normalized_points(
        observations.point_i_xy,
        observations.image_width_px,
        observations.image_height_px,
    )
    point_j = _normalized_points(
        observations.point_j_xy,
        observations.image_width_px,
        observations.image_height_px,
    )
    basis_i = tensor_product_basis(
        point_i,
        columns=config.grid_columns,
        rows=config.grid_rows,
    )
    basis_j = tensor_product_basis(
        point_j,
        columns=config.grid_columns,
        rows=config.grid_rows,
    )
    coefficient_matrix = basis_i - basis_j
    observation_rows = np.arange(observations.count, dtype=np.int64)
    offset_matrix = sparse.csr_matrix(
        (
            np.concatenate(
                (
                    np.ones(observations.count, dtype=np.float64),
                    -np.ones(observations.count, dtype=np.float64),
                )
            ),
            (
                np.concatenate((observation_rows, observation_rows)),
                np.concatenate((observations.frame_i, observations.frame_j)),
            ),
        ),
        shape=(observations.count, observations.frame_count),
    )
    observation_matrix = sparse.hstack(
        (coefficient_matrix, offset_matrix),
        format="csr",
    )

    regularization_rows: list[sparse.csr_matrix] = []
    if config.smooth_lambda > 0.0:
        smooth = smoothness_regularization(config.grid_columns, config.grid_rows)
        regularization_rows.append(
            sparse.hstack(
                (
                    smooth * np.sqrt(config.smooth_lambda),
                    sparse.csr_matrix((smooth.shape[0], observations.frame_count)),
                ),
                format="csr",
            )
        )
    if config.frame_offset_lambda > 0.0:
        offset_regularization = sparse.eye(
            observations.frame_count,
            format="csr",
        ) * np.sqrt(config.frame_offset_lambda)
        regularization_rows.append(
            sparse.hstack(
                (
                    sparse.csr_matrix((observations.frame_count, coefficient_count)),
                    offset_regularization,
                ),
                format="csr",
            )
        )

    mean_weights = mean_field_weights(config.grid_columns, config.grid_rows)
    mean_field_row = sparse.csr_matrix(
        np.concatenate((mean_weights, np.zeros(observations.frame_count)))[None, :]
        * config.gauge_weight
    )
    mean_offset_row = sparse.csr_matrix(
        np.concatenate(
            (
                np.zeros(coefficient_count),
                np.full(observations.frame_count, 1.0 / observations.frame_count),
            )
        )[None, :]
        * config.gauge_weight
    )
    regularization_rows.extend((mean_field_row, mean_offset_row))
    regularization_matrix = sparse.vstack(regularization_rows, format="csr")

    observability = np.asarray(
        np.abs(coefficient_matrix).sum(axis=0),
        dtype=np.float64,
    ).ravel()
    return observation_matrix, regularization_matrix, observability


def _robust_sigma(residual: np.ndarray) -> float:
    values = np.asarray(residual, dtype=np.float64)
    center = float(np.median(values))
    return max(1e-6, 1.4826 * float(np.median(np.abs(values - center))))


def _solve_channel(
    observation_matrix: sparse.csr_matrix,
    regularization_matrix: sparse.csr_matrix,
    values: np.ndarray,
    *,
    config: SolverConfig,
    cancel_callback: CancelCallback | None,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(values.size, dtype=np.float64)
    solution = np.zeros(observation_matrix.shape[1], dtype=np.float64)
    iteration_count = config.irls_iterations if config.robust_loss == "huber" else 1
    zero_regularization = np.zeros(regularization_matrix.shape[0], dtype=np.float64)
    for _iteration in range(iteration_count):
        if cancel_callback is not None and cancel_callback():
            raise InterruptedError("用户已取消周边梯度优化。")
        square_root_weights = np.sqrt(weights)
        weighted_observations = observation_matrix.multiply(square_root_weights[:, None])
        system_matrix = sparse.vstack((weighted_observations, regularization_matrix), format="csr")
        system_values = np.concatenate((values * square_root_weights, zero_regularization))
        solution = lsqr(
            system_matrix,
            system_values,
            atol=1e-8,
            btol=1e-8,
            iter_lim=max(500, system_matrix.shape[1] * 20),
        )[0]
        residual = values - observation_matrix @ solution
        if config.robust_loss != "huber":
            break
        scale = _robust_sigma(residual)
        cutoff = max(1e-6, config.huber_delta * scale)
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        outside = absolute > cutoff
        weights[outside] = cutoff / absolute[outside]
    return solution, values - observation_matrix @ solution


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=0))


def _mad(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0)
    return np.median(np.abs(values - center[None, :]), axis=0)


def _group_rms(
    before: np.ndarray,
    after: np.ndarray,
    group_masks: dict[str, np.ndarray],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    before_result: dict[str, list[float]] = {}
    after_result: dict[str, list[float]] = {}
    for key, mask in group_masks.items():
        if np.any(mask):
            before_result[key] = [float(value) for value in _rms(before[mask])]
            after_result[key] = [float(value) for value in _rms(after[mask])]
    return before_result, after_result


def solve_photometric_model(
    observations: ObservationSet,
    *,
    frame_records: tuple[dict[str, object], ...],
    config: SolverConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> PhotometricSolution:
    config.validated()
    if observations.count < max(100, config.grid_columns * config.grid_rows * 3):
        raise ValueError(
            f"有效观测仅 {observations.count} 条，不足以稳定求解 "
            f"{config.grid_columns}×{config.grid_rows} B-spline。"
        )
    observation_matrix, regularization_matrix, observability = build_observation_matrix(
        observations,
        config,
    )
    coefficient_count = config.grid_columns * config.grid_rows
    coefficients = np.zeros((3, coefficient_count), dtype=np.float64)
    offsets = np.zeros((observations.frame_count, 3), dtype=np.float64)
    residual_after = np.zeros_like(observations.difference_rgb, dtype=np.float64)
    for channel, channel_name in enumerate(("R", "G", "B")):
        if progress_callback is not None:
            progress_callback(f"求解 {channel_name} 通道稀疏系统…", channel, 3)
        channel_solution, channel_residual = _solve_channel(
            observation_matrix,
            regularization_matrix,
            observations.difference_rgb[:, channel].astype(np.float64),
            config=config,
            cancel_callback=cancel_callback,
        )
        coefficients[channel] = channel_solution[:coefficient_count]
        offsets[:, channel] = channel_solution[coefficient_count:]
        residual_after[:, channel] = channel_residual

    preview = rasterize_correction(
        coefficients,
        columns=config.grid_columns,
        rows=config.grid_rows,
        width_px=256,
        height_px=256,
    )
    frame_counts = np.bincount(
        np.concatenate((observations.frame_i, observations.frame_j)),
        minlength=observations.frame_count,
    )
    overlap_counter = Counter(
        f"{min(int(frame_i), int(frame_j))}-{max(int(frame_i), int(frame_j))}"
        for frame_i, frame_j in zip(observations.frame_i, observations.frame_j)
    )
    per_frame_before = np.zeros((observations.frame_count, 3), dtype=np.float64)
    per_frame_after = np.zeros((observations.frame_count, 3), dtype=np.float64)
    for frame_index in range(observations.frame_count):
        frame_mask = (
            (observations.frame_i == frame_index)
            | (observations.frame_j == frame_index)
        )
        if np.any(frame_mask):
            per_frame_before[frame_index] = _rms(observations.difference_rgb[frame_mask])
            per_frame_after[frame_index] = _rms(residual_after[frame_mask])
    overlap_masks = {
        key: (
            (
                (observations.frame_i == int(key.split("-")[0]))
                & (observations.frame_j == int(key.split("-")[1]))
            )
            | (
                (observations.frame_i == int(key.split("-")[1]))
                & (observations.frame_j == int(key.split("-")[0]))
            )
        )
        for key in overlap_counter
    }
    overlap_before, overlap_after = _group_rms(
        observations.difference_rgb,
        residual_after,
        overlap_masks,
    )
    diagnostics = DiagnosticsResult(
        observation_count=observations.count,
        frame_observation_counts=frame_counts.astype(np.int64),
        overlap_observation_counts=dict(sorted(overlap_counter.items())),
        rms_before_rgb=_rms(observations.difference_rgb),
        rms_after_rgb=_rms(residual_after),
        mad_before_rgb=_mad(observations.difference_rgb),
        mad_after_rgb=_mad(residual_after),
        correction_min_rgb=np.min(preview, axis=(0, 1)).astype(np.float64),
        correction_max_rgb=np.max(preview, axis=(0, 1)).astype(np.float64),
        observability=observability.reshape(config.grid_rows, config.grid_columns),
        per_frame_rms_before_rgb=per_frame_before,
        per_frame_rms_after_rgb=per_frame_after,
        per_overlap_rms_before_rgb=overlap_before,
        per_overlap_rms_after_rgb=overlap_after,
        residual_before_rgb=observations.difference_rgb.astype(np.float64),
        residual_after_rgb=residual_after,
    )
    if progress_callback is not None:
        progress_callback("RGB 稀疏求解完成。", 3, 3)
    return PhotometricSolution(
        grid_columns=config.grid_columns,
        grid_rows=config.grid_rows,
        coefficients_rgb=coefficients,
        frame_offsets_rgb=offsets,
        frame_records=frame_records,
        solver_config=config,
        diagnostics=diagnostics,
    )
