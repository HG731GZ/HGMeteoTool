from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .bspline import (
    mean_field_weights,
    rasterize_correction,
    smoothness_regularization,
    tensor_product_basis,
)
from .models import (
    brightness_basis,
    corrected_code_values,
    correction_model_display_name,
    derive_brightness_knots,
    observation_values,
)
from .types import (
    CancelCallback,
    DiagnosticsResult,
    ObservationSet,
    PhotometricSolution,
    ProgressCallback,
    SolverConfig,
)


@dataclass(frozen=True)
class UnknownLayout:
    layer_count: int
    coefficients_per_layer: int
    frame_count: int
    include_frame_plane: bool

    @property
    def coefficient_count(self) -> int:
        return self.layer_count * self.coefficients_per_layer

    @property
    def offset_start(self) -> int:
        return self.coefficient_count

    @property
    def plane_x_start(self) -> int:
        return self.offset_start + self.frame_count

    @property
    def plane_y_start(self) -> int:
        return self.plane_x_start + self.frame_count

    @property
    def unknown_count(self) -> int:
        return (
            self.coefficient_count
            + self.frame_count
            + (2 * self.frame_count if self.include_frame_plane else 0)
        )


def _layout(observations: ObservationSet, config: SolverConfig) -> UnknownLayout:
    return UnknownLayout(
        layer_count=(
            config.brightness_knot_count
            if config.enable_brightness_nonlinearity
            else 1
        ),
        coefficients_per_layer=config.grid_columns * config.grid_rows,
        frame_count=observations.frame_count,
        include_frame_plane=config.enable_frame_plane,
    )


def _normalized_points(points_xy: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).copy()
    points[:, 0] /= max(1, width_px - 1)
    points[:, 1] /= max(1, height_px - 1)
    return np.clip(points, 0.0, 1.0)


def _frame_parameter_matrix(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    values_i: np.ndarray,
    values_j: np.ndarray,
    frame_count: int,
) -> sparse.csr_matrix:
    observation_count = frame_i.size
    rows = np.arange(observation_count, dtype=np.int64)
    return sparse.csr_matrix(
        (
            np.concatenate((values_i, -values_j)),
            (
                np.concatenate((rows, rows)),
                np.concatenate((frame_i, frame_j)),
            ),
        ),
        shape=(observation_count, frame_count),
    )


def _pad_parameter_matrix(
    matrix: sparse.spmatrix,
    *,
    left_columns: int,
    right_columns: int,
) -> sparse.csr_matrix:
    return sparse.hstack(
        (
            sparse.csr_matrix((matrix.shape[0], left_columns)),
            matrix,
            sparse.csr_matrix((matrix.shape[0], right_columns)),
        ),
        format="csr",
    )


def _brightness_second_difference(layer_count: int, coefficient_count: int) -> sparse.csr_matrix:
    if layer_count < 3:
        return sparse.csr_matrix((0, layer_count * coefficient_count))
    diagonals = (
        np.ones(layer_count - 2),
        -2.0 * np.ones(layer_count - 2),
        np.ones(layer_count - 2),
    )
    layer_difference = sparse.diags(diagonals, offsets=(0, 1, 2), shape=(layer_count - 2, layer_count))
    return sparse.kron(
        layer_difference,
        sparse.eye(coefficient_count, format="csr"),
        format="csr",
    )


def build_observation_matrix(
    observations: ObservationSet,
    config: SolverConfig,
    channel: int = 0,
    brightness_knot_values: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Build one channel matrix with optional channel-specific brightness weights."""

    layout = _layout(observations, config)
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
    if config.enable_brightness_nonlinearity:
        if observations.measurement_i_rgb is None or observations.measurement_j_rgb is None:
            raise ValueError("自适应亮度分段需要重叠两侧的原始 RGB 测量。")
        knots = (
            derive_brightness_knots(observations, config)
            if brightness_knot_values is None
            else brightness_knot_values
        )
        weights_i = brightness_basis(
            observations.measurement_i_rgb[:, channel],
            config,
            knots,
        )
        weights_j = brightness_basis(
            observations.measurement_j_rgb[:, channel],
            config,
            knots,
        )
    else:
        weights_i = np.ones((observations.count, 1), dtype=np.float64)
        weights_j = np.ones((observations.count, 1), dtype=np.float64)
    coefficient_blocks = [
        basis_i.multiply(weights_i[:, layer, None])
        - basis_j.multiply(weights_j[:, layer, None])
        for layer in range(layout.layer_count)
    ]
    coefficient_matrix = sparse.hstack(coefficient_blocks, format="csr")
    ones = np.ones(observations.count, dtype=np.float64)
    offset_matrix = _frame_parameter_matrix(
        observations.frame_i,
        observations.frame_j,
        ones,
        ones,
        observations.frame_count,
    )
    matrix_parts: list[sparse.spmatrix] = [coefficient_matrix, offset_matrix]
    if config.enable_frame_plane:
        centered_i = point_i - 0.5
        centered_j = point_j - 0.5
        matrix_parts.extend(
            (
                _frame_parameter_matrix(
                    observations.frame_i,
                    observations.frame_j,
                    centered_i[:, 0],
                    centered_j[:, 0],
                    observations.frame_count,
                ),
                _frame_parameter_matrix(
                    observations.frame_i,
                    observations.frame_j,
                    centered_i[:, 1],
                    centered_j[:, 1],
                    observations.frame_count,
                ),
            )
        )
    observation_matrix = sparse.hstack(matrix_parts, format="csr")

    regularization_rows: list[sparse.csr_matrix] = []
    if config.smooth_lambda > 0.0:
        smooth = smoothness_regularization(config.grid_columns, config.grid_rows)
        layered_smooth = sparse.block_diag(
            [smooth] * layout.layer_count,
            format="csr",
        ) * np.sqrt(config.smooth_lambda / layout.layer_count)
        regularization_rows.append(
            _pad_parameter_matrix(
                layered_smooth,
                left_columns=0,
                right_columns=layout.unknown_count - layout.coefficient_count,
            )
        )
    if config.enable_brightness_nonlinearity and config.brightness_smooth_lambda > 0.0:
        brightness_smooth = _brightness_second_difference(
            layout.layer_count,
            layout.coefficients_per_layer,
        ) * np.sqrt(config.brightness_smooth_lambda)
        regularization_rows.append(
            _pad_parameter_matrix(
                brightness_smooth,
                left_columns=0,
                right_columns=layout.unknown_count - layout.coefficient_count,
            )
        )
    if config.frame_offset_lambda > 0.0:
        offsets = sparse.eye(layout.frame_count, format="csr") * np.sqrt(
            config.frame_offset_lambda
        )
        regularization_rows.append(
            _pad_parameter_matrix(
                offsets,
                left_columns=layout.offset_start,
                right_columns=layout.unknown_count - layout.plane_x_start,
            )
        )
    if config.enable_frame_plane and config.frame_plane_lambda > 0.0:
        plane_regularization = sparse.eye(
            2 * layout.frame_count,
            format="csr",
        ) * np.sqrt(config.frame_plane_lambda)
        regularization_rows.append(
            _pad_parameter_matrix(
                plane_regularization,
                left_columns=layout.plane_x_start,
                right_columns=0,
            )
        )

    mean_weights = mean_field_weights(config.grid_columns, config.grid_rows)
    for layer in range(layout.layer_count):
        gauge_values = np.zeros(layout.unknown_count, dtype=np.float64)
        start = layer * layout.coefficients_per_layer
        gauge_values[start : start + layout.coefficients_per_layer] = mean_weights
        regularization_rows.append(
            sparse.csr_matrix(gauge_values[None, :] * config.gauge_weight)
        )
    for start in (
        layout.offset_start,
        *(
            (layout.plane_x_start, layout.plane_y_start)
            if config.enable_frame_plane
            else ()
        ),
    ):
        gauge_values = np.zeros(layout.unknown_count, dtype=np.float64)
        gauge_values[start : start + layout.frame_count] = 1.0 / layout.frame_count
        regularization_rows.append(
            sparse.csr_matrix(gauge_values[None, :] * config.gauge_weight)
        )
    regularization_matrix = sparse.vstack(regularization_rows, format="csr")
    observability = np.asarray(
        np.abs(coefficient_matrix).sum(axis=0),
        dtype=np.float64,
    ).ravel()
    return observation_matrix, regularization_matrix, observability


def _robust_sigma(residual: np.ndarray) -> float:
    values = np.asarray(residual, dtype=np.float64)
    center = float(np.median(values))
    return max(1e-9, 1.4826 * float(np.median(np.abs(values - center))))


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
        cutoff = max(1e-9, config.huber_delta * _robust_sigma(residual))
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        outside = absolute > cutoff
        weights[outside] = cutoff / absolute[outside]
    return solution, values - observation_matrix @ solution


def _observation_parameters(
    observations: ObservationSet,
    config: SolverConfig,
    channel: int,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    gradients: np.ndarray,
    brightness_knot_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    if config.enable_brightness_nonlinearity:
        assert observations.measurement_i_rgb is not None
        assert observations.measurement_j_rgb is not None
        weights_i = brightness_basis(
            observations.measurement_i_rgb[:, channel],
            config,
            brightness_knot_values,
        )
        weights_j = brightness_basis(
            observations.measurement_j_rgb[:, channel],
            config,
            brightness_knot_values,
        )
    else:
        weights_i = np.ones((observations.count, 1), dtype=np.float64)
        weights_j = np.ones((observations.count, 1), dtype=np.float64)
    parameter_i = np.zeros(observations.count, dtype=np.float64)
    parameter_j = np.zeros(observations.count, dtype=np.float64)
    for layer in range(coefficients.shape[0]):
        parameter_i += weights_i[:, layer] * (basis_i @ coefficients[layer])
        parameter_j += weights_j[:, layer] * (basis_j @ coefficients[layer])
    parameter_i += offsets[observations.frame_i]
    parameter_j += offsets[observations.frame_j]
    if config.enable_frame_plane:
        centered_i = point_i - 0.5
        centered_j = point_j - 0.5
        parameter_i += (
            gradients[observations.frame_i, 0] * centered_i[:, 0]
            + gradients[observations.frame_i, 1] * centered_i[:, 1]
        )
        parameter_j += (
            gradients[observations.frame_j, 0] * centered_j[:, 0]
            + gradients[observations.frame_j, 1] * centered_j[:, 1]
        )
    return parameter_i, parameter_j


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


def _brightness_diagnostics(
    observations: ObservationSet,
    before: np.ndarray,
    after: np.ndarray,
    config: SolverConfig,
    brightness_knot_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_count = int(brightness_knot_values.size)
    if bin_count == 1:
        coordinate_edges = np.array([0.0, 1.0], dtype=np.float64)
    else:
        coordinate_edges = np.concatenate(
            (
                [0.0],
                0.5 * (
                    brightness_knot_values[:-1]
                    + brightness_knot_values[1:]
                ),
                [1.0],
            )
        )
    floor = float(config.intensity_floor_code)
    edges_code = floor * (
        np.exp(coordinate_edges * np.log1p(65535.0 / floor))
        - 1.0
    )
    counts = np.zeros((3, bin_count), dtype=np.int64)
    rms_before = np.full((3, bin_count), np.nan, dtype=np.float64)
    rms_after = np.full((3, bin_count), np.nan, dtype=np.float64)
    if observations.measurement_i_rgb is None or observations.measurement_j_rgb is None:
        return edges_code, counts, rms_before, rms_after
    mean_brightness = (
        np.asarray(observations.measurement_i_rgb, dtype=np.float64)
        + np.asarray(observations.measurement_j_rgb, dtype=np.float64)
    ) * 0.5
    for channel in range(3):
        bin_indices = np.searchsorted(
            edges_code,
            mean_brightness[:, channel],
            side="right",
        ) - 1
        bin_indices = np.clip(bin_indices, 0, bin_count - 1)
        for bin_index in range(bin_count):
            mask = bin_indices == bin_index
            counts[channel, bin_index] = int(np.count_nonzero(mask))
            if np.any(mask):
                rms_before[channel, bin_index] = float(
                    np.sqrt(np.mean(np.square(before[mask, channel])))
                )
                rms_after[channel, bin_index] = float(
                    np.sqrt(np.mean(np.square(after[mask, channel])))
                )
    return edges_code, counts, rms_before, rms_after


def solve_photometric_model(
    observations: ObservationSet,
    *,
    frame_records: tuple[dict[str, object], ...],
    config: SolverConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> PhotometricSolution:
    config.validated()
    layout = _layout(observations, config)
    brightness_knot_values = derive_brightness_knots(observations, config)
    minimum_observations = max(100, layout.unknown_count * 3)
    if observations.count < minimum_observations:
        raise ValueError(
            f"有效观测仅 {observations.count} 条，不足以稳定求解 "
            f"{layout.layer_count} 层 {config.grid_columns}×{config.grid_rows} B-spline。"
        )
    coefficients = np.zeros(
        (3, layout.layer_count, layout.coefficients_per_layer),
        dtype=np.float64,
    )
    offsets = np.zeros((observations.frame_count, 3), dtype=np.float64)
    gradients = np.zeros((observations.frame_count, 3, 2), dtype=np.float64)
    fit_residual_after = np.zeros_like(observations.difference_rgb, dtype=np.float64)
    fit_values = np.zeros_like(observations.difference_rgb, dtype=np.float64)
    code_residual_after = np.zeros_like(observations.difference_rgb, dtype=np.float64)
    observability_channels = np.zeros(
        (3, layout.layer_count, layout.coefficients_per_layer),
        dtype=np.float64,
    )
    for channel, channel_name in enumerate(("R", "G", "B")):
        if progress_callback is not None:
            progress_callback(
                f"求解 {channel_name} 通道（{correction_model_display_name(config)}）…",
                channel,
                3,
            )
        observation_matrix, regularization_matrix, observability = build_observation_matrix(
            observations,
            config,
            channel,
            brightness_knot_values,
        )
        values = observation_values(observations, config, channel)
        fit_values[:, channel] = values
        channel_solution, channel_residual = _solve_channel(
            observation_matrix,
            regularization_matrix,
            values,
            config=config,
            cancel_callback=cancel_callback,
        )
        coefficients[channel] = channel_solution[: layout.coefficient_count].reshape(
            layout.layer_count,
            layout.coefficients_per_layer,
        )
        offsets[:, channel] = channel_solution[
            layout.offset_start : layout.offset_start + layout.frame_count
        ]
        if config.enable_frame_plane:
            gradients[:, channel, 0] = channel_solution[
                layout.plane_x_start : layout.plane_x_start + layout.frame_count
            ]
            gradients[:, channel, 1] = channel_solution[
                layout.plane_y_start : layout.plane_y_start + layout.frame_count
            ]
        fit_residual_after[:, channel] = channel_residual
        observability_channels[channel] = observability.reshape(
            layout.layer_count,
            layout.coefficients_per_layer,
        )
        if observations.measurement_i_rgb is None or observations.measurement_j_rgb is None:
            code_residual_after[:, channel] = channel_residual
            continue
        parameter_i, parameter_j = _observation_parameters(
            observations,
            config,
            channel,
            coefficients[channel],
            offsets[:, channel],
            gradients[:, channel, :],
            brightness_knot_values,
        )
        corrected_i = corrected_code_values(
            observations.measurement_i_rgb[:, channel],
            parameter_i,
            config,
        )
        corrected_j = corrected_code_values(
            observations.measurement_j_rgb[:, channel],
            parameter_j,
            config,
        )
        code_residual_after[:, channel] = corrected_i - corrected_j

    preview_layers = np.stack(
        [
            rasterize_correction(
                coefficients[:, layer, :],
                columns=config.grid_columns,
                rows=config.grid_rows,
                width_px=256,
                height_px=256,
            )
            for layer in range(layout.layer_count)
        ],
        axis=0,
    )
    frame_counts = np.bincount(
        np.concatenate((observations.frame_i, observations.frame_j)),
        minlength=observations.frame_count,
    )
    overlap_counter = Counter(
        f"{min(int(frame_i), int(frame_j))}-{max(int(frame_i), int(frame_j))}"
        for frame_i, frame_j in zip(observations.frame_i, observations.frame_j)
    )
    code_residual_before = observations.difference_rgb.astype(np.float64)
    per_frame_before = np.zeros((observations.frame_count, 3), dtype=np.float64)
    per_frame_after = np.zeros((observations.frame_count, 3), dtype=np.float64)
    for frame_index in range(observations.frame_count):
        frame_mask = (
            (observations.frame_i == frame_index)
            | (observations.frame_j == frame_index)
        )
        if np.any(frame_mask):
            per_frame_before[frame_index] = _rms(code_residual_before[frame_mask])
            per_frame_after[frame_index] = _rms(code_residual_after[frame_mask])
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
        code_residual_before,
        code_residual_after,
        overlap_masks,
    )
    (
        brightness_edges,
        brightness_counts,
        brightness_before,
        brightness_after,
    ) = _brightness_diagnostics(
        observations,
        code_residual_before,
        code_residual_after,
        config,
        brightness_knot_values,
    )
    observability = np.mean(observability_channels, axis=0).reshape(
        (layout.layer_count, config.grid_rows, config.grid_columns)
    )
    if layout.layer_count == 1:
        observability = observability[0]
    diagnostics = DiagnosticsResult(
        observation_count=observations.count,
        frame_observation_counts=frame_counts.astype(np.int64),
        overlap_observation_counts=dict(sorted(overlap_counter.items())),
        rms_before_rgb=_rms(code_residual_before),
        rms_after_rgb=_rms(code_residual_after),
        mad_before_rgb=_mad(code_residual_before),
        mad_after_rgb=_mad(code_residual_after),
        correction_min_rgb=np.min(preview_layers, axis=(0, 1, 2)).astype(np.float64),
        correction_max_rgb=np.max(preview_layers, axis=(0, 1, 2)).astype(np.float64),
        observability=observability,
        per_frame_rms_before_rgb=per_frame_before,
        per_frame_rms_after_rgb=per_frame_after,
        per_overlap_rms_before_rgb=overlap_before,
        per_overlap_rms_after_rgb=overlap_after,
        residual_before_rgb=code_residual_before,
        residual_after_rgb=code_residual_after,
        fit_domain=correction_model_display_name(config),
        fit_rms_before_rgb=_rms(fit_values),
        fit_rms_after_rgb=_rms(fit_residual_after),
        brightness_bin_edges_code=brightness_edges,
        brightness_bin_counts_rgb=brightness_counts,
        brightness_bin_rms_before_rgb=brightness_before,
        brightness_bin_rms_after_rgb=brightness_after,
    )
    if progress_callback is not None:
        progress_callback("RGB 稀疏求解完成。", 3, 3)
    solution_coefficients = coefficients[:, 0, :] if layout.layer_count == 1 else coefficients
    return PhotometricSolution(
        grid_columns=config.grid_columns,
        grid_rows=config.grid_rows,
        coefficients_rgb=solution_coefficients,
        frame_offsets_rgb=offsets,
        frame_records=frame_records,
        solver_config=config,
        diagnostics=diagnostics,
        frame_gradients_rgb=gradients,
        brightness_knots=brightness_knot_values,
    )
