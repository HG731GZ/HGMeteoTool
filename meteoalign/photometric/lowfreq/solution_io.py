from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .types import DiagnosticsResult, PhotometricSolution, SolverConfig


SOLUTION_SCHEMA = "hgmeteo_lowfreq_photometric_correction"
SOLUTION_VERSION = 2
SUPPORTED_SOLUTION_VERSIONS = {1, 2}


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).ravel()]


def _nullable_nested(values: np.ndarray) -> list[Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return [None if not np.isfinite(value) else float(value) for value in array]
    return [_nullable_nested(row) for row in array]


def solution_to_payload(solution: PhotometricSolution) -> dict[str, Any]:
    diagnostics = solution.diagnostics
    config = solution.solver_config
    frame_payloads: list[dict[str, Any]] = []
    gradients = solution.frame_gradient_values_rgb
    for index, record in enumerate(solution.frame_records):
        payload = dict(record)
        payload["offset_rgb"] = _float_list(solution.frame_offsets_rgb[index])
        payload["gradient_x_rgb"] = _float_list(gradients[index, :, 0])
        payload["gradient_y_rgb"] = _float_list(gradients[index, :, 1])
        frame_payloads.append(payload)
    coefficient_layers = solution.coefficient_layers_rgb
    model_type = {
        "additive": "additive_bspline",
        "multiplicative": "multiplicative_bspline",
        "log_gain": "log_gain_bspline",
    }[config.correction_model]
    if config.enable_brightness_nonlinearity:
        model_type = f"brightness_dependent_{model_type}"
    return {
        "schema": SOLUTION_SCHEMA,
        "version": SOLUTION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "correction_domain": (
            "camera_raw_tiff_code_value"
            if config.correction_model == "additive"
            else "camera_raw_tiff_relative_gain"
        ),
        "model_type": model_type,
        "grid": {
            "columns": int(solution.grid_columns),
            "rows": int(solution.grid_rows),
            "degree": 3,
            "coordinate_system": "normalized_source_image_pixel",
            "brightness_knots_log_coordinate": _float_list(
                solution.brightness_knot_values
            ),
        },
        "channels": {
            channel_name: {
                "coefficients": _float_list(coefficient_layers[channel, 0]),
                "coefficient_layers": [
                    _float_list(layer)
                    for layer in coefficient_layers[channel]
                ],
            }
            for channel, channel_name in enumerate(("R", "G", "B"))
        },
        "frames": frame_payloads,
        "solver": {
            "sampling_strategy": "model_icrs_coverage",
            "smooth_lambda": float(config.smooth_lambda),
            "frame_offset_lambda": float(config.frame_offset_lambda),
            "correction_model": config.correction_model,
            "intensity_floor_code": float(config.intensity_floor_code),
            "minimum_gain": float(config.minimum_gain),
            "maximum_gain": float(config.maximum_gain),
            "enable_frame_plane": bool(config.enable_frame_plane),
            "frame_plane_lambda": float(config.frame_plane_lambda),
            "enable_brightness_nonlinearity": bool(
                config.enable_brightness_nonlinearity
            ),
            "brightness_knot_count": int(coefficient_layers.shape[1]),
            "brightness_knot_strategy": (
                "observation_log_brightness_percentile_1_99_with_margin"
                if config.enable_brightness_nonlinearity
                else "single_shared_field"
            ),
            "brightness_smooth_lambda": float(config.brightness_smooth_lambda),
            "gauge": [
                "mean(V_k)=0",
                "sum(e_i)=0",
                "sum(frame_gradient_x_i)=0",
                "sum(frame_gradient_y_i)=0",
            ],
            "robust_loss": config.robust_loss,
            "huber_delta": float(config.huber_delta),
            "irls_iterations": int(config.irls_iterations),
            "sample_long_side_px": int(config.sample_long_side_px),
            "downsample_factor": int(config.downsample_factor),
            "patch_size_px": int(config.patch_size_px),
        },
        "diagnostics": {
            "observation_count": int(diagnostics.observation_count),
            "frame_observation_counts": [
                int(value) for value in diagnostics.frame_observation_counts
            ],
            "overlap_observation_counts": diagnostics.overlap_observation_counts,
            "rms_before_rgb": _float_list(diagnostics.rms_before_rgb),
            "rms_after_rgb": _float_list(diagnostics.rms_after_rgb),
            "mad_before_rgb": _float_list(diagnostics.mad_before_rgb),
            "mad_after_rgb": _float_list(diagnostics.mad_after_rgb),
            "correction_min_rgb": _float_list(diagnostics.correction_min_rgb),
            "correction_max_rgb": _float_list(diagnostics.correction_max_rgb),
            "observability": np.asarray(diagnostics.observability).tolist(),
            "per_frame_rms_before_rgb": np.asarray(
                diagnostics.per_frame_rms_before_rgb
            ).tolist(),
            "per_frame_rms_after_rgb": np.asarray(
                diagnostics.per_frame_rms_after_rgb
            ).tolist(),
            "per_overlap_rms_before_rgb": diagnostics.per_overlap_rms_before_rgb,
            "per_overlap_rms_after_rgb": diagnostics.per_overlap_rms_after_rgb,
            "fit_domain": diagnostics.fit_domain,
            "fit_rms_before_rgb": _float_list(diagnostics.fit_rms_before_rgb),
            "fit_rms_after_rgb": _float_list(diagnostics.fit_rms_after_rgb),
            "brightness_bin_edges_code": _float_list(
                diagnostics.brightness_bin_edges_code
            ),
            "brightness_bin_counts_rgb": np.asarray(
                diagnostics.brightness_bin_counts_rgb,
                dtype=np.int64,
            ).tolist(),
            "brightness_bin_rms_before_rgb": _nullable_nested(
                diagnostics.brightness_bin_rms_before_rgb
            ),
            "brightness_bin_rms_after_rgb": _nullable_nested(
                diagnostics.brightness_bin_rms_after_rgb
            ),
        },
    }


def write_solution(path: str | Path, solution: PhotometricSolution) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(solution_to_payload(solution), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def read_solution_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("亮度校正 solution JSON 根对象必须是对象。")
    if (
        payload.get("schema") != SOLUTION_SCHEMA
        or int(payload.get("version", 0)) not in SUPPORTED_SOLUTION_VERSIONS
    ):
        raise ValueError("不支持的亮度校正 solution JSON。")
    return payload


def read_solution(path: str | Path) -> PhotometricSolution:
    payload = read_solution_payload(path)
    grid = payload.get("grid")
    channels = payload.get("channels")
    frames = payload.get("frames")
    solver = payload.get("solver")
    diagnostic_payload = payload.get("diagnostics")
    if not all(
        isinstance(value, dict)
        for value in (grid, channels, solver, diagnostic_payload)
    ) or not isinstance(frames, list):
        raise ValueError("亮度校正 solution JSON 缺少 grid/channels/frames/solver/diagnostics。")
    columns = int(grid.get("columns", 0))
    rows = int(grid.get("rows", 0))
    channel_layers: list[np.ndarray] = []
    for channel_name in ("R", "G", "B"):
        channel_payload = channels.get(channel_name, {})
        if not isinstance(channel_payload, dict):
            raise ValueError(f"solution channels.{channel_name} 必须是对象。")
        layers_payload = channel_payload.get("coefficient_layers")
        if isinstance(layers_payload, list) and layers_payload:
            channel_layers.append(np.asarray(layers_payload, dtype=np.float64))
        else:
            channel_layers.append(
                np.asarray(
                    [channel_payload.get("coefficients", [])],
                    dtype=np.float64,
                )
            )
    coefficients = np.stack(channel_layers, axis=0)
    if coefficients.shape[1] == 1:
        coefficients = coefficients[:, 0, :]
    offsets = np.asarray(
        [
            frame.get("offset_rgb", [])
            if isinstance(frame, dict)
            else []
            for frame in frames
        ],
        dtype=np.float64,
    )
    gradients = np.asarray(
        [
            np.column_stack(
                (
                    frame.get("gradient_x_rgb", [0.0, 0.0, 0.0]),
                    frame.get("gradient_y_rgb", [0.0, 0.0, 0.0]),
                )
            )
            if isinstance(frame, dict)
            else np.zeros((3, 2), dtype=np.float64)
            for frame in frames
        ],
        dtype=np.float64,
    )
    knots_payload = grid.get("brightness_knots_log_coordinate")
    coefficient_layer_count = coefficients.shape[1] if coefficients.ndim == 3 else 1
    knots = (
        np.asarray(knots_payload, dtype=np.float64)
        if isinstance(knots_payload, list)
        else np.linspace(0.0, 1.0, coefficient_layer_count, dtype=np.float64)
    )
    config = SolverConfig(
        grid_columns=columns,
        grid_rows=rows,
        sample_long_side_px=int(solver.get("sample_long_side_px", 360)),
        downsample_factor=int(solver.get("downsample_factor", 8)),
        patch_size_px=int(solver.get("patch_size_px", 11)),
        smooth_lambda=float(solver.get("smooth_lambda", 30.0)),
        frame_offset_lambda=float(solver.get("frame_offset_lambda", 0.05)),
        correction_model=str(solver.get("correction_model", "additive")),
        intensity_floor_code=float(solver.get("intensity_floor_code", 64.0)),
        minimum_gain=float(solver.get("minimum_gain", 0.2)),
        maximum_gain=float(solver.get("maximum_gain", 5.0)),
        enable_frame_plane=bool(solver.get("enable_frame_plane", False)),
        frame_plane_lambda=float(solver.get("frame_plane_lambda", 5000.0)),
        enable_brightness_nonlinearity=bool(
            solver.get("enable_brightness_nonlinearity", coefficient_layer_count > 1)
        ),
        brightness_knot_count=int(
            solver.get("brightness_knot_count", coefficient_layer_count)
        ),
        brightness_smooth_lambda=float(
            solver.get("brightness_smooth_lambda", 300.0)
        ),
        robust_loss=str(solver.get("robust_loss", "huber")),
        huber_delta=float(solver.get("huber_delta", 1.5)),
        irls_iterations=int(solver.get("irls_iterations", 4)),
    ).validated()
    observation_count = int(diagnostic_payload.get("observation_count", 0))
    diagnostics = DiagnosticsResult(
        observation_count=observation_count,
        frame_observation_counts=np.asarray(
            diagnostic_payload.get("frame_observation_counts", []),
            dtype=np.int64,
        ),
        overlap_observation_counts={
            str(key): int(value)
            for key, value in dict(
                diagnostic_payload.get("overlap_observation_counts", {})
            ).items()
        },
        rms_before_rgb=np.asarray(diagnostic_payload.get("rms_before_rgb", []), dtype=np.float64),
        rms_after_rgb=np.asarray(diagnostic_payload.get("rms_after_rgb", []), dtype=np.float64),
        mad_before_rgb=np.asarray(diagnostic_payload.get("mad_before_rgb", []), dtype=np.float64),
        mad_after_rgb=np.asarray(diagnostic_payload.get("mad_after_rgb", []), dtype=np.float64),
        correction_min_rgb=np.asarray(
            diagnostic_payload.get("correction_min_rgb", []),
            dtype=np.float64,
        ),
        correction_max_rgb=np.asarray(
            diagnostic_payload.get("correction_max_rgb", []),
            dtype=np.float64,
        ),
        observability=np.asarray(
            diagnostic_payload.get("observability", []),
            dtype=np.float64,
        ),
        per_frame_rms_before_rgb=np.asarray(
            diagnostic_payload.get("per_frame_rms_before_rgb", []),
            dtype=np.float64,
        ),
        per_frame_rms_after_rgb=np.asarray(
            diagnostic_payload.get("per_frame_rms_after_rgb", []),
            dtype=np.float64,
        ),
        per_overlap_rms_before_rgb={
            str(key): [float(item) for item in value]
            for key, value in dict(
                diagnostic_payload.get("per_overlap_rms_before_rgb", {})
            ).items()
        },
        per_overlap_rms_after_rgb={
            str(key): [float(item) for item in value]
            for key, value in dict(
                diagnostic_payload.get("per_overlap_rms_after_rgb", {})
            ).items()
        },
        residual_before_rgb=np.empty((0, 3), dtype=np.float64),
        residual_after_rgb=np.empty((0, 3), dtype=np.float64),
        fit_domain=str(diagnostic_payload.get("fit_domain", "code_value")),
        fit_rms_before_rgb=np.asarray(
            diagnostic_payload.get("fit_rms_before_rgb", []),
            dtype=np.float64,
        ),
        fit_rms_after_rgb=np.asarray(
            diagnostic_payload.get("fit_rms_after_rgb", []),
            dtype=np.float64,
        ),
        brightness_bin_edges_code=np.asarray(
            diagnostic_payload.get("brightness_bin_edges_code", []),
            dtype=np.float64,
        ),
        brightness_bin_counts_rgb=np.asarray(
            diagnostic_payload.get("brightness_bin_counts_rgb", []),
            dtype=np.int64,
        ),
        brightness_bin_rms_before_rgb=np.asarray(
            diagnostic_payload.get("brightness_bin_rms_before_rgb", []),
            dtype=np.float64,
        ),
        brightness_bin_rms_after_rgb=np.asarray(
            diagnostic_payload.get("brightness_bin_rms_after_rgb", []),
            dtype=np.float64,
        ),
    )
    return PhotometricSolution(
        grid_columns=columns,
        grid_rows=rows,
        coefficients_rgb=coefficients,
        frame_offsets_rgb=offsets,
        frame_records=tuple(dict(frame) for frame in frames if isinstance(frame, dict)),
        solver_config=config,
        diagnostics=diagnostics,
        frame_gradients_rgb=gradients,
        brightness_knots=knots,
    )
