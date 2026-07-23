from __future__ import annotations

import numpy as np

from .types import ObservationSet, SolverConfig


def brightness_coordinate(values: np.ndarray, config: SolverConfig) -> np.ndarray:
    """Map Camera Raw TIFF code values to a stable [0, 1] log-brightness axis."""

    code_values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    floor = float(config.intensity_floor_code)
    maximum = 65535.0
    coordinate = np.log1p(code_values / floor) / np.log1p(maximum / floor)
    return np.clip(coordinate, 0.0, 1.0)


def brightness_knots(config: SolverConfig) -> np.ndarray:
    layer_count = config.brightness_knot_count if config.enable_brightness_nonlinearity else 1
    return np.linspace(0.0, 1.0, layer_count, dtype=np.float64)


def derive_brightness_knots(
    observations: ObservationSet,
    config: SolverConfig,
) -> np.ndarray:
    if not config.enable_brightness_nonlinearity:
        return np.array([0.0], dtype=np.float64)
    if observations.measurement_i_rgb is None or observations.measurement_j_rgb is None:
        raise ValueError("无法从缺失的原始测量中确定亮度节点。")
    measurements = np.concatenate(
        (
            np.asarray(observations.measurement_i_rgb, dtype=np.float64).ravel(),
            np.asarray(observations.measurement_j_rgb, dtype=np.float64).ravel(),
        )
    )
    coordinates = brightness_coordinate(measurements, config)
    coordinates = coordinates[np.isfinite(coordinates)]
    if coordinates.size < config.brightness_knot_count * 10:
        raise ValueError("有效亮度样本不足，无法建立自适应节点。")
    lower, upper = np.percentile(coordinates, (1.0, 99.0))
    lower = max(0.0, float(lower) - 0.02)
    upper = min(1.0, float(upper) + 0.02)
    if upper - lower < 0.05:
        center = 0.5 * (lower + upper)
        lower = max(0.0, center - 0.025)
        upper = min(1.0, center + 0.025)
    return np.linspace(
        lower,
        upper,
        config.brightness_knot_count,
        dtype=np.float64,
    )


def brightness_interpolation_indices(
    values: np.ndarray,
    config: SolverConfig,
    knots: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = brightness_coordinate(values, config)
    knot_values = brightness_knots(config) if knots is None else np.asarray(knots, dtype=np.float64)
    if knot_values.size == 1:
        zeros = np.zeros(coordinates.shape, dtype=np.int64)
        return zeros, zeros, np.zeros(coordinates.shape, dtype=np.float64)
    upper = np.searchsorted(knot_values, coordinates, side="right")
    upper = np.clip(upper, 1, knot_values.size - 1)
    lower = upper - 1
    below = coordinates <= knot_values[0]
    above = coordinates >= knot_values[-1]
    lower[below] = 0
    upper[below] = 0
    lower[above] = knot_values.size - 1
    upper[above] = knot_values.size - 1
    denominator = knot_values[upper] - knot_values[lower]
    fraction = np.divide(
        coordinates - knot_values[lower],
        denominator,
        out=np.zeros_like(coordinates, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return lower, upper, np.clip(fraction, 0.0, 1.0)


def brightness_basis(
    values: np.ndarray,
    config: SolverConfig,
    knots: np.ndarray | None = None,
) -> np.ndarray:
    """Piecewise-linear hat basis on the adaptive log-brightness coordinate."""

    knot_values = brightness_knots(config) if knots is None else np.asarray(knots, dtype=np.float64)
    lower, upper, fraction = brightness_interpolation_indices(values, config, knot_values)
    lower = lower.ravel()
    upper = upper.ravel()
    fraction = fraction.ravel()
    weights = np.zeros((lower.size, knot_values.size), dtype=np.float64)
    row = np.arange(lower.size, dtype=np.int64)
    weights[row, lower] += 1.0 - fraction
    weights[row, upper] += fraction
    return weights


def observation_values(
    observations: ObservationSet,
    config: SolverConfig,
    channel: int,
) -> np.ndarray:
    if config.correction_model == "additive":
        return observations.difference_rgb[:, channel].astype(np.float64)
    if observations.measurement_i_rgb is None or observations.measurement_j_rgb is None:
        raise ValueError(f"{config.correction_model} 模式需要保存重叠两侧的原始亮度观测。")
    measurement_i = np.asarray(observations.measurement_i_rgb[:, channel], dtype=np.float64)
    measurement_j = np.asarray(observations.measurement_j_rgb[:, channel], dtype=np.float64)
    floor = float(config.intensity_floor_code)
    if config.correction_model == "multiplicative":
        denominator = measurement_i + measurement_j + 2.0 * floor
        return 2.0 * (measurement_i - measurement_j) / denominator
    return np.log(measurement_i + floor) - np.log(measurement_j + floor)


def corrected_code_values(
    values: np.ndarray,
    parameter: np.ndarray,
    config: SolverConfig,
) -> np.ndarray:
    """Apply one solved correction parameter to code values without integer clipping."""

    source = np.asarray(values, dtype=np.float64)
    correction = np.asarray(parameter, dtype=np.float64)
    if config.correction_model == "additive":
        return source - correction
    if config.correction_model == "multiplicative":
        gain = np.clip(
            1.0 + correction,
            config.minimum_gain,
            config.maximum_gain,
        )
        return source / gain
    log_gain = np.clip(
        correction,
        np.log(config.minimum_gain),
        np.log(config.maximum_gain),
    )
    floor = float(config.intensity_floor_code)
    return (source + floor) * np.exp(-log_gain) - floor


def correction_model_display_name(config: SolverConfig) -> str:
    names = {
        "additive": "additive code-value",
        "multiplicative": "linearized multiplicative gain",
        "log_gain": "log gain",
    }
    return names[config.correction_model]
