from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...frame_astrometry import FrameAstrometricModel


ProgressCallback = Callable[[str, int, int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class PhotometricFrame:
    """One source TIFF and its independent astrometric model."""

    index: int
    model_path: Path
    image_path: Path
    mask_path: Path | None
    model: FrameAstrometricModel
    source_payload: dict[str, Any] = field(repr=False)
    model_sha256: str = ""

    @property
    def width_px(self) -> int:
        return int(self.model.image_width_px)

    @property
    def height_px(self) -> int:
        return int(self.model.image_height_px)


@dataclass(frozen=True)
class PhotometricObservation:
    """Scalar form used by tests and external callers."""

    frame_i: int
    frame_j: int
    point_i_xy: tuple[float, float]
    point_j_xy: tuple[float, float]
    measurement_i_rgb: tuple[float, float, float]
    measurement_j_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class ObservationSet:
    """Vectorized overlap observations used by the sparse solver."""

    frame_i: np.ndarray
    frame_j: np.ndarray
    point_i_xy: np.ndarray
    point_j_xy: np.ndarray
    difference_rgb: np.ndarray
    target_sample_index: np.ndarray
    frame_count: int
    image_width_px: int
    image_height_px: int
    measurement_i_rgb: np.ndarray | None = field(default=None, repr=False)
    measurement_j_rgb: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        count = int(np.asarray(self.frame_i).size)
        expected = {
            "frame_j": np.asarray(self.frame_j).shape == (count,),
            "point_i_xy": np.asarray(self.point_i_xy).shape == (count, 2),
            "point_j_xy": np.asarray(self.point_j_xy).shape == (count, 2),
            "difference_rgb": np.asarray(self.difference_rgb).shape == (count, 3),
            "target_sample_index": np.asarray(self.target_sample_index).shape == (count,),
        }
        invalid = [name for name, valid in expected.items() if not valid]
        for name, values in (
            ("measurement_i_rgb", self.measurement_i_rgb),
            ("measurement_j_rgb", self.measurement_j_rgb),
        ):
            if values is not None and np.asarray(values).shape != (count, 3):
                invalid.append(name)
        if invalid:
            raise ValueError(f"ObservationSet 数组尺寸不一致：{', '.join(invalid)}")

    @property
    def count(self) -> int:
        return int(self.frame_i.size)


@dataclass(frozen=True)
class SolverConfig:
    grid_columns: int = 8
    grid_rows: int = 6
    sample_long_side_px: int = 360
    downsample_factor: int = 8
    patch_size_px: int = 11
    minimum_patch_valid_fraction: float = 0.45
    smooth_lambda: float = 30.0
    frame_offset_lambda: float = 0.05
    correction_model: str = "additive"
    intensity_floor_code: float = 64.0
    minimum_gain: float = 0.2
    maximum_gain: float = 5.0
    enable_frame_plane: bool = False
    frame_plane_lambda: float = 5000.0
    enable_brightness_nonlinearity: bool = False
    brightness_knot_count: int = 6
    brightness_smooth_lambda: float = 300.0
    gauge_weight: float = 1000.0
    robust_loss: str = "huber"
    huber_delta: float = 1.5
    irls_iterations: int = 4
    saturation_fraction: float = 0.995
    star_sigma: float = 5.0
    apply_correction: bool = False
    output_block_rows: int = 256

    def validated(self) -> "SolverConfig":
        if self.grid_columns < 4 or self.grid_rows < 4:
            raise ValueError("三次 B-spline 控制网格每个方向至少需要 4 个控制点。")
        if self.sample_long_side_px < 32:
            raise ValueError("公共取景采样长边至少需要 32 px。")
        if self.downsample_factor < 1:
            raise ValueError("缩小倍率必须至少为 1。")
        if self.patch_size_px < 3 or self.patch_size_px % 2 == 0:
            raise ValueError("Patch 尺寸必须是至少为 3 的奇数。")
        if not 0.05 <= self.minimum_patch_valid_fraction <= 1.0:
            raise ValueError("Patch 最小有效比例必须位于 0.05～1.0。")
        if self.smooth_lambda < 0 or self.frame_offset_lambda < 0:
            raise ValueError("正则强度不能为负数。")
        if self.correction_model not in {"additive", "multiplicative", "log_gain"}:
            raise ValueError("校正模型只支持 additive、multiplicative 或 log_gain。")
        if self.intensity_floor_code <= 0:
            raise ValueError("亮度变换 floor 必须大于 0。")
        if not 0 < self.minimum_gain < 1.0 < self.maximum_gain:
            raise ValueError("安全增益范围必须满足 0 < minimum < 1 < maximum。")
        if self.frame_plane_lambda < 0:
            raise ValueError("帧内平面正则强度不能为负数。")
        if self.enable_brightness_nonlinearity and not 3 <= self.brightness_knot_count <= 12:
            raise ValueError("亮度节点数必须位于 3～12。")
        if self.brightness_smooth_lambda < 0:
            raise ValueError("亮度平滑正则不能为负数。")
        if self.robust_loss not in {"none", "huber"}:
            raise ValueError("robust_loss 只支持 none 或 huber。")
        if self.irls_iterations < 1:
            raise ValueError("IRLS 迭代次数必须至少为 1。")
        return self


@dataclass(frozen=True)
class DiagnosticsResult:
    observation_count: int
    frame_observation_counts: np.ndarray
    overlap_observation_counts: dict[str, int]
    rms_before_rgb: np.ndarray
    rms_after_rgb: np.ndarray
    mad_before_rgb: np.ndarray
    mad_after_rgb: np.ndarray
    correction_min_rgb: np.ndarray
    correction_max_rgb: np.ndarray
    observability: np.ndarray
    per_frame_rms_before_rgb: np.ndarray
    per_frame_rms_after_rgb: np.ndarray
    per_overlap_rms_before_rgb: dict[str, list[float]]
    per_overlap_rms_after_rgb: dict[str, list[float]]
    residual_before_rgb: np.ndarray = field(repr=False)
    residual_after_rgb: np.ndarray = field(repr=False)
    fit_domain: str = "code_value"
    fit_rms_before_rgb: np.ndarray = field(default_factory=lambda: np.empty(0))
    fit_rms_after_rgb: np.ndarray = field(default_factory=lambda: np.empty(0))
    brightness_bin_edges_code: np.ndarray = field(default_factory=lambda: np.empty(0))
    brightness_bin_counts_rgb: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    brightness_bin_rms_before_rgb: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0))
    )
    brightness_bin_rms_after_rgb: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0))
    )


@dataclass(frozen=True)
class PhotometricSolution:
    grid_columns: int
    grid_rows: int
    coefficients_rgb: np.ndarray
    frame_offsets_rgb: np.ndarray
    frame_records: tuple[dict[str, Any], ...]
    solver_config: SolverConfig
    diagnostics: DiagnosticsResult
    frame_gradients_rgb: np.ndarray | None = None
    brightness_knots: np.ndarray | None = None

    def __post_init__(self) -> None:
        coefficient_count = int(self.grid_columns * self.grid_rows)
        coefficient_shape = np.asarray(self.coefficients_rgb).shape
        valid_coefficient_shape = (
            coefficient_shape == (3, coefficient_count)
            or (
                len(coefficient_shape) == 3
                and coefficient_shape[0] == 3
                and coefficient_shape[2] == coefficient_count
            )
        )
        if not valid_coefficient_shape:
            raise ValueError("coefficients_rgb 尺寸必须是 3×N 或 3×K×N。")
        if np.asarray(self.frame_offsets_rgb).ndim != 2 or self.frame_offsets_rgb.shape[1] != 3:
            raise ValueError("frame_offsets_rgb 尺寸必须是 frame_count × 3。")
        frame_count = int(self.frame_offsets_rgb.shape[0])
        gradients = self.frame_gradients_rgb
        if gradients is not None and np.asarray(gradients).shape != (frame_count, 3, 2):
            raise ValueError("frame_gradients_rgb 尺寸必须是 frame_count × 3 × 2。")
        layer_count = coefficient_shape[1] if len(coefficient_shape) == 3 else 1
        if layer_count > 1 and not self.solver_config.enable_brightness_nonlinearity:
            raise ValueError("多亮度层系数要求启用自适应亮度分段。")
        if (
            self.solver_config.enable_brightness_nonlinearity
            and layer_count != self.solver_config.brightness_knot_count
        ):
            raise ValueError("亮度层系数数量与 brightness_knot_count 不一致。")
        knots = self.brightness_knots
        if knots is not None and np.asarray(knots).shape != (layer_count,):
            raise ValueError("brightness_knots 数量必须与系数亮度层数一致。")
        if knots is not None:
            knot_values = np.asarray(knots, dtype=np.float64)
            if not np.all(np.isfinite(knot_values)) or (
                knot_values.size > 1 and np.any(np.diff(knot_values) <= 0.0)
            ):
                raise ValueError("brightness_knots 必须有限并严格递增。")

    @property
    def coefficient_layers_rgb(self) -> np.ndarray:
        values = np.asarray(self.coefficients_rgb, dtype=np.float64)
        return values[:, None, :] if values.ndim == 2 else values

    @property
    def frame_gradient_values_rgb(self) -> np.ndarray:
        if self.frame_gradients_rgb is None:
            return np.zeros((self.frame_offsets_rgb.shape[0], 3, 2), dtype=np.float64)
        return np.asarray(self.frame_gradients_rgb, dtype=np.float64)

    @property
    def brightness_knot_values(self) -> np.ndarray:
        layer_count = self.coefficient_layers_rgb.shape[1]
        if self.brightness_knots is None:
            return np.linspace(0.0, 1.0, layer_count, dtype=np.float64)
        return np.asarray(self.brightness_knots, dtype=np.float64)


@dataclass(frozen=True)
class LowFrequencyRunResult:
    solution_path: Path
    diagnostics_directory: Path
    corrected_paths: tuple[Path, ...]
    solution: PhotometricSolution
