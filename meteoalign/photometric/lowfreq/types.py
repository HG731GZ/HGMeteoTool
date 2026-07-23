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
    gauge_weight: float = 1000.0
    robust_loss: str = "huber"
    huber_delta: float = 1.5
    irls_iterations: int = 4
    saturation_fraction: float = 0.995
    star_sigma: float = 5.0
    apply_correction: bool = True
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


@dataclass(frozen=True)
class PhotometricSolution:
    grid_columns: int
    grid_rows: int
    coefficients_rgb: np.ndarray
    frame_offsets_rgb: np.ndarray
    frame_records: tuple[dict[str, Any], ...]
    solver_config: SolverConfig
    diagnostics: DiagnosticsResult

    def __post_init__(self) -> None:
        coefficient_count = int(self.grid_columns * self.grid_rows)
        if np.asarray(self.coefficients_rgb).shape != (3, coefficient_count):
            raise ValueError("coefficients_rgb 尺寸必须是 3 × (grid_columns * grid_rows)。")
        if np.asarray(self.frame_offsets_rgb).ndim != 2 or self.frame_offsets_rgb.shape[1] != 3:
            raise ValueError("frame_offsets_rgb 尺寸必须是 frame_count × 3。")


@dataclass(frozen=True)
class LowFrequencyRunResult:
    solution_path: Path
    diagnostics_directory: Path
    corrected_paths: tuple[Path, ...]
    solution: PhotometricSolution
