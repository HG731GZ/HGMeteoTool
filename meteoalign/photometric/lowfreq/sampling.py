from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...coordinates import radec_to_unit_vectors
from ...domain.settings import CameraSettings
from ...mosaic.export.target_transform import target_image_points_to_icrs_vectors
from .types import PhotometricFrame


@dataclass(frozen=True)
class TargetSampleGrid:
    vectors_icrs: np.ndarray
    output_points_xy: np.ndarray
    valid: np.ndarray
    sample_width: int
    sample_height: int
    output_width_px: int
    output_height_px: int
    sampling_mode: str = "framing_json"
    source_frame_count: int = 0
    candidate_count: int = 0

    @property
    def sample_count(self) -> int:
        return int(self.vectors_icrs.shape[0])


def load_framing_transform(path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    framing_path = Path(path)
    payload = json.loads(framing_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("取景 JSON 根对象必须是对象。")
    transform = payload.get("target_icrs_to_pixel_transform")
    if not isinstance(transform, dict):
        raise ValueError("取景 JSON 缺少 target_icrs_to_pixel_transform。")
    if str(transform.get("type") or "") != "icrs_to_cropped_output_pixel":
        raise ValueError("取景 JSON 的目标变换类型不受支持。")
    return payload, transform


def build_target_sample_grid(
    framing_path: str | Path,
    *,
    long_side_px: int,
) -> TargetSampleGrid:
    _payload, transform = load_framing_transform(framing_path)
    output_width = int(transform.get("output_width_px", 0) or 0)
    output_height = int(transform.get("output_height_px", 0) or 0)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("取景 JSON 的输出尺寸无效。")
    scale = float(long_side_px) / max(output_width, output_height)
    sample_width = max(2, int(round(output_width * scale)))
    sample_height = max(2, int(round(output_height * scale)))
    x_output = np.linspace(0.5, output_width - 0.5, sample_width, dtype=np.float64)
    y_output = np.linspace(0.5, output_height - 0.5, sample_height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_output, y_output)

    camera_payload = transform.get("camera")
    basis_payload = transform.get("icrs_camera_basis")
    if not isinstance(camera_payload, dict) or not isinstance(basis_payload, dict):
        raise ValueError("取景 JSON 缺少 camera 或 icrs_camera_basis。")
    camera = CameraSettings(
        sensor_width_mm=float(camera_payload.get("sensor_width_mm", 36.0)),
        sensor_height_mm=float(camera_payload.get("sensor_height_mm", 24.0)),
        image_width_px=int(camera_payload.get("image_width_px", 0)),
        image_height_px=int(camera_payload.get("image_height_px", 0)),
        focal_length_mm=float(camera_payload.get("focal_length_mm", 24.0)),
        lens_model=str(camera_payload.get("lens_model", "")),
        fisheye_fov_deg=float(camera_payload.get("fisheye_fov_deg", 180.0)),
    )
    basis = tuple(
        np.asarray(basis_payload.get(name), dtype=np.float64)
        for name in ("right", "up", "forward")
    )
    if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in basis):
        raise ValueError("取景 JSON 的 ICRS 相机基向量无效。")

    full_x = grid_x.ravel() + float(transform.get("crop_left_px", 0.0))
    full_y = grid_y.ravel() + float(transform.get("crop_top_px", 0.0))
    vectors, valid = target_image_points_to_icrs_vectors(
        full_x,
        full_y,
        camera=camera,
        icrs_basis=basis,  # type: ignore[arg-type]
    )
    return TargetSampleGrid(
        vectors_icrs=vectors,
        output_points_xy=np.column_stack((grid_x.ravel(), grid_y.ravel())),
        valid=valid,
        sample_width=sample_width,
        sample_height=sample_height,
        output_width_px=output_width,
        output_height_px=output_height,
        candidate_count=int(vectors.shape[0]),
    )


def _source_sample_dimensions(
    width_px: int,
    height_px: int,
    *,
    long_side_px: int,
) -> tuple[int, int]:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("源图模型尺寸无效。")
    scale = float(long_side_px) / max(width_px, height_px)
    return (
        max(2, int(round(width_px * scale))),
        max(2, int(round(height_px * scale))),
    )


def _pixel_points_to_icrs_vectors(
    frame: PhotometricFrame,
    pixel_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixel_points, dtype=np.float64)
    radec = np.asarray(frame.model.pixel_to_sky_points(pixels), dtype=np.float64)
    if radec.shape != (pixels.shape[0], 2):
        raise ValueError(
            f"模型 Pixel→ICRS 输出尺寸异常：{frame.model_path.name} → {radec.shape}"
        )
    valid = np.all(np.isfinite(radec), axis=1)
    vectors = np.full((pixels.shape[0], 3), np.nan, dtype=np.float64)
    if np.any(valid):
        vectors[valid] = radec_to_unit_vectors(
            radec[valid, 0],
            radec[valid, 1],
        )
    return vectors, valid


def _model_coverage(
    frame: PhotometricFrame,
    vectors_icrs: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(
        frame.model.icrs_vectors_to_pixel_points(vectors_icrs),
        dtype=np.float64,
    )
    if pixels.shape != (vectors_icrs.shape[0], 2):
        raise ValueError(
            f"模型 ICRS→Pixel 输出尺寸异常：{frame.model_path.name} → {pixels.shape}"
        )
    return (
        np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= frame.width_px - 1.0)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= frame.height_px - 1.0)
    )


def build_model_coverage_sample_grid(
    frames: tuple[PhotometricFrame, ...],
    *,
    long_side_px: int,
) -> TargetSampleGrid:
    """Build samples directly from the union of source-model sky footprints.

    Each frame contributes a source-pixel grid. The per-frame density is scaled
    by ``sqrt(frame_count)`` so that ``long_side_px`` continues to control the
    approximate total work rather than multiplying it by the number of frames.
    Directions not covered by at least two models are discarded before TIFF
    measurement starts.
    """

    if len(frames) < 2:
        raise ValueError("模型覆盖采样至少需要两帧。")
    if long_side_px < 2:
        raise ValueError("采样长边至少需要 2 px。")

    reference_size = (frames[0].width_px, frames[0].height_px)
    if any((frame.width_px, frame.height_px) != reference_size for frame in frames):
        raise ValueError("模型覆盖采样要求所有源图模型具有相同尺寸。")

    per_frame_long_side = max(
        8,
        int(round(float(long_side_px) / math.sqrt(len(frames)))),
    )
    sample_width, sample_height = _source_sample_dimensions(
        *reference_size,
        long_side_px=per_frame_long_side,
    )
    x_values = np.linspace(
        0.0,
        reference_size[0] - 1.0,
        sample_width,
        dtype=np.float64,
    )
    y_values = np.linspace(
        0.0,
        reference_size[1] - 1.0,
        sample_height,
        dtype=np.float64,
    )
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    source_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    vector_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    for frame in frames:
        vectors, valid = _pixel_points_to_icrs_vectors(frame, source_points)
        if np.any(valid):
            vector_parts.append(vectors[valid])
            point_parts.append(source_points[valid])
    if not vector_parts:
        raise ValueError("无法从任何 model.json 生成有效 ICRS 方向。")

    candidate_vectors = np.concatenate(vector_parts, axis=0)
    candidate_points = np.concatenate(point_parts, axis=0)
    coverage_count = np.zeros(candidate_vectors.shape[0], dtype=np.uint16)
    for frame in frames:
        coverage_count += _model_coverage(frame, candidate_vectors)
    overlap = coverage_count >= 2
    if not np.any(overlap):
        raise ValueError("各 model.json 的 ICRS 覆盖范围之间没有有效重叠。")

    vectors = candidate_vectors[overlap]
    return TargetSampleGrid(
        vectors_icrs=vectors,
        output_points_xy=candidate_points[overlap],
        valid=np.ones(vectors.shape[0], dtype=bool),
        sample_width=sample_width,
        sample_height=sample_height,
        output_width_px=reference_size[0],
        output_height_px=reference_size[1],
        sampling_mode="model_icrs_coverage",
        source_frame_count=len(frames),
        candidate_count=int(candidate_vectors.shape[0]),
    )
