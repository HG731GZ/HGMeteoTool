from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .masking import MeasurementMap, build_measurement_map
from .sampling import TargetSampleGrid
from .types import CancelCallback, ObservationSet, PhotometricFrame, ProgressCallback, SolverConfig


def _sample_measurement_map(
    measurement_map: MeasurementMap,
    source_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(source_pixels, dtype=np.float64)
    reduced_x = (pixels[:, 0] + 0.5) * measurement_map.scale_x - 0.5
    reduced_y = (pixels[:, 1] + 0.5) * measurement_map.scale_y - 0.5
    coordinates = np.vstack((reduced_y, reduced_x))
    sampled = np.column_stack(
        [
            map_coordinates(
                measurement_map.rgb[:, :, channel],
                coordinates,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            for channel in range(3)
        ]
    )
    sampled_valid = map_coordinates(
        measurement_map.valid.astype(np.uint8),
        coordinates,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) != 0
    sampled_valid &= np.all(np.isfinite(sampled), axis=1)
    return sampled.astype(np.float32), sampled_valid


def generate_observations(
    frames: tuple[PhotometricFrame, ...],
    sample_grid: TargetSampleGrid,
    *,
    config: SolverConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> ObservationSet:
    if not frames:
        raise ValueError("没有可用于生成观测的源图。")
    sample_count = int(sample_grid.vectors_icrs.shape[0])
    reference_frame = np.full(sample_count, -1, dtype=np.int32)
    reference_pixels = np.full((sample_count, 2), np.nan, dtype=np.float32)
    reference_rgb = np.full((sample_count, 3), np.nan, dtype=np.float32)

    all_frame_i: list[np.ndarray] = []
    all_frame_j: list[np.ndarray] = []
    all_point_i: list[np.ndarray] = []
    all_point_j: list[np.ndarray] = []
    all_difference: list[np.ndarray] = []
    all_sample_index: list[np.ndarray] = []

    for current_index, frame in enumerate(frames):
        if cancel_callback is not None and cancel_callback():
            raise InterruptedError("用户已取消周边梯度优化。")
        if progress_callback is not None:
            progress_callback(
                f"生成重叠观测：{frame.image_path.name}",
                current_index,
                len(frames),
            )
        source_pixels = frame.model.icrs_vectors_to_pixel_points(sample_grid.vectors_icrs)
        coverage = (
            sample_grid.valid
            & np.all(np.isfinite(source_pixels), axis=1)
            & (source_pixels[:, 0] >= 0.0)
            & (source_pixels[:, 0] <= frame.width_px - 1.0)
            & (source_pixels[:, 1] >= 0.0)
            & (source_pixels[:, 1] <= frame.height_px - 1.0)
        )
        measurement_map = build_measurement_map(
            frame.image_path,
            mask_path=frame.mask_path,
            config=config,
        )
        sampled_rgb, measurement_valid = _sample_measurement_map(measurement_map, source_pixels)
        valid = coverage & measurement_valid

        overlap = valid & (reference_frame >= 0)
        overlap_indices = np.flatnonzero(overlap)
        if overlap_indices.size:
            all_frame_i.append(reference_frame[overlap_indices].copy())
            all_frame_j.append(np.full(overlap_indices.size, current_index, dtype=np.int32))
            all_point_i.append(reference_pixels[overlap_indices].copy())
            all_point_j.append(source_pixels[overlap_indices].astype(np.float32))
            all_difference.append(reference_rgb[overlap_indices] - sampled_rgb[overlap_indices])
            all_sample_index.append(overlap_indices.astype(np.int64))

        first_coverage = valid & (reference_frame < 0)
        reference_frame[first_coverage] = current_index
        reference_pixels[first_coverage] = source_pixels[first_coverage]
        reference_rgb[first_coverage] = sampled_rgb[first_coverage]

    if not all_frame_i:
        raise ValueError("公共取景网格中没有形成有效的跨帧重叠亮度观测。")
    observations = ObservationSet(
        frame_i=np.concatenate(all_frame_i),
        frame_j=np.concatenate(all_frame_j),
        point_i_xy=np.concatenate(all_point_i),
        point_j_xy=np.concatenate(all_point_j),
        difference_rgb=np.concatenate(all_difference),
        target_sample_index=np.concatenate(all_sample_index),
        frame_count=len(frames),
        image_width_px=frames[0].width_px,
        image_height_px=frames[0].height_px,
    )
    if progress_callback is not None:
        progress_callback(f"已生成 {observations.count:,} 条重叠观测。", len(frames), len(frames))
    return observations

