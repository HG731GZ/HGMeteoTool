from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

from ...native_image import load_native_image_array
from .types import SolverConfig


@dataclass(frozen=True)
class MeasurementMap:
    rgb: np.ndarray
    valid: np.ndarray
    scale_x: float
    scale_y: float
    rejected_fraction: float


def read_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Read a TIFF/JPEG/PNG mask; any non-zero channel means valid sky."""

    values = np.asarray(load_native_image_array(Path(mask_path).expanduser()))
    if values.ndim == 2:
        return values != 0
    if values.ndim == 3 and values.shape[2] >= 1:
        return np.any(values != 0, axis=2)
    raise ValueError(f"蒙版图像形状不受支持：{values.shape}")


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"只支持灰度或 RGB TIFF，实际尺寸为 {array.shape}。")
    return array[:, :, :3]


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    return max(1e-6, 1.4826 * float(np.median(np.abs(finite - median))))


def build_measurement_map(
    image_path: str | Path,
    *,
    mask_path: str | Path | None,
    config: SolverConfig,
) -> MeasurementMap:
    source = _as_rgb(tifffile.imread(image_path))
    source_height, source_width = source.shape[:2]
    target_width = max(8, int(round(source_width / config.downsample_factor)))
    target_height = max(8, int(round(source_height / config.downsample_factor)))
    reduced = cv2.resize(
        source.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    valid = np.ones((target_height, target_width), dtype=bool)
    if mask_path is not None:
        user_mask = read_binary_mask(mask_path)
        if user_mask.shape != (source_height, source_width):
            raise ValueError(
                f"蒙版尺寸 {user_mask.shape[::-1]} 与源图 "
                f"{source_width}×{source_height} 不一致：{mask_path}"
            )
        reduced_mask = cv2.resize(
            user_mask.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        valid &= reduced_mask != 0

    dtype_max = (
        float(np.iinfo(source.dtype).max)
        if np.issubdtype(source.dtype, np.integer)
        else float(np.nanmax(source))
    )
    saturated = np.max(reduced, axis=2) >= dtype_max * config.saturation_fraction
    gray = (
        reduced[:, :, 0] * 0.2126
        + reduced[:, :, 1] * 0.7152
        + reduced[:, :, 2] * 0.0722
    )
    background_sigma = max(1.0, config.patch_size_px * 1.5)
    background = cv2.GaussianBlur(gray, (0, 0), background_sigma)
    detail = gray - background
    detail_center = float(np.median(detail[valid])) if np.any(valid) else 0.0
    detail_sigma = _robust_sigma(detail[valid])
    stars = detail > detail_center + config.star_sigma * detail_sigma

    gradient_x = cv2.Sobel(background, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(background, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    gradient_values = gradient[valid]
    gradient_median = float(np.median(gradient_values)) if gradient_values.size else 0.0
    gradient_limit = max(
        gradient_median + 6.0 * _robust_sigma(gradient_values),
        float(np.percentile(gradient_values, 98.5)) if gradient_values.size else 0.0,
    )
    high_gradient = gradient > gradient_limit

    valid_gray = gray[valid]
    bright_limit = float(np.percentile(valid_gray, 99.8)) if valid_gray.size else float("inf")
    exceptionally_bright = gray > bright_limit
    rejected = saturated | stars | high_gradient | exceptionally_bright
    kernel = np.ones((3, 3), dtype=np.uint8)
    rejected = cv2.dilate(rejected.astype(np.uint8), kernel, iterations=1) != 0
    valid &= ~rejected

    patch_area = config.patch_size_px * config.patch_size_px
    valid_float = valid.astype(np.float32)
    valid_count = cv2.boxFilter(
        valid_float,
        ddepth=-1,
        ksize=(config.patch_size_px, config.patch_size_px),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    local_rgb = np.empty_like(reduced, dtype=np.float32)
    for channel in range(3):
        channel_sum = cv2.boxFilter(
            reduced[:, :, channel] * valid_float,
            ddepth=-1,
            ksize=(config.patch_size_px, config.patch_size_px),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        np.divide(
            channel_sum,
            valid_count,
            out=local_rgb[:, :, channel],
            where=valid_count > 0,
        )
    local_valid = valid_count >= patch_area * config.minimum_patch_valid_fraction
    radius = config.patch_size_px // 2
    if radius:
        local_valid[:radius, :] = False
        local_valid[-radius:, :] = False
        local_valid[:, :radius] = False
        local_valid[:, -radius:] = False
    local_rgb[~local_valid] = np.nan
    return MeasurementMap(
        rgb=local_rgb,
        valid=local_valid,
        scale_x=target_width / source_width,
        scale_y=target_height / source_height,
        rejected_fraction=1.0 - float(np.mean(valid)),
    )
