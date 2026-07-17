"""Qt 图像与纯数值 PSF 模块之间的兼容适配层。"""

from __future__ import annotations

import math

import numpy as np
from PyQt5.QtGui import QImage

from .psf.fitting import (
    StarFitError,
    detect_star_candidates_from_array,
    fit_star_position_from_array,
)
from .psf.models import FittedStarPosition, StarSourceCandidate


def qimage_to_luminance_array(image: QImage) -> np.ndarray:
    """把 QImage 转成连续的浮点亮度数组。"""

    if image.isNull():
        raise StarFitError("图像为空，无法进行星点拟合。", code="invalid_image")

    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    buffer_size = rgb_image.sizeInBytes() if hasattr(rgb_image, "sizeInBytes") else rgb_image.byteCount()
    image_bits = rgb_image.bits()
    image_bits.setsize(buffer_size)

    raw = np.frombuffer(image_bits, dtype=np.uint8)
    rows = raw.reshape((height, bytes_per_line))
    rgb = rows[:, : width * 3].reshape((height, width, 3)).astype(np.float64)
    return np.ascontiguousarray(0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2])


def qimage_to_grayscale_array(image: QImage) -> np.ndarray:
    """把整张 QImage 快速转换为序列处理使用的连续 8 位灰度数组。"""

    if image.isNull():
        raise StarFitError("图像为空，无法进行星点检测。", code="invalid_image")

    gray_image = image.convertToFormat(QImage.Format_Grayscale8)
    width = gray_image.width()
    height = gray_image.height()
    bytes_per_line = gray_image.bytesPerLine()
    buffer_size = gray_image.sizeInBytes() if hasattr(gray_image, "sizeInBytes") else gray_image.byteCount()
    image_bits = gray_image.bits()
    image_bits.setsize(buffer_size)
    rows = np.frombuffer(image_bits, dtype=np.uint8).reshape((height, bytes_per_line))
    # 临时 QImage 会在函数返回后销毁，必须强制复制，不能返回其底层内存视图。
    return rows[:, :width].copy()


def _fit_crop_geometry(
    image: QImage,
    click_x: float,
    click_y: float,
    search_radius_px: int,
    max_fit_radius_px: int,
) -> tuple[int, int, int, int]:
    padding = max(4, min(max_fit_radius_px, 48))
    crop_radius = int(search_radius_px + padding)
    center_x = int(round(click_x))
    center_y = int(round(click_y))
    return (
        max(0, center_x - crop_radius),
        max(0, center_y - crop_radius),
        min(image.width(), center_x + crop_radius + 1),
        min(image.height(), center_y + crop_radius + 1),
    )


def fit_star_position(
    image: QImage,
    click_x: float,
    click_y: float,
    radius_px: int = 12,
    *,
    max_fit_radius_px: int | None = None,
    reject_ambiguous: bool = False,
    selection_mode: str = "manual",
) -> FittedStarPosition:
    """在搜索圆中选择星源，再用独立的自适应窗口测量 PSF。"""

    if image.isNull():
        raise StarFitError("图像为空，无法进行星点拟合。", code="invalid_image")
    if not (0.0 <= click_x < image.width() and 0.0 <= click_y < image.height()):
        raise StarFitError("点击位置不在真实图像范围内。", code="outside_image")
    search_radius = max(4, int(radius_px))
    fit_radius_limit = (
        max(8, min(64, int(max_fit_radius_px)))
        if max_fit_radius_px is not None
        else 48
    )
    crop_x0, crop_y0, crop_x1, crop_y1 = _fit_crop_geometry(
        image,
        click_x,
        click_y,
        search_radius,
        fit_radius_limit,
    )
    cropped_image = image.copy(crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0)
    luminance = qimage_to_luminance_array(cropped_image)
    fitted = fit_star_position_from_array(
        luminance,
        click_x=click_x - crop_x0,
        click_y=click_y - crop_y0,
        radius_px=search_radius,
        max_fit_radius_px=fit_radius_limit,
        reject_ambiguous=reject_ambiguous,
        saturation_level=255.0,
        selection_mode=selection_mode,
    )
    if not (math.isfinite(fitted.x) and math.isfinite(fitted.y)):
        raise StarFitError("PSF 拟合返回了无效坐标。", code="invalid_result")
    return FittedStarPosition(
        x=fitted.x + crop_x0,
        y=fitted.y + crop_y0,
        amplitude=fitted.amplitude,
        background=fitted.background,
        sigma_x=fitted.sigma_x,
        sigma_y=fitted.sigma_y,
        theta_rad=fitted.theta_rad,
        fwhm_x=fitted.fwhm_x,
        fwhm_y=fitted.fwhm_y,
        snr=fitted.snr,
        fit_error=fitted.fit_error,
        saturated=fitted.saturated,
        saturation_fraction=fitted.saturation_fraction,
        blended=fitted.blended,
        quality_score=fitted.quality_score,
    )


def detect_star_candidates(
    image: QImage,
    click_x: float,
    click_y: float,
    radius_px: int,
) -> list[StarSourceCandidate]:
    """从 QImage 的局部搜索圆检测星源，返回整图坐标。"""

    if image.isNull():
        raise StarFitError("图像为空，无法检测星点。", code="invalid_image")
    if not (0.0 <= click_x < image.width() and 0.0 <= click_y < image.height()):
        raise StarFitError("搜索位置不在真实图像范围内。", code="outside_image")
    search_radius = max(4, int(radius_px))
    padding = max(6, min(24, search_radius // 2))
    crop_radius = search_radius + padding
    center_x = int(round(click_x))
    center_y = int(round(click_y))
    crop_x0 = max(0, center_x - crop_radius)
    crop_y0 = max(0, center_y - crop_radius)
    crop_x1 = min(image.width(), center_x + crop_radius + 1)
    crop_y1 = min(image.height(), center_y + crop_radius + 1)
    cropped = image.copy(crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0)
    luminance = qimage_to_luminance_array(cropped)
    local_candidates = detect_star_candidates_from_array(
        luminance,
        click_x=click_x - crop_x0,
        click_y=click_y - crop_y0,
        search_radius_px=search_radius,
        saturation_level=255.0,
    )
    return [
        StarSourceCandidate(
            x=candidate.x + crop_x0,
            y=candidate.y + crop_y0,
            major_axis=candidate.major_axis,
            minor_axis=candidate.minor_axis,
            theta_rad=candidate.theta_rad,
            flux=candidate.flux,
            peak=candidate.peak,
            snr=candidate.snr,
            npix=candidate.npix,
            label=candidate.label,
            saturated=candidate.saturated,
            saturation_fraction=candidate.saturation_fraction,
            blended=candidate.blended,
            quality_score=candidate.quality_score,
        )
        for candidate in local_candidates
    ]


__all__ = [
    "FittedStarPosition",
    "StarFitError",
    "StarSourceCandidate",
    "detect_star_candidates_from_array",
    "detect_star_candidates",
    "fit_star_position",
    "fit_star_position_from_array",
    "qimage_to_grayscale_array",
    "qimage_to_luminance_array",
]
