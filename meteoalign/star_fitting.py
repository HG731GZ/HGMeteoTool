from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtGui import QImage
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FittedStarPosition:
    x: float
    y: float
    amplitude: float
    background: float
    sigma_x: float
    sigma_y: float


def qimage_to_luminance_array(image: QImage) -> np.ndarray:
    if image.isNull():
        raise ValueError("图像为空，无法进行星点拟合。")

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
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def fit_star_position(image: QImage, click_x: float, click_y: float, radius_px: int = 12) -> FittedStarPosition:
    if image.isNull():
        raise ValueError("图像为空，无法进行星点拟合。")
    if not (0.0 <= click_x < image.width() and 0.0 <= click_y < image.height()):
        raise ValueError("点击位置不在真实图像范围内。")

    center_x = int(round(click_x))
    center_y = int(round(click_y))
    crop_x0 = max(0, center_x - radius_px)
    crop_y0 = max(0, center_y - radius_px)
    crop_x1 = min(image.width(), center_x + radius_px + 1)
    crop_y1 = min(image.height(), center_y + radius_px + 1)
    cropped_image = image.copy(crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0)
    luminance = qimage_to_luminance_array(cropped_image)
    fitted = fit_star_position_from_array(
        luminance,
        click_x=click_x - crop_x0,
        click_y=click_y - crop_y0,
        radius_px=radius_px,
    )
    return FittedStarPosition(
        x=fitted.x + crop_x0,
        y=fitted.y + crop_y0,
        amplitude=fitted.amplitude,
        background=fitted.background,
        sigma_x=fitted.sigma_x,
        sigma_y=fitted.sigma_y,
    )


def fit_star_position_from_array(
    luminance: np.ndarray,
    click_x: float,
    click_y: float,
    radius_px: int = 12,
) -> FittedStarPosition:
    if luminance.ndim != 2:
        raise ValueError("星点拟合需要二维亮度图像。")
    if radius_px < 4:
        raise ValueError("PSF 拟合半径过小。")

    height, width = luminance.shape
    if not (0.0 <= click_x < width and 0.0 <= click_y < height):
        raise ValueError("点击位置不在真实图像范围内。")

    center_x = int(round(click_x))
    center_y = int(round(click_y))
    x0 = max(0, center_x - radius_px)
    x1 = min(width, center_x + radius_px + 1)
    y0 = max(0, center_y - radius_px)
    y1 = min(height, center_y + radius_px + 1)
    patch = np.asarray(luminance[y0:y1, x0:x1], dtype=np.float64)
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        raise ValueError("点击位置太靠近图像边缘，无法截取足够的 PSF 拟合窗口。")
    if not np.all(np.isfinite(patch)):
        raise ValueError("PSF 拟合窗口内存在无效像素。")

    local_y, local_x = np.indices(patch.shape, dtype=np.float64)
    background0 = float(np.percentile(patch, 20.0))
    signal = np.clip(patch - background0, 0.0, None)
    signal_sum = float(signal.sum())
    peak_y, peak_x = np.unravel_index(int(np.argmax(patch)), patch.shape)
    if signal_sum > 1e-6:
        initial_x = float((signal * local_x).sum() / signal_sum)
        initial_y = float((signal * local_y).sum() / signal_sum)
    else:
        initial_x = float(peak_x)
        initial_y = float(peak_y)

    peak_value = float(patch[peak_y, peak_x])
    amplitude0 = max(peak_value - background0, 1.0)
    if amplitude0 < 2.0 and float(np.std(patch)) < 1.0:
        raise ValueError("点击区域内没有足够明显的星点信号。")

    max_patch_value = max(float(np.max(patch)), 1.0)
    initial = np.asarray([background0, amplitude0, initial_x, initial_y, 1.8, 1.8], dtype=np.float64)
    lower = np.asarray([0.0, 0.0, 0.0, 0.0, 0.45, 0.45], dtype=np.float64)
    upper = np.asarray(
        [
            max(255.0, max_patch_value * 1.5),
            max(255.0, amplitude0 * 8.0),
            float(patch.shape[1] - 1),
            float(patch.shape[0] - 1),
            float(radius_px),
            float(radius_px),
        ],
        dtype=np.float64,
    )

    def residual(params: np.ndarray) -> np.ndarray:
        background, amplitude, fit_x, fit_y, sigma_x, sigma_y = params
        model = background + amplitude * np.exp(
            -0.5 * (((local_x - fit_x) / sigma_x) ** 2 + ((local_y - fit_y) / sigma_y) ** 2)
        )
        return (model - patch).ravel()

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(amplitude0 * 0.2, 1.0),
        max_nfev=250,
    )
    if not result.success:
        raise ValueError("PSF 拟合未收敛，请放大图像后重新点选该星点。")

    background, amplitude, fit_x, fit_y, sigma_x, sigma_y = (float(value) for value in result.x)
    image_x = x0 + fit_x
    image_y = y0 + fit_y
    if not (0.0 <= image_x < width and 0.0 <= image_y < height):
        raise ValueError("PSF 拟合结果超出图像范围。")
    if amplitude < 1.0:
        raise ValueError("PSF 拟合得到的星点信号过弱。")

    return FittedStarPosition(
        x=image_x,
        y=image_y,
        amplitude=amplitude,
        background=background,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )
