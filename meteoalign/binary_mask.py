"""QImage 与二值蒙版之间的通用转换和显示处理。"""

from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QImage


def qimage_to_binary_mask(image: QImage) -> np.ndarray:
    """将 QImage 转为二值蒙版，任一颜色通道非零即视为有效。"""

    if image.isNull():
        raise ValueError("蒙版图像为空。")

    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    buffer_size = rgb_image.sizeInBytes() if hasattr(rgb_image, "sizeInBytes") else rgb_image.byteCount()
    image_bits = rgb_image.bits()
    image_bits.setsize(buffer_size)

    raw = np.frombuffer(image_bits, dtype=np.uint8)
    rows = raw.reshape((height, bytes_per_line))
    rgb = rows[:, : width * 3].reshape((height, width, 3))
    return np.any(rgb != 0, axis=2)


def image_with_binary_mask(image: QImage, mask: np.ndarray) -> QImage:
    """将二值蒙版应用到图像上，蒙版外像素置零。"""

    if image.isNull():
        return QImage()

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != (image.height(), image.width()):
        raise ValueError("蒙版尺寸必须与图像尺寸一致。")

    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    buffer_size = rgb_image.sizeInBytes() if hasattr(rgb_image, "sizeInBytes") else rgb_image.byteCount()
    image_bits = rgb_image.bits()
    image_bits.setsize(buffer_size)

    raw = np.frombuffer(image_bits, dtype=np.uint8)
    rows = np.array(raw.reshape((height, bytes_per_line)), copy=True)
    pixels = rows[:, : width * 3].reshape((height, width, 3))
    pixels[~mask_array] = 0
    return QImage(rows.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def scale_binary_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """以最近邻方式缩放二值蒙版，避免有效区域边缘出现灰色过渡。"""

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2:
        raise ValueError("二值蒙版必须是二维数组。")
    mask_height, mask_width = mask_array.shape
    if mask_height <= 0 or mask_width <= 0 or width <= 0 or height <= 0:
        raise ValueError("蒙版或目标图像尺寸无效。")
    if mask_array.shape == (height, width):
        return mask_array

    y_indices = np.minimum(
        (np.arange(height, dtype=np.float64) * mask_height / float(height)).astype(np.int64),
        mask_height - 1,
    )
    x_indices = np.minimum(
        (np.arange(width, dtype=np.float64) * mask_width / float(width)).astype(np.int64),
        mask_width - 1,
    )
    return np.asarray(mask_array[np.ix_(y_indices, x_indices)], dtype=bool)
