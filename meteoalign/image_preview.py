from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QImageReader

from .native_image import native_image_luminance


SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
IMAGE_FILE_FILTER = "图像文件 (*.tif *.tiff *.jpg *.jpeg *.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;PNG (*.png)"
DEFAULT_PREVIEW_LONG_SIDE_PX = 2400
PREVIEW_ALLOCATION_LIMIT_MB = 512
FULL_IMAGE_ALLOCATION_LIMIT_MB = 4096


@dataclass(frozen=True)
class ImagePreview:
    path: Path
    image: QImage
    original_width: int
    original_height: int
    native_luminance: np.ndarray | None = None


def _scaled_preview_size(width: int, height: int, max_long_side_px: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸无效，无法生成预览。")

    long_side = max(width, height)
    scale = min(1.0, max_long_side_px / float(long_side))
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    return scaled_width, scaled_height


def _reader_error(reader: QImageReader) -> str:
    error = reader.errorString()
    if error:
        return error
    return "未知图像读取错误"


def load_image_preview(
    path: str | Path,
    max_long_side_px: int | None = DEFAULT_PREVIEW_LONG_SIDE_PX,
    *,
    include_native_luminance: bool = False,
) -> ImagePreview:
    image_path = Path(path).expanduser()
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("当前只支持 TIFF、JPG 与 PNG 图像。")
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在：{image_path}")
    image_path = image_path.resolve()

    if hasattr(QImageReader, "setAllocationLimit"):
        # 整图读取需要允许 TIFF 解码器分配临时高位深图像，随后会立即转成 8-bit。
        allocation_limit_mb = FULL_IMAGE_ALLOCATION_LIMIT_MB if max_long_side_px is None else PREVIEW_ALLOCATION_LIMIT_MB
        QImageReader.setAllocationLimit(allocation_limit_mb)

    reader = QImageReader(str(image_path))
    reader.setAutoTransform(True)
    if not reader.canRead():
        raise ValueError(f"无法读取图像：{_reader_error(reader)}")

    original_size = reader.size()
    if not original_size.isValid():
        raise ValueError(f"无法读取图像尺寸：{_reader_error(reader)}")

    if max_long_side_px is not None:
        scaled_width, scaled_height = _scaled_preview_size(
            original_size.width(),
            original_size.height(),
            max_long_side_px,
        )
        reader.setScaledSize(QSize(scaled_width, scaled_height))

    image = reader.read()
    if image.isNull():
        raise ValueError(f"无法生成显示图像：{_reader_error(reader)}")

    image_8bit = image.convertToFormat(QImage.Format_RGB888)
    del image

    native_luminance = native_image_luminance(image_path) if include_native_luminance else None
    if native_luminance is not None and native_luminance.shape != (
        image_8bit.height(),
        image_8bit.width(),
    ):
        raise ValueError(
            "原生位深图像方向或尺寸与显示图不一致："
            f"显示 {image_8bit.width()} x {image_8bit.height()} px，"
            f"计算数据 {native_luminance.shape[1]} x {native_luminance.shape[0]} px。"
        )

    if max_long_side_px is not None and max(image_8bit.width(), image_8bit.height()) > max_long_side_px:
        # 少数解码器可能忽略 setScaledSize；这里再压一次，保证界面只持有缩放图。
        scaled_width, scaled_height = _scaled_preview_size(
            image_8bit.width(),
            image_8bit.height(),
            max_long_side_px,
        )
        image_8bit = image_8bit.scaled(QSize(scaled_width, scaled_height), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return ImagePreview(
        path=image_path,
        image=image_8bit,
        original_width=original_size.width(),
        original_height=original_size.height(),
        native_luminance=native_luminance,
    )
