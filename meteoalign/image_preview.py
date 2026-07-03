from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QImageReader


SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
IMAGE_FILE_FILTER = "图像文件 (*.tif *.tiff *.jpg *.jpeg *.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;PNG (*.png)"
DEFAULT_PREVIEW_LONG_SIDE_PX = 2400
PREVIEW_ALLOCATION_LIMIT_MB = 512


@dataclass(frozen=True)
class ImagePreview:
    path: Path
    image: QImage
    original_width: int
    original_height: int
    preview_width: int
    preview_height: int
    preview_scale: float


def _scaled_preview_size(width: int, height: int, max_long_side_px: int) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸无效，无法生成预览。")

    long_side = max(width, height)
    scale = min(1.0, max_long_side_px / float(long_side))
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    return preview_width, preview_height, scale


def _reader_error(reader: QImageReader) -> str:
    error = reader.errorString()
    if error:
        return error
    return "未知图像读取错误"


def load_image_preview(path: str | Path, max_long_side_px: int = DEFAULT_PREVIEW_LONG_SIDE_PX) -> ImagePreview:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("当前只支持 TIFF、JPG 与 PNG 图像。")
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在：{image_path}")

    if hasattr(QImageReader, "setAllocationLimit"):
        # 给 Qt 解码器设置内存护栏，避免缩略读取失败时尝试分配整张超大图。
        QImageReader.setAllocationLimit(PREVIEW_ALLOCATION_LIMIT_MB)

    reader = QImageReader(str(image_path))
    reader.setAutoTransform(True)
    if not reader.canRead():
        raise ValueError(f"无法读取图像：{_reader_error(reader)}")

    original_size = reader.size()
    if not original_size.isValid():
        raise ValueError(f"无法读取图像尺寸：{_reader_error(reader)}")

    preview_width, preview_height, scale = _scaled_preview_size(
        original_size.width(),
        original_size.height(),
        max_long_side_px,
    )
    reader.setScaledSize(QSize(preview_width, preview_height))
    image = reader.read()
    if image.isNull():
        raise ValueError(f"无法生成图像预览：{_reader_error(reader)}")

    if max(image.width(), image.height()) > max_long_side_px:
        # 少数解码器可能忽略 setScaledSize；这里再压一次，保证界面只持有预览图。
        fallback_width, fallback_height, scale = _scaled_preview_size(
            image.width(),
            image.height(),
            max_long_side_px,
        )
        image = image.scaled(QSize(fallback_width, fallback_height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        preview_width = image.width()
        preview_height = image.height()

    return ImagePreview(
        path=image_path,
        image=image,
        original_width=original_size.width(),
        original_height=original_size.height(),
        preview_width=preview_width,
        preview_height=preview_height,
        preview_scale=scale,
    )
