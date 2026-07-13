"""流星框选页面专用的 LibRaw 图像预览读取。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage

from .image_path_resolution import RAW_IMAGE_SUFFIXES
from .image_preview import DEFAULT_PREVIEW_LONG_SIDE_PX, ImagePreview, _scaled_preview_size, load_image_preview


RAW_IMAGE_SUFFIX_SET = frozenset(RAW_IMAGE_SUFFIXES)
_RAW_FILTER_PATTERNS = " ".join(f"*{suffix}" for suffix in RAW_IMAGE_SUFFIXES)
METEOR_IMAGE_FILE_FILTER = (
    "流星图像 (*.tif *.tiff "
    + _RAW_FILTER_PATTERNS
    + " *.png *.jpg *.jpeg);;TIFF (*.tif *.tiff);;RAW ("
    + _RAW_FILTER_PATTERNS
    + ");;PNG (*.png);;JPEG (*.jpg *.jpeg)"
)


def is_raw_image_path(path: str | Path) -> bool:
    """判断文件后缀是否属于 LibRaw 支持的常见 RAW 格式。"""

    return Path(path).suffix.casefold() in RAW_IMAGE_SUFFIX_SET


def _raw_array_to_qimage(rgb) -> QImage:  # type: ignore[no-untyped-def]
    """把 LibRaw 返回的连续 RGB 数组复制为独立 QImage。"""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype.name != "uint8":
        raise ValueError("LibRaw 返回了不受支持的像素格式。")
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, int(width), int(height), int(rgb.strides[0]), QImage.Format_RGB888)
    if image.isNull():
        raise ValueError("无法把 RAW 像素转换为显示图像。")
    return image.copy()


def load_raw_image_preview(
    path: str | Path,
    max_long_side_px: int | None = DEFAULT_PREVIEW_LONG_SIDE_PX,
) -> ImagePreview:
    """通过 rawpy/LibRaw 解码 RAW，并生成与原始坐标等比例的 8 位预览。"""

    image_path = Path(path).expanduser()
    if not is_raw_image_path(image_path):
        raise ValueError("文件后缀不是受支持的 RAW 图像格式。")
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在：{image_path}")
    image_path = image_path.resolve()

    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover - rawpy 已由项目环境声明。
        raise RuntimeError("当前环境缺少 rawpy，无法通过 LibRaw 读取 RAW 图像。") from exc

    try:
        with rawpy.imread(str(image_path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    except Exception as exc:  # noqa: BLE001 - LibRaw 的多种解码错误统一转为界面可读信息。
        raise ValueError(f"LibRaw 无法读取 RAW 图像：{exc}") from exc

    original_height, original_width = (int(rgb.shape[0]), int(rgb.shape[1]))
    image = _raw_array_to_qimage(rgb)
    del rgb

    if max_long_side_px is not None and max(original_width, original_height) > max_long_side_px:
        scaled_width, scaled_height = _scaled_preview_size(
            original_width,
            original_height,
            max_long_side_px,
        )
        image = image.scaled(QSize(scaled_width, scaled_height), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return ImagePreview(
        path=image_path,
        image=image,
        original_width=original_width,
        original_height=original_height,
    )


def load_meteor_image_preview(
    path: str | Path,
    max_long_side_px: int | None = DEFAULT_PREVIEW_LONG_SIDE_PX,
) -> ImagePreview:
    """仅为流星框选页分派普通图像或 RAW 图像读取。"""

    if is_raw_image_path(path):
        return load_raw_image_preview(path, max_long_side_px=max_long_side_px)
    return load_image_preview(path, max_long_side_px=max_long_side_px)


__all__ = [
    "METEOR_IMAGE_FILE_FILTER",
    "RAW_IMAGE_SUFFIX_SET",
    "is_raw_image_path",
    "load_meteor_image_preview",
    "load_raw_image_preview",
]
