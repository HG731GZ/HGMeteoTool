"""按原始像素位深读取图像，供数值计算与最终导出共用。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - 环境文件已声明 OpenCV，兜底用于给出清晰错误。
    cv2 = None

try:
    import tifffile
except ImportError:  # pragma: no cover - 环境文件已声明 tifffile，兜底用于给出清晰错误。
    tifffile = None


_TIFF_SUFFIXES = {".tif", ".tiff"}


def _normalize_channel_axis(array: np.ndarray) -> np.ndarray:
    """把平面存储的通道轴移到末尾，保留普通灰度与交错 RGB。"""

    values = np.asarray(array)
    if values.ndim != 3:
        return values
    if values.shape[-1] in (1, 2, 3, 4):
        return values
    if values.shape[0] in (1, 2, 3, 4):
        return np.moveaxis(values, 0, -1)
    return values


def _apply_exif_orientation(array: np.ndarray, orientation: int) -> np.ndarray:
    """按照 EXIF Orientation 把原生像素变换到界面显示方向。"""

    values = np.asarray(array)
    if orientation == 2:
        values = np.flip(values, axis=1)
    elif orientation == 3:
        values = np.rot90(values, 2, axes=(0, 1))
    elif orientation == 4:
        values = np.flip(values, axis=0)
    elif orientation == 5:
        values = np.swapaxes(values, 0, 1)
    elif orientation == 6:
        values = np.rot90(values, -1, axes=(0, 1))
    elif orientation == 7:
        values = np.flip(np.swapaxes(values, 0, 1), axis=(0, 1))
    elif orientation == 8:
        values = np.rot90(values, 1, axes=(0, 1))
    return np.ascontiguousarray(values)


def _pil_exif_orientation(path: Path) -> int:
    """仅读取方向标签，不通过 Pillow 解码高位深 RGB 像素。"""

    try:
        with Image.open(path) as image:
            return int(image.getexif().get(274, 1))
    except Exception:
        return 1


def _load_tiff_native(path: Path) -> tuple[np.ndarray, int]:
    if tifffile is None:
        raise RuntimeError("当前环境缺少 tifffile，无法按原始位深读取 TIFF。")
    with tifffile.TiffFile(path) as tiff:
        if not tiff.pages:
            raise ValueError(f"TIFF 中没有可读取的图像页：{path}")
        page = tiff.pages[0]
        array = page.asarray()
        orientation_tag = page.tags.get(274)
        orientation = int(orientation_tag.value) if orientation_tag is not None else 1
    return _normalize_channel_axis(np.asarray(array)), orientation


def _load_opencv_native(path: Path) -> tuple[np.ndarray, int]:
    if cv2 is None:
        raise RuntimeError("当前环境缺少 OpenCV，无法按原始位深读取图像。")
    flags = int(cv2.IMREAD_UNCHANGED)
    if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
        flags |= int(cv2.IMREAD_IGNORE_ORIENTATION)
    # imdecode 配合 NumPy 读取可避免 Windows 下 cv2.imread 对非 ASCII 路径的兼容问题。
    encoded = np.fromfile(path, dtype=np.uint8)
    array = cv2.imdecode(encoded, flags)
    if array is None:
        raise ValueError(f"无法读取图像像素：{path}")
    if array.ndim == 3:
        if array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
    return np.asarray(array), _pil_exif_orientation(path)


def load_native_image_array(path: str | Path) -> np.ndarray:
    """读取首个图像平面，保留 uint8/uint16 等原始数值类型与 EXIF 方向。"""

    image_path = Path(path).expanduser().resolve()
    if image_path.suffix.lower() in _TIFF_SUFFIXES:
        array, orientation = _load_tiff_native(image_path)
    else:
        array, orientation = _load_opencv_native(image_path)
    array = _normalize_channel_axis(array)
    if array.ndim not in (2, 3):
        raise ValueError(f"源图像素形状不支持：{array.shape}")
    return _apply_exif_orientation(array, orientation)


def native_array_to_luminance(array: np.ndarray) -> np.ndarray:
    """把原生灰度或 RGB 像素转成同 dtype 的连续亮度数组。"""

    values = np.asarray(array)
    if values.ndim == 2:
        return np.ascontiguousarray(values)
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError(f"源图像素形状不支持：{values.shape}")
    if values.shape[2] == 1:
        return np.ascontiguousarray(values[:, :, 0])
    if values.shape[2] == 2:
        return np.ascontiguousarray(values[:, :, 0])

    rgb = values[:, :, :3]
    if values.dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
        # 定点权重总和为 65536，可避免大图转换时同时分配三个 float64 通道。
        luminance_u32 = rgb[:, :, 0].astype(np.uint32) * 13933
        luminance_u32 += rgb[:, :, 1].astype(np.uint32) * 46871
        luminance_u32 += rgb[:, :, 2].astype(np.uint32) * 4732
        luminance_u32 += 32768
        return np.ascontiguousarray((luminance_u32 >> 16).astype(values.dtype))

    luminance = (
        0.2126 * rgb[:, :, 0].astype(np.float64)
        + 0.7152 * rgb[:, :, 1].astype(np.float64)
        + 0.0722 * rgb[:, :, 2].astype(np.float64)
    )
    if np.issubdtype(values.dtype, np.integer):
        limits = np.iinfo(values.dtype)
        luminance = np.clip(np.rint(luminance), limits.min, limits.max).astype(values.dtype)
    elif np.issubdtype(values.dtype, np.floating):
        luminance = luminance.astype(values.dtype)
    return np.ascontiguousarray(luminance)


def native_image_luminance(path: str | Path) -> np.ndarray:
    """按源图位深读取并生成 PSF 使用的亮度平面。"""

    return native_array_to_luminance(load_native_image_array(path))


__all__ = ["load_native_image_array", "native_array_to_luminance", "native_image_luminance"]
