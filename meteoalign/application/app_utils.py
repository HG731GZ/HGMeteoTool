from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QImage


# ---------------------------------------------------------------------------
# 星点配对会话 — 图像路径解析
# ---------------------------------------------------------------------------

def _session_image_candidate(path_value: object, source_path: Path, *, force_same_dir_name: bool = False) -> Path | None:
    """从 JSON 中解析候选图像路径。"""
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    image_path = Path(path_value.strip()).expanduser()
    if force_same_dir_name:
        image_path = source_path.parent / image_path.name
    elif not image_path.is_absolute():
        image_path = source_path.parent / image_path
    return image_path.resolve()


def _append_unique_path(paths: list[Path], candidate: Path | None) -> None:
    """按顺序追加候选路径，同时避免同一路径重复出现。"""

    if candidate is None:
        return
    if candidate not in paths:
        paths.append(candidate)


def _session_image_file_name(real_image: dict[str, object]) -> str:
    """优先使用 file_name，旧 JSON 缺失时从相对或绝对路径提取文件名。"""

    for key in ("file_name", "relative_path", "path"):
        value = real_image.get(key)
        if isinstance(value, str) and value.strip():
            name = Path(value.strip()).name
            if name:
                return name
    return ""


def _resolve_star_pair_session_real_image_path(payload: object, source_path: Path) -> Path:
    """从星点配对 JSON 中解析真实图像路径。"""
    if not isinstance(payload, dict):
        raise ValueError("JSON 根对象必须是字典。")
    if payload.get("format") != "meteoalign_star_pair_session":
        raise ValueError("当前只支持 HoshinoPanoAssistant 星点配对 JSON。")
    real_image = payload.get("real_image")
    if not isinstance(real_image, dict):
        raise ValueError("JSON 缺少 real_image 字段。")

    searched_paths: list[Path] = []
    _append_unique_path(
        searched_paths,
        _session_image_candidate(_session_image_file_name(real_image), source_path, force_same_dir_name=True),
    )
    _append_unique_path(searched_paths, _session_image_candidate(real_image.get("relative_path"), source_path))
    raw_path = real_image.get("path")
    absolute_path = Path(raw_path).expanduser() if isinstance(raw_path, str) and raw_path.strip() else None
    if absolute_path is not None:
        if absolute_path.is_absolute():
            _append_unique_path(searched_paths, absolute_path.resolve())
        else:
            _append_unique_path(searched_paths, _session_image_candidate(raw_path, source_path))

    for image_path in searched_paths:
        if image_path.exists():
            return image_path

    if not searched_paths:
        raise ValueError("JSON 缺少真实图像文件名、相对路径与完整路径。")
    searched_text = "\n".join(str(path) for path in searched_paths)
    raise FileNotFoundError(f"真实图像不存在，已按同目录文件名、相对路径和完整路径查找：\n{searched_text}")


def _relative_image_path_for_session(image_path: Path, json_path: Path) -> str:
    """计算图像相对于 JSON 文件的路径，用于会话导出。"""
    json_dir = json_path.expanduser().resolve().parent
    try:
        return os.path.relpath(str(image_path), start=str(json_dir))
    except ValueError:
        # Windows 不同盘符之间没有有效相对路径，此时保留文件名并继续依赖完整路径兜底。
        return image_path.name


# ---------------------------------------------------------------------------
# QImage ↔ numpy 蒙版互转
# ---------------------------------------------------------------------------

def _qimage_to_binary_mask(image: QImage) -> np.ndarray:
    """将 QImage 转为二值蒙版 (bool ndarray)，非零像素视为有效。"""
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


def _image_with_binary_mask(image: QImage, mask: np.ndarray) -> QImage:
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
