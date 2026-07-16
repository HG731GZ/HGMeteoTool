from __future__ import annotations

import os
from pathlib import Path

from ..binary_mask import image_with_binary_mask as _image_with_binary_mask
from ..binary_mask import qimage_to_binary_mask as _qimage_to_binary_mask

from ..image_path_resolution import associated_image_candidates, expected_image_size, first_matching_image_path


# ---------------------------------------------------------------------------
# 星点匹配会话 — 图像路径解析
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


def _session_image_file_stem(real_image: dict[str, object]) -> str:
    """读取 JSON 记录的无后缀图像名，并兼容旧版路径字段。"""

    file_stem = real_image.get("file_stem")
    if isinstance(file_stem, str) and file_stem.strip():
        return Path(file_stem.strip()).name
    file_name = _session_image_file_name(real_image)
    return Path(file_name).stem if file_name else ""


def _star_pair_session_real_image_metadata(payload: object) -> dict[str, object]:
    """校验星点匹配 JSON 的基本结构并返回真实图像元数据。"""

    if not isinstance(payload, dict):
        raise ValueError("JSON 根对象必须是字典。")
    if payload.get("format") != "meteoalign_star_pair_session":
        raise ValueError("当前只支持 HoshinoPanoAssistant 星点匹配 JSON。")
    real_image = payload.get("real_image")
    if not isinstance(real_image, dict):
        raise ValueError("JSON 缺少 real_image 字段。")
    return real_image


def _validate_star_pair_session_current_image(
    payload: object,
    image_path: str | Path,
    image_size: tuple[int, int],
) -> Path:
    """仅按无后缀文件名和尺寸确认当前图像是否属于匹配 JSON。"""

    real_image = _star_pair_session_real_image_metadata(payload)
    current_path = Path(image_path).expanduser().resolve()
    recorded_stem = _session_image_file_stem(real_image)
    if recorded_stem and recorded_stem.casefold() != current_path.stem.casefold():
        raise ValueError(
            "当前图像名称与 JSON 记录不一致："
            f"JSON 记录为 {recorded_stem}，当前图像为 {current_path.stem}。"
        )

    recorded_size = expected_image_size(real_image)
    current_size = (int(image_size[0]), int(image_size[1]))
    if recorded_size is not None and recorded_size != current_size:
        raise ValueError(
            "当前图像尺寸与 JSON 记录不一致："
            f"JSON 记录为 {recorded_size[0]} x {recorded_size[1]} px，"
            f"当前图像为 {current_size[0]} x {current_size[1]} px。"
        )
    return current_path


def _resolve_star_pair_session_real_image_path(payload: object, source_path: Path) -> Path:
    """从星点匹配 JSON 中解析真实图像路径。"""
    real_image = _star_pair_session_real_image_metadata(payload)

    searched_paths = associated_image_candidates(real_image, source_path)
    image_path = first_matching_image_path(searched_paths, expected_image_size(real_image))
    if image_path is not None:
        return image_path

    if not searched_paths:
        raise ValueError("JSON 缺少真实图像文件名、相对路径与完整路径。")
    searched_text = "\n".join(str(path) for path in searched_paths)
    raise FileNotFoundError(
        "真实图像不存在或尺寸与 JSON 记录不一致，"
        f"已按同目录主文件名、相对路径和完整路径查找：\n{searched_text}"
    )


def _relative_image_path_for_session(image_path: Path, json_path: Path) -> str:
    """计算图像相对于 JSON 文件的路径，用于会话导出。"""
    json_dir = json_path.expanduser().resolve().parent
    try:
        return os.path.relpath(str(image_path), start=str(json_dir))
    except ValueError:
        # Windows 不同盘符之间没有有效相对路径，此时保留文件名并继续依赖完整路径兜底。
        return image_path.name
