"""JSON 关联图像的跨后缀路径解析工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from PyQt5.QtGui import QImageReader


STANDARD_IMAGE_SUFFIX_PRIORITY: tuple[tuple[str, ...], ...] = (
    (".tif", ".tiff"),
    (".png",),
    (".jpg", ".jpeg"),
)

RAW_IMAGE_SUFFIXES: tuple[str, ...] = (
    ".3fr",
    ".ari",
    ".arw",
    ".bay",
    ".cap",
    ".cr2",
    ".cr3",
    ".crw",
    ".dcr",
    ".dcs",
    ".dng",
    ".drf",
    ".eip",
    ".erf",
    ".fff",
    ".gpr",
    ".iiq",
    ".k25",
    ".kdc",
    ".mdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".ptx",
    ".pxn",
    ".r3d",
    ".raf",
    ".raw",
    ".rw2",
    ".rwl",
    ".rwz",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
)

METEOR_IMAGE_SUFFIX_PRIORITY: tuple[tuple[str, ...], ...] = (
    STANDARD_IMAGE_SUFFIX_PRIORITY[0],
    RAW_IMAGE_SUFFIXES,
    STANDARD_IMAGE_SUFFIX_PRIORITY[1],
    STANDARD_IMAGE_SUFFIX_PRIORITY[2],
)


def image_file_stem(path_value: str | Path) -> str:
    """返回路径最后一级文件名去掉最终后缀后的内容。"""

    return Path(path_value).stem


def _metadata_file_stem(metadata: Mapping[str, object]) -> str:
    """读取新 JSON 的 file_stem，并兼容只记录带后缀路径的旧 JSON。"""

    value = metadata.get("file_stem")
    if isinstance(value, str) and value.strip():
        return Path(value.strip()).name
    for key in ("file_name", "relative_path", "path"):
        path_value = metadata.get(key)
        if isinstance(path_value, str) and path_value.strip():
            return image_file_stem(path_value.strip())
    return ""


def _append_unique(paths: list[Path], candidate: Path) -> None:
    """追加未出现过的绝对候选路径。"""

    resolved = candidate.expanduser().resolve()
    if resolved not in paths:
        paths.append(resolved)


def _same_stem_candidates(
    directory: Path,
    file_stem: str,
    suffix_priority: tuple[tuple[str, ...], ...],
) -> list[Path]:
    """按指定后缀组优先级返回目录内同主文件名的文件。"""

    if not file_stem or not directory.is_dir():
        return []
    stem_key = file_stem.casefold()
    by_suffix: dict[str, list[Path]] = {}
    try:
        directory_items = tuple(directory.iterdir())
    except OSError:
        return []
    for candidate in directory_items:
        if not candidate.is_file() or image_file_stem(candidate.name).casefold() != stem_key:
            continue
        by_suffix.setdefault(candidate.suffix.casefold(), []).append(candidate)

    ordered: list[Path] = []
    for suffix_group in suffix_priority:
        for suffix in suffix_group:
            ordered.extend(sorted(by_suffix.get(suffix, ()), key=lambda path: path.name.casefold()))
    return ordered


def associated_image_candidates(
    metadata: Mapping[str, object],
    json_path: str | Path,
    *,
    include_raw: bool = False,
) -> list[Path]:
    """生成 JSON 关联图像候选，位置优先于同一位置内的图像格式优先级。"""

    source_path = Path(json_path).expanduser().resolve()
    suffix_priority = METEOR_IMAGE_SUFFIX_PRIORITY if include_raw else STANDARD_IMAGE_SUFFIX_PRIORITY
    file_stem = _metadata_file_stem(metadata)
    candidates: list[Path] = []

    def append_location(directory: Path, exact_path: Path | None = None) -> None:
        for candidate in _same_stem_candidates(directory, file_stem, suffix_priority):
            _append_unique(candidates, candidate)
        if exact_path is not None and (include_raw or exact_path.suffix.casefold() not in RAW_IMAGE_SUFFIXES):
            _append_unique(candidates, exact_path)

    file_name = metadata.get("file_name")
    same_directory_exact = (
        source_path.parent / Path(file_name.strip()).name
        if isinstance(file_name, str) and file_name.strip()
        else None
    )
    append_location(source_path.parent, same_directory_exact)

    relative_path = metadata.get("relative_path")
    if isinstance(relative_path, str) and relative_path.strip():
        exact_path = source_path.parent / Path(relative_path.strip()).expanduser()
        append_location(exact_path.parent.resolve(), exact_path)

    raw_path = metadata.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        exact_path = Path(raw_path.strip()).expanduser()
        if not exact_path.is_absolute():
            exact_path = source_path.parent / exact_path
        append_location(exact_path.parent.resolve(), exact_path)
    return candidates


def expected_image_size(metadata: Mapping[str, object]) -> tuple[int, int] | None:
    """从图像元数据中读取用于排除误匹配文件的原始尺寸。"""

    try:
        width = int(metadata.get("original_width_px", 0))
        height = int(metadata.get("original_height_px", 0))
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def image_size_matches(path: Path, expected_size: tuple[int, int] | None) -> bool:
    """检查普通图像头部尺寸；没有期望尺寸时直接接受。"""

    if expected_size is None:
        return True
    reader = QImageReader(str(path))
    size = reader.size()
    if not size.isValid():
        return False
    return (size.width(), size.height()) == expected_size


def first_matching_image_path(
    candidates: Iterable[Path],
    expected_size: tuple[int, int] | None = None,
) -> Path | None:
    """返回第一个存在且尺寸符合 JSON 记录的普通图像。"""

    for candidate in candidates:
        if candidate.is_file() and image_size_matches(candidate, expected_size):
            return candidate
    return None


__all__ = [
    "METEOR_IMAGE_SUFFIX_PRIORITY",
    "RAW_IMAGE_SUFFIXES",
    "STANDARD_IMAGE_SUFFIX_PRIORITY",
    "associated_image_candidates",
    "expected_image_size",
    "first_matching_image_path",
    "image_file_stem",
    "image_size_matches",
]
