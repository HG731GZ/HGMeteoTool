from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .runtime_paths import frozen_app_sibling_dir, is_frozen_app, source_project_root


LAST_IMPORT_DIRECTORY_KEY = "last_import_directory"

# 这里保存新安装首次启动时使用的完整配置；已有文件只补缺项，不覆盖用户设置。
DEFAULT_PREFERENCE_VALUES: dict[str, object] = {
    "controls_font_size_pt": 10,
    "status_bar_font_size_pt": 10,
    "direction_label_font_size_pt": 16,
    "star_name_font_size_pt": 14,
    "reference_label_font_size_pt": 15,
    "star_color_mag_limit": 4.0,
    "aligned_reference_scale_multiplier": 1.6,
    "star_pick_circle_default_diameter_px": 50,
    "star_pick_circle_min_diameter_px": 20,
    "star_pick_circle_max_diameter_px": 300,
    "star_pick_psf_radius_scale": 0.4,
    "star_pick_psf_max_radius_px": 40,
    "default_latitude_deg": 40.0,
    "default_longitude_deg": 116.0,
    "default_elevation_m": 200.0,
    "auto_match_default_new_count": 200,
    "auto_match_default_constraint_mode": "soft",
    "auto_match_default_soft_weight": 0.3,
    "wheel_zoom_enabled": True,
    "touchpad_pinch_zoom_enabled": True,
    "mosaic_texture_scale_percent": 25.0,
    "mosaic_texture_max_long_side_px": 1920,
    "mosaic_grid_precision_default": 24,
    "mosaic_render_fps_limit": 60,
    "mosaic_export_block_rows": 128,
    "mosaic_map_tile_size_px": 16,
    "mosaic_export_tiff_lzw_compression": True,
    LAST_IMPORT_DIRECTORY_KEY: "",
}

PREFERENCE_COMMENTS: dict[str, str] = {
    "controls_font_size_pt": "全局界面字体大小，单位为 pt。",
    "status_bar_font_size_pt": "状态栏字体大小，单位为 pt。",
    "direction_label_font_size_pt": "星空模拟图中方位、地平线等方向标签字体大小。",
    "star_name_font_size_pt": "星空模拟图中恒星名称字体大小。",
    "reference_label_font_size_pt": "参考星标注编号和名称的字体大小。",
    "star_color_mag_limit": "只有视觉星等不大于该值的恒星才渲染色度；更暗的恒星使用灰阶。",
    "aligned_reference_scale_multiplier": "叠加参考星图时的显示缩放倍率。",
    "star_pick_circle_default_diameter_px": "点选星点时默认圆圈直径，单位为像素。",
    "star_pick_circle_min_diameter_px": "点选星点圆圈允许的最小直径，单位为像素。",
    "star_pick_circle_max_diameter_px": "点选星点圆圈允许的最大直径，单位为像素。",
    "star_pick_psf_radius_scale": "PSF 精修半径相对于点选圆圈半径的比例。",
    "star_pick_psf_max_radius_px": "PSF 精修半径上限，单位为像素。",
    "default_latitude_deg": "默认纬度，北纬为正。",
    "default_longitude_deg": "默认经度，东经为正。",
    "default_elevation_m": "默认海拔，单位为米。",
    "auto_match_default_new_count": "每次自动扩展匹配新增的候选星数量。",
    "auto_match_default_constraint_mode": "自动扩展匹配的默认约束模式，可选 anchor 或 soft。",
    "auto_match_default_soft_weight": "soft 约束模式下新增匹配点的默认权重。",
    "wheel_zoom_enabled": "是否启用鼠标滚轮缩放预览视图。",
    "touchpad_pinch_zoom_enabled": "是否启用触控板双指捏合缩放预览视图。",
    "mosaic_texture_scale_percent": "自由拼图预览贴图使用原图长边的百分比，数值越大越清晰但越占内存。",
    "mosaic_texture_max_long_side_px": "自由拼图预览贴图长边上限，单位为像素。",
    "mosaic_grid_precision_default": "自由拼图贴图网格默认长边点数，数值越大越精细但越慢。",
    "mosaic_render_fps_limit": "自由拼图预览最高刷新率。",
    "mosaic_export_block_rows": "导出重投影 TIFF 时每个处理块的行数，数值越大(可能)越快但越占内存。",
    "mosaic_map_tile_size_px": "导出重投影图时源图映射网格大小，单位为像素，数值越小越精细，计算速度越慢。",
    "mosaic_export_tiff_lzw_compression": "导出 16-bit TIFF 时是否启用 LZW 压缩。",
    LAST_IMPORT_DIRECTORY_KEY: "最近一次导入文件所在目录，所有导入对话框共用；由软件自动更新。",
}

PREFERENCE_SECTION_START_KEYS = {
    "star_pick_circle_default_diameter_px",
    "default_latitude_deg",
    "auto_match_default_new_count",
    "wheel_zoom_enabled",
    "mosaic_texture_scale_percent",
    LAST_IMPORT_DIRECTORY_KEY,
}


def default_preference_path() -> Path:
    """返回源码或打包程序使用的外置配置路径。"""

    if is_frozen_app():
        return frozen_app_sibling_dir() / "preference.json"
    return source_project_root() / "preference.json"


def strip_json_comments(text: str) -> str:
    """移除 JSONC 风格注释，保留字符串内部的斜杠字符。"""

    result: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index = min(index + 2, len(text))
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _read_preference_values(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _write_preference_values(path: Path, values: dict[str, object]) -> None:
    """用临时文件替换配置，避免程序中断时留下半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        _preference_jsonc_text(values),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _preference_jsonc_text(values: dict[str, object]) -> str:
    """生成带中文字段说明的 JSONC 配置文本。"""

    entries = list(values.items())
    lines = ["{"]
    for index, (key, value) in enumerate(entries):
        if index > 0 and key in PREFERENCE_SECTION_START_KEYS:
            lines.append("")
        comment = PREFERENCE_COMMENTS.get(key)
        if comment:
            lines.append(f"  // {comment}")
        value_text = json.dumps(value, ensure_ascii=False, separators=(",", ": "))
        comma = "," if index + 1 < len(entries) else ""
        lines.append(f"  {json.dumps(key, ensure_ascii=False)}: {value_text}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _preference_has_required_comments(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return all(f"// {comment}" in text for comment in PREFERENCE_COMMENTS.values())


def ensure_preference_file(path: str | Path | None = None) -> dict[str, object]:
    """创建缺失配置并补齐字段，同时保留已有值和未知扩展字段。"""

    preference_path = Path(path).expanduser() if path is not None else default_preference_path()
    existing = _read_preference_values(preference_path)
    values = dict(DEFAULT_PREFERENCE_VALUES)
    if existing is not None:
        values.update(existing)
    if not isinstance(values.get(LAST_IMPORT_DIRECTORY_KEY), str):
        values[LAST_IMPORT_DIRECTORY_KEY] = ""

    needs_write = (
        existing is None
        or any(key not in existing for key in DEFAULT_PREFERENCE_VALUES)
        or not _preference_has_required_comments(preference_path)
    )
    if existing is not None and not isinstance(existing.get(LAST_IMPORT_DIRECTORY_KEY), str):
        needs_write = True
    if needs_write:
        try:
            _write_preference_values(preference_path, values)
        except OSError:
            # 配置目录只读时仍允许软件使用内置默认值启动。
            pass
    return values


def recent_import_directory(
    fallback: str | Path,
    *,
    path: str | Path | None = None,
) -> Path:
    """优先返回最近导入目录，无有效记录时使用调用方提供的默认目录。"""

    values = ensure_preference_file(path)
    saved_text = str(values.get(LAST_IMPORT_DIRECTORY_KEY) or "").strip()
    if saved_text:
        saved_path = Path(saved_text).expanduser()
        if saved_path.is_file():
            saved_path = saved_path.parent
        if saved_path.is_dir():
            return saved_path.resolve()

    fallback_path = Path(fallback).expanduser()
    if fallback_path.is_file():
        fallback_path = fallback_path.parent
    return fallback_path.resolve() if fallback_path.exists() else Path.cwd().resolve()


def _first_selected_path(selected: str | Path | Sequence[str | Path]) -> Path | None:
    if isinstance(selected, (str, Path)):
        raw_path: str | Path | None = selected
    else:
        raw_path = next(iter(selected), None)
    if raw_path is None or not str(raw_path).strip():
        return None
    return Path(raw_path).expanduser()


def remember_import_path(
    selected: str | Path | Sequence[str | Path],
    *,
    path: str | Path | None = None,
) -> bool:
    """把本次选择所在目录覆盖写入唯一的最近导入记录。"""

    selected_path = _first_selected_path(selected)
    if selected_path is None:
        return False
    import_directory = selected_path if selected_path.is_dir() else selected_path.parent
    if not import_directory.exists():
        return False

    preference_path = Path(path).expanduser() if path is not None else default_preference_path()
    values = ensure_preference_file(preference_path)
    values[LAST_IMPORT_DIRECTORY_KEY] = str(import_directory.resolve())
    try:
        _write_preference_values(preference_path, values)
    except OSError:
        return False
    return True


__all__ = [
    "DEFAULT_PREFERENCE_VALUES",
    "LAST_IMPORT_DIRECTORY_KEY",
    "PREFERENCE_COMMENTS",
    "default_preference_path",
    "ensure_preference_file",
    "recent_import_directory",
    "remember_import_path",
    "strip_json_comments",
]
