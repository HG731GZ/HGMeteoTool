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
    "constellation_name_font_size_pt": 20,
    "reference_label_font_size_pt": 15,
    "show_constellation_names": True,
    "show_constellation_lines": True,
    "base_star_marker_radius_px": 0.8,
    "star_marker_size_multiplier": 1.0,
    "star_color_mag_limit": 4.0,
    "aligned_reference_scale_multiplier": 1.6,
    "constellation_line_width_px": 1.2,
    "constellation_line_color_hex": "#E6E6E6",
    "constellation_line_opacity": 0.9,
    "star_pick_circle_default_diameter_px": 50,
    "star_pick_circle_min_diameter_px": 20,
    "star_pick_circle_max_diameter_px": 300,
    "star_pick_psf_max_radius_px": 40,
    "star_pair_psf_outer_diameter_multiplier": 1.5,
    "auto_pair_search_rms_multiplier": 3.0,
    "auto_pair_search_base_radius_px": 6,
    "auto_pair_search_max_radius_px": 120,
    "double_click_focus_auto_pair_enabled": False,
    "default_latitude_deg": 40.0,
    "default_longitude_deg": 116.0,
    "default_elevation_m": 200.0,
    "auto_match_default_new_count": 200,
    "auto_match_default_constraint_mode": "soft",
    "auto_match_default_soft_weight": 0.3,
    "auto_match_default_search_radius_px": 30,
    "sequence_psf_search_radius_px": 30,
    "adjacent_alignment_max_correspondences": 160,
    "adjacent_star_background_bw_px": 128,
    "adjacent_star_background_bh_px": 128,
    "adjacent_star_background_fw_px": 3,
    "adjacent_star_background_fh_px": 3,
    "adjacent_star_detection_sigma": 5.0,
    "adjacent_star_detection_min_area_px": 3,
    "adjacent_star_deblend_nthresh": 32,
    "adjacent_star_deblend_cont": 0.005,
    "adjacent_star_detection_edge_margin_px": 12,
    "adjacent_star_min_major_axis_px": 0.45,
    "adjacent_star_max_major_axis_px": 9.0,
    "adjacent_star_min_minor_axis_px": 0.35,
    "adjacent_star_max_axis_ratio": 3.2,
    "adjacent_star_max_detected_stars": 700,
    "adjacent_star_max_alignment_stars": 54,
    "adjacent_star_min_triangle_side_deg": 0.35,
    "adjacent_star_triangle_match_tolerance_deg": 0.14,
    "adjacent_star_rotation_inlier_tolerance_deg": 0.18,
    "adjacent_star_max_triangle_hypotheses": 7000,
    "adjacent_star_min_initial_rotation_inliers": 4,
    "adjacent_star_initial_match_distance_px": 28.0,
    "adjacent_star_homography_match_distance_px": 12.0,
    "adjacent_star_final_match_distance_px": 4.5,
    "adjacent_star_homography_ransac_max_iterations": 5000,
    "adjacent_star_homography_ransac_confidence": 0.999,
    "adjacent_star_min_match_count": 6,
    "adjacent_star_focal_scale_min_ratio": 0.25,
    "adjacent_landscape_normalization_low_percentile": 1.0,
    "adjacent_landscape_normalization_high_percentile": 99.8,
    "adjacent_landscape_clahe_clip_limit": 2.0,
    "adjacent_landscape_clahe_grid_size": 8,
    "adjacent_landscape_sift_max_features": 50000,
    "adjacent_landscape_sift_contrast_threshold": 0.004,
    "adjacent_landscape_sift_edge_threshold": 20.0,
    "adjacent_landscape_sift_sigma": 1.6,
    "adjacent_landscape_flann_trees": 5,
    "adjacent_landscape_flann_checks": 500,
    "adjacent_landscape_ratio_test_threshold": 0.70,
    "adjacent_landscape_ransac_reprojection_threshold_px": 4.0,
    "adjacent_landscape_ransac_max_iterations": 20000,
    "adjacent_landscape_ransac_confidence": 0.999,
    "adjacent_landscape_min_inlier_matches": 6,
    "wheel_zoom_enabled": True,
    "touchpad_pinch_zoom_enabled": True,
    "meteor_detection_engine_path": "",
    "meteor_detection_model_path": "",
    "meteor_detection_confidence_threshold": 0.25,
    "meteor_detection_nms_threshold": 0.45,
    "meteor_detection_multiscale": 2,
    "meteor_detection_partition": 2,
    "meteor_detection_provider": "auto",
    "meteor_detection_box_expansion_ratio": 0.10,
    "mosaic_texture_scale_percent": 25.0,
    "mosaic_texture_max_long_side_px": 1920,
    "mosaic_grid_precision_default": 24,
    "mosaic_render_fps_limit": 60,
    "mosaic_font_size_multiplier": 0.5,
    "mosaic_star_marker_size_multiplier": 0.5,
    "mosaic_export_block_rows": 128,
    "mosaic_map_tile_size_px": 16,
    "mosaic_export_tiff_lzw_compression": True,
    "show_meteor_showers": False,
    "meteor_radiant_only": False,
    "meteor_radiant_label_font_size_pt": 12,
    "meteor_count_multiplier": 0.3,
    "meteor_max_length_deg": 35.0,
    "meteor_min_length_deg": 2.0,
    "meteor_opacity": 0.9,
    "meteor_thickness_ratio": 0.025,
    "meteor_random_seed": 20260715,
    "selected_meteor_shower_ids": [],
    LAST_IMPORT_DIRECTORY_KEY: "",
}

PREFERENCE_COMMENTS: dict[str, str] = {
    "controls_font_size_pt": "全局界面字体大小，单位为 pt。",
    "status_bar_font_size_pt": "状态栏字体大小，单位为 pt。",
    "direction_label_font_size_pt": "星空模拟图中方位、地平线等方向标签字体大小。",
    "star_name_font_size_pt": "星空模拟图中恒星名称字体大小。",
    "constellation_name_font_size_pt": "星空模拟图中星座名称字体大小。",
    "reference_label_font_size_pt": "参考星标注编号和名称的字体大小。",
    "show_constellation_names": "是否显示星座名称。",
    "show_constellation_lines": "是否显示星座连线。",
    "base_star_marker_radius_px": "最暗可见恒星的基础渲染半径，单位为像素。",
    "star_marker_size_multiplier": "恒星星点渲染大小倍率，保留不同星等之间的相对大小。",
    "star_color_mag_limit": "只有视觉星等不大于该值的恒星才渲染色度；更暗的恒星使用灰阶。",
    "aligned_reference_scale_multiplier": "叠加参考星图时的显示缩放倍率。",
    "constellation_line_width_px": "星座连线宽度，单位为像素。",
    "constellation_line_color_hex": "星座连线颜色，格式为 #RRGGBB。",
    "constellation_line_opacity": "星座连线透明度，范围为 0 到 1。",
    "star_pick_circle_default_diameter_px": "点选星点时默认圆圈直径，单位为像素。",
    "star_pick_circle_min_diameter_px": "点选星点圆圈允许的最小直径，单位为像素。",
    "star_pick_circle_max_diameter_px": "点选星点圆圈允许的最大直径，单位为像素。",
    "star_pick_psf_max_radius_px": "自适应 PSF 拟合半径的绝对上限，单位为像素。",
    "star_pair_psf_outer_diameter_multiplier": "真实图像 PSF 黄圈外侧绿圈相对于黄圈的直径倍率。",
    "auto_pair_search_rms_multiplier": "右键自动匹配搜索半径中配准 RMS 的倍率。",
    "auto_pair_search_base_radius_px": "右键自动匹配搜索半径的基础增量，单位为像素。",
    "auto_pair_search_max_radius_px": "右键自动匹配搜索半径的绝对上限，单位为像素。",
    "double_click_focus_auto_pair_enabled": "双击聚焦未匹配参考星时是否联动执行静默自动匹配。",
    "default_latitude_deg": "默认纬度，北纬为正。",
    "default_longitude_deg": "默认经度，东经为正。",
    "default_elevation_m": "默认海拔，单位为米。",
    "auto_match_default_new_count": "每次自动扩展匹配新增的候选星数量。",
    "auto_match_default_constraint_mode": "自动扩展匹配的默认约束模式，可选 anchor 或 soft。",
    "auto_match_default_soft_weight": "soft 约束模式下新增匹配点的默认权重。",
    "auto_match_default_search_radius_px": "星点匹配页预测位置到检测星源的默认允许距离，单位为像素。",
    "sequence_psf_search_radius_px": "图像序列解析使用的独立星源匹配距离，单位为像素。",
    "adjacent_alignment_max_correspondences": "参考图像粗略取景最多保留的 PixelA↔PixelB 对应点数；较大值更稳健但姿态拟合更慢。",
    "adjacent_star_background_bw_px": "星点对齐 SEP 背景估计网格宽度，单位为像素。",
    "adjacent_star_background_bh_px": "星点对齐 SEP 背景估计网格高度，单位为像素。",
    "adjacent_star_background_fw_px": "星点对齐 SEP 背景估计的水平滤波网格大小。",
    "adjacent_star_background_fh_px": "星点对齐 SEP 背景估计的垂直滤波网格大小。",
    "adjacent_star_detection_sigma": "星点对齐 SEP 检测阈值，单位为背景全局 RMS；越小检出越多也越容易误检。",
    "adjacent_star_detection_min_area_px": "星点对齐中星点连通域的最小面积，单位为像素。",
    "adjacent_star_deblend_nthresh": "星点对齐 SEP 去混叠的阈值层数。",
    "adjacent_star_deblend_cont": "星点对齐 SEP 去混叠对比度阈值；越小越倾向拆分相邻星点。",
    "adjacent_star_detection_edge_margin_px": "星点对齐时忽略图像边缘的宽度，单位为像素。",
    "adjacent_star_min_major_axis_px": "星点对齐保留目标的最小长轴尺度，单位为像素。",
    "adjacent_star_max_major_axis_px": "星点对齐保留目标的最大长轴尺度，单位为像素；可排除云团和大面积光斑。",
    "adjacent_star_min_minor_axis_px": "星点对齐保留目标的最小短轴尺度，单位为像素。",
    "adjacent_star_max_axis_ratio": "星点对齐保留目标的最大长短轴比；越小越严格排除拖线和飞机轨迹。",
    "adjacent_star_max_detected_stars": "星点对齐按亮度保留的最大检测星点数。",
    "adjacent_star_max_alignment_stars": "星点对齐用于 Astroalign 和球面三角形初配准的最亮星点数。",
    "adjacent_star_min_triangle_side_deg": "球面三角形初配准允许的最小边长，单位为度。",
    "adjacent_star_triangle_match_tolerance_deg": "球面三角形边长描述子的匹配容差，单位为度。",
    "adjacent_star_rotation_inlier_tolerance_deg": "球面旋转假设的恒星内点角距离阈值，单位为度。",
    "adjacent_star_max_triangle_hypotheses": "球面三角形 RANSAC 最多评估的候选假设数；越大越稳健也越慢。",
    "adjacent_star_min_initial_rotation_inliers": "接受球面三角形初始旋转所需的最少内点星数。",
    "adjacent_star_initial_match_distance_px": "初始变换后寻找星点对应的最大像素距离。",
    "adjacent_star_homography_match_distance_px": "星点单应性 RANSAC 的重投影阈值，单位为像素。",
    "adjacent_star_final_match_distance_px": "星点精配准后保留最终对应点的最大像素距离。",
    "adjacent_star_homography_ransac_max_iterations": "星点单应性 RANSAC 的最大迭代次数。",
    "adjacent_star_homography_ransac_confidence": "星点单应性 RANSAC 的目标置信度。",
    "adjacent_star_min_match_count": "星点模式计算成功所需的最少可靠重合星点对数。",
    "adjacent_star_focal_scale_min_ratio": "导出 Profile 焦距尺度小于图像长边该比例时，球面初配准改用图像长边作为焦距近似。",
    "adjacent_landscape_normalization_low_percentile": "地景对齐将高位深图压缩为 8-bit 时使用的低百分位。",
    "adjacent_landscape_normalization_high_percentile": "地景对齐将高位深图压缩为 8-bit 时使用的高百分位。",
    "adjacent_landscape_clahe_clip_limit": "地景对齐 CLAHE 局部对比度增强的裁剪上限。",
    "adjacent_landscape_clahe_grid_size": "地景对齐 CLAHE 网格边长，单位为像素。",
    "adjacent_landscape_sift_max_features": "地景对齐 SIFT 最多提取的特征点数。",
    "adjacent_landscape_sift_contrast_threshold": "地景对齐 SIFT 对比度阈值；越小越容易保留弱特征。",
    "adjacent_landscape_sift_edge_threshold": "地景对齐 SIFT 边缘响应阈值。",
    "adjacent_landscape_sift_sigma": "地景对齐 SIFT 高斯尺度参数。",
    "adjacent_landscape_flann_trees": "地景对齐 FLANN KD 树数量。",
    "adjacent_landscape_flann_checks": "地景对齐 FLANN 每次近邻搜索的检查次数；越大越准确也越慢。",
    "adjacent_landscape_ratio_test_threshold": "地景对齐 Lowe ratio test 阈值；越小越严格。",
    "adjacent_landscape_ransac_reprojection_threshold_px": "地景单应性 RANSAC 的重投影阈值，单位为像素。",
    "adjacent_landscape_ransac_max_iterations": "地景单应性 RANSAC 的最大迭代次数。",
    "adjacent_landscape_ransac_confidence": "地景单应性 RANSAC 的目标置信度。",
    "adjacent_landscape_min_inlier_matches": "地景模式计算成功所需的最少 RANSAC 内点匹配数。",
    "wheel_zoom_enabled": "是否启用鼠标滚轮缩放预览视图。",
    "touchpad_pinch_zoom_enabled": "是否启用触控板双指捏合缩放预览视图。",
    "meteor_detection_engine_path": "MetDet worker 的可执行文件、源码文件或完整 onedir 目录；留空时自动查找。",
    "meteor_detection_model_path": "流星检测 ONNX 模型路径；留空时使用 worker 自带模型。",
    "meteor_detection_confidence_threshold": "流星检测置信度阈值，范围为 0 到 1。",
    "meteor_detection_nms_threshold": "流星检测非极大值抑制阈值，范围为 0 到 1。",
    "meteor_detection_multiscale": "流星检测的多尺度层数；0 最快，2 对大图召回更好。",
    "meteor_detection_partition": "流星检测每层横纵方向的分块数，最小为 2。",
    "meteor_detection_provider": "流星检测推理设备，可选 auto、cpu、dml、cuda 或 coreml。",
    "meteor_detection_box_expansion_ratio": "自动检测框宽度和高度的额外扩大比例。",
    "mosaic_texture_scale_percent": "自由拼图预览贴图使用原图长边的百分比，数值越大越清晰但越占内存。",
    "mosaic_texture_max_long_side_px": "自由拼图预览贴图长边上限，单位为像素。",
    "mosaic_grid_precision_default": "自由拼图贴图网格默认长边点数，数值越大越精细但越慢。",
    "mosaic_render_fps_limit": "自由拼图预览最高刷新率。",
    "mosaic_font_size_multiplier": "全景构图星空文字相对于普通星图的字号倍率。",
    "mosaic_star_marker_size_multiplier": "全景构图星点相对于普通星图的大小倍率。",
    "mosaic_export_block_rows": "导出重投影 TIFF 时每个处理块的行数，数值越大(可能)越快但越占内存。",
    "mosaic_map_tile_size_px": "导出重投影图时源图映射网格大小，单位为像素，数值越小越精细，计算速度越慢。",
    "mosaic_export_tiff_lzw_compression": "导出 16-bit TIFF 时是否启用 LZW 压缩。",
    "show_meteor_showers": "是否在星空模拟与全景构图中渲染所选流星雨。",
    "meteor_radiant_only": "是否只显示所选流星雨的辐射点，不绘制随机流星轨迹。",
    "meteor_radiant_label_font_size_pt": "流星雨辐射点名称的标注字号，单位为 pt。",
    "meteor_count_multiplier": "渲染流星数与各流星雨峰值 ZHR 的比例。",
    "meteor_max_length_deg": "单颗流星允许的最长天球角长度，单位为度。",
    "meteor_min_length_deg": "单颗流星允许的最短天球角长度，单位为度。",
    "meteor_opacity": "流星覆盖层整体透明度，范围为 0 到 1。",
    "meteor_thickness_ratio": "流星最粗处相对于屏幕轨迹长度的比例；0 表示单像素线。",
    "meteor_random_seed": "流星位置、长度和亮度的随机种子，相同参数与种子可复现。",
    "selected_meteor_shower_ids": "选择用于取景参考的流星雨 IMO 三字母代号列表。",
    LAST_IMPORT_DIRECTORY_KEY: "最近一次导入文件所在目录，所有导入对话框共用；由软件自动更新。",
}

PREFERENCE_SECTION_START_KEYS = {
    "constellation_line_width_px",
    "star_pick_circle_default_diameter_px",
    "default_latitude_deg",
    "auto_match_default_new_count",
    "sequence_psf_search_radius_px",
    "adjacent_alignment_max_correspondences",
    "wheel_zoom_enabled",
    "meteor_detection_engine_path",
    "mosaic_texture_scale_percent",
    "show_meteor_showers",
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
    # 该选项已改为固定覆盖，移除旧配置以免继续造成可以切换的误解。
    values.pop("meteor_detection_overwrite", None)
    if not isinstance(values.get(LAST_IMPORT_DIRECTORY_KEY), str):
        values[LAST_IMPORT_DIRECTORY_KEY] = ""

    needs_write = (
        existing is None
        or any(key not in existing for key in DEFAULT_PREFERENCE_VALUES)
        or "meteor_detection_overwrite" in existing
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


def update_preference_values(
    updates: dict[str, object],
    *,
    path: str | Path | None = None,
) -> bool:
    """将指定配置项写回 preference.json，并保留其余用户设置。"""

    preference_path = Path(path).expanduser() if path is not None else default_preference_path()
    values = ensure_preference_file(preference_path)
    values.update(updates)
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
    "update_preference_values",
]
