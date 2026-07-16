from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .preference_manager import default_preference_path, ensure_preference_file


@dataclass(frozen=True)
class StarMapUiConfig:
    controls_font_size_pt: int = 10
    status_bar_font_size_pt: int = 10
    direction_label_font_size_pt: int = 16
    star_name_font_size_pt: int = 11
    constellation_name_font_size_pt: int = 12
    reference_label_font_size_pt: int = 15
    show_constellation_names: bool = True
    show_constellation_lines: bool = True
    base_star_marker_radius_px: float = 0.8
    star_marker_size_multiplier: float = 1.0
    star_color_mag_limit: float = 6.0
    aligned_reference_scale_multiplier: float = 1.6
    constellation_line_width_px: float = 1.2
    constellation_line_color_hex: str = "#E6E6E6"
    constellation_line_opacity: float = 0.9
    star_pick_circle_default_diameter_px: int = 50
    star_pick_circle_min_diameter_px: int = 20
    star_pick_circle_max_diameter_px: int = 200
    star_pick_psf_max_radius_px: int = 40
    star_pair_psf_outer_diameter_multiplier: float = 1.5
    auto_pair_search_rms_multiplier: float = 3.0
    auto_pair_search_base_radius_px: int = 6
    auto_pair_search_max_radius_px: int = 120
    double_click_focus_auto_pair_enabled: bool = False
    default_latitude_deg: float = 40.0
    default_longitude_deg: float = 116.0
    default_elevation_m: float = 50.0
    auto_match_default_new_count: int = 200
    auto_match_default_constraint_mode: str = "soft"
    auto_match_default_soft_weight: float = 0.3
    auto_match_default_search_radius_px: int = 30
    sequence_psf_search_radius_px: int = 30
    wheel_zoom_enabled: bool = True
    touchpad_pinch_zoom_enabled: bool = True
    mosaic_texture_scale_percent: float = 25.0
    mosaic_texture_max_long_side_px: int = 1920
    mosaic_grid_precision_default: int = 36
    mosaic_render_fps_limit: int = 60
    mosaic_font_size_multiplier: float = 0.5
    mosaic_star_marker_size_multiplier: float = 0.5
    mosaic_export_block_rows: int = 1024
    mosaic_map_tile_size_px: int = 4
    mosaic_export_tiff_lzw_compression: bool = True
    show_meteor_showers: bool = True
    meteor_radiant_only: bool = False
    meteor_radiant_label_font_size_pt: int = 12
    meteor_count_multiplier: float = 0.3
    meteor_max_length_deg: float = 35.0
    meteor_min_length_deg: float = 2.0
    meteor_opacity: float = 0.9
    meteor_thickness_ratio: float = 0.025
    meteor_random_seed: int = 20260715
    selected_meteor_shower_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdjacentStarAlignmentConfig:
    """参考图像星点对齐模式的检测、初配准与精配准参数。"""

    background_bw_px: int = 128
    background_bh_px: int = 128
    background_fw_px: int = 3
    background_fh_px: int = 3
    detection_sigma: float = 5.0
    detection_min_area_px: int = 3
    deblend_nthresh: int = 32
    deblend_cont: float = 0.005
    detection_edge_margin_px: int = 12
    min_major_axis_px: float = 0.45
    max_major_axis_px: float = 9.0
    min_minor_axis_px: float = 0.35
    max_axis_ratio: float = 3.2
    max_detected_stars: int = 700
    max_alignment_stars: int = 54
    min_triangle_side_deg: float = 0.35
    triangle_match_tolerance_deg: float = 0.14
    rotation_inlier_tolerance_deg: float = 0.18
    max_triangle_hypotheses: int = 7000
    min_initial_rotation_inliers: int = 4
    initial_match_distance_px: float = 28.0
    homography_match_distance_px: float = 12.0
    final_match_distance_px: float = 4.5
    homography_ransac_max_iterations: int = 5000
    homography_ransac_confidence: float = 0.999
    min_match_count: int = 6
    focal_scale_min_ratio: float = 0.25


@dataclass(frozen=True)
class AdjacentLandscapeAlignmentConfig:
    """参考图像地景对齐模式的特征提取与 RANSAC 参数。"""

    normalization_low_percentile: float = 1.0
    normalization_high_percentile: float = 99.8
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    sift_max_features: int = 30000
    sift_contrast_threshold: float = 0.004
    sift_edge_threshold: float = 20.0
    sift_sigma: float = 1.6
    flann_trees: int = 5
    flann_checks: int = 500
    ratio_test_threshold: float = 0.75
    ransac_reprojection_threshold_px: float = 6.0
    ransac_max_iterations: int = 20000
    ransac_confidence: float = 0.999
    min_inlier_matches: int = 6


@dataclass(frozen=True)
class AdjacentAlignmentConfig:
    """参考图像粗略取景功能的完整超参数集合。"""

    max_correspondences: int = 160
    stars: AdjacentStarAlignmentConfig = field(default_factory=AdjacentStarAlignmentConfig)
    landscape: AdjacentLandscapeAlignmentConfig = field(default_factory=AdjacentLandscapeAlignmentConfig)


def default_config_path() -> Path:
    return default_preference_path()


def _read_int(config: dict[str, object], key: str, default_value: int, minimum: int, maximum: int) -> int:
    value = config.get(key, default_value)
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return default_value
    return min(max(int_value, minimum), maximum)


def _read_float(config: dict[str, object], key: str, default_value: float, minimum: float, maximum: float) -> float:
    value = config.get(key, default_value)
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return default_value
    return min(max(float_value, minimum), maximum)


def _read_choice(config: dict[str, object], key: str, default_value: str, choices: tuple[str, ...]) -> str:
    value = config.get(key, default_value)
    value_text = str(value).strip()
    if value_text in choices:
        return value_text
    return default_value


def _read_bool(config: dict[str, object], key: str, default_value: bool) -> bool:
    value = config.get(key, default_value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_text = value.strip().lower()
        if value_text in {"1", "true", "yes", "on"}:
            return True
        if value_text in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default_value


def _read_color_hex(config: dict[str, object], key: str, default_value: str) -> str:
    """读取 #RRGGBB 颜色；无效配置回退到默认值。"""

    value = str(config.get(key, default_value)).strip().upper()
    if re.fullmatch(r"#[0-9A-F]{6}", value):
        return value
    return default_value


def _read_string_tuple(
    config: dict[str, object],
    key: str,
    default_value: tuple[str, ...],
) -> tuple[str, ...]:
    """读取去重后的字符串列表，非法类型回退到默认值。"""

    value = config.get(key, default_value)
    if not isinstance(value, (list, tuple)):
        return default_value
    result: list[str] = []
    for item in value:
        item_text = str(item).strip()
        if item_text and item_text not in result:
            result.append(item_text)
    return tuple(result)


def load_star_map_ui_config(path: Path | None = None) -> StarMapUiConfig:
    config_path = path or default_config_path()
    raw_config = ensure_preference_file(config_path)

    circle_min = _read_int(raw_config, "star_pick_circle_min_diameter_px", 20, 4, 1000)
    circle_max = _read_int(raw_config, "star_pick_circle_max_diameter_px", 200, circle_min, 2000)
    circle_default = _read_int(raw_config, "star_pick_circle_default_diameter_px", 50, circle_min, circle_max)

    return StarMapUiConfig(
        controls_font_size_pt=_read_int(raw_config, "controls_font_size_pt", 10, 6, 32),
        status_bar_font_size_pt=_read_int(raw_config, "status_bar_font_size_pt", 10, 6, 32),
        direction_label_font_size_pt=_read_int(raw_config, "direction_label_font_size_pt", 16, 6, 48),
        star_name_font_size_pt=_read_int(raw_config, "star_name_font_size_pt", 11, 6, 48),
        constellation_name_font_size_pt=_read_int(
            raw_config,
            "constellation_name_font_size_pt",
            12,
            6,
            48,
        ),
        reference_label_font_size_pt=_read_int(raw_config, "reference_label_font_size_pt", 15, 6, 64),
        show_constellation_names=_read_bool(raw_config, "show_constellation_names", True),
        show_constellation_lines=_read_bool(raw_config, "show_constellation_lines", True),
        base_star_marker_radius_px=_read_float(raw_config, "base_star_marker_radius_px", 0.8, 0.1, 10.0),
        star_marker_size_multiplier=_read_float(
            raw_config,
            "star_marker_size_multiplier",
            1.0,
            0.2,
            5.0,
        ),
        star_color_mag_limit=_read_float(raw_config, "star_color_mag_limit", 6.0, -10.0, 30.0),
        aligned_reference_scale_multiplier=_read_float(
            raw_config,
            "aligned_reference_scale_multiplier",
            1.6,
            0.2,
            6.0,
        ),
        constellation_line_width_px=_read_float(raw_config, "constellation_line_width_px", 1.2, 0.1, 20.0),
        constellation_line_color_hex=_read_color_hex(
            raw_config,
            "constellation_line_color_hex",
            "#E6E6E6",
        ),
        constellation_line_opacity=_read_float(raw_config, "constellation_line_opacity", 0.9, 0.0, 1.0),
        star_pick_circle_default_diameter_px=circle_default,
        star_pick_circle_min_diameter_px=circle_min,
        star_pick_circle_max_diameter_px=circle_max,
        star_pick_psf_max_radius_px=_read_int(raw_config, "star_pick_psf_max_radius_px", 40, 8, 200),
        star_pair_psf_outer_diameter_multiplier=_read_float(
            raw_config,
            "star_pair_psf_outer_diameter_multiplier",
            1.5,
            1.05,
            10.0,
        ),
        auto_pair_search_rms_multiplier=_read_float(
            raw_config,
            "auto_pair_search_rms_multiplier",
            3.0,
            0.0,
            20.0,
        ),
        auto_pair_search_base_radius_px=_read_int(
            raw_config,
            "auto_pair_search_base_radius_px",
            6,
            0,
            500,
        ),
        auto_pair_search_max_radius_px=_read_int(
            raw_config,
            "auto_pair_search_max_radius_px",
            120,
            4,
            2000,
        ),
        double_click_focus_auto_pair_enabled=_read_bool(
            raw_config,
            "double_click_focus_auto_pair_enabled",
            False,
        ),
        default_latitude_deg=_read_float(raw_config, "default_latitude_deg", 40.0, -90.0, 90.0),
        default_longitude_deg=_read_float(raw_config, "default_longitude_deg", 116.0, -180.0, 180.0),
        default_elevation_m=_read_float(raw_config, "default_elevation_m", 50.0, -500.0, 9000.0),
        auto_match_default_new_count=_read_int(raw_config, "auto_match_default_new_count", 200, 1, 10000),
        auto_match_default_constraint_mode=_read_choice(
            raw_config,
            "auto_match_default_constraint_mode",
            "soft",
            ("anchor", "soft"),
        ),
        auto_match_default_soft_weight=_read_float(raw_config, "auto_match_default_soft_weight", 0.3, 0.01, 1.0),
        auto_match_default_search_radius_px=_read_int(
            raw_config,
            "auto_match_default_search_radius_px",
            30,
            4,
            300,
        ),
        sequence_psf_search_radius_px=_read_int(raw_config, "sequence_psf_search_radius_px", 30, 4, 200),
        wheel_zoom_enabled=_read_bool(raw_config, "wheel_zoom_enabled", True),
        touchpad_pinch_zoom_enabled=_read_bool(raw_config, "touchpad_pinch_zoom_enabled", True),
        mosaic_texture_scale_percent=_read_float(raw_config, "mosaic_texture_scale_percent", 25.0, 1.0, 100.0),
        mosaic_texture_max_long_side_px=_read_int(raw_config, "mosaic_texture_max_long_side_px", 1920, 64, 20000),
        mosaic_grid_precision_default=_read_int(raw_config, "mosaic_grid_precision_default", 36, 12, 180),
        mosaic_render_fps_limit=_read_int(raw_config, "mosaic_render_fps_limit", 60, 1, 240),
        mosaic_font_size_multiplier=_read_float(
            raw_config,
            "mosaic_font_size_multiplier",
            0.5,
            0.1,
            2.0,
        ),
        mosaic_star_marker_size_multiplier=_read_float(
            raw_config,
            "mosaic_star_marker_size_multiplier",
            0.5,
            0.1,
            2.0,
        ),
        mosaic_export_block_rows=_read_int(raw_config, "mosaic_export_block_rows", 1024, 8, 4096),
        mosaic_map_tile_size_px=_read_int(
            raw_config,
            "mosaic_map_tile_size_px",
            4,
            1,
            512,
        ),
        mosaic_export_tiff_lzw_compression=_read_bool(
            raw_config,
            "mosaic_export_tiff_lzw_compression",
            True,
        ),
        show_meteor_showers=_read_bool(raw_config, "show_meteor_showers", True),
        meteor_radiant_only=_read_bool(raw_config, "meteor_radiant_only", False),
        meteor_radiant_label_font_size_pt=_read_int(
            raw_config,
            "meteor_radiant_label_font_size_pt",
            12,
            6,
            48,
        ),
        meteor_count_multiplier=_read_float(raw_config, "meteor_count_multiplier", 0.3, 0.0, 10.0),
        meteor_max_length_deg=_read_float(raw_config, "meteor_max_length_deg", 35.0, 0.1, 150.0),
        meteor_min_length_deg=_read_float(raw_config, "meteor_min_length_deg", 2.0, 0.1, 150.0),
        meteor_opacity=_read_float(raw_config, "meteor_opacity", 0.9, 0.0, 1.0),
        meteor_thickness_ratio=_read_float(raw_config, "meteor_thickness_ratio", 0.025, 0.0, 0.2),
        meteor_random_seed=_read_int(raw_config, "meteor_random_seed", 20260715, 0, 2147483647),
        selected_meteor_shower_ids=_read_string_tuple(
            raw_config,
            "selected_meteor_shower_ids",
            (),
        ),
    )


def load_adjacent_alignment_config(path: Path | None = None) -> AdjacentAlignmentConfig:
    """从 preference.json 读取参考图像两种配准模式的超参数，并限制到安全范围。"""

    config_path = path or default_config_path()
    raw_config = ensure_preference_file(config_path)
    star_defaults = AdjacentStarAlignmentConfig()
    landscape_defaults = AdjacentLandscapeAlignmentConfig()

    min_major_axis = _read_float(
        raw_config,
        "adjacent_star_min_major_axis_px",
        star_defaults.min_major_axis_px,
        0.01,
        100.0,
    )
    max_major_axis = _read_float(
        raw_config,
        "adjacent_star_max_major_axis_px",
        star_defaults.max_major_axis_px,
        min_major_axis,
        1000.0,
    )
    percentile_low = _read_float(
        raw_config,
        "adjacent_landscape_normalization_low_percentile",
        landscape_defaults.normalization_low_percentile,
        0.0,
        99.0,
    )
    percentile_high = _read_float(
        raw_config,
        "adjacent_landscape_normalization_high_percentile",
        landscape_defaults.normalization_high_percentile,
        percentile_low + 0.01,
        100.0,
    )

    max_alignment_stars = _read_int(
        raw_config,
        "adjacent_star_max_alignment_stars",
        star_defaults.max_alignment_stars,
        3,
        100,
    )
    min_initial_rotation_inliers = min(
        _read_int(
            raw_config,
            "adjacent_star_min_initial_rotation_inliers",
            star_defaults.min_initial_rotation_inliers,
            3,
            100,
        ),
        max_alignment_stars,
    )

    stars = AdjacentStarAlignmentConfig(
        background_bw_px=_read_int(raw_config, "adjacent_star_background_bw_px", star_defaults.background_bw_px, 16, 2048),
        background_bh_px=_read_int(raw_config, "adjacent_star_background_bh_px", star_defaults.background_bh_px, 16, 2048),
        background_fw_px=_read_int(raw_config, "adjacent_star_background_fw_px", star_defaults.background_fw_px, 1, 63),
        background_fh_px=_read_int(raw_config, "adjacent_star_background_fh_px", star_defaults.background_fh_px, 1, 63),
        detection_sigma=_read_float(raw_config, "adjacent_star_detection_sigma", star_defaults.detection_sigma, 0.1, 100.0),
        detection_min_area_px=_read_int(raw_config, "adjacent_star_detection_min_area_px", star_defaults.detection_min_area_px, 1, 10000),
        deblend_nthresh=_read_int(raw_config, "adjacent_star_deblend_nthresh", star_defaults.deblend_nthresh, 1, 256),
        deblend_cont=_read_float(raw_config, "adjacent_star_deblend_cont", star_defaults.deblend_cont, 0.0, 1.0),
        detection_edge_margin_px=_read_int(raw_config, "adjacent_star_detection_edge_margin_px", star_defaults.detection_edge_margin_px, 0, 10000),
        min_major_axis_px=min_major_axis,
        max_major_axis_px=max_major_axis,
        min_minor_axis_px=_read_float(raw_config, "adjacent_star_min_minor_axis_px", star_defaults.min_minor_axis_px, 0.01, 100.0),
        max_axis_ratio=_read_float(raw_config, "adjacent_star_max_axis_ratio", star_defaults.max_axis_ratio, 1.0, 100.0),
        max_detected_stars=_read_int(raw_config, "adjacent_star_max_detected_stars", star_defaults.max_detected_stars, 3, 10000),
        max_alignment_stars=max_alignment_stars,
        min_triangle_side_deg=_read_float(raw_config, "adjacent_star_min_triangle_side_deg", star_defaults.min_triangle_side_deg, 0.001, 45.0),
        triangle_match_tolerance_deg=_read_float(raw_config, "adjacent_star_triangle_match_tolerance_deg", star_defaults.triangle_match_tolerance_deg, 0.001, 10.0),
        rotation_inlier_tolerance_deg=_read_float(raw_config, "adjacent_star_rotation_inlier_tolerance_deg", star_defaults.rotation_inlier_tolerance_deg, 0.001, 10.0),
        max_triangle_hypotheses=_read_int(raw_config, "adjacent_star_max_triangle_hypotheses", star_defaults.max_triangle_hypotheses, 1, 500000),
        min_initial_rotation_inliers=min_initial_rotation_inliers,
        initial_match_distance_px=_read_float(raw_config, "adjacent_star_initial_match_distance_px", star_defaults.initial_match_distance_px, 0.1, 10000.0),
        homography_match_distance_px=_read_float(raw_config, "adjacent_star_homography_match_distance_px", star_defaults.homography_match_distance_px, 0.1, 10000.0),
        final_match_distance_px=_read_float(raw_config, "adjacent_star_final_match_distance_px", star_defaults.final_match_distance_px, 0.1, 10000.0),
        homography_ransac_max_iterations=_read_int(raw_config, "adjacent_star_homography_ransac_max_iterations", star_defaults.homography_ransac_max_iterations, 1, 1000000),
        homography_ransac_confidence=_read_float(raw_config, "adjacent_star_homography_ransac_confidence", star_defaults.homography_ransac_confidence, 0.5, 1.0),
        min_match_count=_read_int(raw_config, "adjacent_star_min_match_count", star_defaults.min_match_count, 4, 10000),
        focal_scale_min_ratio=_read_float(raw_config, "adjacent_star_focal_scale_min_ratio", star_defaults.focal_scale_min_ratio, 0.01, 10.0),
    )
    landscape = AdjacentLandscapeAlignmentConfig(
        normalization_low_percentile=percentile_low,
        normalization_high_percentile=percentile_high,
        clahe_clip_limit=_read_float(raw_config, "adjacent_landscape_clahe_clip_limit", landscape_defaults.clahe_clip_limit, 0.1, 100.0),
        clahe_grid_size=_read_int(raw_config, "adjacent_landscape_clahe_grid_size", landscape_defaults.clahe_grid_size, 1, 512),
        sift_max_features=_read_int(raw_config, "adjacent_landscape_sift_max_features", landscape_defaults.sift_max_features, 4, 1000000),
        sift_contrast_threshold=_read_float(raw_config, "adjacent_landscape_sift_contrast_threshold", landscape_defaults.sift_contrast_threshold, 0.000001, 1.0),
        sift_edge_threshold=_read_float(raw_config, "adjacent_landscape_sift_edge_threshold", landscape_defaults.sift_edge_threshold, 0.1, 10000.0),
        sift_sigma=_read_float(raw_config, "adjacent_landscape_sift_sigma", landscape_defaults.sift_sigma, 0.1, 100.0),
        flann_trees=_read_int(raw_config, "adjacent_landscape_flann_trees", landscape_defaults.flann_trees, 1, 100),
        flann_checks=_read_int(raw_config, "adjacent_landscape_flann_checks", landscape_defaults.flann_checks, 1, 100000),
        ratio_test_threshold=_read_float(raw_config, "adjacent_landscape_ratio_test_threshold", landscape_defaults.ratio_test_threshold, 0.01, 0.999999),
        ransac_reprojection_threshold_px=_read_float(raw_config, "adjacent_landscape_ransac_reprojection_threshold_px", landscape_defaults.ransac_reprojection_threshold_px, 0.1, 10000.0),
        ransac_max_iterations=_read_int(raw_config, "adjacent_landscape_ransac_max_iterations", landscape_defaults.ransac_max_iterations, 1, 1000000),
        ransac_confidence=_read_float(raw_config, "adjacent_landscape_ransac_confidence", landscape_defaults.ransac_confidence, 0.5, 1.0),
        min_inlier_matches=_read_int(raw_config, "adjacent_landscape_min_inlier_matches", landscape_defaults.min_inlier_matches, 4, 100000),
    )
    return AdjacentAlignmentConfig(
        max_correspondences=_read_int(raw_config, "adjacent_alignment_max_correspondences", 160, 4, 10000),
        stars=stars,
        landscape=landscape,
    )
