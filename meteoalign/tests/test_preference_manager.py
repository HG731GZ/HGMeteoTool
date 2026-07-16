from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from meteoalign.config import (
    AdjacentLandscapeAlignmentConfig,
    AdjacentStarAlignmentConfig,
    StarMapUiConfig,
    load_adjacent_alignment_config,
    load_star_map_ui_config,
)
from meteoalign.preference_manager import (
    DEFAULT_PREFERENCE_VALUES,
    LAST_IMPORT_DIRECTORY_KEY,
    PREFERENCE_COMMENTS,
    ensure_preference_file,
    recent_import_directory,
    remember_import_path,
    strip_json_comments,
)


def _read_jsonc(path: Path) -> dict[str, object]:
    return json.loads(strip_json_comments(path.read_text(encoding="utf-8")))


def test_missing_preference_file_is_created_with_all_defaults(tmp_path: Path) -> None:
    preference_path = tmp_path / "preference.json"

    values = ensure_preference_file(preference_path)
    written = _read_jsonc(preference_path)
    written_text = preference_path.read_text(encoding="utf-8")

    assert preference_path.exists()
    assert values == written
    assert set(DEFAULT_PREFERENCE_VALUES).issubset(written)
    assert written[LAST_IMPORT_DIRECTORY_KEY] == ""
    for comment in PREFERENCE_COMMENTS.values():
        assert f"// {comment}" in written_text


def test_preference_defaults_cover_every_ui_config_field() -> None:
    config_field_names = {field.name for field in fields(StarMapUiConfig)}

    assert config_field_names.issubset(DEFAULT_PREFERENCE_VALUES)
    assert set(PREFERENCE_COMMENTS) == set(DEFAULT_PREFERENCE_VALUES)


def test_constellation_style_preferences_are_loaded_and_bounded(tmp_path: Path) -> None:
    preference_path = tmp_path / "preference.json"
    preference_path.write_text(
        """
        {
          "constellation_line_width_px": 2.5,
          "constellation_line_color_hex": "#aBcDeF",
          "constellation_line_opacity": 1.5
        }
        """,
        encoding="utf-8",
    )

    config = load_star_map_ui_config(preference_path)

    assert config.constellation_line_width_px == 2.5
    assert config.constellation_line_color_hex == "#ABCDEF"
    assert config.constellation_line_opacity == 1.0


def test_constellation_and_star_marker_preferences_are_loaded_and_bounded(tmp_path: Path) -> None:
    """星座显示开关、星点基础大小与倍率应独立读取并限制到安全范围。"""

    preference_path = tmp_path / "preference.json"
    preference_path.write_text(
        """
        {
          "star_name_font_size_pt": 17,
          "constellation_name_font_size_pt": 23,
          "show_constellation_names": false,
          "show_constellation_lines": false,
          "base_star_marker_radius_px": 20.0,
          "star_marker_size_multiplier": 9.0
        }
        """,
        encoding="utf-8",
    )

    config = load_star_map_ui_config(preference_path)

    assert config.star_name_font_size_pt == 17
    assert config.constellation_name_font_size_pt == 23
    assert not config.show_constellation_names
    assert not config.show_constellation_lines
    assert config.base_star_marker_radius_px == 10.0
    assert config.star_marker_size_multiplier == 5.0


def test_mosaic_display_scales_and_search_radii_have_independent_defaults(tmp_path: Path) -> None:
    """全景显示比例以及两处搜索半径都应具有独立默认配置。"""

    preference_path = tmp_path / "preference.json"
    config = load_star_map_ui_config(preference_path)

    assert config.mosaic_font_size_multiplier == 0.5
    assert config.mosaic_star_marker_size_multiplier == 0.5
    assert config.auto_match_default_search_radius_px == 30
    assert config.sequence_psf_search_radius_px == 30


def test_psf_interaction_preferences_are_loaded_and_bounded(tmp_path: Path) -> None:
    """单星自动匹配搜索公式和 PSF 外圈倍率都应从配置读取并限制安全范围。"""

    preference_path = tmp_path / "preference.json"
    preference_path.write_text(
        """
        {
          "star_pair_psf_outer_diameter_multiplier": 20.0,
          "auto_pair_search_rms_multiplier": -1.0,
          "auto_pair_search_base_radius_px": 23,
          "auto_pair_search_max_radius_px": 9999,
          "double_click_focus_auto_pair_enabled": true
        }
        """,
        encoding="utf-8",
    )

    config = load_star_map_ui_config(preference_path)

    assert config.star_pair_psf_outer_diameter_multiplier == 10.0
    assert config.auto_pair_search_rms_multiplier == 0.0
    assert config.auto_pair_search_base_radius_px == 23
    assert config.auto_pair_search_max_radius_px == 2000
    assert config.double_click_focus_auto_pair_enabled is True


def test_adjacent_alignment_hyperparameters_are_loaded_and_bounded(tmp_path: Path) -> None:
    """相邻图像两种配准模式都应从 preference.json 获取超参数并限制有效范围。"""

    preference_path = tmp_path / "preference.json"
    preference_path.write_text(
        """
        {
          "adjacent_alignment_max_correspondences": 72,
          "adjacent_star_detection_sigma": 3.5,
          "adjacent_star_max_detected_stars": 480,
          "adjacent_star_final_match_distance_px": 2.25,
          "adjacent_landscape_sift_max_features": 24000,
          "adjacent_landscape_ratio_test_threshold": 0.72,
          "adjacent_landscape_normalization_low_percentile": 95.0,
          "adjacent_landscape_normalization_high_percentile": 10.0
        }
        """,
        encoding="utf-8",
    )

    config = load_adjacent_alignment_config(preference_path)

    assert config.max_correspondences == 72
    assert config.stars.detection_sigma == 3.5
    assert config.stars.max_detected_stars == 480
    assert config.stars.final_match_distance_px == 2.25
    assert config.landscape.sift_max_features == 24000
    assert config.landscape.ratio_test_threshold == 0.72
    assert config.landscape.normalization_low_percentile == 95.0
    assert config.landscape.normalization_high_percentile > config.landscape.normalization_low_percentile
    assert isinstance(config.stars, AdjacentStarAlignmentConfig)
    assert isinstance(config.landscape, AdjacentLandscapeAlignmentConfig)
    assert AdjacentLandscapeAlignmentConfig().sift_max_features == 30000
    assert AdjacentLandscapeAlignmentConfig().flann_checks == 500
    assert AdjacentLandscapeAlignmentConfig().min_inlier_matches == 6


def test_existing_preference_only_adds_missing_keys_and_preserves_extensions(tmp_path: Path) -> None:
    preference_path = tmp_path / "preference.json"
    preference_path.write_text(
        """
        {
          // 用户已经修改过的配置不能被默认值覆盖。
          "controls_font_size_pt": 18,
          "custom_extension": "keep-me"
        }
        """,
        encoding="utf-8",
    )

    values = ensure_preference_file(preference_path)
    written = _read_jsonc(preference_path)

    assert values["controls_font_size_pt"] == 18
    assert written["controls_font_size_pt"] == 18
    assert written["custom_extension"] == "keep-me"
    assert set(DEFAULT_PREFERENCE_VALUES).issubset(written)


def test_invalid_preference_is_rebuilt_instead_of_breaking_startup(tmp_path: Path) -> None:
    preference_path = tmp_path / "preference.json"
    preference_path.write_text("{ invalid json", encoding="utf-8")

    values = ensure_preference_file(preference_path)
    written = _read_jsonc(preference_path)

    assert values == written
    assert set(DEFAULT_PREFERENCE_VALUES).issubset(written)


def test_latest_import_directory_is_shared_and_overwritten(tmp_path: Path) -> None:
    preference_path = tmp_path / "preference.json"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    fallback_dir = tmp_path / "fallback"
    first_dir.mkdir()
    second_dir.mkdir()
    fallback_dir.mkdir()
    first_file = first_dir / "one.json"
    second_file = second_dir / "two.tif"
    first_file.write_text("{}", encoding="utf-8")
    second_file.write_bytes(b"TIFF")

    assert remember_import_path(first_file, path=preference_path)
    assert recent_import_directory(fallback_dir, path=preference_path) == first_dir.resolve()

    assert remember_import_path([second_file], path=preference_path)
    assert recent_import_directory(fallback_dir, path=preference_path) == second_dir.resolve()
    written = _read_jsonc(preference_path)
    written_text = preference_path.read_text(encoding="utf-8")
    assert written[LAST_IMPORT_DIRECTORY_KEY] == str(second_dir.resolve())
    for comment in PREFERENCE_COMMENTS.values():
        assert f"// {comment}" in written_text


def test_missing_saved_import_directory_falls_back_to_caller_directory(tmp_path: Path) -> None:
    preference_path = tmp_path / "preference.json"
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    values = ensure_preference_file(preference_path)
    values[LAST_IMPORT_DIRECTORY_KEY] = str(tmp_path / "removed")
    preference_path.write_text(json.dumps(values), encoding="utf-8")

    assert recent_import_directory(fallback_dir, path=preference_path) == fallback_dir.resolve()
