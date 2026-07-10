from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .preference_manager import default_preference_path, ensure_preference_file


@dataclass(frozen=True)
class StarMapUiConfig:
    controls_font_size_pt: int = 10
    status_bar_font_size_pt: int = 10
    direction_label_font_size_pt: int = 16
    star_name_font_size_pt: int = 11
    reference_label_font_size_pt: int = 15
    star_color_mag_limit: float = 6.0
    aligned_reference_scale_multiplier: float = 1.6
    star_pick_circle_default_diameter_px: int = 50
    star_pick_circle_min_diameter_px: int = 20
    star_pick_circle_max_diameter_px: int = 200
    star_pick_psf_radius_scale: float = 0.4
    star_pick_psf_max_radius_px: int = 40
    default_latitude_deg: float = 40.0
    default_longitude_deg: float = 116.0
    default_elevation_m: float = 50.0
    auto_match_default_new_count: int = 200
    auto_match_default_constraint_mode: str = "soft"
    auto_match_default_soft_weight: float = 0.3
    wheel_zoom_enabled: bool = True
    touchpad_pinch_zoom_enabled: bool = True
    mosaic_texture_scale_percent: float = 25.0
    mosaic_texture_max_long_side_px: int = 1920
    mosaic_grid_precision_default: int = 36
    mosaic_render_fps_limit: int = 60
    mosaic_export_block_rows: int = 1024
    mosaic_map_tile_size_px: int = 4
    mosaic_export_tiff_lzw_compression: bool = True


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
        reference_label_font_size_pt=_read_int(raw_config, "reference_label_font_size_pt", 15, 6, 64),
        star_color_mag_limit=_read_float(raw_config, "star_color_mag_limit", 6.0, -10.0, 30.0),
        aligned_reference_scale_multiplier=_read_float(
            raw_config,
            "aligned_reference_scale_multiplier",
            1.6,
            0.2,
            6.0,
        ),
        star_pick_circle_default_diameter_px=circle_default,
        star_pick_circle_min_diameter_px=circle_min,
        star_pick_circle_max_diameter_px=circle_max,
        star_pick_psf_radius_scale=_read_float(raw_config, "star_pick_psf_radius_scale", 0.4, 0.1, 1.0),
        star_pick_psf_max_radius_px=_read_int(raw_config, "star_pick_psf_max_radius_px", 40, 8, 200),
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
        wheel_zoom_enabled=_read_bool(raw_config, "wheel_zoom_enabled", True),
        touchpad_pinch_zoom_enabled=_read_bool(raw_config, "touchpad_pinch_zoom_enabled", True),
        mosaic_texture_scale_percent=_read_float(raw_config, "mosaic_texture_scale_percent", 25.0, 1.0, 100.0),
        mosaic_texture_max_long_side_px=_read_int(raw_config, "mosaic_texture_max_long_side_px", 1920, 64, 20000),
        mosaic_grid_precision_default=_read_int(raw_config, "mosaic_grid_precision_default", 36, 12, 180),
        mosaic_render_fps_limit=_read_int(raw_config, "mosaic_render_fps_limit", 60, 1, 240),
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
    )
