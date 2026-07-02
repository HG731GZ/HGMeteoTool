from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StarMapUiConfig:
    controls_font_size_pt: int = 10
    status_bar_font_size_pt: int = 10
    direction_label_font_size_pt: int = 16
    star_name_font_size_pt: int = 11
    reference_label_font_size_pt: int = 15


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "meteoalign_ui_config.json"


def _read_int(config: dict[str, object], key: str, default_value: int, minimum: int, maximum: int) -> int:
    value = config.get(key, default_value)
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return default_value
    return min(max(int_value, minimum), maximum)


def load_star_map_ui_config(path: Path | None = None) -> StarMapUiConfig:
    config_path = path or default_config_path()
    if not config_path.exists():
        return StarMapUiConfig()

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StarMapUiConfig()
    if not isinstance(raw_config, dict):
        return StarMapUiConfig()

    return StarMapUiConfig(
        controls_font_size_pt=_read_int(raw_config, "controls_font_size_pt", 10, 6, 32),
        status_bar_font_size_pt=_read_int(raw_config, "status_bar_font_size_pt", 10, 6, 32),
        direction_label_font_size_pt=_read_int(raw_config, "direction_label_font_size_pt", 16, 6, 48),
        star_name_font_size_pt=_read_int(raw_config, "star_name_font_size_pt", 11, 6, 48),
        reference_label_font_size_pt=_read_int(raw_config, "reference_label_font_size_pt", 15, 6, 64),
    )
