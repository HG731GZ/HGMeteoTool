from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from meteoalign.config import StarMapUiConfig
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
