from __future__ import annotations

import sys
from pathlib import Path

from meteoalign.config import StarMapUiConfig, default_config_path, load_star_map_ui_config
from meteoalign.runtime_paths import runtime_catalog_dir


def test_default_config_path_uses_source_project_root() -> None:
    expected = Path(__file__).resolve().parents[2] / "preference.json"

    assert default_config_path() == expected


def test_star_color_mag_limit_defaults_to_six() -> None:
    assert StarMapUiConfig().star_color_mag_limit == 6.0


def test_star_color_mag_limit_is_loaded_from_preference(tmp_path: Path) -> None:
    config_path = tmp_path / "preference.json"
    config_path.write_text('{"star_color_mag_limit": 5.25}', encoding="utf-8")

    assert load_star_map_ui_config(config_path).star_color_mag_limit == 5.25


def test_default_config_path_uses_windows_exe_sibling(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", "/release/HoshinoPanoAssistant/HoshinoPanoAssistant.exe")

    assert default_config_path() == Path("/release/HoshinoPanoAssistant") / "preference.json"


def test_default_config_path_uses_macos_app_sibling(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        sys,
        "executable",
        "/Applications/HoshinoPanoAssistant.app/Contents/MacOS/HoshinoPanoAssistant",
    )

    assert default_config_path() == Path("/Applications") / "preference.json"


def test_runtime_catalog_dir_prefers_frozen_executable_sibling_catalog(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    app_dir = tmp_path / "release" / "HoshinoPanoAssistant"
    catalog_dir = app_dir / "catalog"
    catalog_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(app_dir / "HoshinoPanoAssistant.exe"))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert runtime_catalog_dir() == catalog_dir


def test_runtime_catalog_dir_uses_pyinstaller_meipass_catalog_when_external_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    app_dir = tmp_path / "release" / "HoshinoPanoAssistant"
    bundled_dir = tmp_path / "bundle" / "catalog"
    bundled_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(app_dir / "HoshinoPanoAssistant.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)

    assert runtime_catalog_dir() == bundled_dir


def test_runtime_catalog_dir_uses_macos_resources_catalog_when_external_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    app_bundle = tmp_path / "dist" / "HoshinoPanoAssistant.app"
    catalog_dir = app_bundle / "Contents" / "Resources" / "catalog"
    catalog_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "executable", str(app_bundle / "Contents" / "MacOS" / "HoshinoPanoAssistant"))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert runtime_catalog_dir() == catalog_dir
