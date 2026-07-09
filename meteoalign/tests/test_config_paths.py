from __future__ import annotations

import sys
from pathlib import Path

from meteoalign.config import default_config_path


def test_default_config_path_uses_source_project_root() -> None:
    expected = Path(__file__).resolve().parents[2] / "preference.json"

    assert default_config_path() == expected


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
