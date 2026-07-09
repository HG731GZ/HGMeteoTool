from __future__ import annotations

from pathlib import Path

from meteoalign.app_utils import _resolve_star_pair_session_real_image_path
from meteoalign.mosaic_model_io import _resolve_source_image_path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def test_star_pair_session_image_lookup_prefers_same_dir_name_then_relative_then_absolute(tmp_path: Path) -> None:
    json_path = tmp_path / "session.json"
    same_dir_image = _touch(tmp_path / "same.jpg")
    relative_image = _touch(tmp_path / "images" / "relative.jpg")
    absolute_image = _touch(tmp_path / "elsewhere" / "absolute.jpg")
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_name": same_dir_image.name,
            "relative_path": "images/relative.jpg",
            "path": str(absolute_image),
        },
    }

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == same_dir_image.resolve()

    same_dir_image.unlink()

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == relative_image.resolve()

    relative_image.unlink()

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == absolute_image.resolve()


def test_mosaic_source_image_lookup_prefers_same_dir_name_then_relative_then_absolute(tmp_path: Path) -> None:
    json_path = tmp_path / "model.json"
    same_dir_image = _touch(tmp_path / "same.fit")
    relative_image = _touch(tmp_path / "images" / "relative.fit")
    absolute_image = _touch(tmp_path / "elsewhere" / "absolute.fit")
    payload = {
        "source_image": {
            "file_name": same_dir_image.name,
            "relative_path": "images/relative.fit",
            "path": str(absolute_image),
        },
    }

    assert _resolve_source_image_path(payload, json_path)[0] == same_dir_image.resolve()

    same_dir_image.unlink()

    assert _resolve_source_image_path(payload, json_path)[0] == relative_image.resolve()

    relative_image.unlink()

    assert _resolve_source_image_path(payload, json_path)[0] == absolute_image.resolve()
