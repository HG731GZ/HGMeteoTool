from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QImage

from meteoalign.adjacent_alignment import resolve_model_source_image_path
from meteoalign.application.app_utils import _resolve_star_pair_session_real_image_path
from meteoalign.image_path_resolution import associated_image_candidates
from meteoalign.mosaic_model_io import _resolve_source_image_path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def _write_image(path: Path, width: int, height: int) -> Path:
    """写入可由 QImageReader 读取尺寸的测试图像。"""

    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(0)
    assert image.save(str(path))
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


def test_json_image_lookup_uses_tif_png_jpg_priority_and_skips_wrong_size(tmp_path: Path) -> None:
    """同主文件名多格式共存时应按优先级选择，并排除尺寸不符的 TIFF。"""

    json_path = tmp_path / "session.json"
    jpg_path = _write_image(tmp_path / "scene.jpg", 120, 80)
    png_path = _write_image(tmp_path / "scene.png", 120, 80)
    tif_path = _write_image(tmp_path / "scene.tif", 120, 80)
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_name": jpg_path.name,
            "file_stem": "scene",
            "original_width_px": 120,
            "original_height_px": 80,
        },
    }

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == tif_path.resolve()

    tif_path.unlink()
    _write_image(tif_path, 60, 40)
    assert _resolve_star_pair_session_real_image_path(payload, json_path) == png_path.resolve()


def test_source_model_and_adjacent_lookup_accept_converted_extension(tmp_path: Path) -> None:
    """源模型仅记录旧 JPG 名称时，也应按主文件名找到尺寸一致的 TIFF。"""

    json_path = tmp_path / "model.json"
    converted_path = _write_image(tmp_path / "frame.tif", 160, 90)
    payload = {
        "source_image": {
            "file_name": "frame.jpg",
            "file_stem": "frame",
            "original_width_px": 160,
            "original_height_px": 90,
        },
        "image_geometry": {"width_px": 160, "height_px": 90},
    }

    assert _resolve_source_image_path(payload, json_path)[0] == converted_path.resolve()
    assert resolve_model_source_image_path(payload, json_path) == converted_path.resolve()


def test_standard_json_lookup_does_not_add_raw_candidates(tmp_path: Path) -> None:
    """流星框选以外的 JSON 解析不得把 RAW 加入普通图片候选。"""

    raw_path = _touch(tmp_path / "frame.dng")
    metadata = {"file_name": "frame.dng", "file_stem": "frame"}

    assert raw_path.resolve() not in associated_image_candidates(metadata, tmp_path / "model.json")


def test_meteor_json_candidate_order_places_raw_between_tif_and_png(tmp_path: Path) -> None:
    """流星关联图像候选应使用 TIFF、RAW、PNG、JPG 的顺序。"""

    for suffix in (".jpg", ".png", ".dng", ".tif"):
        _touch(tmp_path / f"meteor{suffix}")
    candidates = associated_image_candidates(
        {"file_name": "meteor.jpg", "file_stem": "meteor"},
        tmp_path / "meteor_Meteor.json",
        include_raw=True,
    )

    assert [path.suffix for path in candidates[:4]] == [".tif", ".dng", ".png", ".jpg"]
