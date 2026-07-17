from __future__ import annotations

from datetime import timezone, timedelta
from types import SimpleNamespace

from PIL import Image

import pytest

from meteoalign.application.app_sequence_import import SequenceImportMixin
from meteoalign.image_sequence import collect_image_sequence, read_image_capture_time


def _write_jpeg_with_exif_time(path, time_text: str, offset_text: str | None = None) -> None:
    image = Image.new("RGB", (8, 8), (12, 14, 18))
    exif = Image.Exif()
    exif[36867] = time_text
    exif[37521] = "250"
    if offset_text is not None:
        exif[36881] = offset_text
    image.save(path, exif=exif)


def test_read_image_capture_time_prefers_datetime_original_with_offset(tmp_path) -> None:
    image_path = tmp_path / "frame_001.jpg"
    _write_jpeg_with_exif_time(image_path, "2026:08:12 23:59:58", "+08:00")

    item = read_image_capture_time(image_path)

    assert item.path == image_path.resolve()
    assert item.capture_datetime.tzinfo == timezone(timedelta(hours=8))
    assert item.capture_datetime.microsecond == 250000
    assert item.capture_datetime_utc is not None
    assert item.capture_datetime_utc.hour == 15
    assert item.capture_time_source == "EXIF DateTimeOriginal+OffsetTime"


def test_collect_image_sequence_sorts_and_rejects_missing_capture_time(tmp_path) -> None:
    later_path = tmp_path / "later.jpg"
    earlier_path = tmp_path / "earlier.jpg"
    missing_path = tmp_path / "missing.jpg"
    _write_jpeg_with_exif_time(later_path, "2026:08:13 00:00:10")
    _write_jpeg_with_exif_time(earlier_path, "2026:08:13 00:00:01")
    Image.new("RGB", (8, 8), (0, 0, 0)).save(missing_path)

    items, rejected = collect_image_sequence([str(later_path), str(missing_path), str(earlier_path)])

    assert [item.path.name for item in items] == ["earlier.jpg", "later.jpg"]
    assert len(rejected) == 1
    assert rejected[0].path == missing_path.resolve()
    assert "EXIF" in rejected[0].reason


def test_read_image_capture_time_rejects_datetime_without_original_time(tmp_path) -> None:
    image_path = tmp_path / "exported_only.jpg"
    image = Image.new("RGB", (8, 8), (12, 14, 18))
    exif = Image.Exif()
    exif[306] = "2026:01:02 03:04:05"
    image.save(image_path, exif=exif)

    with pytest.raises(ValueError, match="原始拍摄时间"):
        read_image_capture_time(image_path)


def test_sequence_first_session_import_does_not_reuse_previous_loaded_image(tmp_path) -> None:
    """新序列首帧 JSON 不应与星点匹配页中遗留的旧图像校验。"""

    calls: list[tuple[object, dict[str, object]]] = []

    def _load_star_pair_session(file_path: object, **kwargs: object) -> None:
        calls.append((file_path, kwargs))

    harness = SimpleNamespace(
        _json_import_thread=None,
        ui=SimpleNamespace(statusbar=SimpleNamespace(showMessage=lambda _message: None)),
        load_star_pair_session=_load_star_pair_session,
    )
    json_path = tmp_path / "新序列_starpairs.json"

    SequenceImportMixin._start_first_sequence_session_import(harness, json_path, 27)

    assert calls == [
        (
            json_path,
            {
                "switch_to_reference": False,
                "show_progress": False,
                "clear_input_name": "第一帧匹配 JSON",
                "reuse_current_image": False,
            },
        )
    ]
