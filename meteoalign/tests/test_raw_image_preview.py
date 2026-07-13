"""流星框选页面专用 RAW 预览测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rawpy

from meteoalign.raw_image_preview import (
    METEOR_IMAGE_FILE_FILTER,
    is_raw_image_path,
    load_meteor_image_preview,
)


class _FakeRaw:
    """提供 rawpy 上下文管理器和后处理接口的最小替身。"""

    def __enter__(self) -> "_FakeRaw":
        return self

    def __exit__(self, *_unused: object) -> None:
        return None

    def postprocess(self, **_unused: object) -> np.ndarray:
        return np.zeros((120, 240, 3), dtype=np.uint8)


def test_meteor_raw_preview_uses_libraw_and_preserves_original_geometry(tmp_path: Path, monkeypatch) -> None:
    """RAW 预览可以缩放显示，但框选坐标仍使用 LibRaw 输出的原始尺寸。"""

    raw_path = tmp_path / "meteor.CR3"
    raw_path.write_bytes(b"raw")
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeRaw())

    preview = load_meteor_image_preview(raw_path, max_long_side_px=100)

    assert preview.path == raw_path.resolve()
    assert (preview.original_width, preview.original_height) == (240, 120)
    assert (preview.image.width(), preview.image.height()) == (100, 50)
    assert is_raw_image_path(raw_path)
    assert "*.cr3" in METEOR_IMAGE_FILE_FILTER


def test_meteor_raw_support_does_not_change_standard_image_filter() -> None:
    """RAW 只出现在流星框选专用筛选器中。"""

    from meteoalign.image_preview import IMAGE_FILE_FILTER

    assert "*.dng" in METEOR_IMAGE_FILE_FILTER
    assert "*.dng" not in IMAGE_FILE_FILTER
