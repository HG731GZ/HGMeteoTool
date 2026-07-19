"""流星框选页面专用 RAW 预览测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy

from meteoalign.raw_image_preview import (
    METEOR_IMAGE_FILE_FILTER,
    is_raw_image_path,
    load_meteor_image_preview,
)


class _FakeRaw:
    """提供 rawpy 上下文管理器和后处理接口的最小替身。"""

    last_postprocess_kwargs: dict[str, object] = {}

    sizes = SimpleNamespace(
        width=240,
        height=120,
        crop_left_margin=0,
        crop_top_margin=0,
        crop_width=240,
        crop_height=120,
        flip=0,
    )

    def __enter__(self) -> "_FakeRaw":
        return self

    def __exit__(self, *_unused: object) -> None:
        return None

    def postprocess(self, **kwargs: object) -> np.ndarray:
        type(self).last_postprocess_kwargs = dict(kwargs)
        if kwargs.get("half_size"):
            return np.zeros((60, 120, 3), dtype=np.uint8)
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
    assert _FakeRaw.last_postprocess_kwargs["half_size"] is True
    assert _FakeRaw.last_postprocess_kwargs["user_flip"] == 0
    assert is_raw_image_path(raw_path)
    assert "*.cr3" in METEOR_IMAGE_FILE_FILTER


def test_meteor_raw_preview_uses_camera_recommended_crop(tmp_path: Path, monkeypatch) -> None:
    """RAW 内嵌标准裁切区应成为预览、框选和 Photoshop 蒙版的统一尺寸。"""

    class CroppedFakeRaw(_FakeRaw):
        sizes = SimpleNamespace(
            width=240,
            height=120,
            crop_left_margin=4,
            crop_top_margin=3,
            crop_width=230,
            crop_height=110,
            flip=0,
        )

    raw_path = tmp_path / "meteor.ARW"
    raw_path.write_bytes(b"raw")
    monkeypatch.setattr(rawpy, "imread", lambda _path: CroppedFakeRaw())

    preview = load_meteor_image_preview(raw_path, max_long_side_px=None)

    assert (preview.original_width, preview.original_height) == (230, 110)
    assert (preview.image.width(), preview.image.height()) == (230, 110)


def test_meteor_raw_preview_ignores_camera_orientation(tmp_path: Path, monkeypatch) -> None:
    """即使 RAW 带有竖拍标记，预览和框选坐标也必须保持传感器原始方向。"""

    class PortraitTaggedFakeRaw(_FakeRaw):
        sizes = SimpleNamespace(
            width=240,
            height=120,
            crop_left_margin=4,
            crop_top_margin=3,
            crop_width=230,
            crop_height=110,
            flip=6,
        )

    raw_path = tmp_path / "portrait-tagged.ARW"
    raw_path.write_bytes(b"raw")
    monkeypatch.setattr(rawpy, "imread", lambda _path: PortraitTaggedFakeRaw())

    preview = load_meteor_image_preview(raw_path, max_long_side_px=None)

    assert (preview.original_width, preview.original_height) == (230, 110)
    assert (preview.image.width(), preview.image.height()) == (230, 110)


def test_sony_a7s_raw_preview_matches_photoshop_dimensions() -> None:
    """Sony A7S 的标准裁切应与 Photoshop 导出的 4240×2832 蒙版一致。"""

    image_path = Path(__file__).resolve().parents[2] / "testimages" / "RAW文件" / "DSC03817.ARW"
    if not image_path.exists():
        pytest.skip("未提供 Sony A7S RAW 实图回归素材。")

    preview = load_meteor_image_preview(image_path, max_long_side_px=240)

    assert (preview.original_width, preview.original_height) == (4240, 2832)
    assert max(preview.image.width(), preview.image.height()) <= 240
    assert preview.image.width() / preview.image.height() == pytest.approx(4240 / 2832, rel=0.01)


def test_meteor_raw_support_does_not_change_standard_image_filter() -> None:
    """RAW 只出现在流星框选专用筛选器中。"""

    from meteoalign.image_preview import IMAGE_FILE_FILTER

    assert "*.dng" in METEOR_IMAGE_FILE_FILTER
    assert "*.dng" not in IMAGE_FILE_FILTER
