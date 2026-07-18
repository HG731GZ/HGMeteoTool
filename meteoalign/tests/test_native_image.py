"""原始位深图像读取测试。"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from PIL import Image

from meteoalign.image_preview import load_image_preview
from meteoalign.native_image import load_native_image_array, native_array_to_luminance


def test_native_loader_and_preview_keep_source_integer_depth(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """显示图可保持 8 位，但 PSF 亮度平面必须沿用源图的整数 dtype。"""

    path_8bit = tmp_path / "scene.jpg"
    path_16bit = tmp_path / "scene.tif"
    Image.fromarray(np.full((7, 9, 3), (30, 60, 90), dtype=np.uint8), mode="RGB").save(path_8bit)
    tifffile.imwrite(
        path_16bit,
        np.full((7, 9, 3), (10000, 30000, 50000), dtype=np.uint16),
        photometric="rgb",
    )

    preview_8bit = load_image_preview(path_8bit, max_long_side_px=None, include_native_luminance=True)
    preview_16bit = load_image_preview(path_16bit, max_long_side_px=None, include_native_luminance=True)

    assert preview_8bit.image.depth() == 24
    assert preview_8bit.native_luminance is not None
    assert preview_8bit.native_luminance.dtype == np.uint8
    assert preview_16bit.image.depth() == 24
    assert preview_16bit.native_luminance is not None
    assert preview_16bit.native_luminance.dtype == np.uint16
    assert int(preview_16bit.native_luminance[0, 0]) > 25000


def test_native_rgb_luminance_preserves_dtype() -> None:
    """RGB 转亮度只改变通道，不应改变 8/16 位量化类型。"""

    rgb_8bit = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    rgb_16bit = rgb_8bit.astype(np.uint16) * 257

    luminance_8bit = native_array_to_luminance(rgb_8bit)
    luminance_16bit = native_array_to_luminance(rgb_16bit)

    assert luminance_8bit.dtype == np.uint8
    assert luminance_16bit.dtype == np.uint16
    assert int(luminance_16bit[0, 0]) / int(luminance_8bit[0, 0]) == pytest.approx(257.0, rel=0.03)


def test_tiff_native_loader_returns_rgb_uint16_without_pillow_downconversion(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """16 位 RGB TIFF 应直接保留通道数值。"""

    path = tmp_path / "native.tif"
    expected = np.arange(5 * 7 * 3, dtype=np.uint16).reshape(5, 7, 3) * 521
    tifffile.imwrite(path, expected, photometric="rgb")

    actual = load_native_image_array(path)

    assert actual.dtype == np.uint16
    assert np.array_equal(actual, expected)
