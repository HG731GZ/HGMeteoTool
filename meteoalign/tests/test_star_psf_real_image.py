"""项目内 28mm 星空与地景样片的 PSF 回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meteoalign.image_preview import load_image_preview
from meteoalign.star_fitting import StarFitError, fit_star_position


def test_img_0067_accepts_saturated_stars_and_rejects_landscape_samples() -> None:
    """真实星野图应保留亮星，并稳定拒绝蒙版内典型地景纹理。"""

    root = Path(__file__).resolve().parents[2] / "testimages" / "28mm测试"
    image_path = root / "IMG_0067.TIF"
    model_path = root / "IMG_0067_model.json"
    if not image_path.exists() or not model_path.exists():
        pytest.skip("未提供 IMG_0067 实图回归素材。")

    preview = load_image_preview(
        image_path,
        max_long_side_px=None,
        include_native_luminance=True,
    )
    assert preview.native_luminance is not None
    image = preview.native_luminance
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    pairs_by_id = {str(pair["star_id"]): pair for pair in payload["fit_pairs"]}
    regression_star_ids = ("HR2618", "HR2827", "HR3165", "HR2040", "HR2445", "HR2579", "HR2942")
    for star_id in regression_star_ids:
        old_pair = pairs_by_id[star_id]
        fitted = fit_star_position(
            image,
            float(old_pair["image_x_px"]),
            float(old_pair["image_y_px"]),
            30,
            max_fit_radius_px=40,
            selection_mode="manual",
        )
        offset = float(
            np.hypot(
                fitted.x - float(old_pair["image_x_px"]),
                fitted.y - float(old_pair["image_y_px"]),
            )
        )
        assert offset < 2.0
        assert fitted.saturated
        assert fitted.quality_score >= 0.62

    landscape_points = (
        (300.0, 300.0),
        (1500.0, 300.0),
        (2700.0, 800.0),
        (900.0, 1400.0),
        (2100.0, 2000.0),
        (2700.0, 2700.0),
        (1500.0, 3300.0),
        (2700.0, 3300.0),
    )
    for x, y in landscape_points:
        with pytest.raises(StarFitError):
            fit_star_position(
                image,
                x,
                y,
                20,
                max_fit_radius_px=40,
                selection_mode="manual",
            )


def test_img_0888_problematic_stars_keep_stable_shape_and_size() -> None:
    """HR1605 的次峰、HR1612 的地景边缘和昴星团邻星不能污染主星 PSF。"""

    root = Path(__file__).resolve().parents[2] / "testimages" / "椭圆星点问题"
    image_path = root / "IMG_0888.tif"
    pairs_path = root / "IMG_0888_starpairs.json"
    if not image_path.exists() or not pairs_path.exists():
        pytest.skip("未提供 IMG_0888 星点问题实图回归素材。")

    payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs_by_id = {str(item["star_id"]): item for item in payload["pairs"]}
    preview = load_image_preview(
        image_path,
        max_long_side_px=None,
        include_native_luminance=True,
    )
    assert preview.native_luminance is not None

    hr1605 = pairs_by_id["HR1605"]
    for image in (preview.image, preview.native_luminance):
        fitted = fit_star_position(
            image,
            float(hr1605["image_x_px"]),
            float(hr1605["image_y_px"]),
            10,
            max_fit_radius_px=40,
            selection_mode="manual",
        )

        assert not fitted.forced
        assert fitted.x == pytest.approx(float(hr1605["image_x_px"]), abs=0.1)
        assert fitted.y == pytest.approx(float(hr1605["image_y_px"]), abs=0.1)
        assert fitted.fwhm_x == pytest.approx(fitted.fwhm_y)
        assert fitted.quality_score >= 0.85

    hr1612 = pairs_by_id["HR1612"]
    hr1612_fits = []
    for image in (preview.image, preview.native_luminance):
        fitted = fit_star_position(
            image,
            float(hr1612["image_x_px"]),
            float(hr1612["image_y_px"]),
            10,
            max_fit_radius_px=40,
            selection_mode="manual",
        )
        hr1612_fits.append(fitted)

        assert not fitted.forced
        assert fitted.fwhm_x == pytest.approx(fitted.fwhm_y)
        assert 3.0 < fitted.fwhm_x < 8.0
        assert fitted.quality_score >= 0.80

    display_fit, native_fit = hr1612_fits
    assert display_fit.x == pytest.approx(native_fit.x, abs=0.15)
    assert display_fit.y == pytest.approx(native_fit.y, abs=0.15)
    assert display_fit.fwhm_x == pytest.approx(native_fit.fwhm_x, abs=0.5)

    # 昴宿一的星翼与昴宿增九等邻星在大窗口中相连；旧的强制矩心会被拖到左侧约 18 px。
    electra_click = (3487.0, 2091.0)
    electra_center = (3487.2, 2091.3)
    for image in (preview.image, preview.native_luminance):
        fitted = fit_star_position(
            image,
            *electra_click,
            25,
            max_fit_radius_px=40,
            selection_mode="manual",
            force_reliable_source=True,
        )
        assert not fitted.forced
        assert fitted.x == pytest.approx(electra_center[0], abs=0.35)
        assert fitted.y == pytest.approx(electra_center[1], abs=0.35)

        forced = fit_star_position(
            image,
            *electra_click,
            25,
            max_fit_radius_px=40,
            selection_mode="manual",
            saturated_fit_error_limit=0.000001,
            force_reliable_source=True,
        )
        assert forced.forced
        assert forced.x == pytest.approx(electra_center[0], abs=0.6)
        assert forced.y == pytest.approx(electra_center[1], abs=0.7)
