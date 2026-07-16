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

    image = load_image_preview(image_path, max_long_side_px=None).image
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
