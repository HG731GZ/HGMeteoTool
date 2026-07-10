from __future__ import annotations

import numpy as np

from meteoalign.simulator import _star_rgb, _star_style


def test_star_rgb_applies_configured_magnitude_threshold() -> None:
    magnitudes = np.asarray([2.9, 3.1, 6.0, 6.1], dtype=np.float64)
    _radius, intensity = _star_style(magnitudes, visible_mag_limit=6.5)

    rgb = _star_rgb(
        mag_v=magnitudes,
        star_color_mag_limit=6.0,
        intensity=intensity,
        color_index_bv=np.asarray([0.80, 0.80, 0.80, 0.80], dtype=np.float64),
        spectral_type=np.asarray(["", "", "", ""], dtype=str),
    )

    assert rgb[:3].tolist() == [[255, 220, 135]] * 3
    assert rgb[3].tolist() == [int(intensity[3])] * 3


def test_star_rgb_uses_all_available_chromaticity_information() -> None:
    rgb = _star_rgb(
        mag_v=np.asarray([5.0, 5.0, 5.0], dtype=np.float64),
        star_color_mag_limit=6.0,
        intensity=np.asarray([70, 90, 110], dtype=np.uint8),
        color_index_bv=np.asarray([1.30, np.nan, np.nan], dtype=np.float64),
        spectral_type=np.asarray(["", "K2III", ""], dtype=str),
    )

    assert rgb.tolist() == [
        [255, 105, 80],
        [255, 165, 85],
        [110, 110, 110],
    ]
