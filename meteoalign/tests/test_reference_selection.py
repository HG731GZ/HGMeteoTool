from __future__ import annotations

import numpy as np

from meteoalign.simulator import ProjectedStarMap, select_reference_stars


def _projected_star_map_for_selection() -> ProjectedStarMap:
    return ProjectedStarMap(
        width=1000,
        height=800,
        source_name="test",
        x_px=np.asarray([500.0, 650.0], dtype=np.float64),
        y_px=np.asarray([400.0, 420.0], dtype=np.float64),
        radius_px=np.asarray([4.0, 3.0], dtype=np.float64),
        intensity=np.asarray([255, 220], dtype=np.uint8),
        alpha=np.asarray([128, 255], dtype=np.uint8),
        above_horizon=np.asarray([False, True], dtype=bool),
        star_ids=np.asarray(["below", "above"]),
        display_names=np.asarray(["Below", "Above"]),
        common_names=np.asarray(["", ""]),
        ra_deg=np.asarray([10.0, 11.0], dtype=np.float64),
        dec_deg=np.asarray([20.0, 21.0], dtype=np.float64),
        alt_deg=np.asarray([-5.0, 35.0], dtype=np.float64),
        az_deg=np.asarray([120.0, 121.0], dtype=np.float64),
        mag_v=np.asarray([1.0, 2.0], dtype=np.float64),
        color_index_bv=np.asarray([0.0, 0.0], dtype=np.float64),
        spectral_type=np.asarray(["", ""]),
        star_rgb=np.asarray([[255, 255, 255], [220, 220, 220]], dtype=np.uint8),
        grid_lines=(),
        direction_labels=(),
        catalog_count=2,
    )


def test_reference_selection_does_not_filter_below_horizon_star() -> None:
    selected = select_reference_stars(_projected_star_map_for_selection(), max_count=1)

    assert len(selected) == 1
    assert selected[0].star_id == "below"
