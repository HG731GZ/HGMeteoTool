from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from meteoalign.reference import build_reference_payload
from meteoalign.simulator import (
    CameraSettings,
    ObserverSettings,
    ProjectedStarMap,
    ReferenceStar,
    ViewSettings,
    select_reference_stars,
)


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


def test_reference_payload_keeps_configured_count_separate_from_saved_star_records() -> None:
    """自动匹配附带的星记录再多，也不得覆盖用户设置的标注星数。"""

    star_map = _projected_star_map_for_selection()
    reference_stars = tuple(
        ReferenceStar(
            index=index,
            star_id=str(star_map.star_ids[star_index]),
            name=str(star_map.display_names[star_index]),
            display_name=str(star_map.display_names[star_index]),
            common_name="",
            ra_deg=float(star_map.ra_deg[star_index]),
            dec_deg=float(star_map.dec_deg[star_index]),
            mag_v=float(star_map.mag_v[star_index]),
            sim_x=float(star_map.x_px[star_index]),
            sim_y=float(star_map.y_px[star_index]),
            alt_deg=float(star_map.alt_deg[star_index]),
            az_deg=float(star_map.az_deg[star_index]),
        )
        for index, star_index in enumerate(range(len(star_map)), start=1)
    )
    payload = build_reference_payload(
        star_map=star_map,
        reference_stars=reference_stars,
        observer=ObserverSettings(
            observation_time_utc=datetime(2026, 7, 16, tzinfo=timezone.utc),
            latitude_deg=40.0,
            longitude_deg=116.0,
        ),
        camera=CameraSettings(36.0, 24.0, 1000, 800, 24.0),
        view=ViewSettings(0.0, 30.0),
        visible_mag_limit=6.5,
        reference_star_count=12,
    )

    assert len(payload["stars"]) == 2
    assert payload["render"]["reference_star_count"] == 12
