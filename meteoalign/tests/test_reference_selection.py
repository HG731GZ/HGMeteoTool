from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from meteoalign.application.app_constants import REFERENCE_LABEL_MODE_FIXED_COUNT
from meteoalign.application.app_rendering import RenderingMixin
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


class _ValueControl:
    """提供星空模拟标注限制所需的最小数值控件。"""

    def __init__(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value


def test_simulator_annotations_only_follow_simulator_count_limit() -> None:
    """匹配列表即使保留更多星，模拟页标注仍只能使用自身数量限制。"""

    harness = RenderingMixin()
    harness.ui = type(
        "Ui",
        (),
        {
            "spinBoxReferenceStarCount": _ValueControl(1),
            "doubleSpinBoxReferenceMagLimit": _ValueControl(6.0),
        },
    )()
    harness._reference_label_mode = lambda: REFERENCE_LABEL_MODE_FIXED_COUNT
    harness._auto_match_reference_star_ids = ["above"]

    selected = harness._select_simulator_annotation_stars(_projected_star_map_for_selection())

    assert [star.star_id for star in selected] == ["below"]


def test_render_routes_limited_annotations_and_full_pair_rows_separately() -> None:
    """一次重渲染不得再把完整匹配集合传给星空模拟标注。"""

    harness = RenderingMixin()
    star_map = _projected_star_map_for_selection()
    limited_annotations = ("受限标注",)
    complete_pair_rows = ("匹配一", "匹配二", "匹配三")
    displayed: list[tuple[object, tuple[object, ...]]] = []
    table_updates: list[tuple[object, ...]] = []
    harness._build_projected_star_map = lambda: (None, None, None, None, star_map)
    harness._select_simulator_annotation_stars = lambda _star_map: limited_annotations
    harness._select_current_reference_stars = lambda _star_map: complete_pair_rows
    harness._display_star_map = lambda rendered_map, stars: displayed.append((rendered_map, stars))
    harness._update_star_pair_table = lambda stars: table_updates.append(stars)
    harness.ui = type("Ui", (), {"statusbar": type("Status", (), {"showMessage": lambda *_args: None})()})()

    harness.render_now()

    assert displayed == [(star_map, limited_annotations)]
    assert table_updates == [complete_pair_rows]


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
