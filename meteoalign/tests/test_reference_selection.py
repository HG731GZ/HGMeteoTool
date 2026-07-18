from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

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
from meteoalign.star_pair_model import (
    PAIR_ORIGIN_AUTO_MATCH,
    StarPairRecord,
    star_pair_records_from_payloads,
)
from meteoalign.star_pair_store import StarPairStore


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


def test_matched_star_stays_in_pair_rows_after_simulator_view_crops_it() -> None:
    """模拟器重绘裁掉已匹配星时，表格仍应从统一存储恢复该星。"""

    cropped_star = ReferenceStar(
        index=3,
        star_id="cropped-auto-match",
        name="视场外匹配星",
        display_name="视场外匹配星",
        common_name="",
        ra_deg=42.0,
        dec_deg=18.0,
        mag_v=3.2,
        sim_x=1200.0,
        sim_y=900.0,
        alt_deg=38.0,
        az_deg=210.0,
    )
    store = StarPairStore()
    store.add(
        StarPairRecord(
            reference_star=cropped_star,
            image_x_px=740.0,
            image_y_px=510.0,
            pair_origin=PAIR_ORIGIN_AUTO_MATCH,
            group_id="A",
        )
    )
    harness = RenderingMixin()
    harness._star_pair_store = store
    harness._manual_reference_star_ids = []
    harness._auto_match_reference_star_ids = [cropped_star.star_id]
    harness._excluded_reference_star_ids = []
    harness._mask_excluded_reference_star_ids = set()
    harness._imported_reference_star_by_id = {}
    harness._normalize_auto_match_groups = lambda: None
    harness._auto_match_group_id_for_star_id = lambda _star_id: "A"
    harness._select_simulator_annotation_stars = lambda star_map: (
        harness._reference_star_from_star_map_index(star_map, 0, 1),
    )

    selected = harness._select_current_reference_stars(_projected_star_map_for_selection())

    assert [star.star_id for star in selected] == ["below", cropped_star.star_id]


def test_a7r3_fisheye_matches_survive_rectilinear_simulator_crop() -> None:
    """复现鱼眼匹配配合直线模拟镜头时，两批自动匹配不应被裁成少数几行。"""

    session_path = (
        Path(__file__).resolve().parents[2]
        / "testimages"
        / "A7R3_1214_DSC06601_starpairs.json"
    )
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    records = star_pair_records_from_payloads(payload["pairs"])
    auto_records = [record for record in records if record.is_auto_match]
    group_by_star_id = {record.star_id: record.group_id or "A" for record in auto_records}
    store = StarPairStore()
    store.add_records(records)

    harness = RenderingMixin()
    harness._star_pair_store = store
    harness._manual_reference_star_ids = []
    harness._auto_match_reference_star_ids = [record.star_id for record in auto_records]
    harness._excluded_reference_star_ids = []
    harness._mask_excluded_reference_star_ids = set()
    harness._imported_reference_star_by_id = {}
    harness._normalize_auto_match_groups = lambda: None
    harness._auto_match_group_id_for_star_id = lambda star_id: group_by_star_id.get(star_id, "")
    harness._select_simulator_annotation_stars = lambda star_map: (
        harness._reference_star_from_star_map_index(star_map, 0, 1),
    )

    selected = harness._select_current_reference_stars(_projected_star_map_for_selection())
    selected_star_ids = {star.star_id for star in selected}

    assert len(records) == 54
    assert len(auto_records) == 37
    assert {record.star_id for record in records} <= selected_star_ids


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
