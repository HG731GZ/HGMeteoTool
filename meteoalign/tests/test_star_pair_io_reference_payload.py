from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from meteoalign.application.app_star_pair_io import StarPairIOMixin
from meteoalign.simulator import ObserverSettings, ReferenceStar


def _star(star_id: str, index: int = 1) -> ReferenceStar:
    return ReferenceStar(
        index=index,
        star_id=star_id,
        name=star_id,
        display_name=star_id,
        common_name="",
        ra_deg=10.0 + index,
        dec_deg=20.0 + index,
        mag_v=1.0,
        sim_x=100.0 + index,
        sim_y=200.0 + index,
        alt_deg=45.0,
        az_deg=180.0,
    )


def _pair_record(star_id: str, index: int = 1) -> dict[str, object]:
    return {
        "star_id": star_id,
        "name": star_id,
        "display_name": star_id,
        "common_name": "",
        "ra_deg": 10.0 + index,
        "dec_deg": 20.0 + index,
        "mag_v": 1.0,
        "image_x_px": 100.0 + index,
        "image_y_px": 200.0 + index,
        "sim_x": 100.0 + index,
        "sim_y": 200.0 + index,
        "alt_deg": 45.0,
        "az_deg": 180.0,
        "object_type": "star",
        "pair_origin": "manual",
    }


def _observer() -> ObserverSettings:
    return ObserverSettings(
        observation_time_utc=datetime(2025, 12, 14, 19, 15, 45, tzinfo=timezone.utc),
        latitude_deg=40.0,
        longitude_deg=116.0,
        elevation_m=200.0,
    )


class _Harness(StarPairIOMixin):
    def __init__(self, observer: ObserverSettings | None = None) -> None:
        self._observer = observer or _observer()
        self._imported_reference_star_by_id: dict[str, ReferenceStar] = {}

    def _observer_settings(self) -> ObserverSettings:
        return self._observer

    def _reference_star_with_index(self, star: ReferenceStar, index: int) -> ReferenceStar:
        return replace(star, index=index)


def test_reference_payload_keeps_pair_stars_missing_from_current_reference_list() -> None:
    harness = _Harness()

    merged = harness._reference_stars_with_pair_records(
        (_star("HR1", 1),),
        [_pair_record("HR1", 1), _pair_record("HR2", 2), _pair_record("HR3", 3)],
        _observer(),
    )

    assert [star.star_id for star in merged] == ["HR1", "HR2", "HR3"]
    assert [star.index for star in merged] == [1, 2, 3]
    assert merged[1].sim_x == 102.0


def test_import_lookup_can_recover_pair_stars_missing_from_reference_payload() -> None:
    harness = _Harness()
    harness._imported_reference_star_by_id = {"HR1": _star("HR1", 1)}

    harness._merge_imported_reference_stars_from_pairs([_pair_record("HR2", 2)])

    assert set(harness._imported_reference_star_by_id) == {"HR1", "HR2"}
    assert harness._imported_reference_star_by_id["HR2"].display_name == "HR2"


def test_testseq_json_pair_ids_are_recoverable_when_reference_payload_misses_some() -> None:
    json_path = Path(__file__).resolve().parents[2] / "testseq" / "A7M3_1214_DSC04975_starpairs.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    reference_payload = payload["reference_payload"]
    observer_payload = reference_payload["observer"]
    observer = ObserverSettings(
        observation_time_utc=datetime.fromisoformat(
            str(observer_payload["observation_time_utc"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc),
        latitude_deg=float(observer_payload["latitude_deg"]),
        longitude_deg=float(observer_payload["longitude_deg"]),
        elevation_m=float(observer_payload["elevation_m"]),
    )
    harness = _Harness(observer)

    original_reference_ids = {
        str(record.get("star_id", "")).strip()
        for record in reference_payload["stars"]
        if isinstance(record, dict)
    }
    removed_pair_ids: set[str] = set()
    for pair_record in payload["pairs"]:
        if not isinstance(pair_record, dict):
            continue
        star_id = str(pair_record.get("star_id", "")).strip()
        if star_id in original_reference_ids:
            removed_pair_ids.add(star_id)
        if len(removed_pair_ids) >= 9:
            break
    assert len(removed_pair_ids) == 9

    pruned_reference_records = [
        record
        for record in reference_payload["stars"]
        if not isinstance(record, dict) or str(record.get("star_id", "")).strip() not in removed_pair_ids
    ]
    reference_lookup = harness._reference_star_lookup_from_records(pruned_reference_records, observer=observer)
    pair_lookup = harness._reference_star_lookup_from_records(payload["pairs"], observer=observer)
    missing_pair_ids = set(pair_lookup) - set(reference_lookup)
    merged_stars = harness._reference_stars_with_pair_records(
        tuple(reference_lookup.values()),
        payload["pairs"],
        observer,
    )

    assert removed_pair_ids.issubset(missing_pair_ids)
    assert set(pair_lookup).issubset({star.star_id for star in merged_stars})
