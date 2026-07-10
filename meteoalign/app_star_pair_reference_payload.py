from __future__ import annotations

from pathlib import Path

from .image_sequence import ImageSequenceItem, read_image_capture_time, sequence_item_observation_time_utc
from .reference import build_reference_payload
from .simulator import ObserverSettings, ReferenceStar, project_horizontal_catalog
from .star_pair_model import StarPairRecord, reference_star_from_pair_payload

class StarPairReferencePayloadMixin:
    """参考星 payload 构建与导入配对中的参考星合并。"""

    def _is_catalog_reference_star(self, star: ReferenceStar) -> bool:
        star_id = star.star_id.strip()
        return star.object_type == "star" and bool(star_id) and not star_id.startswith("solar_system:")

    def _current_real_image_capture_item(self) -> ImageSequenceItem | None:
        if self.current_image_preview is None:
            return None
        try:
            return read_image_capture_time(Path(self.current_image_preview.path))
        except Exception:  # noqa: BLE001 - JSON 时间源允许退回到星空模拟页时间。
            return None

    def _current_real_image_capture_payload(self) -> dict[str, object]:
        item = self._current_real_image_capture_item()
        if item is None:
            return {}
        payload: dict[str, object] = {
            "capture_time_source": item.capture_time_source,
            "exif_capture_time": item.capture_datetime.isoformat(),
        }
        if item.capture_datetime_utc is not None:
            payload["capture_time_utc"] = item.capture_datetime_utc.isoformat()
        return payload

    def _reference_payload_observer(self) -> tuple[ObserverSettings, dict[str, object]]:
        base_observer = self._observer_settings()
        item = self._current_real_image_capture_item()
        if item is None:
            return base_observer, {
                "observation_time_source": "star_simulator_ui",
            }

        utc_offset_hours = float(self.ui.doubleSpinBoxUtcOffset.value())
        observer = ObserverSettings(
            observation_time_utc=sequence_item_observation_time_utc(item, utc_offset_hours),
            latitude_deg=base_observer.latitude_deg,
            longitude_deg=base_observer.longitude_deg,
            elevation_m=base_observer.elevation_m,
        )
        payload: dict[str, object] = {
            "observation_time_source": "real_image_exif",
            "capture_time_source": item.capture_time_source,
            "exif_capture_time": item.capture_datetime.isoformat(),
        }
        if item.capture_datetime_utc is not None:
            payload["capture_time_utc"] = item.capture_datetime_utc.isoformat()
        return observer, payload

    def _build_reference_payload_for_current_settings(self) -> dict[str, object]:
        return self._build_reference_payload_for_records(self._star_pair_records())

    def _build_reference_payload_for_records(self, pair_records: list[dict[str, object]]) -> dict[str, object]:
        output_camera = self._output_camera_settings()
        observer, observer_time_payload = self._reference_payload_observer()
        camera = output_camera
        view = self._view_settings()
        mag_limit = float(self.ui.doubleSpinBoxMagLimit.value())
        horizontal_catalog = self._get_horizontal_catalog(observer, mag_limit)
        horizontal_milky_way = self._get_horizontal_milky_way(observer)
        horizontal_solar_system = self._get_horizontal_solar_system(observer)
        star_map = project_horizontal_catalog(
            horizontal_catalog=horizontal_catalog,
            camera=camera,
            view=view,
            visible_mag_limit=mag_limit,
            horizontal_milky_way=horizontal_milky_way,
            horizontal_solar_system=horizontal_solar_system,
            star_color_mag_limit=self.ui_config.star_color_mag_limit,
        )
        reference_stars = self._select_current_reference_stars(star_map)
        reference_stars = self._reference_stars_with_pair_records(reference_stars, pair_records, observer)
        payload = build_reference_payload(
            star_map=star_map,
            reference_stars=reference_stars,
            observer=observer,
            camera=camera,
            view=view,
            visible_mag_limit=mag_limit,
            utc_offset_hours=self.ui.doubleSpinBoxUtcOffset.value(),
            reference_label_mode=self._reference_label_mode(),
            reference_mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            manual_reference_star_ids=tuple(self._manual_reference_star_ids),
        )
        observer_payload = payload.get("observer")
        if isinstance(observer_payload, dict):
            observer_payload.update(observer_time_payload)
        return payload

    def _record_reference_star(
        self,
        record: object,
        *,
        observer: ObserverSettings | None = None,
        output_index: int = 0,
    ) -> ReferenceStar | None:
        return reference_star_from_pair_payload(record, observer=observer, output_index=output_index)

    def _reference_star_lookup_from_records(
        self,
        records: object,
        *,
        observer: ObserverSettings | None = None,
    ) -> dict[str, ReferenceStar]:
        lookup: dict[str, ReferenceStar] = {}
        if not isinstance(records, list):
            return lookup
        for record in records:
            if isinstance(record, StarPairRecord):
                star = record.reference_star
            else:
                star = self._record_reference_star(record, observer=observer)
            if star is not None and star.star_id not in lookup:
                lookup[star.star_id] = star
        return lookup

    def _reference_stars_with_pair_records(
        self,
        reference_stars: tuple[ReferenceStar, ...],
        pair_records: list[dict[str, object]],
        observer: ObserverSettings,
    ) -> tuple[ReferenceStar, ...]:
        merged: list[ReferenceStar] = list(reference_stars)
        seen_star_ids = {star.star_id.strip() for star in merged if star.star_id.strip()}
        pair_lookup = self._reference_star_lookup_from_records(pair_records, observer=observer)
        for pair_record in pair_records:
            star_id = pair_record.star_id if isinstance(pair_record, StarPairRecord) else self._session_pair_star_id(pair_record)
            if not star_id or star_id in seen_star_ids:
                continue
            fallback_star = pair_lookup.get(star_id)
            if fallback_star is None:
                continue
            seen_star_ids.add(star_id)
            merged.append(fallback_star)
        return tuple(self._reference_star_with_index(star, index) for index, star in enumerate(merged, start=1))

    def _merge_imported_reference_stars_from_pairs(self, pair_payloads: list[object]) -> None:
        imported_lookup = getattr(self, "_imported_reference_star_by_id", {})
        if not isinstance(imported_lookup, dict):
            imported_lookup = {}
        merged_lookup: dict[str, ReferenceStar] = dict(imported_lookup)
        pair_lookup = self._reference_star_lookup_from_records(pair_payloads, observer=self._observer_settings())
        for star_id, star in pair_lookup.items():
            merged_lookup.setdefault(star_id, star)
        self._imported_reference_star_by_id = merged_lookup
