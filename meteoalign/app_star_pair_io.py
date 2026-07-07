from __future__ import annotations
from .app_constants import *

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QDateTime, Qt, QThread, QTimer
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QInputDialog, QMessageBox, QProgressDialog,
    QTableWidgetItem,
)

from .app_utils import _relative_image_path_for_session, _resolve_star_pair_session_real_image_path
from .app_workers import StarPairSessionImportWorker, ReferenceJsonImportWorker
from .catalog import project_root
from .fixed_camera_model import (
    FixedCameraModel,
    FixedCameraTimeFitResult,
    estimate_frame_time_correction,
)
from .image_preview import load_image_preview, ImagePreview
from .image_sequence import ImageSequenceItem, read_image_capture_time, sequence_item_observation_time_utc
from .mapping_validation import MappingValidationDialog
from .reference import build_reference_payload
from .simulator import ObserverSettings, ReferenceStar, compute_altaz_from_radec, project_horizontal_catalog
from .alignment import MIN_ALIGNMENT_PAIRS

from .app_constants import (
    STAR_PAIR_SESSION_FORMAT, STAR_PAIR_SESSION_VERSION,
    STAR_PAIR_SESSION_JSON_FILTER,
    STAR_PAIR_INDEX_COLUMN, STAR_PAIR_FIT_ROLE, STAR_PAIR_POSITION_ROLE,
    STAR_PAIR_CONSTRAINT_MODE_ROLE, STAR_PAIR_FIT_WEIGHT_ROLE,
    STAR_PAIR_ROW_TYPE_AUTO_MATCH, STAR_PAIR_ROW_TYPE_MANUAL,
    AUTO_MATCH_CONSTRAINT_ANCHOR, AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_CONSTRAINT_MODES, REFERENCE_LABEL_MODE_FIXED_COUNT,
    REFERENCE_LABEL_MODES, RECTILINEAR_LENS_MODEL, LENS_MODELS,
)


class StarPairIOMixin:
    """星对数据导入导出 Mixin：星对会话 JSON、源模型 JSON、参考 JSON。"""

    ui: object
    _json_import_thread: QThread | None
    _json_import_worker: object | None
    _json_import_progress: QProgressDialog | None
    _star_pair_session_import_switch_to_reference: bool
    _star_pair_session_import_clear_input_name: str
    _alignment_model: object  # method
    _set_alignment_model: object  # method
    _auto_match_constraint_for_star_id: object  # method
    _auto_match_group_label: object  # method
    _ensure_auto_match_group: object  # method
    _normalize_auto_match_groups: object  # method
    _normalized_auto_match_constraint: object  # method
    _auto_match_constraint_mode: object  # method
    _auto_match_soft_weight: object  # method
    _reference_label_mode: object  # method
    _observer_settings: object  # method
    _output_camera_settings: object  # method
    _camera_settings_for_image_size: object  # method
    _build_projected_star_map: object  # method
    _view_settings: object  # method
    _get_horizontal_catalog: object  # method
    _get_horizontal_milky_way: object  # method
    _get_horizontal_solar_system: object  # method
    _select_current_reference_stars: object  # method
    _reference_star_with_index: object  # method
    _lens_model: object  # method
    _update_lens_model_controls: object  # method
    _update_reference_label_controls: object  # method
    _is_auto_match_row: object  # method
    _is_star_pair_group_row: object  # method
    _star_pair_star_id: object  # method
    _star_pair_fit_constraint: object  # method
    _star_pair_alignment_residual: object  # method
    _parse_star_pair_position_text: object  # method
    _star_pair_position_count: object  # method
    _sequence_base_templates: object  # method
    _fit_sequence_fixed_camera_model: object  # method
    _first_frame_matched_pairs: object  # method
    _sequence_pair_fit_arrays: object  # method
    _apply_sequence_time_fit: object  # method
    _sequence_pair_records: object  # method
    _collect_star_pair_states: object  # method
    _clear_star_pair_positions: object  # method
    _clear_star_pair_annotations: object  # method
    _refresh_star_pair_table_styles: object  # method
    _refresh_reference_stars_from_current_map: object  # method
    _update_reference_alignment_transform: object  # method
    _update_reference_alignment_controls: object  # method
    _set_json_import_controls_enabled: object  # method
    _row_auto_match_group_id: object  # method
    _auto_match_reference_star_ids: list
    _auto_match_constraint_by_star_id: dict
    _auto_match_group_order: list
    _auto_match_group_by_star_id: dict
    _auto_match_group_expanded_by_id: dict
    _auto_match_next_group_index: int
    _manual_reference_star_ids: list
    _imported_reference_star_by_id: dict
    _excluded_reference_star_ids: list
    _current_star_map: object | None
    _sky_alignment_transform: object | None
    _source_astrometric_model: object | None
    _source_model_error_message: str
    _sky_alignment_error_message: str
    _reference_alignment_error_message: str
    _mapping_validation_dialog: object | None
    current_image_preview: object | None
    current_sky_mask: np.ndarray | None
    current_sky_mask_path: Path | None

    def _clear_star_pair_positions_for_new_input(self, input_name: str) -> int:
        pair_count = self._star_pair_position_count()
        if pair_count <= 0:
            return 0
        self._clear_star_pair_positions()
        self.ui.statusbar.showMessage(f"导入{input_name}前已清除 {pair_count} 个已有匹配。")
        return pair_count

    def _show_json_import_progress(
        self,
        title: str,
        label_text: str,
        status_text: str,
    ) -> QProgressDialog:
        dialog = QProgressDialog(self)
        dialog.setWindowTitle(title)
        dialog.setLabelText(label_text)
        dialog.setRange(0, 0)
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.show()
        self.ui.statusbar.showMessage(status_text)
        QApplication.processEvents()
        return dialog

    def _cleanup_json_import(self) -> None:
        if self._json_import_progress is not None:
            self._json_import_progress.close()
        self._json_import_thread = None
        self._json_import_worker = None
        self._json_import_progress = None
        self._star_pair_session_import_switch_to_reference = True
        self._star_pair_session_import_clear_input_name = "新的配对 JSON"
        self._set_json_import_controls_enabled(True)
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()

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

    def _record_float(self, record: dict[str, object], key: str, default_value: float = float("nan")) -> float:
        try:
            value = float(record[key])
        except (KeyError, TypeError, ValueError):
            return default_value
        return value if math.isfinite(value) else default_value

    def _record_reference_star(
        self,
        record: object,
        *,
        observer: ObserverSettings | None = None,
        output_index: int = 0,
    ) -> ReferenceStar | None:
        if not isinstance(record, dict):
            return None
        star_id = self._session_pair_star_id(record)
        if not star_id:
            return None
        ra_deg = self._record_float(record, "ra_deg")
        dec_deg = self._record_float(record, "dec_deg")
        if not math.isfinite(ra_deg) or not math.isfinite(dec_deg):
            return None

        alt_deg = self._record_float(record, "alt_deg")
        az_deg = self._record_float(record, "az_deg")
        if observer is not None and (not math.isfinite(alt_deg) or not math.isfinite(az_deg)):
            try:
                alt_values, az_values = compute_altaz_from_radec(
                    np.asarray([ra_deg], dtype=np.float64),
                    np.asarray([dec_deg], dtype=np.float64),
                    observer,
                )
                alt_deg = float(alt_values[0])
                az_deg = float(az_values[0])
            except Exception:  # noqa: BLE001 - 兜底参考星缺少地平坐标时不应阻断旧 JSON 导入。
                pass

        display_name = str(record.get("display_name") or "").strip()
        common_name = str(record.get("common_name") or "").strip()
        name = str(record.get("name") or common_name or display_name or star_id).strip()
        sim_x = self._record_float(
            record,
            "sim_x",
            self._record_float(record, "theoretical_x_px", self._record_float(record, "image_x_px")),
        )
        sim_y = self._record_float(
            record,
            "sim_y",
            self._record_float(record, "theoretical_y_px", self._record_float(record, "image_y_px")),
        )
        return ReferenceStar(
            index=output_index,
            star_id=star_id,
            name=name or star_id,
            display_name=display_name or name or star_id,
            common_name=common_name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            mag_v=self._record_float(record, "mag_v"),
            sim_x=sim_x,
            sim_y=sim_y,
            alt_deg=alt_deg,
            az_deg=az_deg,
            object_type="star",
            index_label=str(record.get("index_label") or ""),
        )

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
            star_id = self._session_pair_star_id(pair_record)
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

    def _star_pair_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            reference_star = self._reference_star_for_row(row)
            target_position = self._parse_star_pair_position_text(row)
            if reference_star is None or target_position is None:
                continue
            if not self._is_catalog_reference_star(reference_star):
                continue

            image_x, image_y = target_position
            if not all(math.isfinite(value) for value in (image_x, image_y, reference_star.ra_deg, reference_star.dec_deg)):
                continue

            record: dict[str, object] = {
                "reference_index": reference_star.index,
                "star_id": reference_star.star_id,
                "name": reference_star.name,
                "display_name": reference_star.display_name,
                "common_name": reference_star.common_name,
                "ra_deg": reference_star.ra_deg,
                "dec_deg": reference_star.dec_deg,
                "mag_v": reference_star.mag_v,
                "image_x_px": image_x,
                "image_y_px": image_y,
                "sim_x": reference_star.sim_x,
                "sim_y": reference_star.sim_y,
                "alt_deg": reference_star.alt_deg,
                "az_deg": reference_star.az_deg,
                "object_type": "star",
                "pair_origin": "auto_match" if self._is_auto_match_row(row) else "manual",
            }
            if self._is_auto_match_row(row):
                group_id = self._row_auto_match_group_id(row) or self._auto_match_group_by_star_id.get(reference_star.star_id, "")
                if group_id:
                    record["auto_match_group_id"] = group_id
                    record["auto_match_group_name"] = self._auto_match_group_label(group_id)
            constraint_mode, fit_weight = self._star_pair_fit_constraint(row)
            record["fit_constraint_mode"] = constraint_mode
            record["fit_weight"] = fit_weight
            fit_payload = self._star_pair_fit_payload(row)
            if fit_payload is not None:
                for key in ("amplitude", "background", "sigma_x", "sigma_y"):
                    if key in fit_payload:
                        record[key] = fit_payload[key]
            residual = self._star_pair_alignment_residual(row)
            if residual is not None:
                dx, dy, distance = residual
                record["residual_dx_px"] = dx
                record["residual_dy_px"] = dy
                record["residual_px"] = distance
            records.append(record)
        return records

    def _auto_match_constraints_payload(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for star_id in self._auto_match_reference_star_ids:
            mode, fit_weight = self._auto_match_constraint_for_star_id(star_id)
            payload[star_id] = {
                "fit_constraint_mode": mode,
                "fit_weight": fit_weight,
            }
        return payload

    def _auto_match_groups_payload(self) -> list[dict[str, object]]:
        self._normalize_auto_match_groups()
        groups: list[dict[str, object]] = []
        for group_id in self._auto_match_group_order:
            star_ids = self._auto_match_group_star_ids(group_id)
            if not star_ids:
                continue
            groups.append(
                {
                    "group_id": group_id,
                    "name": self._auto_match_group_label(group_id),
                    "star_ids": star_ids,
                    "expanded": bool(self._auto_match_group_expanded_by_id.get(group_id, True)),
                }
            )
        return groups

    def _default_star_pair_session_path(self) -> Path:
        if self.current_image_preview is not None:
            return self._star_pair_session_path_for_image(Path(self.current_image_preview.path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return project_root() / "outputs" / f"star_pairs_{timestamp}.json"

    def _star_pair_session_path_for_image(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_starpairs.json")

    def _source_model_path_for_image(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_model.json")

    def _existing_pair_count_from_json(self, json_path: Path, *, model_json: bool = False) -> int | None:
        if not json_path.exists():
            return None
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 已有文件可能不是本程序生成的 JSON，覆盖提示里只做保守兜底。
            return None
        if not isinstance(payload, dict):
            return None
        if model_json:
            diagnostics = payload.get("diagnostics")
            if isinstance(diagnostics, dict):
                try:
                    return int(diagnostics["pair_count"])
                except (KeyError, TypeError, ValueError):
                    pass
            fit_pairs = payload.get("fit_pairs")
            return len(fit_pairs) if isinstance(fit_pairs, list) else None
        try:
            return int(payload["pair_count"])
        except (KeyError, TypeError, ValueError):
            pairs = payload.get("pairs")
            return len(pairs) if isinstance(pairs, list) else None

    def _confirm_overwrite_if_existing_has_more_pairs(
        self,
        json_path: Path,
        current_pair_count: int,
        *,
        model_json: bool = False,
    ) -> bool:
        existing_pair_count = self._existing_pair_count_from_json(json_path, model_json=model_json)
        if existing_pair_count is None or existing_pair_count <= int(current_pair_count):
            return True
        reply = QMessageBox.question(
            self,
            "已有 JSON 配对更多",
            (
                f"已有文件包含 {existing_pair_count} 个配对，当前只有 {current_pair_count} 个配对。\n"
                f"继续会覆盖：{json_path}\n\n是否仍然覆盖？"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _maybe_auto_import_star_pair_session_for_image(self, image_path: Path) -> None:
        if self._json_import_thread is not None:
            return
        json_path = self._star_pair_session_path_for_image(image_path)
        if not json_path.exists():
            return
        self.ui.statusbar.showMessage(f"发现同名配对 JSON，正在自动导入: {json_path}")
        self.load_star_pair_session(json_path)

    def _build_star_pair_session_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像，再导出星点配对 JSON。")

        preview = self.current_image_preview
        image_path = Path(preview.path).expanduser().resolve()
        relative_image_path = _relative_image_path_for_session(image_path, json_path)
        pair_records = self._star_pair_records()
        image_model = "manual_star_pair_session"
        try:
            fixed_bundle = self._single_image_fixed_camera_export_bundle()
        except Exception:  # noqa: BLE001 - 配对 JSON 需要支持中途保存，固定模型未就绪时保留普通配对记录。
            reference_payload = self._build_reference_payload_for_records(pair_records)
        else:
            pair_records = fixed_bundle["fit_pairs"]
            reference_payload = fixed_bundle["reference_payload"]
            image_model = "fixed_camera_model"
        generated_time = datetime.now(timezone.utc)
        real_image_payload = {
            "path": str(image_path),
            "relative_path": relative_image_path,
            "file_name": image_path.name,
            "original_width_px": preview.original_width,
            "original_height_px": preview.original_height,
            "display_width_px": preview.image.width(),
            "display_height_px": preview.image.height(),
        }
        real_image_payload.update(self._current_real_image_capture_payload())
        return {
            "format": STAR_PAIR_SESSION_FORMAT,
            "version": STAR_PAIR_SESSION_VERSION,
            "generated_at_utc": generated_time.isoformat(),
            "real_image": real_image_payload,
            "reference_payload": reference_payload,
            "sky_alignment_model": self._alignment_model(),
            "image_model": image_model,
            "auto_match_star_ids": list(self._auto_match_reference_star_ids),
            "auto_match_groups": self._auto_match_groups_payload(),
            "auto_match_constraints": self._auto_match_constraints_payload(),
            "pair_count": len(pair_records),
            "pairs": pair_records,
            "mask": self._sky_mask_payload(json_path),
            "matching": self._auto_match_settings_payload(),
        }

    def export_star_pair_session(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导出星点配对 JSON。")
            return

        default_path = self._default_star_pair_session_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = default_path
        try:
            payload = self._build_star_pair_session_payload(json_path)
            pair_count = int(payload.get("pair_count", 0))
            if not self._confirm_overwrite_if_existing_has_more_pairs(json_path, pair_count):
                self.ui.statusbar.showMessage("已取消导出星点配对 JSON。")
                return
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.ui.statusbar.showMessage(f"已导出星点配对 JSON: {json_path}  配对数: {pair_count}")
            QMessageBox.information(self, "配对 JSON 已导出", f"JSON：{json_path}\n配对数：{pair_count}")
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把文件和字段错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导出星点配对 JSON 失败: {exc}")
            QMessageBox.critical(self, "导出星点配对 JSON 失败", str(exc))

    def _default_source_model_path(self) -> Path:
        if self.current_image_preview is not None:
            return self._source_model_path_for_image(Path(self.current_image_preview.path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return project_root() / "outputs" / f"source_model_{timestamp}.json"

    def _source_image_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像。")
        preview = self.current_image_preview
        image_path = Path(preview.path).expanduser().resolve()
        payload = {
            "path": str(image_path),
            "relative_path": _relative_image_path_for_session(image_path, json_path),
            "file_name": image_path.name,
            "original_width_px": preview.original_width,
            "original_height_px": preview.original_height,
            "model_width_px": preview.image.width(),
            "model_height_px": preview.image.height(),
        }
        payload.update(self._current_real_image_capture_payload())
        return payload

    def _sky_mask_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_sky_mask is None:
            return {"active": False}
        mask_height, mask_width = self.current_sky_mask.shape
        payload: dict[str, object] = {}
        if self.current_sky_mask_path is not None:
            mask_path = self.current_sky_mask_path.expanduser().resolve()
            payload["path"] = str(mask_path)
            payload["relative_path"] = _relative_image_path_for_session(mask_path, json_path)
        payload.update(
            {
                "active": True,
                "width_px": int(mask_width),
                "height_px": int(mask_height),
                "valid_fraction": float(np.count_nonzero(self.current_sky_mask))
                / max(float(self.current_sky_mask.size), 1.0),
                "zero_pixels_excluded": True,
            }
        )
        return payload

    def _auto_match_settings_payload(self) -> dict[str, object]:
        return {
            "sky_alignment_model": self._alignment_model(),
            "new_star_count": int(self.ui.spinBoxAutoMatchCount.value()),
            "new_constraint_mode": self._auto_match_constraint_mode(),
            "soft_constraint_weight": float(self.ui.doubleSpinBoxAutoMatchSoftWeight.value()),
            "search_radius_px": int(self.ui.spinBoxAutoMatchRadius.value()),
            "mask_enabled": self.current_sky_mask is not None,
        }

    def _current_source_model(self) -> SourceAstrometricModel:
        if self._source_astrometric_model is None:
            self._update_reference_alignment_transform()
        if self._source_astrometric_model is None:
            raise ValueError(self._source_model_error_message or f"至少需要 {MIN_ALIGNMENT_PAIRS} 对星点才能导出映射。")
        return self._source_astrometric_model

    def _single_image_time_fit_for_pairs(
        self,
        pairs: list[object],
        fixed_model: FixedCameraModel,
        observer: ObserverSettings,
    ) -> FixedCameraTimeFitResult:
        ra_dec_points, pixel_points, point_weights = self._sequence_pair_fit_arrays(pairs)
        return estimate_frame_time_correction(
            fixed_model=fixed_model,
            ra_dec_points=ra_dec_points,
            observed_pixels=pixel_points,
            nominal_time_utc=observer.observation_time_utc,
            latitude_deg=observer.latitude_deg,
            longitude_deg=observer.longitude_deg,
            elevation_m=observer.elevation_m,
            initial_delta_seconds=0.0,
            point_weights=point_weights,
            max_iterations=0,
        )

    def _single_image_dynamic_sky_payload(
        self,
        observer: ObserverSettings,
        observer_time_payload: dict[str, object],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "observation_time_utc": observer.observation_time_utc.astimezone(timezone.utc).isoformat(),
            "latitude_deg": observer.latitude_deg,
            "longitude_deg": observer.longitude_deg,
            "elevation_m": observer.elevation_m,
            "utc_offset_hours": self.ui.doubleSpinBoxUtcOffset.value(),
        }
        payload.update(observer_time_payload)
        return payload

    def _single_image_fixed_camera_diagnostics(
        self,
        records: list[dict[str, object]],
        time_fit: FixedCameraTimeFitResult,
    ) -> dict[str, object]:
        return {
            "pair_count": len(records),
            "accepted_count": time_fit.accepted_count,
            "inlier_ratio": float(time_fit.inlier_ratio),
            "rms_px": float(time_fit.rms_px),
            "median_residual_px": float(time_fit.median_residual_px),
            "max_residual_px": float(time_fit.max_residual_px),
            "projected_motion_px_per_s_median": float(time_fit.projected_motion_px_per_s_median),
        }

    def _single_image_fixed_camera_export_bundle(self) -> dict[str, object]:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像。")
        if self._sky_alignment_transform is None:
            self._update_reference_alignment_transform()
        if self._sky_alignment_transform is None:
            raise ValueError(
                self._sky_alignment_error_message
                or self._reference_alignment_error_message
                or f"至少需要 {MIN_ALIGNMENT_PAIRS} 对星点才能导出映射。"
            )

        preview = self.current_image_preview
        templates = self._sequence_base_templates()
        target_size = (preview.image.width(), preview.image.height())
        fixed_model = self._fit_sequence_fixed_camera_model(templates, target_size)
        observer, observer_time_payload = self._reference_payload_observer()
        pairs = self._first_frame_matched_pairs(templates)
        time_fit = self._single_image_time_fit_for_pairs(pairs, fixed_model, observer)
        pairs = self._apply_sequence_time_fit(pairs, time_fit, require_accepted=False)
        pairs = [replace(pair, time_delta_seconds=None) for pair in pairs]
        fit_pairs = self._sequence_pair_records(pairs)
        reference_payload = self._build_reference_payload_for_records(fit_pairs)
        return {
            "fixed_model": fixed_model,
            "time_fit": time_fit,
            "fit_pairs": fit_pairs,
            "reference_payload": reference_payload,
            "observer": observer,
            "observer_time_payload": observer_time_payload,
        }

    def _build_source_model_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像。")
        source_model = self._current_source_model()
        fit_pairs = self._star_pair_records()
        reference_payload = self._build_reference_payload_for_records(fit_pairs)
        observer, observer_time_payload = self._reference_payload_observer()
        frame_model = source_model.to_frame_astrometric_model(
            fit_metadata={
                "scene_observer_hint": {
                    "observation_time_utc": observer.observation_time_utc.astimezone(timezone.utc).isoformat(),
                    "latitude_deg": float(observer.latitude_deg),
                    "longitude_deg": float(observer.longitude_deg),
                    "elevation_m": float(observer.elevation_m),
                    "utc_offset_hours": float(self.ui.doubleSpinBoxUtcOffset.value()),
                    **observer_time_payload,
                },
                "scene_observer_hint_role": "metadata_only_not_required_for_pixel_icrs_model",
            }
        )
        return frame_model.to_json_payload(
            source_image=self._source_image_payload(json_path),
            mask=self._sky_mask_payload(json_path),
            matching=self._auto_match_settings_payload(),
            fit_pairs=fit_pairs,
            reference_payload=reference_payload,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def export_source_model_json(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导出 xy→RA/Dec 映射 JSON。")
            return

        default_path = self._default_source_model_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = default_path
        try:
            payload = self._build_source_model_payload(json_path)
            diagnostics = payload.get("diagnostics", {})
            pair_count = int(diagnostics.get("pair_count", 0)) if isinstance(diagnostics, dict) else 0
            rms_px = float(diagnostics.get("rms_px", float("nan"))) if isinstance(diagnostics, dict) else float("nan")
            if not self._confirm_overwrite_if_existing_has_more_pairs(json_path, pair_count, model_json=True):
                self.ui.statusbar.showMessage("已取消导出 xy→RA/Dec 映射 JSON。")
                return
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.ui.statusbar.showMessage(
                f"已导出 xy→RA/Dec 映射 JSON: {json_path}  配对数: {pair_count}  RMS: {rms_px:.2f}px"
            )
            QMessageBox.information(
                self,
                "映射 JSON 已导出",
                f"JSON：{json_path}\n配对数：{pair_count}\nRMS：{rms_px:.2f} px",
            )
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把模型生成与文件错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导出 xy→RA/Dec 映射 JSON 失败: {exc}")
            QMessageBox.critical(self, "导出 xy→RA/Dec 映射 JSON 失败", str(exc))

    def show_mapping_validation_dialog(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再进行映射验证。")
            return

        try:
            model = self._current_source_model().to_frame_astrometric_model()
        except Exception as exc:  # noqa: BLE001 - 验证入口需要把模型未就绪原因直接反馈给用户。
            QMessageBox.information(self, "映射尚未就绪", str(exc))
            return

        old_dialog = getattr(self, "_mapping_validation_dialog", None)
        if old_dialog is not None:
            try:
                if old_dialog.isVisible():
                    old_dialog.raise_()
                    old_dialog.activateWindow()
                    return
            except RuntimeError:
                pass

        observer = self._observer_settings()
        base_camera = self._output_camera_settings()
        initial_view = self._view_settings()
        visible_mag_limit = float(self.ui.doubleSpinBoxMagLimit.value())
        dialog = MappingValidationDialog(
            parent=self,
            renderer=self.renderer,
            model=model,
            source_image=self.current_image_preview.image,
            observer=observer,
            base_camera=base_camera,
            initial_view=initial_view,
            visible_mag_limit=visible_mag_limit,
            horizontal_catalog=self._get_horizontal_catalog(observer, visible_mag_limit),
            horizontal_milky_way=self._get_horizontal_milky_way(observer),
            horizontal_solar_system=self._get_horizontal_solar_system(observer),
            ui_config=self.ui_config,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda _obj=None: setattr(self, "_mapping_validation_dialog", None))
        parent_size = self.size()
        dialog.resize(max(720, int(parent_size.width() * 0.88)), max(520, int(parent_size.height() * 0.88)))
        dialog_geometry = dialog.frameGeometry()
        dialog_geometry.moveCenter(self.geometry().center())
        dialog.move(dialog_geometry.topLeft())
        self._mapping_validation_dialog = dialog
        dialog.show()

    def import_star_pair_session(self) -> None:
        default_dir = project_root() / "outputs"
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入星点配对 JSON",
            str(default_dir),
            STAR_PAIR_SESSION_JSON_FILTER,
        )
        if not file_path:
            return
        self.load_star_pair_session(file_path)

    def load_star_pair_session(
        self,
        file_path: str | Path,
        *,
        switch_to_reference: bool = True,
        show_progress: bool = True,
        clear_input_name: str = "新的配对 JSON",
    ) -> None:
        if self._json_import_thread is not None:
            QMessageBox.information(self, "正在导入 JSON", "当前已有 JSON 正在导入，请稍候。")
            return
        json_path = Path(file_path)
        self._set_json_import_controls_enabled(False)
        self._star_pair_session_import_switch_to_reference = bool(switch_to_reference)
        self._star_pair_session_import_clear_input_name = clear_input_name
        if show_progress:
            self._json_import_progress = self._show_json_import_progress(
                title="正在导入配对 JSON",
                label_text=f"正在读取配对 JSON 并恢复真实图像...\n{json_path}",
                status_text=f"正在导入星点配对 JSON: {json_path}",
            )
        else:
            self._json_import_progress = None
            self.ui.statusbar.showMessage(f"正在后台导入星点配对 JSON: {json_path}")

        thread = QThread(self)
        worker = StarPairSessionImportWorker(json_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_star_pair_session_import_finished)
        worker.failed.connect(self._handle_star_pair_session_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_json_import)

        self._json_import_thread = thread
        self._json_import_worker = worker
        thread.start()

    def _session_pair_star_id(self, pair_payload: object) -> str:
        if not isinstance(pair_payload, dict):
            return ""
        object_type = str(pair_payload.get("object_type", "star")).strip()
        if object_type != "star":
            return ""
        star_id = str(pair_payload.get("star_id", "")).strip()
        if not star_id or star_id.startswith("solar_system:"):
            return ""
        return star_id

    def _session_pair_position(self, pair_payload: object) -> tuple[float, float] | None:
        if not isinstance(pair_payload, dict):
            return None
        try:
            image_x = float(pair_payload["image_x_px"])
            image_y = float(pair_payload["image_y_px"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(image_x) or not math.isfinite(image_y):
            return None
        return image_x, image_y

    def _session_pair_fit_payload(self, pair_payload: object) -> dict[str, float] | None:
        if not isinstance(pair_payload, dict):
            return None

        fit_payload: dict[str, float] = {}
        for key in ("amplitude", "background", "sigma_x", "sigma_y"):
            try:
                value = float(pair_payload[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                fit_payload[key] = value
        return fit_payload or None

    def _session_auto_match_star_ids(self, payload: dict[str, object], pair_payloads: list[object]) -> list[str]:
        auto_match_star_ids: list[str] = []
        raw_star_ids = payload.get("auto_match_star_ids")
        if isinstance(raw_star_ids, list):
            for raw_star_id in raw_star_ids:
                star_id = str(raw_star_id).strip()
                if star_id and star_id not in auto_match_star_ids:
                    auto_match_star_ids.append(star_id)

        for pair_payload in pair_payloads:
            if not isinstance(pair_payload, dict):
                continue
            if str(pair_payload.get("pair_origin", "")).strip() != "auto_match":
                continue
            star_id = self._session_pair_star_id(pair_payload)
            if star_id and star_id not in auto_match_star_ids:
                auto_match_star_ids.append(star_id)
        return auto_match_star_ids

    def _session_auto_match_pair_group_id(self, pair_payload: object) -> str:
        if not isinstance(pair_payload, dict):
            return ""
        group_id = str(pair_payload.get("auto_match_group_id", "")).strip()
        if group_id:
            return group_id
        group_name = str(pair_payload.get("auto_match_group_name", "")).strip()
        if group_name.startswith("自动") and len(group_name) >= 3:
            return group_name[2:].strip()
        return ""

    def _session_auto_match_groups(
        self,
        payload: dict[str, object],
        pair_payloads: list[object],
        auto_match_star_ids: list[str],
    ) -> tuple[list[str], dict[str, str], dict[str, bool], int]:
        group_order: list[str] = []
        group_by_star_id: dict[str, str] = {}
        expanded_by_group_id: dict[str, bool] = {}
        auto_star_set = set(auto_match_star_ids)

        raw_groups = payload.get("auto_match_groups")
        if isinstance(raw_groups, list):
            for raw_group in raw_groups:
                if not isinstance(raw_group, dict):
                    continue
                group_id = str(raw_group.get("group_id", "")).strip()
                if not group_id:
                    group_id = str(raw_group.get("name", "")).replace("自动", "", 1).strip()
                if not group_id:
                    continue
                if group_id not in group_order:
                    group_order.append(group_id)
                expanded_by_group_id[group_id] = bool(raw_group.get("expanded", True))
                raw_star_ids = raw_group.get("star_ids", [])
                if not isinstance(raw_star_ids, list):
                    continue
                for raw_star_id in raw_star_ids:
                    star_id = str(raw_star_id).strip()
                    if star_id in auto_star_set:
                        group_by_star_id[star_id] = group_id

        for pair_payload in pair_payloads:
            if not isinstance(pair_payload, dict):
                continue
            if str(pair_payload.get("pair_origin", "")).strip() != "auto_match":
                continue
            star_id = self._session_pair_star_id(pair_payload)
            if not star_id or star_id not in auto_star_set:
                continue
            group_id = self._session_auto_match_pair_group_id(pair_payload)
            if not group_id:
                continue
            if group_id not in group_order:
                group_order.append(group_id)
            expanded_by_group_id.setdefault(group_id, True)
            group_by_star_id.setdefault(star_id, group_id)

        if not group_order and auto_match_star_ids:
            group_order.append("A")
            expanded_by_group_id["A"] = True
        fallback_group_id = group_order[0] if group_order else "A"
        for star_id in auto_match_star_ids:
            group_by_star_id.setdefault(star_id, fallback_group_id)

        next_group_index = 0
        for group_id in group_order:
            if len(group_id) == 1 and "A" <= group_id <= "Z":
                next_group_index = max(next_group_index, ord(group_id) - ord("A") + 1)
        return group_order, group_by_star_id, expanded_by_group_id, next_group_index

    def _normalized_auto_match_constraint(self, raw_mode: object, raw_weight: object) -> tuple[str, float]:
        mode = str(raw_mode or AUTO_MATCH_CONSTRAINT_ANCHOR).strip()
        if mode not in AUTO_MATCH_CONSTRAINT_MODES:
            mode = AUTO_MATCH_CONSTRAINT_ANCHOR
        try:
            fit_weight = float(raw_weight)
        except (TypeError, ValueError):
            fit_weight = 1.0
        if mode == AUTO_MATCH_CONSTRAINT_SOFT:
            fit_weight = max(0.01, min(1.0, fit_weight))
        else:
            fit_weight = 1.0
        return mode, fit_weight

    def _session_auto_match_constraints(
        self,
        payload: dict[str, object],
        pair_payloads: list[object],
    ) -> dict[str, tuple[str, float]]:
        constraints: dict[str, tuple[str, float]] = {}
        raw_constraints = payload.get("auto_match_constraints")
        if isinstance(raw_constraints, dict):
            for raw_star_id, raw_constraint in raw_constraints.items():
                star_id = str(raw_star_id).strip()
                if not star_id or not isinstance(raw_constraint, dict):
                    continue
                constraints[star_id] = self._normalized_auto_match_constraint(
                    raw_constraint.get("fit_constraint_mode"),
                    raw_constraint.get("fit_weight", 1.0),
                )

        for pair_payload in pair_payloads:
            if not isinstance(pair_payload, dict):
                continue
            if str(pair_payload.get("pair_origin", "")).strip() != "auto_match":
                continue
            star_id = self._session_pair_star_id(pair_payload)
            if not star_id:
                continue
            constraints[star_id] = self._normalized_auto_match_constraint(
                pair_payload.get("fit_constraint_mode"),
                pair_payload.get("fit_weight", 1.0),
            )
        return constraints

    def _ensure_pair_record_stars_visible(self, pair_payloads: list[object]) -> None:
        visible_star_ids = {
            self._star_pair_star_id(row)
            for row in range(self.ui.tableWidgetStarPairs.rowCount())
            if self._star_pair_star_id(row)
        }
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        added_any = False
        for pair_payload in pair_payloads:
            star_id = self._session_pair_star_id(pair_payload)
            if (
                not star_id
                or star_id in visible_star_ids
                or star_id in self._manual_reference_star_ids
                or star_id in auto_match_star_ids
            ):
                continue
            if star_id in self._excluded_reference_star_ids:
                self._excluded_reference_star_ids.remove(star_id)
            if isinstance(pair_payload, dict) and str(pair_payload.get("pair_origin", "")).strip() == "auto_match":
                group_id = (
                    self._session_auto_match_pair_group_id(pair_payload)
                    or self._auto_match_group_by_star_id.get(star_id, "")
                    or "A"
                )
                self._ensure_auto_match_group(group_id, expanded=True)
                self._auto_match_reference_star_ids.append(star_id)
                self._auto_match_group_by_star_id[star_id] = group_id
                auto_match_star_ids.add(star_id)
                self._auto_match_constraint_by_star_id[star_id] = self._normalized_auto_match_constraint(
                    pair_payload.get("fit_constraint_mode"),
                    pair_payload.get("fit_weight", 1.0),
                )
            else:
                self._manual_reference_star_ids.append(star_id)
            visible_star_ids.add(star_id)
            added_any = True
        if added_any:
            self._refresh_reference_stars_from_current_map()

    def _restore_star_pair_records(self, pair_payloads: list[object], update_alignment: bool = True) -> int:
        recorded_positions: dict[str, tuple[float, float]] = {}
        recorded_fit_payloads: dict[str, dict[str, float] | None] = {}
        recorded_constraints: dict[str, tuple[str, float]] = {}
        for pair_payload in pair_payloads:
            star_id = self._session_pair_star_id(pair_payload)
            position = self._session_pair_position(pair_payload)
            if star_id and position is not None:
                recorded_positions[star_id] = position
                fit_payload = self._session_pair_fit_payload(pair_payload)
                if fit_payload is not None:
                    fit_payload["x"] = position[0]
                    fit_payload["y"] = position[1]
                recorded_fit_payloads[star_id] = fit_payload
                if isinstance(pair_payload, dict):
                    recorded_constraints[star_id] = self._normalized_auto_match_constraint(
                        pair_payload.get("fit_constraint_mode"),
                        pair_payload.get("fit_weight", 1.0),
                    )

        table = self.ui.tableWidgetStarPairs
        signals_were_blocked = table.blockSignals(True)
        self._clear_star_pair_annotations()
        restored_count = 0
        for row in range(table.rowCount()):
            star_id = self._star_pair_star_id(row)
            position_item = table.item(row, STAR_PAIR_POSITION_COLUMN)
            if position_item is None:
                position_item = QTableWidgetItem()
                table.setItem(row, STAR_PAIR_POSITION_COLUMN, position_item)
            position_item.setData(Qt.UserRole, star_id)
            position = recorded_positions.get(star_id)
            if position is None:
                position_item.setText("")
                position_item.setData(STAR_PAIR_POSITION_ROLE, None)
                position_item.setData(STAR_PAIR_FIT_ROLE, None)
                continue
            image_x, image_y = position
            position_item.setData(STAR_PAIR_POSITION_ROLE, (float(image_x), float(image_y)))
            position_item.setData(STAR_PAIR_FIT_ROLE, recorded_fit_payloads.get(star_id))
            mode, fit_weight = recorded_constraints.get(star_id, self._star_pair_fit_constraint(row))
            self._set_star_pair_constraint(row, mode, fit_weight)
            restored_count += 1
        table.blockSignals(signals_were_blocked)

        self._restore_star_pair_annotations_from_table()
        self._refresh_star_pair_table_styles()
        if update_alignment:
            self._update_reference_alignment_transform()
        QTimer.singleShot(0, table.scrollToBottom)
        return restored_count

    def _session_real_image_path(self, payload: dict[str, object], source_path: Path) -> Path:
        return _resolve_star_pair_session_real_image_path(payload, source_path)

    def _handle_star_pair_session_import_finished(self, result: object) -> None:
        try:
            source_path, payload, preview = result  # type: ignore[misc]
            if not isinstance(source_path, Path):
                source_path = Path(source_path)
            self._clear_star_pair_positions_for_new_input(self._star_pair_session_import_clear_input_name)
            self._apply_star_pair_session_payload(
                payload,
                source_path,
                preview=preview,
                switch_to_reference=self._star_pair_session_import_switch_to_reference,
            )
        except Exception as exc:  # noqa: BLE001 - 主线程恢复界面时也需要把错误反馈给用户。
            self.ui.statusbar.showMessage(f"导入星点配对 JSON 失败: {exc}")
            QMessageBox.critical(self, "导入星点配对 JSON 失败", str(exc))

    def _handle_star_pair_session_import_failed(self, error_message: str) -> None:
        self.ui.statusbar.showMessage(f"导入星点配对 JSON 失败: {error_message}")
        QMessageBox.critical(self, "导入星点配对 JSON 失败", error_message)

    def _apply_star_pair_session_payload(
        self,
        payload: object,
        source_path: Path,
        preview: ImagePreview | None = None,
        *,
        switch_to_reference: bool = True,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("JSON 根对象必须是字典。")
        payload_format = payload.get("format")
        if payload_format != STAR_PAIR_SESSION_FORMAT:
            raise ValueError("当前只支持 MeteoAlign 星点配对 JSON。")

        reference_payload = payload.get("reference_payload")
        pair_payloads = payload.get("pairs", [])
        if not isinstance(pair_payloads, list):
            raise ValueError("JSON 中 pairs 字段必须是列表。")
        if not isinstance(reference_payload, dict):
            raise ValueError("JSON 缺少 reference_payload 字段。")
        auto_match_star_ids = self._session_auto_match_star_ids(payload, pair_payloads)
        auto_match_constraints = self._session_auto_match_constraints(payload, pair_payloads)
        (
            auto_match_group_order,
            auto_match_group_by_star_id,
            auto_match_group_expanded_by_id,
            auto_match_next_group_index,
        ) = self._session_auto_match_groups(payload, pair_payloads, auto_match_star_ids)

        image_path = self._session_real_image_path(payload, source_path)
        self._active_star_pair_row = None
        current_tab = self.ui.tabWidgetMain.currentWidget() if not switch_to_reference else None
        previous_suspend_alignment = self._suspend_alignment_updates
        self._suspend_alignment_updates = True
        try:
            self._set_alignment_model(payload.get("sky_alignment_model"))
            self._apply_reference_payload(reference_payload, source_path)
            if current_tab is not None:
                self.ui.tabWidgetMain.setCurrentWidget(current_tab)
            self._merge_imported_reference_stars_from_pairs(pair_payloads)
            self._auto_match_reference_star_ids = auto_match_star_ids
            self._auto_match_constraint_by_star_id = {
                star_id: auto_match_constraints.get(star_id, (AUTO_MATCH_CONSTRAINT_ANCHOR, 1.0))
                for star_id in auto_match_star_ids
            }
            self._auto_match_group_order = auto_match_group_order
            self._auto_match_group_by_star_id = auto_match_group_by_star_id
            self._auto_match_group_expanded_by_id = auto_match_group_expanded_by_id
            self._auto_match_next_group_index = auto_match_next_group_index
            self._normalize_auto_match_groups()
            self._refresh_reference_stars_from_current_map()
            self._ensure_pair_record_stars_visible(pair_payloads)
            if preview is None:
                preview = load_image_preview(image_path, max_long_side_px=None)
            self._apply_loaded_image_preview(
                preview,
                clear_existing_pairs=False,
                switch_to_reference=switch_to_reference,
            )
            restored_count = self._restore_star_pair_records(pair_payloads, update_alignment=False)
        finally:
            self._suspend_alignment_updates = previous_suspend_alignment
        self._update_reference_alignment_transform()
        if switch_to_reference:
            self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        elif current_tab is not None:
            self.ui.tabWidgetMain.setCurrentWidget(current_tab)
        self.ui.statusbar.showMessage(
            f"已导入星点配对 JSON: {source_path}  真实图像: {image_path}  恢复配对: {restored_count}"
        )

    def import_reference_json(self) -> None:
        default_dir = project_root() / "outputs"
        if not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入预览 JSON",
            str(default_dir),
            "MeteoAlign 参考图 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)",
        )
        if not file_path:
            return
        self.load_reference_json(file_path)

    def load_reference_json(self, file_path: str | Path) -> None:
        if self._json_import_thread is not None:
            QMessageBox.information(self, "正在导入 JSON", "当前已有 JSON 正在导入，请稍候。")
            return
        json_path = Path(file_path)
        self._set_json_import_controls_enabled(False)
        self._json_import_progress = self._show_json_import_progress(
            title="正在导入预览 JSON",
            label_text=f"正在读取预览 JSON 并恢复星空模拟参数...\n{json_path}",
            status_text=f"正在导入预览 JSON: {json_path}",
        )

        thread = QThread(self)
        worker = ReferenceJsonImportWorker(json_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_reference_json_import_finished)
        worker.failed.connect(self._handle_reference_json_import_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_json_import)

        self._json_import_thread = thread
        self._json_import_worker = worker
        thread.start()

    def _handle_reference_json_import_finished(self, result: object) -> None:
        try:
            source_path, payload = result  # type: ignore[misc]
            if not isinstance(source_path, Path):
                source_path = Path(source_path)
            self._clear_star_pair_positions_for_new_input("新的预览 JSON")
            self._apply_reference_payload(payload, source_path)
        except Exception as exc:  # noqa: BLE001 - 主线程恢复界面时也需要把错误反馈给用户。
            self.ui.statusbar.showMessage(f"导入预览 JSON 失败: {exc}")
            QMessageBox.critical(self, "导入预览 JSON 失败", str(exc))

    def _handle_reference_json_import_failed(self, error_message: str) -> None:
        self.ui.statusbar.showMessage(f"导入预览 JSON 失败: {error_message}")
        QMessageBox.critical(self, "导入预览 JSON 失败", error_message)

    def _payload_section(self, payload: dict[str, object], section_name: str) -> dict[str, object]:
        section = payload.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"JSON 缺少 {section_name} 字段。")
        return section

    def _payload_float(self, section: dict[str, object], key: str) -> float:
        try:
            return float(section[key])
        except KeyError as exc:
            raise ValueError(f"JSON 缺少 {key} 字段。") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效数字。") from exc

    def _payload_int(self, section: dict[str, object], key: str) -> int:
        try:
            return int(section[key])
        except KeyError as exc:
            raise ValueError(f"JSON 缺少 {key} 字段。") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效整数。") from exc

    def _payload_optional_float(self, section: dict[str, object], key: str, default_value: float) -> float:
        if key not in section or section.get(key) is None:
            return default_value
        try:
            return float(section[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效数字。") from exc

    def _payload_datetime_utc(self, section: dict[str, object], key: str) -> datetime:
        raw_value = section.get(key)
        if not isinstance(raw_value, str):
            raise ValueError(f"JSON 缺少 {key} 字段。")
        try:
            parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"JSON 中 {key} 字段不是有效 ISO 时间。") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _apply_reference_payload(self, payload: object, source_path: Path) -> None:
        if not isinstance(payload, dict):
            raise ValueError("JSON 根对象必须是字典。")
        if payload.get("format") != "meteoalign_phase1_reference":
            raise ValueError("当前只支持 MeteoAlign 导出的参考图 JSON。")

        observer = self._payload_section(payload, "observer")
        camera = self._payload_section(payload, "camera")
        view = self._payload_section(payload, "view")
        render = self._payload_section(payload, "render")

        observation_time_utc = self._payload_datetime_utc(observer, "observation_time_utc")
        utc_offset_hours = self._payload_optional_float(observer, "utc_offset_hours", 0.0)
        utc_offset_hours = min(
            max(utc_offset_hours, self.ui.doubleSpinBoxUtcOffset.minimum()),
            self.ui.doubleSpinBoxUtcOffset.maximum(),
        )
        local_observation_time = observation_time_utc.astimezone(timezone(timedelta(hours=utc_offset_hours)))
        local_datetime_text = local_observation_time.strftime("%Y-%m-%d %H:%M:%S")
        qt_observation_time = QDateTime.fromString(local_datetime_text, "yyyy-MM-dd HH:mm:ss")
        if not qt_observation_time.isValid():
            raise ValueError("JSON 中的观测时间无法转换为界面时间。")

        widgets_to_block = (
            self.ui.dateTimeEditObservation,
            self.ui.doubleSpinBoxUtcOffset,
            self.ui.doubleSpinBoxLatitude,
            self.ui.doubleSpinBoxLongitude,
            self.ui.doubleSpinBoxElevation,
            self.ui.doubleSpinBoxSensorWidth,
            self.ui.doubleSpinBoxSensorHeight,
            self.ui.spinBoxImageWidth,
            self.ui.spinBoxImageHeight,
            self.ui.doubleSpinBoxFocalLength,
            self.ui.comboBoxLensModel,
            self.ui.doubleSpinBoxFisheyeFov,
            self.ui.doubleSpinBoxMagLimit,
            self.ui.doubleSpinBoxAz,
            self.ui.doubleSpinBoxAlt,
            self.ui.doubleSpinBoxRoll,
            self.ui.comboBoxReferenceLabelMode,
            self.ui.spinBoxReferenceStarCount,
            self.ui.doubleSpinBoxReferenceMagLimit,
        )
        previous_signal_states = [widget.blockSignals(True) for widget in widgets_to_block]
        previous_syncing = self._syncing_camera_dimensions
        self._syncing_camera_dimensions = True
        try:
            self.ui.dateTimeEditObservation.setDateTime(qt_observation_time)
            self.ui.doubleSpinBoxUtcOffset.setValue(utc_offset_hours)
            self.ui.doubleSpinBoxLatitude.setValue(self._payload_float(observer, "latitude_deg"))
            self.ui.doubleSpinBoxLongitude.setValue(self._payload_float(observer, "longitude_deg"))
            self.ui.doubleSpinBoxElevation.setValue(self._payload_float(observer, "elevation_m"))

            self.ui.doubleSpinBoxSensorWidth.setValue(self._payload_float(camera, "sensor_width_mm"))
            self.ui.doubleSpinBoxSensorHeight.setValue(self._payload_float(camera, "sensor_height_mm"))
            self.ui.spinBoxImageWidth.setValue(self._payload_int(camera, "image_width_px"))
            self.ui.spinBoxImageHeight.setValue(self._payload_int(camera, "image_height_px"))
            self.ui.doubleSpinBoxFocalLength.setValue(self._payload_float(camera, "focal_length_mm"))
            lens_model = str(camera.get("lens_model", RECTILINEAR_LENS_MODEL))
            lens_index = LENS_MODELS.index(lens_model) if lens_model in LENS_MODELS else 0
            self.ui.comboBoxLensModel.setCurrentIndex(lens_index)
            self.ui.doubleSpinBoxFisheyeFov.setValue(self._payload_float(camera, "fisheye_fov_deg"))

            self.ui.doubleSpinBoxAz.setValue(self._payload_float(view, "center_az_deg"))
            self.ui.doubleSpinBoxAlt.setValue(self._payload_float(view, "center_alt_deg"))
            self.ui.doubleSpinBoxRoll.setValue(self._payload_float(view, "roll_deg"))

            self.ui.doubleSpinBoxMagLimit.setValue(self._payload_float(render, "visible_mag_limit"))
            reference_label_mode = str(render.get("reference_label_mode", REFERENCE_LABEL_MODE_FIXED_COUNT))
            if reference_label_mode not in REFERENCE_LABEL_MODES:
                reference_label_mode = REFERENCE_LABEL_MODE_FIXED_COUNT
            self.ui.comboBoxReferenceLabelMode.setCurrentIndex(REFERENCE_LABEL_MODES.index(reference_label_mode))
            self.ui.spinBoxReferenceStarCount.setValue(self._payload_int(render, "reference_star_count"))
            self.ui.doubleSpinBoxReferenceMagLimit.setValue(
                self._payload_optional_float(render, "reference_mag_limit", self.ui.doubleSpinBoxReferenceMagLimit.value())
            )
        finally:
            self._syncing_camera_dimensions = previous_syncing
            for widget, was_blocked in zip(widgets_to_block, previous_signal_states):
                widget.blockSignals(was_blocked)

        restored_manual_star_ids: list[str] = []
        manual_ids_payload = payload.get("manual_reference_star_ids")
        if isinstance(manual_ids_payload, list):
            for raw_star_id in manual_ids_payload:
                star_id = str(raw_star_id).strip()
                if star_id and star_id not in restored_manual_star_ids:
                    restored_manual_star_ids.append(star_id)
        self._manual_reference_star_ids = restored_manual_star_ids
        self._imported_reference_star_by_id = self._reference_star_lookup_from_records(
            payload.get("stars", []),
            observer=self._observer_settings(),
        )
        self._auto_match_reference_star_ids = []
        self._auto_match_constraint_by_star_id = {}
        self._auto_match_group_order = []
        self._auto_match_group_by_star_id = {}
        self._auto_match_group_expanded_by_id = {}
        self._auto_match_next_group_index = 0
        self._excluded_reference_star_ids = []
        self._update_reference_label_controls()
        self._update_lens_model_controls()
        self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabSimulator)
        self.render_now()
        self.ui.statusbar.showMessage(f"已导入预览 JSON 并恢复星空模拟参数: {source_path}")
