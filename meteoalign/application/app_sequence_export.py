from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ..alignment.constants import MIN_ALIGNMENT_PAIRS
from .app_constants import (
    STAR_PAIR_SESSION_FORMAT,
    STAR_PAIR_SESSION_VERSION,
)
from .app_constants import AUTO_MATCH_SEARCH_MAG_LIMIT
from .app_utils import _relative_image_path_for_session
from ..fixed_camera_model import FixedCameraModel, FixedCameraTimeFitResult
from ..image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from ..image_sequence import ImageSequenceItem, sequence_item_time_delta_seconds
from ..reference import build_reference_payload
from ..sequence_geometry import frame_astrometric_model_from_fixed_camera
from ..sequence_types import _SequenceMatchedPair
from ..simulator import ObserverSettings
from ..star_pair_model import PsfFit, StarPairRecord

class SequenceExportMixin:
    """序列批处理结果 payload、JSON 输出和覆盖确认。"""

    def _sequence_pair_records(
        self,
        pairs: list[_SequenceMatchedPair],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for output_index, pair in enumerate(pairs, start=1):
            reference_star = pair.reference_star
            fitted = pair.fitted_position
            predicted_x = pair.predicted_x_px if pair.predicted_x_px is not None else float("nan")
            predicted_y = pair.predicted_y_px if pair.predicted_y_px is not None else float("nan")
            residual_dx = float(predicted_x - fitted.x)
            residual_dy = float(predicted_y - fitted.y)
            extra_fields: dict[str, object] = {
                "fixed_model_x_px": float(predicted_x),
                "fixed_model_y_px": float(predicted_y),
            }
            if pair.predicted_x_px is not None and pair.predicted_y_px is not None:
                extra_fields["theoretical_x_px"] = float(pair.predicted_x_px)
                extra_fields["theoretical_y_px"] = float(pair.predicted_y_px)
                extra_fields["psf_offset_from_theory_px"] = float(
                    np.hypot(fitted.x - pair.predicted_x_px, fitted.y - pair.predicted_y_px)
                )
                extra_fields["adaptive_offset_x_px"] = float(pair.adaptive_offset_x_px)
                extra_fields["adaptive_offset_y_px"] = float(pair.adaptive_offset_y_px)
            if pair.initial_predicted_x_px is not None and pair.initial_predicted_y_px is not None:
                extra_fields["initial_theoretical_x_px"] = float(pair.initial_predicted_x_px)
                extra_fields["initial_theoretical_y_px"] = float(pair.initial_predicted_y_px)
                extra_fields["psf_offset_from_initial_theory_px"] = float(
                    np.hypot(fitted.x - pair.initial_predicted_x_px, fitted.y - pair.initial_predicted_y_px)
                )
            if pair.time_delta_seconds is not None:
                extra_fields["delta_t_seconds"] = float(pair.time_delta_seconds)
            if pair.search_x_px is not None and pair.search_y_px is not None:
                extra_fields["search_x_px"] = float(pair.search_x_px)
                extra_fields["search_y_px"] = float(pair.search_y_px)
                extra_fields["psf_offset_from_search_px"] = float(
                    np.hypot(fitted.x - pair.search_x_px, fitted.y - pair.search_y_px)
                )
            record = StarPairRecord(
                reference_star=reference_star,
                image_x_px=float(fitted.x),
                image_y_px=float(fitted.y),
                psf=PsfFit.from_fitted_position(fitted),
                pair_origin=pair.pair_origin,
                fit_constraint_mode=pair.fit_constraint_mode,
                fit_weight=float(pair.fit_weight),
                residual_dx_px=residual_dx,
                residual_dy_px=residual_dy,
                residual_px=float(np.hypot(residual_dx, residual_dy)),
                extra_fields=extra_fields,
            )
            records.append(record.to_json_payload(reference_index=output_index))
        return records

    def _sequence_image_base_payload(
        self,
        preview: ImagePreview,
        json_path: Path,
        item: ImageSequenceItem,
    ) -> dict[str, object]:
        image_path = Path(preview.path).expanduser().resolve()
        payload = {
            "path": str(image_path),
            "relative_path": _relative_image_path_for_session(image_path, json_path),
            "file_name": image_path.name,
            "file_stem": image_path.stem,
            "original_width_px": int(preview.original_width),
            "original_height_px": int(preview.original_height),
        }
        payload.update(self._sequence_capture_payload(item))
        return payload

    def _sequence_real_image_payload(
        self,
        preview: ImagePreview,
        json_path: Path,
        item: ImageSequenceItem,
    ) -> dict[str, object]:
        payload = self._sequence_image_base_payload(preview, json_path, item)
        payload.update(
            {
                "display_width_px": int(preview.image.width()),
                "display_height_px": int(preview.image.height()),
            }
        )
        return payload

    def _sequence_source_image_payload(
        self,
        preview: ImagePreview,
        json_path: Path,
        item: ImageSequenceItem,
    ) -> dict[str, object]:
        payload = self._sequence_image_base_payload(preview, json_path, item)
        payload.update(
            {
                "model_width_px": int(preview.image.width()),
                "model_height_px": int(preview.image.height()),
            }
        )
        return payload

    def _sequence_capture_payload(self, item: ImageSequenceItem) -> dict[str, object]:
        payload: dict[str, object] = {
            "exif_capture_time": item.capture_datetime.isoformat(),
            "capture_time_source": item.capture_time_source,
        }
        if item.capture_datetime_utc is not None:
            payload["capture_time_utc"] = item.capture_datetime_utc.isoformat()
        return payload

    def _sequence_reference_payload(
        self,
        item: ImageSequenceItem,
        preview: ImagePreview,
        pairs: list[_SequenceMatchedPair],
    ) -> dict[str, object]:
        observer = self._sequence_observer_for_item(item)
        camera = self._camera_settings_for_image_size(preview.image.width(), preview.image.height())
        view = self._view_settings()
        visible_mag_limit = max(
            self._reference_catalog_mag_limit(AUTO_MATCH_SEARCH_MAG_LIMIT),
            AUTO_MATCH_SEARCH_MAG_LIMIT,
        )
        star_map = self._sequence_projected_star_map(
            item,
            (preview.image.width(), preview.image.height()),
            visible_mag_limit,
        )
        reference_stars = tuple(
            self._reference_star_with_index(pair.reference_star, index)
            for index, pair in enumerate(pairs, start=1)
        )
        payload = build_reference_payload(
            star_map=star_map,
            reference_stars=reference_stars,
            observer=observer,
            camera=camera,
            view=view,
            visible_mag_limit=visible_mag_limit,
            utc_offset_hours=self.ui.doubleSpinBoxUtcOffset.value(),
            reference_label_mode=self._reference_label_mode(),
            reference_mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            reference_star_count=self.ui.spinBoxReferenceStarCount.value(),
            manual_reference_star_ids=tuple(pair.reference_star.star_id for pair in pairs),
        )
        observer_payload = payload.get("observer")
        if isinstance(observer_payload, dict):
            observer_payload.update(
                {
                    "observation_time_source": "real_image_exif",
                    "capture_time_source": item.capture_time_source,
                    "exif_capture_time": item.capture_datetime.isoformat(),
                }
            )
            if item.capture_datetime_utc is not None:
                observer_payload["capture_time_utc"] = item.capture_datetime_utc.isoformat()
        return payload

    def _sequence_matching_payload(self) -> dict[str, object]:
        return self._auto_match_settings_payload()

    def _sequence_timing_payload(
        self,
        item: ImageSequenceItem,
        first_item: ImageSequenceItem,
        time_fit: FixedCameraTimeFitResult,
    ) -> dict[str, object]:
        nominal_time_utc = self._sequence_nominal_time_utc(item)
        effective_time_utc = nominal_time_utc + timedelta(seconds=float(time_fit.delta_t_seconds))
        first_time_utc = self._sequence_nominal_time_utc(first_item)
        return {
            "time_model": "first_frame_relative_exif_plus_per_frame_delta_t",
            "first_frame_nominal_time_utc": first_time_utc.isoformat(),
            "frame_nominal_time_utc": nominal_time_utc.isoformat(),
            "frame_effective_time_utc": effective_time_utc.isoformat(),
            "exif_delta_from_first_seconds": float(sequence_item_time_delta_seconds(item, first_item)),
            "delta_t_seconds": float(time_fit.delta_t_seconds),
            "delta_t_reference": "relative_to_frame_nominal_time_not_chained",
            "delta_t_0_seconds": 0.0,
            "time_fit": time_fit.to_json_payload(),
        }

    def _sequence_model_payload(
        self,
        *,
        item: ImageSequenceItem,
        first_item: ImageSequenceItem,
        preview: ImagePreview,
        model_path: Path,
        fixed_model: FixedCameraModel,
        time_fit: FixedCameraTimeFitResult,
        records: list[dict[str, object]],
        reference_payload: dict[str, object],
        generated_at_utc: str,
    ) -> dict[str, object]:
        timing_payload = self._sequence_timing_payload(item, first_item, time_fit)
        observer = ObserverSettings(
            observation_time_utc=datetime.fromisoformat(
                str(timing_payload["frame_effective_time_utc"]).replace("Z", "+00:00")
            ).astimezone(timezone.utc),
            latitude_deg=float(self.ui.doubleSpinBoxLatitude.value()),
            longitude_deg=float(self.ui.doubleSpinBoxLongitude.value()),
            elevation_m=float(self.ui.doubleSpinBoxElevation.value()),
        )
        diagnostics = {
            "pair_count": len(records),
            "rms_px": float(time_fit.rms_px),
            "median_residual_px": float(time_fit.median_residual_px),
            "max_residual_px": float(time_fit.max_residual_px),
            **time_fit.to_json_payload(),
        }
        frame_model = frame_astrometric_model_from_fixed_camera(
            fixed_camera_model=fixed_model,
            observer=observer,
            fit_metadata={
                "model_type": "sequence_frame_astrometric_model",
                "source_sequence_model": "fixed_camera_enu_sequence_geometry",
                "control_point_count": len(records),
                "sequence_timing": timing_payload,
                "scene_observer_hint": {
                    "observation_time_utc": observer.observation_time_utc.isoformat(),
                    "latitude_deg": observer.latitude_deg,
                    "longitude_deg": observer.longitude_deg,
                    "elevation_m": observer.elevation_m,
                    "utc_offset_hours": float(self.ui.doubleSpinBoxUtcOffset.value()),
                    **timing_payload,
                },
                "scene_observer_hint_role": "metadata_only_not_required_for_pixel_icrs_model",
            },
            diagnostics=diagnostics,
        )
        return frame_model.to_json_payload(
            source_image=self._sequence_source_image_payload(preview, model_path, item),
            mask=self._sky_mask_payload(model_path),
            matching=self._sequence_matching_payload(),
            fit_pairs=records,
            reference_payload=reference_payload,
            generated_at_utc=generated_at_utc,
        )

    def _sequence_starpair_json_path(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_starpairs.json")

    def _sequence_model_json_path(self, image_path: Path) -> Path:
        resolved_path = Path(image_path).expanduser().resolve()
        return resolved_path.with_name(f"{resolved_path.stem}_model.json")

    def _write_sequence_outputs(
        self,
        item: ImageSequenceItem,
        first_item: ImageSequenceItem,
        preview: ImagePreview,
        pairs: list[_SequenceMatchedPair],
        fixed_model: FixedCameraModel,
        time_fit: FixedCameraTimeFitResult,
    ) -> tuple[Path, Path]:
        image_path = Path(preview.path).expanduser().resolve()
        starpair_path = self._sequence_starpair_json_path(image_path)
        model_path = self._sequence_model_json_path(image_path)
        records = self._sequence_pair_records(pairs)
        reference_payload = self._sequence_reference_payload(item, preview, pairs)
        generated_at_utc = datetime.now(timezone.utc).isoformat()
        timing_payload = self._sequence_timing_payload(item, first_item, time_fit)
        starpair_payload = {
            "format": STAR_PAIR_SESSION_FORMAT,
            "version": STAR_PAIR_SESSION_VERSION,
            "generated_at_utc": generated_at_utc,
            "real_image": self._sequence_real_image_payload(preview, starpair_path, item),
            "reference_payload": reference_payload,
            "sky_alignment_model": self._alignment_model(),
            "image_model": "fixed_camera_model",
            "sequence_timing": timing_payload,
            "pair_count": len(records),
            "pairs": records,
            "mask": self._sky_mask_payload(starpair_path),
            "matching": self._sequence_matching_payload(),
        }
        source_payload = self._sequence_model_payload(
            item=item,
            first_item=first_item,
            preview=preview,
            model_path=model_path,
            fixed_model=fixed_model,
            time_fit=time_fit,
            records=records,
            reference_payload=reference_payload,
            generated_at_utc=generated_at_utc,
        )
        self._with_current_simulator_time(starpair_payload)
        self._with_current_simulator_time(source_payload)

        starpair_path.write_text(json.dumps(starpair_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        model_path.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return starpair_path, model_path

    def _current_preview_is_sequence_first_item(self, first_item: ImageSequenceItem) -> bool:
        if self.current_image_preview is None:
            return False
        try:
            return Path(self.current_image_preview.path).expanduser().resolve() == first_item.path.expanduser().resolve()
        except OSError:
            return False

    def _load_first_sequence_image_without_tab_switch(self, first_item: ImageSequenceItem) -> None:
        preview = load_image_preview(
            first_item.path,
            max_long_side_px=None,
            include_native_luminance=True,
        )
        self._preserve_sequence_on_next_image_load = True
        self._apply_loaded_image_preview(
            preview,
            clear_existing_pairs=True,
            switch_to_reference=False,
        )

    def _ensure_first_sequence_image_ready_for_mask(self) -> ImageSequenceItem:
        first_item = self._current_sequence_first_item()
        if self._current_preview_is_sequence_first_item(first_item):
            return first_item
        if self._sequence_starpair_json_path(first_item.path).exists():
            self._load_first_sequence_session_for_reference_page(raise_on_error=True)
        else:
            current_tab = self.ui.tabWidgetMain.currentWidget()
            self._load_first_sequence_image_without_tab_switch(first_item)
            if current_tab is not None:
                self.ui.tabWidgetMain.setCurrentWidget(current_tab)
        if not self._current_preview_is_sequence_first_item(first_item):
            raise ValueError("序列第一帧图像尚未载入，无法导入序列蒙版。")
        return first_item

    def import_image_sequence_sky_mask(self) -> None:
        if not self._sequence_mode_active():
            QMessageBox.information(self, "尚未导入序列", "请先导入图像序列，再导入序列蒙版。")
            return
        if getattr(self, "_mask_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入蒙版", "当前已有蒙版正在导入，请稍候。")
            return
        if getattr(self, "_image_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入图像", "当前已有图像正在导入，请稍候。")
            return
        try:
            first_item = self._ensure_first_sequence_image_ready_for_mask()
        except Exception as exc:  # noqa: BLE001 - 蒙版入口需要把基准图未就绪原因直接反馈给用户。
            QMessageBox.warning(self, "无法导入序列蒙版", str(exc))
            return

        default_dir = self._import_dialog_directory(first_item.path.parent)
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入序列蒙版",
            str(default_dir),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        self._remember_import_path(file_path)
        self.start_sky_mask_import(file_path)

    def clear_image_sequence_sky_mask(self) -> None:
        if not self._sequence_mode_active():
            self.ui.statusbar.showMessage("当前没有正在处理的图像序列。")
            return
        self.clear_sky_mask()
        self._update_image_sequence_controls()
        self._update_image_sequence_preview()
        self.ui.statusbar.showMessage("已清除序列蒙版，后续序列自动匹配将使用整张图像。")

    def _load_first_sequence_session_for_reference_page(self, *, raise_on_error: bool) -> bool:
        first_item = self._current_sequence_first_item()
        json_path = self._sequence_starpair_json_path(first_item.path)
        if not json_path.exists():
            return False
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            preview = load_image_preview(
                first_item.path,
                max_long_side_px=None,
                include_native_luminance=True,
            )
            current_tab = self.ui.tabWidgetMain.currentWidget()
            self._clear_star_pair_positions_for_new_input("第一帧匹配 JSON")
            self._apply_star_pair_session_payload(
                payload,
                json_path,
                preview=preview,
                switch_to_reference=False,
                restore_observation_time=self._auto_sync_simulator_time_from_exif_enabled(),
            )
            if current_tab is not None:
                self.ui.tabWidgetMain.setCurrentWidget(current_tab)
            return True
        except Exception as exc:  # noqa: BLE001 - 序列导入时尽量保留序列页，错误转为界面提示。
            message = f"无法自动载入第一帧匹配 JSON：{json_path}\n{exc}"
            if raise_on_error:
                raise ValueError(message) from exc
            try:
                self._load_first_sequence_image_without_tab_switch(first_item)
            except Exception as image_exc:  # noqa: BLE001 - 兜底载图失败同样反馈给用户。
                message += f"\n\n第一帧图像也无法载入：{image_exc}"
            QMessageBox.warning(self, "第一帧匹配 JSON 载入失败", message)
            self.ui.statusbar.showMessage(f"第一帧匹配 JSON 载入失败: {json_path.name}")
            return False

    def _ensure_first_sequence_session_loaded_for_processing(self) -> None:
        first_item = self._current_sequence_first_item()
        if self._current_preview_is_sequence_first_item(first_item) and self._star_pair_position_count() >= MIN_ALIGNMENT_PAIRS:
            return

        self._load_first_sequence_session_for_reference_page(raise_on_error=True)

    def _ensure_first_sequence_output_jsons(self, first_item: ImageSequenceItem) -> list[Path]:
        """确保第一帧有用户基准 JSON；已有文件绝不覆盖。"""
        starpair_path = self._sequence_starpair_json_path(first_item.path)
        model_path = self._sequence_model_json_path(first_item.path)
        created_paths: list[Path] = []
        if not starpair_path.exists():
            starpair_payload = self._build_star_pair_session_payload(starpair_path)
            starpair_path.write_text(json.dumps(starpair_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            created_paths.append(starpair_path)
        if not model_path.exists():
            model_payload = self._build_source_model_payload(model_path)
            model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            created_paths.append(model_path)
        return created_paths

    def _confirm_overwrite_sequence_outputs(self) -> bool:
        items = getattr(self, "_image_sequence_items", [])
        if not items:
            return False
        existing_paths: list[Path] = []
        for item in items[1:]:
            for output_path in (
                self._sequence_starpair_json_path(item.path),
                self._sequence_model_json_path(item.path),
            ):
                if output_path.exists():
                    existing_paths.append(output_path)

        if existing_paths:
            sample_lines = "\n".join(str(path) for path in existing_paths[:6])
            if len(existing_paths) > 6:
                sample_lines += f"\n... 另有 {len(existing_paths) - 6} 个文件"
            message = (
                "开始处理后会覆盖第二帧及之后已有的匹配 JSON 与模型 JSON。\n"
                "第一帧已有 JSON 会保留，不参与覆盖。\n\n"
                f"已发现 {len(existing_paths)} 个已有输出文件：\n{sample_lines}\n\n是否继续？"
            )
        else:
            message = (
                "开始处理后会为第二帧及之后写入同名匹配 JSON 与模型 JSON。\n"
                "第一帧只作为用户基准，已有 JSON 会保留，缺失 JSON 会先自动补齐。是否继续？"
            )
        reply = QMessageBox.question(
            self,
            "确认处理图像序列",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes
