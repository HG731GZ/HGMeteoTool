from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from ..alignment.constants import MIN_ALIGNMENT_PAIRS
from .app_utils import _relative_image_path_for_session
from ..catalog import project_root
from ..fixed_camera_model import (
    FixedCameraModel,
    FixedCameraTimeFitResult,
    estimate_frame_time_correction,
)
from ..simulator import ObserverSettings
from ..source_model import SourceAstrometricModel

class SourceModelExportMixin:
    """单图源模型 JSON 导出入口。"""

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
            "file_stem": image_path.stem,
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
            payload["file_name"] = mask_path.name
            payload["file_stem"] = mask_path.stem
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
        observer, observer_time_payload = self._reference_payload_observer()
        fixed_model = self._fit_sequence_fixed_camera_model(templates, target_size, observer)
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
        return self._with_current_simulator_time(
            frame_model.to_json_payload(
                source_image=self._source_image_payload(json_path),
                mask=self._sky_mask_payload(json_path),
                matching=self._auto_match_settings_payload(),
                fit_pairs=fit_pairs,
                reference_payload=reference_payload,
                generated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

    def _write_current_source_model(
        self,
        *,
        preload_to_mosaic: bool = True,
    ) -> tuple[Path, int, float, bool] | None:
        """把当前源图映射写入默认同名 JSON；用户拒绝覆盖时返回空。"""

        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像，再导出 xy→RA/Dec 映射 JSON。")
        default_path = self._default_source_model_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = default_path
        payload = self._build_source_model_payload(json_path)
        diagnostics = payload.get("diagnostics", {})
        pair_count = int(diagnostics.get("pair_count", 0)) if isinstance(diagnostics, dict) else 0
        rms_px = float(diagnostics.get("rms_px", float("nan"))) if isinstance(diagnostics, dict) else float("nan")
        if not self._confirm_overwrite_if_existing_has_more_pairs(json_path, pair_count, model_json=True):
            self.ui.statusbar.showMessage("已取消导出 xy→RA/Dec 映射 JSON。")
            return None
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        preloaded_to_mosaic = False
        if preload_to_mosaic and hasattr(self, "load_mosaic_model_json"):
            preloaded_to_mosaic = bool(self.load_mosaic_model_json(json_path, quiet=True))
        if hasattr(self, "_refresh_image_group_assistant_status"):
            self._refresh_image_group_assistant_status()
        return json_path, pair_count, rms_px, preloaded_to_mosaic

    def export_source_model_json(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导出 xy→RA/Dec 映射 JSON。")
            return

        try:
            result = self._write_current_source_model()
            if result is None:
                return
            json_path, pair_count, rms_px, preloaded_to_mosaic = result
            try:
                star_pair_result = self._write_current_star_pair_session()
            except Exception as exc:  # noqa: BLE001 - 映射已落盘时必须单独说明匹配 JSON 的保存失败。
                self.ui.statusbar.showMessage(
                    f"映射 JSON 已导出，但星点匹配 JSON 保存失败: {exc}"
                )
                QMessageBox.critical(
                    self,
                    "星点匹配 JSON 保存失败",
                    f"映射 JSON 已成功导出：\n{json_path}\n\n星点匹配 JSON 保存失败：\n{exc}",
                )
                return
            if star_pair_result is None:
                self.ui.statusbar.showMessage("映射 JSON 已导出，但已取消覆盖星点匹配 JSON。")
                QMessageBox.warning(
                    self,
                    "映射已导出",
                    f"映射 JSON 已成功导出：\n{json_path}\n\n星点匹配 JSON 未覆盖。",
                )
                return
            star_pair_path, star_pair_count = star_pair_result
            preload_status = "，已预载到全景构图" if preloaded_to_mosaic else "，自由投影预载失败"
            self.ui.statusbar.showMessage(
                f"已导出映射并保存星点匹配 JSON: {json_path}  匹配: {star_pair_path}  "
                f"匹配数: {pair_count}  RMS: {rms_px:.2f}px{preload_status}"
            )
            preload_message = "\n已预载到全景构图，可切换页面查看。" if preloaded_to_mosaic else "\n自由投影预载失败，可稍后手动导入检查。"
            QMessageBox.information(
                self,
                "映射与匹配 JSON 已导出",
                (
                    f"映射 JSON：{json_path}\n"
                    f"星点匹配 JSON：{star_pair_path}\n"
                    f"匹配数：{star_pair_count}\n"
                    f"RMS：{rms_px:.2f} px{preload_message}"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把模型生成与文件错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导出 xy→RA/Dec 映射 JSON 失败: {exc}")
            QMessageBox.critical(self, "导出 xy→RA/Dec 映射 JSON 失败", str(exc))
