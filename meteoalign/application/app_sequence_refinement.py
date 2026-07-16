"""图像序列完成后的逐帧映射精修。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

from ..alignment.constants import (
    MIN_ALIGNMENT_PAIRS,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_POLYNOMIAL,
    SKY_MATCHING_MODELS,
)
from ..camera_calibration import CameraCalibrationProfile
from ..frame_astrometry import FrameAstrometricModel
from ..source_model import fit_source_astrometric_model, fit_source_astrometric_model_with_fixed_profile
from ..star_pair_model import StarPairRecord, star_pair_records_from_payloads


SEQUENCE_REFINEMENT_MODE_POSE = "pose"
SEQUENCE_REFINEMENT_MODE_REFIT = "refit"
SEQUENCE_REFINEMENT_MODES = (
    SEQUENCE_REFINEMENT_MODE_POSE,
    SEQUENCE_REFINEMENT_MODE_REFIT,
)


class SequenceRefinementMixin:
    """用序列输出的匹配星对每帧模型进行可选后处理。"""

    def _sequence_refinement_ready(self) -> bool:
        """仅当序列每张图都具有匹配和模型输出时允许精修。"""

        items = list(getattr(self, "_image_sequence_items", []))
        if not items or bool(getattr(self, "_sequence_processing_active", False)):
            return False
        if bool(getattr(self, "_sequence_refinement_active", False)):
            return False
        if getattr(self, "_image_import_thread", None) is not None:
            return False
        if getattr(self, "_sequence_import_thread", None) is not None:
            return False
        if getattr(self, "_json_import_thread", None) is not None:
            return False
        return all(
            self._sequence_starpair_json_path(item.path).is_file()
            and self._sequence_model_json_path(item.path).is_file()
            for item in items
        )

    def _sequence_refinement_mode(self) -> str:
        """返回下拉列表当前选中的精修方式。"""

        if not hasattr(self.ui, "comboBoxSequenceRefinementMode"):
            return SEQUENCE_REFINEMENT_MODE_POSE
        index = int(self.ui.comboBoxSequenceRefinementMode.currentIndex())
        if 0 <= index < len(SEQUENCE_REFINEMENT_MODES):
            return SEQUENCE_REFINEMENT_MODES[index]
        return SEQUENCE_REFINEMENT_MODE_POSE

    @staticmethod
    def _read_sequence_refinement_json(path: Path) -> dict[str, object]:
        """读取并验证单帧 JSON 的顶层对象。"""

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取 {path.name}：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} 的根对象不是 JSON 对象。")
        return payload

    @staticmethod
    def _sequence_refinement_fit_data(starpair_payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """从序列匹配 JSON 提取 RA/Dec、PSF 像素、权重和锚点标记。"""

        pair_payloads = starpair_payload.get("pairs")
        if not isinstance(pair_payloads, list):
            raise ValueError("匹配 JSON 缺少 pairs 列表。")
        records = [record for record in star_pair_records_from_payloads(pair_payloads) if record.is_valid_for_fit()]
        if len(records) < MIN_ALIGNMENT_PAIRS:
            raise ValueError(f"有效匹配只有 {len(records)} 个，至少需要 {MIN_ALIGNMENT_PAIRS} 个。")
        return SequenceRefinementMixin._sequence_refinement_fit_arrays(records)

    @staticmethod
    def _sequence_refinement_fit_arrays(
        records: list[StarPairRecord],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """将匹配记录转换为两种精修共用的数值输入。"""

        ra_dec_points = np.asarray(
            [(record.reference_star.ra_deg, record.reference_star.dec_deg) for record in records],
            dtype=np.float64,
        )
        pixel_points = np.asarray([record.position for record in records], dtype=np.float64)
        point_weights = np.asarray([record.fit_weight for record in records], dtype=np.float64)
        anchor_mask = np.asarray(
            [record.fit_constraint_mode != "soft" for record in records],
            dtype=bool,
        )
        return ra_dec_points, pixel_points, point_weights, anchor_mask

    @staticmethod
    def _base_sequence_profile(
        model_payload: dict[str, object],
        current_model: FrameAstrometricModel,
    ) -> CameraCalibrationProfile:
        """优先取得首次保存的固定机位 Profile，支持两种精修反复切换。"""

        refinement = model_payload.get("sequence_refinement")
        if isinstance(refinement, dict):
            profile_payload = refinement.get("base_sequence_camera_calibration_profile")
            if isinstance(profile_payload, dict):
                return CameraCalibrationProfile.from_json_payload(profile_payload)
        return current_model.camera_calibration_profile

    @staticmethod
    def _refit_matching_model(
        starpair_payload: dict[str, object],
        profile: CameraCalibrationProfile,
    ) -> str:
        """选择单帧重新拟合的基础投影，优先沿用序列匹配时的选择。"""

        selected = str(starpair_payload.get("sky_alignment_model") or "").strip()
        if selected == SKY_MATCHING_MODEL_POLYNOMIAL:
            selected = SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION
        if selected in SKY_MATCHING_MODELS:
            return selected
        if profile.base_projection_type in SKY_KNOWN_PROJECTION_MODELS:
            return profile.base_projection_type
        return SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION

    @staticmethod
    def _preserved_mapping(payload: dict[str, object], key: str) -> dict[str, object] | None:
        value = payload.get(key)
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _preserved_list(payload: dict[str, object], key: str) -> list[object] | None:
        value = payload.get(key)
        return list(value) if isinstance(value, list) else None

    def _refined_model_payload(
        self,
        *,
        original_payload: dict[str, object],
        refined_frame_model: FrameAstrometricModel,
        base_profile: CameraCalibrationProfile,
        mode: str,
        rms_px: float,
        median_residual_px: float,
        max_residual_px: float,
        pair_count: int,
    ) -> dict[str, object]:
        """生成保留序列上下文与两类精修指标的新版模型 JSON。"""

        original_metadata = self._preserved_mapping(original_payload, "fit_metadata") or {}
        metadata = dict(refined_frame_model.fit_metadata)
        for key in ("sequence_timing", "source_sequence_model", "scene_observer_hint", "scene_observer_hint_role"):
            if key in original_metadata:
                metadata.setdefault(key, original_metadata[key])
        metadata["sequence_refinement"] = {
            "mode": mode,
            "base_profile_frozen": mode == SEQUENCE_REFINEMENT_MODE_POSE,
        }
        refined_frame_model = replace(refined_frame_model, fit_metadata=metadata)

        payload = refined_frame_model.to_json_payload(
            source_image=self._preserved_mapping(original_payload, "source_image"),
            mask=self._preserved_mapping(original_payload, "mask"),
            matching=self._preserved_mapping(original_payload, "matching"),
            fit_pairs=self._preserved_list(original_payload, "fit_pairs"),
            reference_payload=self._preserved_mapping(original_payload, "reference_payload"),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        existing_refinement = original_payload.get("sequence_refinement")
        refinement = dict(existing_refinement) if isinstance(existing_refinement, dict) else {}
        existing_results = refinement.get("results")
        results = dict(existing_results) if isinstance(existing_results, dict) else {}
        results[mode] = {
            "rms_px": float(rms_px),
            "median_residual_px": float(median_residual_px),
            "max_residual_px": float(max_residual_px),
            "pair_count": int(pair_count),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        refinement.update(
            {
                "version": 1,
                "base_sequence_camera_calibration_profile": base_profile.to_json_payload(),
                "results": results,
            }
        )
        payload["sequence_refinement"] = refinement
        return payload

    @staticmethod
    def _write_sequence_refinement_model(path: Path, payload: dict[str, object]) -> None:
        """以临时文件替换精修模型，避免中断时留下不完整 JSON。"""

        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(path)

    def _refine_sequence_frame(
        self,
        *,
        starpair_path: Path,
        model_path: Path,
        mode: str,
    ) -> float:
        """精修单帧模型并返回该模式写入的 RMS。"""

        starpair_payload = self._read_sequence_refinement_json(starpair_path)
        model_payload = self._read_sequence_refinement_json(model_path)
        current_model = FrameAstrometricModel.from_json_payload(model_payload)
        base_profile = self._base_sequence_profile(model_payload, current_model)
        ra_dec_points, pixel_points, point_weights, anchor_mask = self._sequence_refinement_fit_data(starpair_payload)
        image_size = (current_model.image_width_px, current_model.image_height_px)
        initial_rotation = current_model.frame_pose.icrs_to_camera

        if mode == SEQUENCE_REFINEMENT_MODE_POSE:
            refined = fit_source_astrometric_model_with_fixed_profile(
                ra_dec_points=ra_dec_points,
                pixel_points=pixel_points,
                image_size=image_size,
                camera_calibration_profile=base_profile,
                initial_rotation_matrix=initial_rotation,
                point_weights=point_weights,
                residual_anchor_mask=anchor_mask,
                profile_source_path=str(model_path),
                solve_mode="sequence_refinement_pose_only",
            )
            refined_frame_model = refined.frame_model
        elif mode == SEQUENCE_REFINEMENT_MODE_REFIT:
            refined = fit_source_astrometric_model(
                ra_dec_points=ra_dec_points,
                pixel_points=pixel_points,
                image_size=image_size,
                matching_model=self._refit_matching_model(starpair_payload, base_profile),
                initial_rotation_matrix=initial_rotation,
                point_weights=point_weights,
                residual_anchor_mask=anchor_mask,
            )
            refined_frame_model = refined.to_frame_astrometric_model()
        else:
            raise ValueError(f"不支持的单帧精修模式：{mode}")

        output_payload = self._refined_model_payload(
            original_payload=model_payload,
            refined_frame_model=refined_frame_model,
            base_profile=base_profile,
            mode=mode,
            rms_px=float(refined.rms_px),
            median_residual_px=float(refined.median_residual_px),
            max_residual_px=float(refined.max_residual_px),
            pair_count=int(refined.pair_count),
        )
        self._write_sequence_refinement_model(model_path, output_payload)
        return float(refined.rms_px)

    def refine_sequence_frames(self) -> None:
        """对序列内全部帧执行所选的单帧结果修正。"""

        if not self._sequence_refinement_ready():
            QMessageBox.information(
                self,
                "单帧结果修正尚不可用",
                "请先完成整个图像序列解析，确保每张图像均已有 starpairs.json 与 model.json。",
            )
            return
        mode = self._sequence_refinement_mode()
        mode_name = "优化取景角度" if mode == SEQUENCE_REFINEMENT_MODE_POSE else "单帧重新拟合"
        item_count = len(self._image_sequence_items)
        reply = QMessageBox.question(
            self,
            "确认单帧结果修正",
            f"将按“{mode_name}”依次修正 {item_count} 张图像的 model.json，并保留 δt RMS 基准。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            self.ui.statusbar.showMessage("已取消单帧结果修正。")
            return

        progress = QProgressDialog(self)
        progress.setWindowTitle("正在单帧结果修正")
        progress.setRange(0, item_count)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()
        self._sequence_refinement_active = True
        self._update_image_sequence_controls()
        successes = 0
        failures: list[str] = []
        try:
            for index, item in enumerate(self._image_sequence_items, start=1):
                if progress.wasCanceled():
                    failures.append("用户取消了后续精修。")
                    break
                progress.setValue(index - 1)
                progress.setLabelText(f"正在精修 {index}/{item_count}\n{item.path.name}")
                QApplication.processEvents()
                try:
                    self._refine_sequence_frame(
                        starpair_path=self._sequence_starpair_json_path(item.path),
                        model_path=self._sequence_model_json_path(item.path),
                        mode=mode,
                    )
                    successes += 1
                except Exception as exc:  # noqa: BLE001 - 单帧失败不应中断其余图像的精修。
                    failures.append(f"{item.path.name}: {exc}")
                self._refresh_image_sequence_table()
        finally:
            progress.setValue(item_count)
            progress.close()
            self._sequence_refinement_active = False
            self._refresh_image_sequence_table()
            self._update_image_sequence_controls()

        self.ui.statusbar.showMessage(f"单帧结果修正完成：{mode_name}成功 {successes} 张，失败 {len(failures)} 张。")
        message = f"“{mode_name}”已完成：成功 {successes} 张，失败 {len(failures)} 张。"
        if failures:
            message += "\n\n失败明细：\n" + "\n".join(failures[:12])
            if len(failures) > 12:
                message += f"\n... 另有 {len(failures) - 12} 条"
            QMessageBox.warning(self, "单帧结果修正完成", message)
        else:
            QMessageBox.information(self, "单帧结果修正完成", message)


__all__ = [
    "SEQUENCE_REFINEMENT_MODE_POSE",
    "SEQUENCE_REFINEMENT_MODE_REFIT",
    "SEQUENCE_REFINEMENT_MODES",
    "SequenceRefinementMixin",
]
