from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QFileDialog, QMessageBox, QTableWidgetItem,
)

from ..alignment.constants import MIN_ALIGNMENT_PAIRS
from .app_constants import (
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_MODES,
    AUTO_MATCH_CONSTRAINT_SOFT,
    STAR_PAIR_POSITION_COLUMN,
    STAR_PAIR_SESSION_FORMAT,
    STAR_PAIR_SESSION_JSON_FILTER,
    STAR_PAIR_SESSION_VERSION,
)
from .app_utils import (
    _relative_image_path_for_session,
    _resolve_star_pair_session_real_image_path,
    _validate_star_pair_session_current_image,
)
from .app_workers import StarPairSessionImportWorker
from ..catalog import project_root
from ..image_preview import load_image_preview, ImagePreview
from ..qt_tasks import start_qt_worker_task
from ..star_pair_model import (
    StarPairRecord,
    star_pair_records_from_payloads,
)

class StarPairSessionMixin:
    """星对会话导入导出、记录快照和恢复。"""

    def _update_star_pair_export_control(
        self,
        *,
        controls_enabled: bool | None = None,
    ) -> None:
        """仅在 JSON 控件可用且有效匹配达到正式配准门槛时允许导出。"""

        if controls_enabled is None:
            controls_enabled = getattr(self, "_json_import_thread", None) is None
        self.ui.pushButtonExportStarPairs.setEnabled(
            bool(controls_enabled) and self._star_pair_position_count() >= MIN_ALIGNMENT_PAIRS
        )

    def _star_pair_records(self) -> list[dict[str, object]]:
        return self._star_pair_store.records_to_json_payloads()

    def _auto_match_constraints_payload(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for star_id in self._auto_match_reference_star_ids:
            record = self._star_pair_store.get(star_id)
            if record is None:
                continue
            mode, fit_weight = record.fit_constraint_mode, record.fit_weight
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
            # 导入的匹配 JSON 可能位于别处；导出始终跟随当前真实图像，避免覆盖外部 JSON。
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
            "已有 JSON 匹配更多",
            (
                f"已有文件包含 {existing_pair_count} 个匹配，当前只有 {current_pair_count} 个匹配。\n"
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
        self.ui.statusbar.showMessage(f"发现同名匹配 JSON，正在自动导入: {json_path}")
        self.load_star_pair_session(json_path)

    def _build_star_pair_session_payload(self, json_path: Path) -> dict[str, object]:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像，再导出星点匹配 JSON。")

        preview = self.current_image_preview
        image_path = Path(preview.path).expanduser().resolve()
        relative_image_path = _relative_image_path_for_session(image_path, json_path)
        pair_records = self._star_pair_records()
        image_model = "manual_star_pair_session"
        try:
            fixed_bundle = self._single_image_fixed_camera_export_bundle()
        except Exception:  # noqa: BLE001 - 匹配 JSON 需要支持中途保存，固定模型未就绪时保留普通匹配记录。
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
            "file_stem": image_path.stem,
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

    def _write_current_star_pair_session(self) -> tuple[Path, int] | None:
        """把当前匹配写入默认同名 JSON；用户拒绝覆盖时返回空。"""

        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像，再导出星点匹配 JSON。")
        default_path = self._default_star_pair_session_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = default_path
        payload = self._build_star_pair_session_payload(json_path)
        pair_count = int(payload.get("pair_count", 0))
        if not self._confirm_overwrite_if_existing_has_more_pairs(json_path, pair_count):
            self.ui.statusbar.showMessage("已取消导出星点匹配 JSON。")
            return None
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if hasattr(self, "_refresh_image_group_assistant_status"):
            self._refresh_image_group_assistant_status()
        return json_path, pair_count

    def export_star_pair_session(self) -> None:
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入真实图像，再导出星点匹配 JSON。")
            return

        try:
            result = self._write_current_star_pair_session()
            if result is None:
                return
            json_path, pair_count = result
            self.ui.statusbar.showMessage(f"已导出星点匹配 JSON: {json_path}  匹配数: {pair_count}")
            QMessageBox.information(self, "匹配 JSON 已导出", f"JSON：{json_path}\n匹配数：{pair_count}")
        except Exception as exc:  # noqa: BLE001 - 导出入口需要把文件和字段错误直接反馈给用户。
            self.ui.statusbar.showMessage(f"导出星点匹配 JSON 失败: {exc}")
            QMessageBox.critical(self, "导出星点匹配 JSON 失败", str(exc))

    def import_star_pair_session(self) -> None:
        default_dir = project_root() / "outputs"
        if not default_dir.exists():
            default_dir = project_root()
        default_dir = self._import_dialog_directory(default_dir)
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入星点匹配 JSON",
            str(default_dir),
            STAR_PAIR_SESSION_JSON_FILTER,
        )
        if not file_path:
            return
        self._remember_import_path(file_path)
        self.load_star_pair_session(file_path)

    def load_star_pair_session(
        self,
        file_path: str | Path,
        *,
        switch_to_reference: bool = True,
        show_progress: bool = True,
        clear_input_name: str = "新的匹配 JSON",
        reuse_current_image: bool = True,
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
                title="正在导入匹配 JSON",
                label_text=f"正在读取匹配 JSON 并恢复真实图像...\n{json_path}",
                status_text=f"正在导入星点匹配 JSON: {json_path}",
            )
        else:
            self._json_import_progress = None
            self.ui.statusbar.showMessage(f"正在后台导入星点匹配 JSON: {json_path}")

        current_preview = self.current_image_preview if reuse_current_image else None
        if current_preview is None:
            worker = StarPairSessionImportWorker(json_path)
        else:
            worker = StarPairSessionImportWorker(
                json_path,
                current_image_path=current_preview.path,
                current_image_size=(current_preview.original_width, current_preview.original_height),
            )
        task = start_qt_worker_task(
            parent=self,
            worker=worker,
            finished_signal=worker.finished,
            failed_signal=worker.failed,
            on_finished=self._handle_star_pair_session_import_finished,
            on_failed=self._handle_star_pair_session_import_failed,
            on_cleanup=self._cleanup_json_import,
            progress_dialog=self._json_import_progress,
        )

        self._json_import_thread = task.thread
        self._json_import_worker = task.worker

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

    def _ensure_pair_record_stars_visible(self, pair_records: list[StarPairRecord]) -> None:
        visible_star_ids = {
            self._star_pair_star_id(row)
            for row in range(self.ui.tableWidgetStarPairs.rowCount())
            if self._star_pair_star_id(row)
        }
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        added_any = False
        for record in pair_records:
            star_id = record.star_id
            if (
                not star_id
                or star_id in visible_star_ids
                or star_id in self._manual_reference_star_ids
                or star_id in auto_match_star_ids
            ):
                continue
            if star_id in self._excluded_reference_star_ids:
                self._excluded_reference_star_ids.remove(star_id)
            if record.is_auto_match:
                group_id = (
                    record.group_id
                    or self._auto_match_group_id_for_star_id(star_id)
                    or "A"
                )
                self._ensure_auto_match_group(group_id, expanded=True)
                self._auto_match_reference_star_ids.append(star_id)
                self._auto_match_group_by_star_id[star_id] = group_id
                auto_match_star_ids.add(star_id)
            else:
                self._manual_reference_star_ids.append(star_id)
            visible_star_ids.add(star_id)
            added_any = True
        if added_any:
            self._refresh_reference_stars_from_current_map()

    def _restore_star_pair_records(self, pair_records: list[StarPairRecord], update_alignment: bool = True) -> int:
        store = getattr(self, "_star_pair_store", None)
        if store is not None:
            store.add_records(pair_records)
        self._normalize_auto_match_groups()

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
            record = self._star_pair_store.get(star_id)
            if record is None:
                position_item.setText("")
                self._refresh_star_pair_quality_cell(row)
                continue
            position_item.setText(self._star_pair_mode_display_text(row))
            self._refresh_star_pair_quality_cell(row)
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
            self.ui.statusbar.showMessage(f"导入星点匹配 JSON 失败: {exc}")
            QMessageBox.critical(self, "导入星点匹配 JSON 失败", str(exc))

    def _handle_star_pair_session_import_failed(self, error_message: str) -> None:
        self.ui.statusbar.showMessage(f"导入星点匹配 JSON 失败: {error_message}")
        QMessageBox.critical(self, "导入星点匹配 JSON 失败", error_message)

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
            raise ValueError("当前只支持 HoshinoPanoAssistant 星点匹配 JSON。")

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

        if preview is None:
            image_path = self._session_real_image_path(payload, source_path)
        else:
            image_path = _validate_star_pair_session_current_image(
                payload,
                preview.path,
                (preview.original_width, preview.original_height),
            )
        self._active_star_pair_row = None
        current_tab = self.ui.tabWidgetMain.currentWidget() if not switch_to_reference else None
        previous_suspend_alignment = self._suspend_alignment_updates
        self._suspend_alignment_updates = True
        try:
            self._set_alignment_model(payload.get("sky_alignment_model"))
            # 匹配会话恢复图像几何与星空范围，但参考星数量沿用当前会话。
            # 这也兼容旧版本把自动匹配总数误写成 reference_star_count 的 JSON。
            self._apply_reference_payload(
                reference_payload,
                source_path,
                preserve_reference_star_count=True,
            )
            if current_tab is not None:
                self.ui.tabWidgetMain.setCurrentWidget(current_tab)
            pair_records = star_pair_records_from_payloads(pair_payloads, observer=self._observer_settings())
            pair_records = [
                replace(
                    record,
                    fit_constraint_mode=auto_match_constraints.get(
                        record.star_id,
                        (record.fit_constraint_mode, record.fit_weight),
                    )[0],
                    fit_weight=auto_match_constraints.get(
                        record.star_id,
                        (record.fit_constraint_mode, record.fit_weight),
                    )[1],
                    group_id=auto_match_group_by_star_id.get(record.star_id, record.group_id),
                    group_name=(
                        self._auto_match_group_label(auto_match_group_by_star_id[record.star_id])
                        if record.star_id in auto_match_group_by_star_id
                        else record.group_name
                    ),
                )
                for record in pair_records
            ]
            self._merge_imported_reference_stars_from_pairs(pair_records)
            self._auto_match_reference_star_ids = auto_match_star_ids
            self._auto_match_group_order = auto_match_group_order
            self._auto_match_group_by_star_id = auto_match_group_by_star_id
            self._auto_match_group_expanded_by_id = auto_match_group_expanded_by_id
            self._auto_match_next_group_index = auto_match_next_group_index
            self._normalize_auto_match_groups()
            self._refresh_reference_stars_from_current_map()
            self._ensure_pair_record_stars_visible(pair_records)
            if preview is None:
                preview = load_image_preview(image_path, max_long_side_px=None)
            self._apply_loaded_image_preview(
                preview,
                clear_existing_pairs=False,
                switch_to_reference=switch_to_reference,
            )
            restored_count = self._restore_star_pair_records(pair_records, update_alignment=False)
        finally:
            self._suspend_alignment_updates = previous_suspend_alignment
        self._update_reference_alignment_transform()
        if switch_to_reference:
            self.ui.tabWidgetMain.setCurrentWidget(self.ui.tabReferenceImage)
        elif current_tab is not None:
            self.ui.tabWidgetMain.setCurrentWidget(current_tab)
        self.ui.statusbar.showMessage(
            f"已导入星点匹配 JSON: {source_path}  真实图像: {image_path}  恢复匹配: {restored_count}"
        )
