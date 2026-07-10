from __future__ import annotations

from PyQt5.QtWidgets import QApplication, QProgressDialog

from ..qt_tasks import create_progress_dialog

class StarPairJsonTaskMixin:
    """JSON 导入后台任务与进度对话框生命周期。"""

    def _clear_star_pair_positions_for_new_input(self, input_name: str) -> int:
        """导入新的图像或 JSON 前，重置整套星点配对会话状态。

        旧实现只在已有有效坐标时清空坐标；如果上一轮只留下未配对参考星、
        自动匹配分组或导入参考星缓存，新图就会继续复用这些状态。
        """
        pair_count = self._star_pair_position_count()
        table = self.ui.tableWidgetStarPairs
        row_count = table.rowCount()

        if getattr(self, "_active_star_pair_row", None) is not None:
            if hasattr(self, "_leave_star_pick_mode"):
                self._leave_star_pick_mode()
            else:
                self._active_star_pair_row = None

        store = getattr(self, "_star_pair_store", None)
        if store is not None:
            store.clear()

        if hasattr(self, "_clear_star_pair_annotations"):
            self._clear_star_pair_annotations()

        signals_were_blocked = table.blockSignals(True)
        try:
            table.setRowCount(0)
        finally:
            table.blockSignals(signals_were_blocked)

        self._manual_match_group_expanded = True
        self._manual_reference_star_ids = []
        self._imported_reference_star_by_id = {}
        self._auto_match_reference_star_ids = []
        self._auto_match_group_order = []
        self._auto_match_group_by_star_id = {}
        self._auto_match_group_expanded_by_id = {}
        self._auto_match_next_group_index = 0
        self._excluded_reference_star_ids = []
        self._mask_excluded_reference_star_ids = set()

        self._sky_alignment_transform = None
        self._source_astrometric_model = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""

        # 参考星选择必须回到星空模拟原生星图，不能继续使用上一张图的对齐星图。
        self._current_reference_star_map = getattr(self, "_current_star_map", None)
        if self._current_reference_star_map is not None and hasattr(self, "_refresh_reference_stars_from_current_map"):
            self._refresh_reference_stars_from_current_map()
        if hasattr(self, "_update_reference_alignment_transform"):
            self._update_reference_alignment_transform()

        if pair_count > 0:
            self.ui.statusbar.showMessage(f"导入{input_name}前已清除 {pair_count} 个已有匹配。")
        elif row_count > 0:
            self.ui.statusbar.showMessage(f"导入{input_name}前已重置旧的参考星列表。")
        return pair_count

    def _show_json_import_progress(
        self,
        title: str,
        label_text: str,
        status_text: str,
    ) -> QProgressDialog:
        dialog = create_progress_dialog(
            self,
            title=title,
            label_text=label_text,
            minimum=0,
            maximum=0,
        )
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
