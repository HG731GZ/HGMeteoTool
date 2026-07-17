from __future__ import annotations

from PyQt5.QtWidgets import QApplication, QProgressDialog

from ..qt_tasks import create_progress_dialog

class StarPairJsonTaskMixin:
    """JSON 导入后台任务与进度对话框生命周期。"""

    def _clear_star_pair_positions_for_new_input(self, input_name: str) -> int:
        """导入新的图像或 JSON 前，重置整套星点匹配会话状态。

        旧实现只在已有有效坐标时清空坐标；如果上一轮只留下未匹配参考星、
        自动匹配分组或导入参考星缓存，新图就会继续复用这些状态。
        """
        if hasattr(self, "_clear_adjacent_rough_framing"):
            self._clear_adjacent_rough_framing(
                status_text="输入已重置，请重新计算粗略取景",
                refresh_alignment=False,
            )
        pair_count, row_count, _rebuilt_count = self.reset_reference_star_list()

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
        self._star_pair_session_import_clear_input_name = "新的匹配 JSON"
        self._star_pair_session_import_restore_observation_time = True
        self._set_json_import_controls_enabled(True)
        if hasattr(self, "_update_image_sequence_controls"):
            self._update_image_sequence_controls()
