from __future__ import annotations

from .app_star_pair_io_common import *  # noqa: F401, F403

class StarPairJsonTaskMixin:
    """JSON 导入后台任务与进度对话框生命周期。"""

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

