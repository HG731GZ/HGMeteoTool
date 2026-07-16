from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QMessageBox


class ImageGroupMixin:
    """多图导入后的图像组列表、切换确认与状态同步。"""

    def _show_image_group_assistant(self) -> None:
        """显示单实例非模态图像组助手，并将已有窗口提到前台。"""

        if not self._image_group_mode_active():
            return
        self.image_group_assistant.refresh_file_statuses()
        self._sync_image_group_current_image()
        self.image_group_assistant.show()
        self.image_group_assistant.raise_()
        self.image_group_assistant.activateWindow()

    def _normalized_image_group_paths(
        self,
        image_paths: list[str] | tuple[str, ...],
    ) -> tuple[Path, ...]:
        """规范化图像路径，并保持文件对话框中的原始顺序。"""

        normalized: list[Path] = []
        for raw_path in image_paths:
            image_path = Path(raw_path).expanduser().resolve()
            if image_path not in normalized:
                normalized.append(image_path)
        return tuple(normalized)

    def _set_image_group_paths(
        self,
        image_paths: list[str] | tuple[str, ...],
    ) -> tuple[Path, ...]:
        """进入多图模式；不足两张时回到单图模式。"""

        normalized = self._normalized_image_group_paths(image_paths)
        self._image_group_paths = normalized if len(normalized) > 1 else ()
        self.image_group_assistant.set_image_paths(self._image_group_paths)
        if not self._image_group_paths:
            self.image_group_assistant.close()
        self._update_image_group_controls()
        return normalized

    def _reset_image_group_status(self) -> None:
        """清除图像组，使星点匹配页恢复单图模式。"""

        self._image_group_paths = ()
        self.image_group_assistant.set_image_paths(())
        self.image_group_assistant.close()
        self._update_image_group_controls()

    def _image_group_mode_active(self) -> bool:
        sequence_mode = bool(hasattr(self, "_sequence_mode_active") and self._sequence_mode_active())
        return len(getattr(self, "_image_group_paths", ())) > 1 and not sequence_mode

    def _image_group_controls_idle(self) -> bool:
        return (
            getattr(self, "_image_import_thread", None) is None
            and getattr(self, "_json_import_thread", None) is None
            and getattr(self, "_sequence_import_thread", None) is None
            and not bool(getattr(self, "_sequence_processing_active", False))
        )

    def _update_image_group_controls(self) -> None:
        button = getattr(self.ui, "pushButtonOpenImageGroupAssistant", None)
        if button is not None:
            button.setEnabled(self._image_group_mode_active() and self._image_group_controls_idle())
        if hasattr(self, "_update_adjacent_framing_controls"):
            self._update_adjacent_framing_controls()

    def _sync_image_group_current_image(self, image_path: str | Path | None = None) -> None:
        """让助手列表选中主窗口当前图像。"""

        if image_path is None:
            preview = getattr(self, "current_image_preview", None)
            image_path = preview.path if preview is not None else None
        self.image_group_assistant.set_current_image(image_path)

    def _refresh_image_group_assistant_status(self) -> None:
        self.image_group_assistant.refresh_file_statuses()
        self._sync_image_group_current_image()

    def _current_image_group_output_paths(self) -> tuple[Path, Path] | None:
        preview = getattr(self, "current_image_preview", None)
        if preview is None:
            return None
        image_path = Path(preview.path).expanduser().resolve()
        return self._star_pair_session_path_for_image(image_path), self._source_model_path_for_image(image_path)

    def _current_image_group_missing_outputs(self) -> tuple[Path, ...]:
        if self._star_pair_position_count() <= 0:
            return ()
        output_paths = self._current_image_group_output_paths()
        if output_paths is None:
            return ()
        return tuple(path for path in output_paths if not path.is_file())

    def _confirm_image_group_switch(self, target_path: Path) -> bool:
        """当前图像有尚未落盘的输出时，询问用户如何处理。"""

        missing_outputs = self._current_image_group_missing_outputs()
        if not missing_outputs:
            return True

        output_names = "、".join("匹配" if path.name.endswith("_starpairs.json") else "映射" for path in missing_outputs)
        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Warning)
        message_box.setWindowTitle("当前匹配尚未保存")
        message_box.setText(f"当前图像的{output_names}尚未保存。是否保存后跳转到 {target_path.name}？")
        save_button = message_box.addButton("保存并跳转", QMessageBox.AcceptRole)
        discard_button = message_box.addButton("不保存", QMessageBox.DestructiveRole)
        cancel_button = message_box.addButton("取消", QMessageBox.RejectRole)
        message_box.setDefaultButton(save_button)
        message_box.exec_()
        clicked_button = message_box.clickedButton()
        if clicked_button is cancel_button:
            self._sync_image_group_current_image()
            return False
        if clicked_button is discard_button:
            return True
        if clicked_button is save_button:
            return self._save_image_group_outputs_before_switch()
        self._sync_image_group_current_image()
        return False

    def _save_image_group_outputs_before_switch(self) -> bool:
        """静默保存当前匹配与映射；失败时留在当前图像。"""

        try:
            pair_result = self._write_current_star_pair_session()
            if pair_result is None:
                self._sync_image_group_current_image()
                return False
            model_result = self._write_current_source_model(preload_to_mosaic=False)
            if model_result is None:
                self._sync_image_group_current_image()
                return False
        except Exception as exc:  # noqa: BLE001 - 保存失败必须阻止切图并直接反馈原因。
            self.ui.statusbar.showMessage(f"保存当前匹配与映射失败: {exc}")
            QMessageBox.critical(self, "保存当前匹配与映射失败", str(exc))
            self._sync_image_group_current_image()
            return False
        self._refresh_image_group_assistant_status()
        self.ui.statusbar.showMessage("当前图像的匹配与映射已保存，正在切换图像。")
        return True

    def _handle_image_group_image_activated(self, image_path: object) -> None:
        """双击图像组行后，按需确认保存并加载目标图像。"""

        if not self._image_group_mode_active() or not self._image_group_controls_idle():
            self._sync_image_group_current_image()
            return
        target_path = Path(image_path).expanduser().resolve()
        if target_path not in self._image_group_paths:
            self._sync_image_group_current_image()
            return

        current_preview = getattr(self, "current_image_preview", None)
        if current_preview is not None:
            current_path = Path(current_preview.path).expanduser().resolve()
            if current_path == target_path:
                self._sync_image_group_current_image(current_path)
                return
        if not self._confirm_image_group_switch(target_path):
            return

        # 新图应用时会统一重置参考星列表；若存在同名匹配 JSON，随后由现有自动导入流程恢复。
        self.start_single_image_import(target_path, preserve_image_group_status=True)


__all__ = ["ImageGroupMixin"]
