"""流星自动检测选项对话框。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog

from ..meteor_detection import METEOR_DETECTION_PROVIDERS, MeteorDetectionOptions
from ..ui.ui_meteor_detection_options_dialog import Ui_MeteorDetectionOptionsDialog
from .file_dialogs import get_open_file_name


class MeteorDetectionOptionsDialog(QDialog):
    """编辑引擎位置和 MetDet worker 的关键检测参数。"""

    def __init__(
        self,
        options: MeteorDetectionOptions,
        parent=None,  # type: ignore[no-untyped-def]
    ) -> None:
        super().__init__(parent)
        self.ui = Ui_MeteorDetectionOptionsDialog()
        self.ui.setupUi(self)
        self.ui.pushButtonBrowseEngineFile.clicked.connect(self._browse_engine_file)
        self.ui.pushButtonBrowseEngineDirectory.clicked.connect(self._browse_engine_directory)
        self.ui.pushButtonBrowseModel.clicked.connect(self._browse_model)
        restore_button = self.ui.buttonBox.button(QDialogButtonBox.RestoreDefaults)
        if restore_button is not None:
            restore_button.clicked.connect(lambda: self.set_options(MeteorDetectionOptions()))
        self.set_options(options)

    def set_options(self, options: MeteorDetectionOptions) -> None:
        """把配置显示到控件中。"""

        self.ui.lineEditEnginePath.setText(options.engine_path)
        self.ui.lineEditModelPath.setText(options.model_path)
        self.ui.doubleSpinBoxConfidenceThreshold.setValue(options.confidence_threshold)
        self.ui.doubleSpinBoxNmsThreshold.setValue(options.nms_threshold)
        self.ui.spinBoxMultiscale.setValue(options.multiscale)
        self.ui.spinBoxPartition.setValue(options.partition)
        provider_index = self.ui.comboBoxProvider.findData(options.provider)
        self.ui.comboBoxProvider.setCurrentIndex(max(provider_index, 0))
        self.ui.doubleSpinBoxBoxExpansionRatio.setValue(options.box_expansion_ratio)

    def options(self) -> MeteorDetectionOptions:
        """返回当前控件值。"""

        provider = str(self.ui.comboBoxProvider.currentData() or "auto")
        if provider not in METEOR_DETECTION_PROVIDERS:
            provider = "auto"
        return MeteorDetectionOptions(
            engine_path=self.ui.lineEditEnginePath.text().strip(),
            model_path=self.ui.lineEditModelPath.text().strip(),
            confidence_threshold=self.ui.doubleSpinBoxConfidenceThreshold.value(),
            nms_threshold=self.ui.doubleSpinBoxNmsThreshold.value(),
            multiscale=self.ui.spinBoxMultiscale.value(),
            partition=self.ui.spinBoxPartition.value(),
            provider=provider,
            box_expansion_ratio=self.ui.doubleSpinBoxBoxExpansionRatio.value(),
        )

    def _browse_engine_file(self) -> None:
        fallback = self._existing_parent(self.ui.lineEditEnginePath.text())
        selected, _selected_filter = get_open_file_name(
            self,
            "选择 MetDet worker",
            str(fallback),
            "MetDet worker (metdet_worker metdet_worker.exe metdet_worker.py)",
        )
        if selected:
            self.ui.lineEditEnginePath.setText(selected)

    def _browse_engine_directory(self) -> None:
        fallback = self._existing_parent(self.ui.lineEditEnginePath.text())
        selected = QFileDialog.getExistingDirectory(self, "选择 MetDet worker 目录", str(fallback))
        if selected:
            self.ui.lineEditEnginePath.setText(selected)

    def _browse_model(self) -> None:
        fallback = self._existing_parent(self.ui.lineEditModelPath.text())
        selected, _selected_filter = get_open_file_name(
            self,
            "选择流星检测模型",
            str(fallback),
            "ONNX 模型 (*.onnx)",
        )
        if selected:
            self.ui.lineEditModelPath.setText(selected)

    @staticmethod
    def _existing_parent(text: str) -> Path:
        path = Path(text).expanduser() if text.strip() else Path.cwd()
        if path.is_file():
            path = path.parent
        if path.is_dir():
            return path
        return path.parent if path.parent.is_dir() else Path.cwd()


__all__ = ["MeteorDetectionOptionsDialog"]
