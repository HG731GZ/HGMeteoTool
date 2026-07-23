from __future__ import annotations

import sys
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..photometric.lowfreq.pipeline import (
    run_low_frequency_correction,
    validate_low_frequency_inputs,
)
from ..photometric.lowfreq.types import LowFrequencyRunResult, SolverConfig


class LowFrequencyCorrectionWorker(QObject):
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        input_directory: Path,
        framing_path: Path,
        output_directory: Path,
        config: SolverConfig,
    ) -> None:
        super().__init__()
        self.input_directory = input_directory
        self.framing_path = framing_path
        self.output_directory = output_directory
        self.config = config
        self._cancel_requested = Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            result = run_low_frequency_correction(
                input_directory=self.input_directory,
                framing_path=self.framing_path,
                output_directory=self.output_directory,
                config=self.config,
                progress_callback=lambda message, value, total: self.progress.emit(
                    message,
                    int(value),
                    int(total),
                ),
                cancel_callback=self._cancel_requested.is_set,
            )
            self.finished.emit(result)
        except InterruptedError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - 后台任务必须把输入/数值/IO 错误带回日志。
            self.failed.emit(str(exc))


class LowFrequencyGradientMixin:
    """“周边梯度优化”页面及后台任务生命周期。"""

    ui: object

    def _init_low_frequency_gradient_page(self) -> None:
        page = QWidget()
        page.setObjectName("tabLowFrequencyGradient")
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(6 if sys.platform == "darwin" else 9, 6, 6, 6)
        page_layout.setSpacing(6)

        controls_scroll = QScrollArea(page)
        controls_scroll.setObjectName("scrollAreaLowFrequencyControls")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(320 if sys.platform == "darwin" else 350)
        controls_scroll.setMaximumWidth(360 if sys.platform == "darwin" else 390)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        input_group = QGroupBox("输入与输出", controls)
        input_layout = QVBoxLayout(input_group)
        input_form = QFormLayout()
        self.ui.lineEditLowFrequencyInputDirectory = QLineEdit(input_group)
        self.ui.lineEditLowFrequencyFramingJson = QLineEdit(input_group)
        self.ui.lineEditLowFrequencyOutputDirectory = QLineEdit(input_group)
        input_form.addRow("源图目录", self.ui.lineEditLowFrequencyInputDirectory)
        input_form.addRow("取景 JSON", self.ui.lineEditLowFrequencyFramingJson)
        input_form.addRow("输出目录", self.ui.lineEditLowFrequencyOutputDirectory)
        input_layout.addLayout(input_form)
        browse_row = QHBoxLayout()
        self.ui.pushButtonBrowseLowFrequencyInput = QPushButton("源图…", input_group)
        self.ui.pushButtonBrowseLowFrequencyFraming = QPushButton("取景…", input_group)
        self.ui.pushButtonBrowseLowFrequencyOutput = QPushButton("输出…", input_group)
        browse_row.addWidget(self.ui.pushButtonBrowseLowFrequencyInput)
        browse_row.addWidget(self.ui.pushButtonBrowseLowFrequencyFraming)
        browse_row.addWidget(self.ui.pushButtonBrowseLowFrequencyOutput)
        input_layout.addLayout(browse_row)
        controls_layout.addWidget(input_group)

        solver_group = QGroupBox("V1 求解参数", controls)
        solver_form = QFormLayout(solver_group)
        self.ui.spinBoxLowFrequencyGridColumns = self._integer_control(4, 20, 8)
        self.ui.spinBoxLowFrequencyGridRows = self._integer_control(4, 16, 6)
        self.ui.spinBoxLowFrequencySampleLongSide = self._integer_control(64, 1200, 360, 20)
        self.ui.spinBoxLowFrequencyDownsample = self._integer_control(1, 32, 8)
        self.ui.spinBoxLowFrequencyPatchSize = self._integer_control(3, 31, 11, 2)
        self.ui.doubleSpinBoxLowFrequencySmooth = self._decimal_control(0.0, 10000.0, 30.0, 1.0)
        self.ui.doubleSpinBoxLowFrequencyFrameOffset = self._decimal_control(0.0, 100.0, 0.05, 0.01, 3)
        self.ui.checkBoxLowFrequencyHuber = QCheckBox("IRLS + Huber（4 次）", solver_group)
        self.ui.checkBoxLowFrequencyHuber.setChecked(True)
        self.ui.checkBoxLowFrequencySolveOnly = QCheckBox("仅求解和诊断，不写校正 TIFF", solver_group)
        self.ui.checkBoxLowFrequencySolveOnly.setChecked(False)
        solver_form.addRow("控制网格列", self.ui.spinBoxLowFrequencyGridColumns)
        solver_form.addRow("控制网格行", self.ui.spinBoxLowFrequencyGridRows)
        solver_form.addRow("采样长边", self.ui.spinBoxLowFrequencySampleLongSide)
        solver_form.addRow("图像缩小倍率", self.ui.spinBoxLowFrequencyDownsample)
        solver_form.addRow("Patch 边长", self.ui.spinBoxLowFrequencyPatchSize)
        solver_form.addRow("二阶平滑 λ", self.ui.doubleSpinBoxLowFrequencySmooth)
        solver_form.addRow("单帧偏移 λ", self.ui.doubleSpinBoxLowFrequencyFrameOffset)
        solver_form.addRow(self.ui.checkBoxLowFrequencyHuber)
        solver_form.addRow(self.ui.checkBoxLowFrequencySolveOnly)
        controls_layout.addWidget(solver_group)

        action_group = QGroupBox("运行", controls)
        action_layout = QVBoxLayout(action_group)
        self.ui.pushButtonValidateLowFrequencyInputs = QPushButton("检查输入与蒙版绑定", action_group)
        action_row = QHBoxLayout()
        self.ui.pushButtonStartLowFrequencyCorrection = QPushButton("开始处理", action_group)
        self.ui.pushButtonCancelLowFrequencyCorrection = QPushButton("取消", action_group)
        self.ui.pushButtonCancelLowFrequencyCorrection.setEnabled(False)
        action_row.addWidget(self.ui.pushButtonStartLowFrequencyCorrection)
        action_row.addWidget(self.ui.pushButtonCancelLowFrequencyCorrection)
        self.ui.progressBarLowFrequency = QProgressBar(action_group)
        self.ui.progressBarLowFrequency.setRange(0, 100)
        self.ui.progressBarLowFrequency.setValue(0)
        self.ui.labelLowFrequencyStatus = QLabel("待处理", action_group)
        self.ui.labelLowFrequencyStatus.setWordWrap(True)
        action_layout.addWidget(self.ui.pushButtonValidateLowFrequencyInputs)
        action_layout.addLayout(action_row)
        action_layout.addWidget(self.ui.progressBarLowFrequency)
        action_layout.addWidget(self.ui.labelLowFrequencyStatus)
        controls_layout.addWidget(action_group)
        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls)
        page_layout.addWidget(controls_scroll)

        log_group = QGroupBox("运行日志（Debug）", page)
        log_layout = QVBoxLayout(log_group)
        self.ui.plainTextEditLowFrequencyLog = QPlainTextEdit(log_group)
        self.ui.plainTextEditLowFrequencyLog.setReadOnly(True)
        self.ui.plainTextEditLowFrequencyLog.setLineWrapMode(QPlainTextEdit.NoWrap)
        log_layout.addWidget(self.ui.plainTextEditLowFrequencyLog)
        page_layout.addWidget(log_group, 1)
        self.ui.tabLowFrequencyGradient = page
        self.ui.scrollAreaLowFrequencyControls = controls_scroll
        self.ui.horizontalLayoutLowFrequencyGradient = page_layout
        self.ui.tabWidgetMain.addTab(page, "周边梯度优化")

        self._low_frequency_thread: QThread | None = None
        self._low_frequency_worker: LowFrequencyCorrectionWorker | None = None
        self.ui.pushButtonBrowseLowFrequencyInput.clicked.connect(self._browse_low_frequency_input)
        self.ui.pushButtonBrowseLowFrequencyFraming.clicked.connect(self._browse_low_frequency_framing)
        self.ui.pushButtonBrowseLowFrequencyOutput.clicked.connect(self._browse_low_frequency_output)
        self.ui.pushButtonValidateLowFrequencyInputs.clicked.connect(self._validate_low_frequency_inputs)
        self.ui.pushButtonStartLowFrequencyCorrection.clicked.connect(self._start_low_frequency_correction)
        self.ui.pushButtonCancelLowFrequencyCorrection.clicked.connect(self._cancel_low_frequency_correction)

    @staticmethod
    def _integer_control(
        minimum: int,
        maximum: int,
        value: int,
        step: int = 1,
    ) -> QSpinBox:
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setValue(value)
        return control

    @staticmethod
    def _decimal_control(
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 2,
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setDecimals(decimals)
        control.setValue(value)
        return control

    def _append_low_frequency_log(self, message: str) -> None:
        self.ui.plainTextEditLowFrequencyLog.appendPlainText(str(message))

    def _set_low_frequency_paths_from_input(self, directory: Path) -> None:
        self.ui.lineEditLowFrequencyInputDirectory.setText(str(directory))
        framing = directory / "mosaic_framing.json"
        if framing.is_file():
            self.ui.lineEditLowFrequencyFramingJson.setText(str(framing))
        if not self.ui.lineEditLowFrequencyOutputDirectory.text().strip():
            self.ui.lineEditLowFrequencyOutputDirectory.setText(str(directory / "lowfreq_output"))

    def _browse_low_frequency_input(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择 TIFF 与 model.json 所在目录",
            self.ui.lineEditLowFrequencyInputDirectory.text().strip(),
        )
        if selected:
            self._set_low_frequency_paths_from_input(Path(selected))

    def _browse_low_frequency_framing(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "选择自由投影取景 JSON",
            self.ui.lineEditLowFrequencyFramingJson.text().strip(),
            "取景 JSON (*.json);;JSON 文件 (*.json)",
        )
        if selected:
            self.ui.lineEditLowFrequencyFramingJson.setText(selected)

    def _browse_low_frequency_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择空的输出目录",
            self.ui.lineEditLowFrequencyOutputDirectory.text().strip(),
        )
        if selected:
            self.ui.lineEditLowFrequencyOutputDirectory.setText(selected)

    def _low_frequency_paths(self) -> tuple[Path, Path, Path]:
        input_text = self.ui.lineEditLowFrequencyInputDirectory.text().strip()
        framing_text = self.ui.lineEditLowFrequencyFramingJson.text().strip()
        output_text = self.ui.lineEditLowFrequencyOutputDirectory.text().strip()
        if not input_text or not framing_text or not output_text:
            raise ValueError("请完整选择源图目录、取景 JSON 和输出目录。")
        return Path(input_text), Path(framing_text), Path(output_text)

    def _validate_low_frequency_inputs(self) -> None:
        try:
            input_directory, framing_path, _output_directory = self._low_frequency_paths()
            frames = validate_low_frequency_inputs(input_directory, framing_path)
        except Exception as exc:  # noqa: BLE001 - 输入检查结果直接展示在 debug 日志。
            self._append_low_frequency_log(f"[输入错误] {exc}")
            self.ui.labelLowFrequencyStatus.setText("输入检查失败")
            return
        self._append_low_frequency_log(f"输入检查通过：{len(frames)} 张 16-bit RGB TIFF。")
        for frame in frames:
            mask_text = frame.mask_path.name if frame.mask_path is not None else "全天空/未指定蒙版"
            self._append_low_frequency_log(
                f"  [{frame.index:02d}] {frame.image_path.name} ← {frame.model_path.name}；蒙版：{mask_text}"
            )
        self.ui.labelLowFrequencyStatus.setText(f"输入有效：{len(frames)} 帧")

    def _low_frequency_config(self) -> SolverConfig:
        patch_size = int(self.ui.spinBoxLowFrequencyPatchSize.value())
        if patch_size % 2 == 0:
            patch_size += 1
        return SolverConfig(
            grid_columns=int(self.ui.spinBoxLowFrequencyGridColumns.value()),
            grid_rows=int(self.ui.spinBoxLowFrequencyGridRows.value()),
            sample_long_side_px=int(self.ui.spinBoxLowFrequencySampleLongSide.value()),
            downsample_factor=int(self.ui.spinBoxLowFrequencyDownsample.value()),
            patch_size_px=patch_size,
            smooth_lambda=float(self.ui.doubleSpinBoxLowFrequencySmooth.value()),
            frame_offset_lambda=float(self.ui.doubleSpinBoxLowFrequencyFrameOffset.value()),
            robust_loss="huber" if self.ui.checkBoxLowFrequencyHuber.isChecked() else "none",
            irls_iterations=4 if self.ui.checkBoxLowFrequencyHuber.isChecked() else 1,
            apply_correction=not self.ui.checkBoxLowFrequencySolveOnly.isChecked(),
        ).validated()

    def _start_low_frequency_correction(self) -> None:
        if self._low_frequency_thread is not None:
            return
        try:
            input_directory, framing_path, output_directory = self._low_frequency_paths()
            config = self._low_frequency_config()
        except Exception as exc:  # noqa: BLE001 - 参数错误应留在页面日志。
            self._append_low_frequency_log(f"[参数错误] {exc}")
            return
        self._append_low_frequency_log("=" * 72)
        self._append_low_frequency_log(
            f"开始 V1：grid={config.grid_columns}×{config.grid_rows}, "
            f"sample={config.sample_long_side_px}, downsample=1/{config.downsample_factor}, "
            f"patch={config.patch_size_px}"
        )
        worker = LowFrequencyCorrectionWorker(
            input_directory=input_directory,
            framing_path=framing_path,
            output_directory=output_directory,
            config=config,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_low_frequency_progress)
        worker.finished.connect(self._handle_low_frequency_finished)
        worker.cancelled.connect(self._handle_low_frequency_cancelled)
        worker.failed.connect(self._handle_low_frequency_failed)
        worker.finished.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_low_frequency_worker)
        self._low_frequency_worker = worker
        self._low_frequency_thread = thread
        self._set_low_frequency_running(True)
        thread.start()

    def _set_low_frequency_running(self, running: bool) -> None:
        self.ui.pushButtonStartLowFrequencyCorrection.setEnabled(not running)
        self.ui.pushButtonValidateLowFrequencyInputs.setEnabled(not running)
        self.ui.pushButtonCancelLowFrequencyCorrection.setEnabled(running)
        if running:
            self.ui.labelLowFrequencyStatus.setText("处理中…")

    def _handle_low_frequency_progress(self, message: str, value: int, total: int) -> None:
        self._append_low_frequency_log(message)
        self.ui.labelLowFrequencyStatus.setText(message)
        if total > 0:
            self.ui.progressBarLowFrequency.setRange(0, 100)
            self.ui.progressBarLowFrequency.setValue(
                min(100, max(0, int(round(100.0 * value / total))))
            )
        else:
            self.ui.progressBarLowFrequency.setRange(0, 0)

    def _handle_low_frequency_finished(self, result: LowFrequencyRunResult) -> None:
        self.ui.progressBarLowFrequency.setRange(0, 100)
        self.ui.progressBarLowFrequency.setValue(100)
        self.ui.labelLowFrequencyStatus.setText("处理完成")
        self._append_low_frequency_log(f"Solution：{result.solution_path}")
        self._append_low_frequency_log(f"Diagnostics：{result.diagnostics_directory}")
        self._append_low_frequency_log(f"校正 TIFF：{len(result.corrected_paths)} 张")

    def _handle_low_frequency_cancelled(self, message: str) -> None:
        self.ui.labelLowFrequencyStatus.setText("已取消")
        self._append_low_frequency_log(f"[已取消] {message}")

    def _handle_low_frequency_failed(self, message: str) -> None:
        self.ui.labelLowFrequencyStatus.setText("处理失败")
        self._append_low_frequency_log(f"[处理失败] {message}")
        QMessageBox.warning(self, "周边梯度优化失败", message)

    def _cleanup_low_frequency_worker(self) -> None:
        self._low_frequency_thread = None
        self._low_frequency_worker = None
        self._set_low_frequency_running(False)

    def _cancel_low_frequency_correction(self) -> None:
        if self._low_frequency_worker is not None:
            self._low_frequency_worker.request_cancel()
            self.ui.pushButtonCancelLowFrequencyCorrection.setEnabled(False)
            self.ui.labelLowFrequencyStatus.setText("正在取消…")
            self._append_low_frequency_log("已请求取消，将在下一个安全检查点停止。")
