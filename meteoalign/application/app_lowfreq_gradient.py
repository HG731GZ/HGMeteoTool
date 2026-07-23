from __future__ import annotations

import sys
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..photometric.lowfreq.pipeline import (
    run_low_frequency_correction,
    validate_low_frequency_inputs,
)
from ..photometric.lowfreq.types import LowFrequencyRunResult, SolverConfig


_LOW_FREQUENCY_TUNING_GUIDE_HTML = """
<h2>周边梯度优化调参指南</h2>
<p><b>先检查、再求解、最后输出。</b>本功能不会修改 model.json，也不会覆盖原始 TIFF。
V6 会直接分析各 model.json 的 ICRS 覆盖，不再需要取景 JSON。</p>

<h3>推荐起点</h3>
<ul>
  <li>模型：<b>对数增益（V3）</b></li>
  <li>网格 8×6；采样长边 360；缩小倍率 8；Patch 11</li>
  <li>空间平滑 λ=30；单帧偏移 λ=0.05；floor=64</li>
  <li>V4 关闭；V5 开启（6 节点、平滑 λ=300）；Huber 开启</li>
  <li>首次勾选“仅求解和诊断”，确认结果自然后再写校正 TIFF</li>
</ul>

<h3>按这个顺序调</h3>
<ol>
  <li>点击“检查输入与蒙版绑定”，确认 TIFF、model.json 和天空蒙版对应正确。</li>
  <li>先跑推荐起点；若 V5 亮度分段样本不足，关闭 V5，回退到单层 V3。</li>
  <li>只有少数帧仍有明确的方向性渐变时才开 V4；帧内平面 λ 建议 5000～10000。</li>
  <li>每次只改一类参数，并保留一份基线 diagnostics 便于比较。</li>
</ol>

<h3>关键参数</h3>
<table cellspacing="6">
  <tr><td><b>网格</b></td><td>6×4 更保守，8×6 为默认；出现波纹或斑点时降低网格。</td></tr>
  <tr><td><b>采样长边</b></td><td>快速试跑 96～180，正式处理 360；窄重叠可提高到 500～700。</td></tr>
  <tr><td><b>缩小倍率</b></td><td>8 为默认；16 更快，4 适合小图或很窄的重叠。</td></tr>
  <tr><td><b>Patch</b></td><td>11 为默认；噪声或星点影响大时用 13～17，窄重叠用 7～9。</td></tr>
  <tr><td><b>空间平滑 λ</b></td><td>场有波纹就增大；过平且仍有平滑接缝残差时小幅减小。</td></tr>
  <tr><td><b>floor</b></td><td>增益模型暗部不稳时从 64 提高到 128～512。</td></tr>
</table>

<h3>怎样判断结果</h3>
<ul>
  <li>RMS 应明显下降，但不能只追求最低数值。</li>
  <li>correction field 应非常平滑，不应出现星点、银河纹理或局部云结构。</li>
  <li>observability 大面积偏暗：降低网格、增加采样或检查蒙版。</li>
  <li>frame offsets / gradients 个别值异常大：先检查云、蒙版、曝光参数和几何模型。</li>
  <li>V5 仅在多数亮度区间都有样本、各层场连续且分区 RMS 普遍改善时采用。</li>
</ul>

<p><b>常用回退：</b>8×6、关闭 V4/V5、提高空间平滑 λ、重新检查蒙版和 model.json。
最终请用完全相同的 PTGui 参数比较原图与校正图，重点看接缝、银河结构、星点 halo、
星色、地景边缘和 360° 闭环。</p>
"""


class LowFrequencyTuningGuideDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("周边梯度优化 · 调参指南")
        self.setMinimumSize(620, 520)
        self.resize(760, 680)
        layout = QVBoxLayout(self)
        self.guideBrowser = QTextBrowser(self)
        self.guideBrowser.setObjectName("textBrowserLowFrequencyTuningGuide")
        self.guideBrowser.setOpenExternalLinks(False)
        self.guideBrowser.setHtml(_LOW_FREQUENCY_TUNING_GUIDE_HTML)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.close)
        layout.addWidget(self.guideBrowser)
        layout.addWidget(buttons)


class LowFrequencyCorrectionWorker(QObject):
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        input_directory: Path,
        output_directory: Path,
        config: SolverConfig,
    ) -> None:
        super().__init__()
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.config = config
        self._cancel_requested = Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            result = run_low_frequency_correction(
                input_directory=self.input_directory,
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
        self.ui.lineEditLowFrequencyOutputDirectory = QLineEdit(input_group)
        input_form.addRow("源图目录", self.ui.lineEditLowFrequencyInputDirectory)
        input_form.addRow("输出目录", self.ui.lineEditLowFrequencyOutputDirectory)
        input_layout.addLayout(input_form)
        v6_note = QLabel(
            "V6：直接使用各帧 model.json 的 ICRS 覆盖生成重叠采样，"
            "无需选择取景 JSON。",
            input_group,
        )
        v6_note.setWordWrap(True)
        v6_note.setObjectName("labelLowFrequencyV6Sampling")
        input_layout.addWidget(v6_note)
        browse_row = QHBoxLayout()
        self.ui.pushButtonBrowseLowFrequencyInput = QPushButton("源图…", input_group)
        self.ui.pushButtonBrowseLowFrequencyOutput = QPushButton("输出…", input_group)
        browse_row.addWidget(self.ui.pushButtonBrowseLowFrequencyInput)
        browse_row.addWidget(self.ui.pushButtonBrowseLowFrequencyOutput)
        input_layout.addLayout(browse_row)
        controls_layout.addWidget(input_group)

        solver_group = QGroupBox("共享场与 V3–V6 参数", controls)
        solver_form = QFormLayout(solver_group)
        self.ui.comboBoxLowFrequencyCorrectionModel = QComboBox(solver_group)
        self.ui.comboBoxLowFrequencyCorrectionModel.addItem(
            "加性码值（V1/V2）",
            "additive",
        )
        self.ui.comboBoxLowFrequencyCorrectionModel.addItem(
            "乘法增益（V3）",
            "multiplicative",
        )
        self.ui.comboBoxLowFrequencyCorrectionModel.addItem(
            "对数增益（V3，推荐）",
            "log_gain",
        )
        self.ui.spinBoxLowFrequencyGridColumns = self._integer_control(4, 20, 8)
        self.ui.spinBoxLowFrequencyGridRows = self._integer_control(4, 16, 6)
        self.ui.spinBoxLowFrequencySampleLongSide = self._integer_control(64, 1200, 360, 20)
        self.ui.spinBoxLowFrequencyDownsample = self._integer_control(1, 32, 8)
        self.ui.spinBoxLowFrequencyPatchSize = self._integer_control(3, 31, 11, 2)
        self.ui.doubleSpinBoxLowFrequencySmooth = self._decimal_control(0.0, 10000.0, 30.0, 1.0)
        self.ui.doubleSpinBoxLowFrequencyFrameOffset = self._decimal_control(0.0, 100.0, 0.05, 0.01, 3)
        self.ui.doubleSpinBoxLowFrequencyIntensityFloor = self._decimal_control(
            1.0,
            4096.0,
            64.0,
            16.0,
            1,
        )
        self.ui.checkBoxLowFrequencyFramePlane = QCheckBox(
            "允许每帧弱 a·x+b·y（V4）",
            solver_group,
        )
        self.ui.doubleSpinBoxLowFrequencyFramePlaneLambda = self._decimal_control(
            0.0,
            1000000.0,
            5000.0,
            500.0,
            1,
        )
        self.ui.checkBoxLowFrequencyBrightnessNonlinear = QCheckBox(
            "按有效亮度自适应分段（V5）",
            solver_group,
        )
        self.ui.spinBoxLowFrequencyBrightnessKnots = self._integer_control(3, 12, 6)
        self.ui.doubleSpinBoxLowFrequencyBrightnessSmooth = self._decimal_control(
            0.0,
            100000.0,
            300.0,
            50.0,
            1,
        )
        self.ui.checkBoxLowFrequencyHuber = QCheckBox("IRLS + Huber（4 次）", solver_group)
        self.ui.checkBoxLowFrequencyHuber.setChecked(True)
        self.ui.checkBoxLowFrequencySolveOnly = QCheckBox("仅求解和诊断，不写校正 TIFF", solver_group)
        self.ui.checkBoxLowFrequencySolveOnly.setChecked(False)
        solver_form.addRow("校正模型", self.ui.comboBoxLowFrequencyCorrectionModel)
        solver_form.addRow("控制网格列", self.ui.spinBoxLowFrequencyGridColumns)
        solver_form.addRow("控制网格行", self.ui.spinBoxLowFrequencyGridRows)
        solver_form.addRow("采样长边", self.ui.spinBoxLowFrequencySampleLongSide)
        solver_form.addRow("图像缩小倍率", self.ui.spinBoxLowFrequencyDownsample)
        solver_form.addRow("Patch 边长", self.ui.spinBoxLowFrequencyPatchSize)
        solver_form.addRow("二阶平滑 λ", self.ui.doubleSpinBoxLowFrequencySmooth)
        solver_form.addRow("单帧偏移 λ", self.ui.doubleSpinBoxLowFrequencyFrameOffset)
        solver_form.addRow("暗部稳定 floor", self.ui.doubleSpinBoxLowFrequencyIntensityFloor)
        solver_form.addRow(self.ui.checkBoxLowFrequencyFramePlane)
        solver_form.addRow(
            "帧内平面 λ",
            self.ui.doubleSpinBoxLowFrequencyFramePlaneLambda,
        )
        solver_form.addRow(self.ui.checkBoxLowFrequencyBrightnessNonlinear)
        solver_form.addRow(
            "V5 亮度节点",
            self.ui.spinBoxLowFrequencyBrightnessKnots,
        )
        solver_form.addRow(
            "V5 节点平滑 λ",
            self.ui.doubleSpinBoxLowFrequencyBrightnessSmooth,
        )
        solver_form.addRow(self.ui.checkBoxLowFrequencyHuber)
        solver_form.addRow(self.ui.checkBoxLowFrequencySolveOnly)
        controls_layout.addWidget(solver_group)

        action_group = QGroupBox("运行", controls)
        action_layout = QVBoxLayout(action_group)
        self.ui.pushButtonLowFrequencyTuningGuide = QPushButton(
            "调参指南",
            action_group,
        )
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
        action_layout.addWidget(self.ui.pushButtonLowFrequencyTuningGuide)
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
        self._low_frequency_guide_dialog: LowFrequencyTuningGuideDialog | None = None
        self.ui.pushButtonBrowseLowFrequencyInput.clicked.connect(self._browse_low_frequency_input)
        self.ui.pushButtonBrowseLowFrequencyOutput.clicked.connect(self._browse_low_frequency_output)
        self.ui.pushButtonLowFrequencyTuningGuide.clicked.connect(
            self._show_low_frequency_tuning_guide
        )
        self.ui.pushButtonValidateLowFrequencyInputs.clicked.connect(self._validate_low_frequency_inputs)
        self.ui.pushButtonStartLowFrequencyCorrection.clicked.connect(self._start_low_frequency_correction)
        self.ui.pushButtonCancelLowFrequencyCorrection.clicked.connect(self._cancel_low_frequency_correction)
        self.ui.checkBoxLowFrequencyFramePlane.toggled.connect(
            self._update_low_frequency_advanced_controls
        )
        self.ui.checkBoxLowFrequencyBrightnessNonlinear.toggled.connect(
            self._update_low_frequency_advanced_controls
        )
        self._update_low_frequency_advanced_controls()

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

    def _browse_low_frequency_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择空的输出目录",
            self.ui.lineEditLowFrequencyOutputDirectory.text().strip(),
        )
        if selected:
            self.ui.lineEditLowFrequencyOutputDirectory.setText(selected)

    def _low_frequency_paths(self) -> tuple[Path, Path]:
        input_text = self.ui.lineEditLowFrequencyInputDirectory.text().strip()
        output_text = self.ui.lineEditLowFrequencyOutputDirectory.text().strip()
        if not input_text or not output_text:
            raise ValueError("请完整选择源图目录和输出目录。")
        return Path(input_text), Path(output_text)

    def _show_low_frequency_tuning_guide(self) -> None:
        if self._low_frequency_guide_dialog is None:
            self._low_frequency_guide_dialog = LowFrequencyTuningGuideDialog(self)
        self._low_frequency_guide_dialog.show()
        self._low_frequency_guide_dialog.raise_()
        self._low_frequency_guide_dialog.activateWindow()

    def _validate_low_frequency_inputs(self) -> None:
        try:
            input_directory, _output_directory = self._low_frequency_paths()
            frames = validate_low_frequency_inputs(input_directory)
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
            correction_model=str(
                self.ui.comboBoxLowFrequencyCorrectionModel.currentData()
            ),
            intensity_floor_code=float(
                self.ui.doubleSpinBoxLowFrequencyIntensityFloor.value()
            ),
            enable_frame_plane=self.ui.checkBoxLowFrequencyFramePlane.isChecked(),
            frame_plane_lambda=float(
                self.ui.doubleSpinBoxLowFrequencyFramePlaneLambda.value()
            ),
            enable_brightness_nonlinearity=(
                self.ui.checkBoxLowFrequencyBrightnessNonlinear.isChecked()
            ),
            brightness_knot_count=int(
                self.ui.spinBoxLowFrequencyBrightnessKnots.value()
            ),
            brightness_smooth_lambda=float(
                self.ui.doubleSpinBoxLowFrequencyBrightnessSmooth.value()
            ),
            robust_loss="huber" if self.ui.checkBoxLowFrequencyHuber.isChecked() else "none",
            irls_iterations=4 if self.ui.checkBoxLowFrequencyHuber.isChecked() else 1,
            apply_correction=not self.ui.checkBoxLowFrequencySolveOnly.isChecked(),
        ).validated()

    def _update_low_frequency_advanced_controls(self) -> None:
        self.ui.doubleSpinBoxLowFrequencyFramePlaneLambda.setEnabled(
            self.ui.checkBoxLowFrequencyFramePlane.isChecked()
        )
        brightness_enabled = (
            self.ui.checkBoxLowFrequencyBrightnessNonlinear.isChecked()
        )
        self.ui.spinBoxLowFrequencyBrightnessKnots.setEnabled(brightness_enabled)
        self.ui.doubleSpinBoxLowFrequencyBrightnessSmooth.setEnabled(
            brightness_enabled
        )

    def _start_low_frequency_correction(self) -> None:
        if self._low_frequency_thread is not None:
            return
        try:
            input_directory, output_directory = self._low_frequency_paths()
            config = self._low_frequency_config()
        except Exception as exc:  # noqa: BLE001 - 参数错误应留在页面日志。
            self._append_low_frequency_log(f"[参数错误] {exc}")
            return
        self._append_low_frequency_log("=" * 72)
        self._append_low_frequency_log(
            f"开始：model={config.correction_model}, "
            f"V4={'on' if config.enable_frame_plane else 'off'}, "
            f"V5={'on' if config.enable_brightness_nonlinearity else 'off'}, "
            f"grid={config.grid_columns}×{config.grid_rows}, "
            f"sample={config.sample_long_side_px}, downsample=1/{config.downsample_factor}, "
            f"patch={config.patch_size_px}"
        )
        worker = LowFrequencyCorrectionWorker(
            input_directory=input_directory,
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
