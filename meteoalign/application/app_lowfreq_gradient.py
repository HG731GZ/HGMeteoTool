from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..catalog import project_root
from ..image_path_resolution import is_reserved_mask_path
from ..image_preview import IMAGE_FILE_FILTER
from ..photometric.lowfreq.masking import read_binary_mask
from ..photometric.lowfreq.pipeline import (
    photometric_frame_from_image,
    run_low_frequency_correction,
    validate_low_frequency_inputs,
    validate_photometric_frame,
)
from ..photometric.lowfreq.types import (
    LowFrequencyRunResult,
    PhotometricFrame,
    SolverConfig,
)
from .file_dialogs import get_multiple_open_file_names, get_open_file_name


_LOW_FREQUENCY_TIFF_FILTER = "16-bit RGB TIFF (*.tif *.tiff)"
_LOW_FREQUENCY_LOG_SEPARATOR = "=" * 72


_LOW_FREQUENCY_TUNING_GUIDE_HTML = """
<h2>周边梯度优化调参指南</h2>
<p><b>先检查、再求解、最后决定是否导出校正图像。</b>本功能不会修改 model.json，
也不会覆盖原始 TIFF。采样直接来自各 model.json 的 ICRS 覆盖，不需要取景 JSON。</p>

<h3>推荐起点</h3>
<ul>
  <li>校正模型：<b>对数增益（推荐）</b></li>
  <li>控制网格列 8；控制网格行 6；采样长边 360；图像缩小倍率 8；Patch 边长 11</li>
  <li>二阶平滑 λ=30；单帧偏移 λ=0.05；暗部稳定 floor=64</li>
  <li>“允许每帧弱 a·x+b·y”关闭；“按有效亮度自适应分段”开启；
  亮度节点 6；节点平滑 λ=300；“IRLS + Huber（4 次）”开启</li>
  <li>默认只保存方案和诊断；确认结果自然后，再勾选“导出校正图像”</li>
</ul>

<h3>按这个顺序调</h3>
<ol>
  <li>导入多张 16-bit RGB TIFF；缺少同名 model.json 或格式不符的图像会被过滤。</li>
  <li>点击“检查输入与蒙版绑定”，确认 TIFF、model.json 和天空蒙版对应正确。</li>
  <li>先跑推荐起点；若亮度分段样本不足，关闭“按有效亮度自适应分段”，回退到单层对数增益。</li>
  <li>只有少数帧仍有明确的方向性渐变时才开启“允许每帧弱 a·x+b·y”；
  帧内平面 λ 建议 5000～10000。</li>
  <li>每次只改一类参数，并保留一份基线 diagnostics 便于比较。</li>
</ol>

<h3>关键参数</h3>
<table cellspacing="6">
  <tr><td><b>控制网格列/行</b></td><td>6×4 更保守，8×6 为默认；出现波纹或斑点时降低网格。</td></tr>
  <tr><td><b>采样长边</b></td><td>快速试跑 96～180，正式处理 360；窄重叠可提高到 500～700。</td></tr>
  <tr><td><b>图像缩小倍率</b></td><td>8 为默认；16 更快，4 适合小图或很窄的重叠。</td></tr>
  <tr><td><b>Patch 边长</b></td><td>11 为默认；噪声或星点影响大时用 13～17，窄重叠用 7～9。</td></tr>
  <tr><td><b>二阶平滑 λ</b></td><td>场有波纹就增大；过平且仍有平滑接缝残差时小幅减小。</td></tr>
  <tr><td><b>暗部稳定 floor</b></td><td>增益模型暗部不稳时从 64 提高到 128～512。</td></tr>
</table>

<h3>怎样判断结果</h3>
<ul>
  <li>RMS 应明显下降，但不能只追求最低数值。</li>
  <li>correction field 应非常平滑，不应出现星点、银河纹理或局部云结构。</li>
  <li>observability 大面积偏暗：降低控制网格列/行、增加采样长边或检查蒙版。</li>
  <li>frame offsets / gradients 个别值异常大：先检查云、蒙版、曝光参数和几何模型。</li>
  <li>“按有效亮度自适应分段”仅在多数亮度区间都有样本、各层场连续且分区 RMS
  普遍改善时采用。</li>
</ul>

<p><b>常用回退：</b>控制网格列/行使用 8×6、关闭“允许每帧弱 a·x+b·y”和
“按有效亮度自适应分段”、提高二阶平滑 λ、重新检查蒙版和 model.json。
最终请用完全相同的 PTGui 参数比较原图与校正图，重点看接缝、银河结构、星点 halo、
星色、地景边缘和 360° 闭环。</p>

<h3>输出位置</h3>
<p>方案保存为原图同路径下的 photometric_solution.json，诊断保存在
gradient_diagnostics。再次求解会覆盖这些结果。勾选“导出校正图像”后，
原文件名图像会写入 gradient_corrected。全景图批处理也可直接导入方案，
在内存中完成梯度校正后立即重投影。未参与方案计算、但采用相同拍摄和 Camera Raw
处理流程的同尺寸图像也可复用方案；此时只应用公共低频校正场，不应用无法估计的
逐帧亮度偏移或弱平面。</p>
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
        frames: tuple[PhotometricFrame, ...],
        config: SolverConfig,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.config = config
        self._cancel_requested = Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            result = run_low_frequency_correction(
                frames=self.frames,
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

        input_group = QGroupBox("图像", controls)
        input_layout = QVBoxLayout(input_group)
        sampling_note = QLabel(
            "导入多张 16-bit RGB TIFF；缺少同名 model.json 的图像会被过滤。"
            "同路径的“原图名_Mask.*”会自动关联为蒙版。",
            input_group,
        )
        sampling_note.setWordWrap(True)
        sampling_note.setObjectName("labelLowFrequencySampling")
        input_layout.addWidget(sampling_note)
        browse_row = QHBoxLayout()
        self.ui.pushButtonImportLowFrequencyImages = QPushButton("导入图像…", input_group)
        self.ui.pushButtonRemoveLowFrequencyImages = QPushButton("移除所选", input_group)
        self.ui.pushButtonClearLowFrequencyImages = QPushButton("清空", input_group)
        browse_row.addWidget(self.ui.pushButtonImportLowFrequencyImages)
        browse_row.addWidget(self.ui.pushButtonRemoveLowFrequencyImages)
        browse_row.addWidget(self.ui.pushButtonClearLowFrequencyImages)
        input_layout.addLayout(browse_row)
        controls_layout.addWidget(input_group)

        solver_group = QGroupBox("共享场与校正参数", controls)
        solver_form = QFormLayout(solver_group)
        self.ui.comboBoxLowFrequencyCorrectionModel = QComboBox(solver_group)
        self.ui.comboBoxLowFrequencyCorrectionModel.addItem(
            "加性码值",
            "additive",
        )
        self.ui.comboBoxLowFrequencyCorrectionModel.addItem(
            "乘法增益",
            "multiplicative",
        )
        self.ui.comboBoxLowFrequencyCorrectionModel.addItem(
            "对数增益（推荐）",
            "log_gain",
        )
        self.ui.comboBoxLowFrequencyCorrectionModel.setCurrentIndex(
            self.ui.comboBoxLowFrequencyCorrectionModel.findData("log_gain")
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
            "允许每帧弱 a·x+b·y",
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
            "按有效亮度自适应分段",
            solver_group,
        )
        self.ui.checkBoxLowFrequencyBrightnessNonlinear.setChecked(True)
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
        self.ui.checkBoxLowFrequencyExportCorrected = QCheckBox(
            "导出校正图像",
            solver_group,
        )
        self.ui.checkBoxLowFrequencyExportCorrected.setChecked(False)
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
            "亮度节点",
            self.ui.spinBoxLowFrequencyBrightnessKnots,
        )
        solver_form.addRow(
            "节点平滑 λ",
            self.ui.doubleSpinBoxLowFrequencyBrightnessSmooth,
        )
        solver_form.addRow(self.ui.checkBoxLowFrequencyHuber)
        solver_form.addRow(self.ui.checkBoxLowFrequencyExportCorrected)
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

        results_panel = QWidget(page)
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        image_list_group = QGroupBox("图像列表", results_panel)
        image_list_layout = QVBoxLayout(image_list_group)
        self.ui.tableWidgetLowFrequencyImages = QTableWidget(image_list_group)
        self.ui.tableWidgetLowFrequencyImages.setColumnCount(3)
        self.ui.tableWidgetLowFrequencyImages.setHorizontalHeaderLabels(
            ("序号", "文件名", "蒙版")
        )
        self.ui.tableWidgetLowFrequencyImages.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.ui.tableWidgetLowFrequencyImages.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.ui.tableWidgetLowFrequencyImages.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.ui.tableWidgetLowFrequencyImages.setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self.ui.tableWidgetLowFrequencyImages.verticalHeader().setVisible(False)
        image_header = self.ui.tableWidgetLowFrequencyImages.horizontalHeader()
        image_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        image_header.setSectionResizeMode(1, QHeaderView.Stretch)
        image_header.setSectionResizeMode(2, QHeaderView.Stretch)
        image_list_layout.addWidget(self.ui.tableWidgetLowFrequencyImages)
        results_layout.addWidget(image_list_group, 1)

        log_group = QGroupBox("运行日志（Debug）", results_panel)
        log_layout = QVBoxLayout(log_group)
        self.ui.plainTextEditLowFrequencyLog = QPlainTextEdit(log_group)
        self.ui.plainTextEditLowFrequencyLog.setReadOnly(True)
        self.ui.plainTextEditLowFrequencyLog.setLineWrapMode(QPlainTextEdit.NoWrap)
        log_layout.addWidget(self.ui.plainTextEditLowFrequencyLog)
        results_layout.addWidget(log_group, 1)
        page_layout.addWidget(results_panel, 1)
        self.ui.tabLowFrequencyGradient = page
        self.ui.scrollAreaLowFrequencyControls = controls_scroll
        self.ui.horizontalLayoutLowFrequencyGradient = page_layout
        self.ui.tabWidgetMain.addTab(page, "周边梯度优化")

        self._low_frequency_thread: QThread | None = None
        self._low_frequency_worker: LowFrequencyCorrectionWorker | None = None
        self._low_frequency_guide_dialog: LowFrequencyTuningGuideDialog | None = None
        self._low_frequency_frames: list[PhotometricFrame] = []
        self.ui.pushButtonImportLowFrequencyImages.clicked.connect(
            self._import_low_frequency_images
        )
        self.ui.pushButtonRemoveLowFrequencyImages.clicked.connect(
            self._remove_selected_low_frequency_images
        )
        self.ui.pushButtonClearLowFrequencyImages.clicked.connect(
            self._clear_low_frequency_images
        )
        self.ui.tableWidgetLowFrequencyImages.customContextMenuRequested.connect(
            self._show_low_frequency_image_context_menu
        )
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
        self._update_low_frequency_image_controls()

        batch_tab = getattr(self.ui, "tabMosaicBatch", None)
        if batch_tab is not None:
            batch_index = self.ui.tabWidgetMain.indexOf(batch_tab)
            if batch_index >= 0 and batch_index != self.ui.tabWidgetMain.count() - 1:
                batch_text = self.ui.tabWidgetMain.tabText(batch_index)
                self.ui.tabWidgetMain.removeTab(batch_index)
                self.ui.tabWidgetMain.addTab(batch_tab, batch_text)
        install_wheel_filters = getattr(
            self,
            "_install_value_control_wheel_filters",
            None,
        )
        if callable(install_wheel_filters):
            # 本页是运行时动态创建的，必须在控件创建后补装全局滚轮保护。
            install_wheel_filters()

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

    def _append_low_frequency_result_block(
        self,
        status: str,
        *details: str,
    ) -> None:
        """用纯文本分隔块突出一次运行的最终状态。"""

        self._append_low_frequency_log(_LOW_FREQUENCY_LOG_SEPARATOR)
        self._append_low_frequency_log(f"最终运行结果：{status}")
        for detail in details:
            self._append_low_frequency_log(detail)
        self._append_low_frequency_log(_LOW_FREQUENCY_LOG_SEPARATOR)

    def _low_frequency_default_directory(self) -> Path:
        if self._low_frequency_frames:
            default_directory = self._low_frequency_frames[-1].image_path.parent
        else:
            default_directory = project_root() / "testimages"
            if not default_directory.exists():
                default_directory = project_root()
        import_directory = getattr(self, "_import_dialog_directory", None)
        if callable(import_directory):
            return Path(import_directory(default_directory))
        return default_directory

    def _import_low_frequency_images(self) -> None:
        file_paths, _selected_filter = get_multiple_open_file_names(
            self,
            "导入梯度校正图像",
            str(self._low_frequency_default_directory()),
            _LOW_FREQUENCY_TIFF_FILTER,
        )
        if not file_paths:
            return
        remember_path = getattr(self, "_remember_import_path", None)
        if callable(remember_path):
            remember_path(file_paths)
        imported_count, issues = self._add_low_frequency_image_paths(file_paths)
        self._refresh_low_frequency_image_table()
        if issues:
            QMessageBox.warning(
                self,
                "部分图像已过滤",
                "\n".join(issues[:16]) + ("\n…" if len(issues) > 16 else ""),
            )
        self.ui.labelLowFrequencyStatus.setText(
            f"已导入 {len(self._low_frequency_frames)} 帧"
        )
        self._append_low_frequency_log(
            f"本次导入 {imported_count} 张；列表共 {len(self._low_frequency_frames)} 张。"
        )

    def _add_low_frequency_image_paths(
        self,
        file_paths: list[str] | tuple[str, ...],
    ) -> tuple[int, list[str]]:
        existing = {
            frame.image_path.expanduser().resolve()
            for frame in self._low_frequency_frames
        }
        expected_size = (
            None
            if not self._low_frequency_frames
            else (
                self._low_frequency_frames[0].width_px,
                self._low_frequency_frames[0].height_px,
            )
        )
        imported_count = 0
        issues: list[str] = []
        for file_path in file_paths:
            image_path = Path(file_path).expanduser().resolve()
            if image_path in existing:
                continue
            if is_reserved_mask_path(image_path):
                issues.append(f"{image_path.name}：蒙版不能作为原图导入")
                continue
            try:
                frame = photometric_frame_from_image(
                    image_path,
                    index=len(self._low_frequency_frames),
                )
                try:
                    validate_photometric_frame(
                        frame,
                        expected_size=expected_size,
                    )
                except Exception as exc:
                    if frame.mask_path is None:
                        raise
                    mask_name = frame.mask_path.name
                    frame = replace(frame, mask_path=None)
                    validate_photometric_frame(
                        frame,
                        expected_size=expected_size,
                    )
                    issues.append(
                        f"{image_path.name}：自动蒙版 {mask_name} 无效，已忽略（{exc}）"
                    )
            except Exception as exc:  # noqa: BLE001 - 单张异常需要过滤，其余图像继续导入。
                issues.append(f"{image_path.name}：{exc}")
                continue
            self._low_frequency_frames.append(frame)
            existing.add(image_path)
            if expected_size is None:
                expected_size = (frame.width_px, frame.height_px)
            imported_count += 1
        self._renumber_low_frequency_frames()
        return imported_count, issues

    def _renumber_low_frequency_frames(self) -> None:
        self._low_frequency_frames = [
            replace(frame, index=index)
            for index, frame in enumerate(self._low_frequency_frames)
        ]

    def _refresh_low_frequency_image_table(self) -> None:
        table = self.ui.tableWidgetLowFrequencyImages
        table.setRowCount(len(self._low_frequency_frames))
        for row, frame in enumerate(self._low_frequency_frames):
            index_item = QTableWidgetItem(str(row + 1))
            name_item = QTableWidgetItem(frame.image_path.name)
            mask_text = frame.mask_path.name if frame.mask_path is not None else "无"
            mask_item = QTableWidgetItem(mask_text)
            for item in (index_item, name_item, mask_item):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            name_item.setToolTip(str(frame.image_path))
            mask_item.setToolTip(
                "" if frame.mask_path is None else str(frame.mask_path)
            )
            table.setItem(row, 0, index_item)
            table.setItem(row, 1, name_item)
            table.setItem(row, 2, mask_item)
        self._update_low_frequency_image_controls()

    def _selected_low_frequency_rows(self) -> list[int]:
        selection_model = self.ui.tableWidgetLowFrequencyImages.selectionModel()
        if selection_model is None:
            return []
        return sorted({index.row() for index in selection_model.selectedRows()})

    def _remove_selected_low_frequency_images(self) -> None:
        selected_rows = self._selected_low_frequency_rows()
        for row in reversed(selected_rows):
            if 0 <= row < len(self._low_frequency_frames):
                del self._low_frequency_frames[row]
        self._renumber_low_frequency_frames()
        self._refresh_low_frequency_image_table()

    def _clear_low_frequency_images(self) -> None:
        self._low_frequency_frames = []
        self._refresh_low_frequency_image_table()
        self.ui.labelLowFrequencyStatus.setText("待处理")

    def _update_low_frequency_image_controls(self) -> None:
        has_frames = bool(self._low_frequency_frames)
        has_enough_frames = len(self._low_frequency_frames) >= 2
        running = self._low_frequency_thread is not None
        self.ui.pushButtonValidateLowFrequencyInputs.setEnabled(
            has_enough_frames and not running
        )
        self.ui.pushButtonStartLowFrequencyCorrection.setEnabled(
            has_enough_frames and not running
        )
        self.ui.pushButtonRemoveLowFrequencyImages.setEnabled(
            has_frames and not running
        )
        self.ui.pushButtonClearLowFrequencyImages.setEnabled(
            has_frames and not running
        )

    def _show_low_frequency_image_context_menu(self, position) -> None:  # type: ignore[no-untyped-def]
        table = self.ui.tableWidgetLowFrequencyImages
        clicked_index = table.indexAt(position)
        if clicked_index.isValid() and clicked_index.row() not in self._selected_low_frequency_rows():
            table.clearSelection()
            table.selectRow(clicked_index.row())
        selected_rows = self._selected_low_frequency_rows()
        if not selected_rows:
            return
        menu = QMenu(table)
        select_mask_action = menu.addAction("选择蒙版文件")
        clear_mask_action = menu.addAction("清除蒙版")
        selected_action = menu.exec_(table.viewport().mapToGlobal(position))
        if selected_action is select_mask_action:
            self._select_low_frequency_mask_for_rows(selected_rows)
        elif selected_action is clear_mask_action:
            for row in selected_rows:
                self._low_frequency_frames[row] = replace(
                    self._low_frequency_frames[row],
                    mask_path=None,
                )
            self._refresh_low_frequency_image_table()

    def _select_low_frequency_mask_for_rows(self, rows: list[int]) -> None:
        first_frame = self._low_frequency_frames[rows[0]]
        file_path, _selected_filter = get_open_file_name(
            self,
            "选择天空区域蒙版",
            str(first_frame.image_path.parent),
            IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        mask_path = Path(file_path).expanduser().resolve()
        remember_path = getattr(self, "_remember_import_path", None)
        if callable(remember_path):
            remember_path(file_path)
        try:
            mask = read_binary_mask(mask_path)
            expected_shape = (first_frame.height_px, first_frame.width_px)
            if mask.shape != expected_shape:
                raise ValueError(
                    f"蒙版尺寸 {mask.shape[1]}×{mask.shape[0]}，"
                    f"原图尺寸 {expected_shape[1]}×{expected_shape[0]}"
                )
            if not bool(mask.any()):
                raise ValueError("蒙版中没有非零有效区域")
            for row in rows:
                frame = self._low_frequency_frames[row]
                if (frame.height_px, frame.width_px) != expected_shape:
                    raise ValueError("所选图像尺寸不一致，不能共用该蒙版。")
        except Exception as exc:  # noqa: BLE001 - 蒙版错误应以对话框反馈。
            QMessageBox.warning(self, "蒙版不可用", str(exc))
            return
        for row in rows:
            self._low_frequency_frames[row] = replace(
                self._low_frequency_frames[row],
                mask_path=mask_path,
            )
        self._refresh_low_frequency_image_table()

    def _show_low_frequency_tuning_guide(self) -> None:
        if self._low_frequency_guide_dialog is None:
            self._low_frequency_guide_dialog = LowFrequencyTuningGuideDialog(self)
        self._low_frequency_guide_dialog.show()
        self._low_frequency_guide_dialog.raise_()
        self._low_frequency_guide_dialog.activateWindow()

    def _validate_low_frequency_inputs(self) -> None:
        try:
            frames = validate_low_frequency_inputs(
                tuple(self._low_frequency_frames)
            )
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
            apply_correction=self.ui.checkBoxLowFrequencyExportCorrected.isChecked(),
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
            frames = validate_low_frequency_inputs(
                tuple(self._low_frequency_frames)
            )
            config = self._low_frequency_config()
        except Exception as exc:  # noqa: BLE001 - 参数错误应留在页面日志。
            self._append_low_frequency_log(f"[参数错误] {exc}")
            return
        self._append_low_frequency_log(_LOW_FREQUENCY_LOG_SEPARATOR)
        self._append_low_frequency_log(
            f"开始：model={config.correction_model}, "
            f"frame_plane={'on' if config.enable_frame_plane else 'off'}, "
            "brightness_layers="
            f"{'on' if config.enable_brightness_nonlinearity else 'off'}, "
            f"grid={config.grid_columns}×{config.grid_rows}, "
            f"sample={config.sample_long_side_px}, downsample=1/{config.downsample_factor}, "
            f"patch={config.patch_size_px}"
        )
        worker = LowFrequencyCorrectionWorker(
            frames=frames,
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
        self.ui.pushButtonCancelLowFrequencyCorrection.setEnabled(running)
        self.ui.pushButtonImportLowFrequencyImages.setEnabled(not running)
        self.ui.tableWidgetLowFrequencyImages.setEnabled(not running)
        self._update_low_frequency_image_controls()
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
        self._append_low_frequency_result_block(
            "处理完成",
            f"Solution：{result.solution_path}",
            f"Diagnostics：{result.diagnostics_directory}",
            f"校正 TIFF：{len(result.corrected_paths)} 张",
        )

    def _handle_low_frequency_cancelled(self, message: str) -> None:
        self.ui.labelLowFrequencyStatus.setText("已取消")
        self._append_low_frequency_result_block("已取消", f"原因：{message}")

    def _handle_low_frequency_failed(self, message: str) -> None:
        self.ui.labelLowFrequencyStatus.setText("处理失败")
        self._append_low_frequency_result_block("处理失败", f"错误：{message}")
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
