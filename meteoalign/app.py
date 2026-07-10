from __future__ import annotations

import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, QEvent, QObject, QPoint, QPointF, QRectF, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QCursor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QHeaderView,
    QTableWidgetItem,
)

from .alignment.constants import (
    MIN_ALIGNMENT_PAIRS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_POLYNOMIAL,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .alignment.fitting import fit_sky_alignment
from .alignment.models import SkyAlignmentTransform
from .catalog_download import ensure_catalogs_ready_or_handle
from .catalog import load_default_catalog, project_root
from .camera_calibration import CameraCalibrationProfile
from .config import StarMapUiConfig, load_star_map_ui_config
from .preference_manager import ensure_preference_file, recent_import_directory, remember_import_path
from .coordinates import radec_to_unit_vectors
from .image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from .milky_way import MilkyWayCatalog, load_milky_way
from .reference import build_reference_payload, save_reference_outputs
from .renderer import StarMapRenderer
from .source_model import SourceAstrometricModel, fit_source_astrometric_model
from .simulator import (
    CameraSettings,
    FISHEYE_EQUIDISTANT,
    FISHEYE_EQUISOLID,
    HorizontalMilkyWayCatalog,
    HorizontalSolarSystemCatalog,
    HorizontalStarCatalog,
    ObserverSettings,
    ProjectedStarMap,
    ReferenceStar,
    RECTILINEAR_LENS_MODEL,
    ViewSettings,
    camera_basis_from_view,
    compute_horizontal_catalog,
    compute_horizontal_milky_way,
    compute_horizontal_solar_system,
    horizontal_fov_deg,
    local_vectors_from_altaz,
    project_horizontal_catalog,
    select_reference_stars,
    vertical_fov_deg,
)
from .star_fitting import FittedStarPosition, fit_star_position
from .star_pair_store import StarPairStore
from .ui.ui_main_window import Ui_MainWindow
from .app_constants import (
    AUTO_MATCH_CONSTRAINT_MODES,
    AUTO_MATCH_CONSTRAINT_SOFT,
    REAL_IMAGE_MAX_ZOOM_SCALE,
)

# ---------------------------------------------------------------------------
# 导入拆分后的模块
# ---------------------------------------------------------------------------
from .app_graphics_items import GraphicsImageItem, LiveStarMapGraphicsItem
from .app_workers import (
    ImagePreviewLoadWorker,
    ImageSequenceCollectWorker,
    SkyMaskLoadWorker,
    ReferenceJsonImportWorker,
    StarPairSessionImportWorker,
)
from .app_widgets import AppWidgetMixin
from .app_star_pair_table import StarPairTableMixin
from .app_alignment import AlignmentMixin
from .app_star_pair_io import StarPairIOMixin
from .app_image import ImageMixin
from .app_sequence import SequenceBatchMixin
from .app_auto_match import AutoMatchMixin
from .app_rendering import RenderingMixin
from .app_view_controls import ViewControlsMixin
from .app_mosaic import MosaicProjectionMixin
from .app_mosaic_batch import MosaicBatchMixin
from .app_utils import (
    _session_image_candidate,
    _resolve_star_pair_session_real_image_path,
    _relative_image_path_for_session,
    _qimage_to_binary_mask,
    _image_with_binary_mask,
)

# ---------------------------------------------------------------------------
# 重新导出辅助函数（供 app_workers 等模块使用）
# ---------------------------------------------------------------------------
# _session_image_candidate, _resolve_star_pair_session_real_image_path,
# _relative_image_path_for_session, _qimage_to_binary_mask, _image_with_binary_mask
# 均已在上面从 app_utils 导入，此处仅保留名称以便兼容旧引用。


# ===================================================================
# MainWindow
# ===================================================================

class MainWindow(
    QMainWindow,
    AppWidgetMixin,
    StarPairTableMixin,
    AlignmentMixin,
    StarPairIOMixin,
    SequenceBatchMixin,
    ImageMixin,
    AutoMatchMixin,
    MosaicProjectionMixin,
    MosaicBatchMixin,
    RenderingMixin,
    ViewControlsMixin,
):
    """HoshinoPanoAssistant 主窗口。

    采用 Mixin 多重继承模式将不同功能拆分到独立模块：
    - AppWidgetMixin: UI 辅助（标签、字体）
    - StarPairTableMixin: 星对表格管理
    - AlignmentMixin: 天球配准与残差
    - StarPairIOMixin: JSON 导入导出
    - ImageMixin: 图像与蒙版导入
    - AutoMatchMixin: 自动匹配场星
    - RenderingMixin: 渲染与模拟
    - ViewControlsMixin: 视图缩放与事件处理
    """

    def _import_dialog_directory(self, fallback: str | Path) -> Path:
        """让所有导入对话框共享最近一次选择的目录。"""

        return recent_import_directory(fallback)

    def _remember_import_path(self, selected: str | Path | list[str] | tuple[str, ...]) -> None:
        """保存最近一次文件导入目录；写入失败不阻断实际导入。"""

        remember_import_path(selected)

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui_config = load_star_map_ui_config()
        self._apply_ui_font_config(self.ui_config)

        self.catalog = load_default_catalog(mag_limit=None)
        self.milky_way_catalog: MilkyWayCatalog = load_milky_way()
        self.renderer = StarMapRenderer(self.ui_config)
        self.scene = QGraphicsScene(self)
        self.star_map_item = LiveStarMapGraphicsItem(self.renderer)
        self.scene.addItem(self.star_map_item)
        self.ui.starMapView.setScene(self.scene)
        self.ui.starMapView.viewport().installEventFilter(self)

        self.reference_scene = QGraphicsScene(self)
        self.reference_star_map_item = LiveStarMapGraphicsItem(self.renderer)
        self.reference_scene.addItem(self.reference_star_map_item)
        self.ui.referenceImageView.setScene(self.reference_scene)
        self.ui.referenceImageView.installEventFilter(self)
        self.ui.referenceImageView.viewport().installEventFilter(self)
        self.ui.referenceImageView.viewport().setFocusPolicy(Qt.StrongFocus)
        self.ui.referenceImageView.viewport().setMouseTracking(True)

        self.real_image_scene = QGraphicsScene(self)
        self.real_image_item = GraphicsImageItem()
        self.real_image_scene.addItem(self.real_image_item)
        self.real_reference_overlay_item = LiveStarMapGraphicsItem(
            self.renderer,
            draw_background=False,
            draw_horizon_shadow=False,
        )
        self.real_reference_overlay_item.setZValue(5.0)
        self.real_reference_overlay_item.setVisible(False)
        self.real_image_scene.addItem(self.real_reference_overlay_item)
        self.ui.realImageView.setScene(self.real_image_scene)
        self.ui.realImageView.installEventFilter(self)
        self.ui.realImageView.viewport().installEventFilter(self)

        self.image_sequence_scene = QGraphicsScene(self)
        self.image_sequence_item = GraphicsImageItem()
        self.image_sequence_scene.addItem(self.image_sequence_item)
        if hasattr(self.ui, "imageSequenceView"):
            self.ui.imageSequenceView.setScene(self.image_sequence_scene)
            self.ui.imageSequenceView.installEventFilter(self)
            self.ui.imageSequenceView.viewport().installEventFilter(self)

        self._init_mosaic_projection_page()
        self._init_mosaic_batch_page()

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render_now)
        self.drag_start: QPoint | None = None
        self.last_drag_pos: QPoint | None = None
        self._horizontal_cache_key: tuple[object, ...] | None = None
        self._horizontal_cache: HorizontalStarCatalog | None = None
        self._milky_way_cache_key: tuple[object, ...] | None = None
        self._milky_way_cache: HorizontalMilkyWayCatalog | None = None
        self._solar_system_cache_key: tuple[object, ...] | None = None
        self._solar_system_cache: HorizontalSolarSystemCatalog | None = None
        self._last_render_size: tuple[int, int] | None = None
        self._last_reference_render_size: tuple[int, int] | None = None
        self.current_image_preview: ImagePreview | None = None
        self._image_import_thread: object | None = None
        self._image_import_worker: ImagePreviewLoadWorker | None = None
        self._image_import_progress: QProgressDialog | None = None
        self._sequence_import_thread: object | None = None
        self._sequence_import_worker: ImageSequenceCollectWorker | None = None
        self._sequence_import_progress: QProgressDialog | None = None
        self._json_import_thread: object | None = None
        self._json_import_worker: QObject | None = None
        self._json_import_progress: QProgressDialog | None = None
        self._star_pair_session_import_switch_to_reference = True
        self._star_pair_session_import_clear_input_name = "新的配对 JSON"
        self._mask_import_thread: object | None = None
        self._mask_import_worker: QObject | None = None
        self._mask_import_progress: QProgressDialog | None = None
        self._real_image_zoom_max_scale = REAL_IMAGE_MAX_ZOOM_SCALE
        self._syncing_camera_dimensions = False
        self._active_star_pair_row: int | None = None
        self._reference_pick_press_pos: QPoint | None = None
        self._star_pick_cursor: QCursor | None = None
        self._star_pick_circle_diameter_px = self.ui_config.star_pick_circle_default_diameter_px
        self._star_pick_previous_drag_mode = self.ui.realImageView.dragMode()
        self._star_pair_annotations: dict[str, tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        self._focused_star_annotations: list[QGraphicsItem] = []
        self._current_star_map: ProjectedStarMap | None = None
        self._current_reference_star_map: ProjectedStarMap | None = None
        self._current_reference_stars: tuple[ReferenceStar, ...] = ()
        self._sky_alignment_transform: SkyAlignmentTransform | None = None
        self._source_astrometric_model: SourceAstrometricModel | None = None
        self._imported_camera_calibration_profile: CameraCalibrationProfile | None = None
        self._imported_camera_calibration_profile_path: Path | None = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""
        self._syncing_reference_real_views = False
        self._syncing_reference_preview_splitter = False
        self._suspend_alignment_updates = False
        self._simulator_controls_locked = False
        self._manual_match_group_expanded = True
        self._manual_reference_star_ids: list[str] = []
        self._imported_reference_star_by_id: dict[str, ReferenceStar] = {}
        self._auto_match_reference_star_ids: list[str] = []
        self._auto_match_group_order: list[str] = []
        self._auto_match_group_by_star_id: dict[str, str] = {}
        self._auto_match_group_expanded_by_id: dict[str, bool] = {}
        self._auto_match_next_group_index = 0
        self._star_pair_store = StarPairStore(self)
        self._star_pair_sort_key: str | None = None
        self._star_pair_sort_descending = True
        self._excluded_reference_star_ids: list[str] = []
        self._mask_excluded_reference_star_ids: set[str] = set()
        self._star_pick_native_zoom_remainder = 0.0
        self.current_sky_mask_path: Path | None = None
        self.current_sky_mask: np.ndarray | None = None
        self.current_sky_masked_image: QImage | None = None
        self._image_sequence_items = []
        self._image_sequence_current_index = -1
        self._image_sequence_sort_key = "index"
        self._image_sequence_sort_descending = False
        self._image_sequence_preview_cache = OrderedDict()
        self._image_sequence_scaled_mask_cache = OrderedDict()
        self._image_sequence_masked_preview_cache = OrderedDict()
        self._sequence_processing_active = False
        self._preserve_sequence_on_next_image_load = False

        self._init_defaults()
        self._connect_inputs()
        self._connect_mosaic_projection_inputs()
        self._connect_mosaic_batch_inputs()
        self._configure_star_pair_table_columns()
        if hasattr(self, "_configure_image_sequence_table_columns"):
            self._configure_image_sequence_table_columns()
        self._configure_reference_preview_splitter()
        self.schedule_render(delay_ms=0)

    def _init_defaults(self) -> None:
        """初始化所有控件默认值。"""
        self.ui.dateTimeEditObservation.setDateTime(QDateTime.currentDateTime())
        utc_offset = datetime.now().astimezone().utcoffset()
        if utc_offset is None:
            utc_offset = timedelta(hours=8)
        self.ui.doubleSpinBoxUtcOffset.setValue(utc_offset.total_seconds() / 3600.0)
        self.ui.doubleSpinBoxLatitude.setValue(self.ui_config.default_latitude_deg)
        self.ui.doubleSpinBoxLongitude.setValue(self.ui_config.default_longitude_deg)
        self.ui.doubleSpinBoxElevation.setValue(self.ui_config.default_elevation_m)
        self.ui.doubleSpinBoxSensorWidth.setValue(36.0)
        self.ui.doubleSpinBoxSensorHeight.setValue(24.0)
        self.ui.spinBoxImageWidth.setValue(1920)
        self.ui.spinBoxImageHeight.setValue(1280)
        self.ui.doubleSpinBoxFocalLength.setValue(24.0)
        self.ui.comboBoxLensModel.setCurrentIndex(0)
        self.ui.doubleSpinBoxFisheyeFov.setValue(180.0)
        self.ui.doubleSpinBoxMagLimit.setValue(6.5)
        self.ui.doubleSpinBoxAz.setValue(0.0)
        self.ui.doubleSpinBoxAlt.setValue(20.0)
        self.ui.doubleSpinBoxRoll.setValue(0.0)
        self.ui.comboBoxReferenceLabelMode.setCurrentIndex(0)
        self.ui.spinBoxReferenceStarCount.setValue(12)
        self.ui.doubleSpinBoxReferenceMagLimit.setValue(3.0)
        self.ui.comboBoxSkyAlignmentModel.setCurrentIndex(0)
        if hasattr(self.ui, "comboBoxProfileSolveMode"):
            self.ui.comboBoxProfileSolveMode.setCurrentIndex(1)
        self.ui.spinBoxAutoMatchCount.setValue(self.ui_config.auto_match_default_new_count)
        constraint_index = (
            AUTO_MATCH_CONSTRAINT_MODES.index(self.ui_config.auto_match_default_constraint_mode)
            if self.ui_config.auto_match_default_constraint_mode in AUTO_MATCH_CONSTRAINT_MODES
            else AUTO_MATCH_CONSTRAINT_MODES.index(AUTO_MATCH_CONSTRAINT_SOFT)
        )
        self.ui.comboBoxAutoMatchConstraintMode.setCurrentIndex(constraint_index)
        self.ui.doubleSpinBoxAutoMatchSoftWeight.setValue(self.ui_config.auto_match_default_soft_weight)
        self.ui.spinBoxAutoMatchRadius.setValue(30)
        self._reset_imported_image_labels()
        self._reset_sky_mask_status()
        self._reset_image_sequence_status()
        self._update_reference_label_controls()
        self._update_auto_match_controls()
        self._update_lens_model_controls()
        self._update_reference_overlay_opacity_label()
        self._update_camera_profile_controls()
        self._update_reference_alignment_controls()

    def _connect_inputs(self) -> None:
        """连接所有 UI 控件信号到对应处理槽。"""
        widgets = (
            self.ui.dateTimeEditObservation,
            self.ui.doubleSpinBoxUtcOffset,
            self.ui.doubleSpinBoxLatitude,
            self.ui.doubleSpinBoxLongitude,
            self.ui.doubleSpinBoxElevation,
            self.ui.doubleSpinBoxFocalLength,
            self.ui.doubleSpinBoxFisheyeFov,
            self.ui.doubleSpinBoxMagLimit,
            self.ui.doubleSpinBoxAz,
            self.ui.doubleSpinBoxAlt,
            self.ui.doubleSpinBoxRoll,
        )
        for widget in widgets:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.schedule_render)
            elif hasattr(widget, "dateTimeChanged"):
                widget.dateTimeChanged.connect(self.schedule_render)
        self.ui.doubleSpinBoxSensorWidth.valueChanged.connect(self._handle_sensor_size_changed)
        self.ui.doubleSpinBoxSensorHeight.valueChanged.connect(self._handle_sensor_size_changed)
        self.ui.doubleSpinBoxUtcOffset.valueChanged.connect(self._handle_image_sequence_time_context_changed)
        self.ui.spinBoxImageWidth.valueChanged.connect(self._handle_image_width_changed)
        self.ui.spinBoxImageHeight.valueChanged.connect(self._handle_image_height_changed)
        self.ui.comboBoxLensModel.currentIndexChanged.connect(self._handle_lens_model_changed)
        self.ui.comboBoxReferenceLabelMode.currentIndexChanged.connect(self._handle_reference_label_mode_changed)
        self.ui.spinBoxReferenceStarCount.valueChanged.connect(self._handle_reference_label_options_changed)
        self.ui.doubleSpinBoxReferenceMagLimit.valueChanged.connect(self._handle_reference_label_options_changed)
        self.ui.pushButtonSwapOrientation.clicked.connect(self._swap_camera_orientation)
        self.ui.pushButtonExportReference.clicked.connect(self.export_reference_map)
        self.ui.pushButtonImportSingleImage.clicked.connect(self.import_single_image)
        self.ui.pushButtonImportImageSequence.clicked.connect(self.import_image_sequence)
        self.ui.pushButtonProcessImageSequence.clicked.connect(self.process_image_sequence)
        if hasattr(self.ui, "pushButtonImportImageSequenceSkyMask"):
            self.ui.pushButtonImportImageSequenceSkyMask.clicked.connect(self.import_image_sequence_sky_mask)
        if hasattr(self.ui, "pushButtonClearImageSequenceSkyMask"):
            self.ui.pushButtonClearImageSequenceSkyMask.clicked.connect(self.clear_image_sequence_sky_mask)
        if hasattr(self.ui, "checkBoxShowImageSequenceMask"):
            self.ui.checkBoxShowImageSequenceMask.toggled.connect(self._handle_image_sequence_mask_toggled)
        self.ui.pushButtonExportStarPairs.clicked.connect(self.export_star_pair_session)
        self.ui.pushButtonImportStarPairs.clicked.connect(self.import_star_pair_session)
        self.ui.pushButtonClearStarPairs.clicked.connect(self.clear_all_star_pair_positions)
        self.ui.pushButtonImportSkyMask.clicked.connect(self.import_sky_mask)
        self.ui.pushButtonClearSkyMask.clicked.connect(self.clear_sky_mask)
        self.ui.checkBoxShowSkyMask.toggled.connect(self._refresh_real_image_display_for_mask)
        self.ui.comboBoxSkyAlignmentModel.currentIndexChanged.connect(self._handle_alignment_model_changed)
        if hasattr(self.ui, "pushButtonImportCameraProfile"):
            self.ui.pushButtonImportCameraProfile.clicked.connect(self.import_camera_calibration_profile)
        if hasattr(self.ui, "pushButtonClearCameraProfile"):
            self.ui.pushButtonClearCameraProfile.clicked.connect(self.clear_camera_calibration_profile)
        if hasattr(self.ui, "comboBoxProfileSolveMode"):
            self.ui.comboBoxProfileSolveMode.currentIndexChanged.connect(self._handle_profile_reuse_options_changed)
        self.ui.comboBoxAutoMatchConstraintMode.currentIndexChanged.connect(self._update_auto_match_controls)
        self.ui.pushButtonAutoMatchFieldStars.clicked.connect(self.auto_match_field_stars)
        self.ui.pushButtonExportSourceModel.clicked.connect(self.export_source_model_json)
        self.ui.tabWidgetMain.currentChanged.connect(self._handle_tab_changed)
        self.ui.tabWidgetMain.currentChanged.connect(lambda _index: self.schedule_mosaic_render())
        self.ui.tableWidgetStarPairs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.tableWidgetStarPairs.customContextMenuRequested.connect(self._show_star_pair_context_menu)
        self.ui.tableWidgetStarPairs.itemChanged.connect(self._handle_star_pair_item_changed)
        self.ui.tableWidgetStarPairs.cellClicked.connect(self._handle_star_pair_cell_clicked)
        self.ui.tableWidgetStarPairs.cellDoubleClicked.connect(self._handle_star_pair_cell_double_clicked)
        self.ui.tableWidgetStarPairs.horizontalHeader().sectionClicked.connect(self._handle_star_pair_header_clicked)
        self.ui.tableWidgetStarPairs.installEventFilter(self)
        self.ui.tableWidgetStarPairs.viewport().installEventFilter(self)
        if hasattr(self.ui, "tableWidgetImageSequence"):
            self.ui.tableWidgetImageSequence.cellClicked.connect(self._handle_image_sequence_cell_clicked)
            self.ui.tableWidgetImageSequence.horizontalHeader().sectionClicked.connect(
                self._handle_image_sequence_header_clicked
            )
            self.ui.tableWidgetImageSequence.installEventFilter(self)
            self.ui.tableWidgetImageSequence.viewport().installEventFilter(self)
        if hasattr(self.ui, "toolButtonImageSequencePrevious"):
            self.ui.toolButtonImageSequencePrevious.clicked.connect(self.show_previous_image_sequence_frame)
        if hasattr(self.ui, "toolButtonImageSequenceNext"):
            self.ui.toolButtonImageSequenceNext.clicked.connect(self.show_next_image_sequence_frame)
        self.ui.labelImportedImagePath.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.labelImportedImagePath.customContextMenuRequested.connect(self._show_imported_image_path_context_menu)
        self.ui.labelImportedImagePath.installEventFilter(self)
        self.ui.labelSkyMaskStatus.installEventFilter(self)
        self.ui.labelAlignmentTransformStatus.installEventFilter(self)
        if hasattr(self.ui, "labelImageSequenceStatus"):
            self.ui.labelImageSequenceStatus.installEventFilter(self)
        if hasattr(self.ui, "labelImageSequenceSummary"):
            self.ui.labelImageSequenceSummary.installEventFilter(self)
        if hasattr(self.ui, "labelImageSequencePreviewTitle"):
            self.ui.labelImageSequencePreviewTitle.installEventFilter(self)
        self.ui.pushButtonImportReferenceJson.clicked.connect(self.import_reference_json)
        self.ui.checkBoxOverlayReferenceMap.toggled.connect(self._update_reference_alignment_display)
        self.ui.doubleSpinBoxReferenceOverlayOpacity.valueChanged.connect(self._handle_reference_overlay_opacity_changed)
        self.ui.checkBoxSyncReferenceAndRealView.toggled.connect(self._handle_reference_real_sync_toggled)
        self.ui.checkBoxHideReferenceAnnotations.toggled.connect(self._handle_show_reference_annotations_toggled)
        self.ui.checkBoxHideRealImageAnnotations.toggled.connect(self._handle_show_real_image_annotations_toggled)
        self.ui.referenceImageView.horizontalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.referenceImageView)
        )
        self.ui.referenceImageView.verticalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.referenceImageView)
        )
        self.ui.realImageView.horizontalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.realImageView)
        )
        self.ui.realImageView.verticalScrollBar().valueChanged.connect(
            lambda _value: self._sync_reference_real_view_from(self.ui.realImageView)
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        ViewControlsMixin.closeEvent(self, event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        ViewControlsMixin.resizeEvent(self, event)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if self._handle_mosaic_event_filter(watched, event):
            return True
        return ViewControlsMixin.eventFilter(self, watched, event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        ViewControlsMixin.keyPressEvent(self, event)


# ===================================================================
# 程序入口
# ===================================================================

def main(argv: list[str] | None = None) -> int:
    """HoshinoPanoAssistant 桌面程序入口。

    启动 Qt 应用，检查星表数据是否就绪，然后显示主窗口。
    """
    ensure_preference_file()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv or sys.argv)

    icon_path = Path(__file__).resolve().parent.parent / "icon256.png"
    app.setWindowIcon(QIcon(str(icon_path)))

    if not ensure_catalogs_ready_or_handle():
        return 0

    window = MainWindow()
    window.show()
    return int(app.exec_())
