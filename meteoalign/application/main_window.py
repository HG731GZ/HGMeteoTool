from __future__ import annotations

import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, QObject, QPoint, QTimer, Qt
from PyQt5.QtGui import QCursor, QIcon, QImage, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QShortcut,
)

from ..alignment.models import SkyAlignmentTransform
from ..catalog_download import ensure_catalogs_ready_or_handle
from ..catalog import load_constellation_catalog, load_default_catalog
from ..camera_calibration import CameraCalibrationProfile
from ..config import StarMapUiConfig, load_star_map_ui_config
from ..preference_manager import ensure_preference_file, recent_import_directory, remember_import_path
from ..runtime_paths import runtime_icon_path
from ..image_preview import ImagePreview
from ..adjacent_framing_worker import AdjacentFramingWorker
from ..milky_way import MilkyWayCatalog, load_milky_way
from ..renderer import StarMapRenderer
from ..source_model import SourceAstrometricModel
from ..simulator import (
    HorizontalConstellationCatalog,
    HorizontalMilkyWayCatalog,
    HorizontalSolarSystemCatalog,
    HorizontalStarCatalog,
    ProjectedStarMap,
    ReferenceStar,
)
from ..star_pair_store import StarPairStore
from ..ui.ui_main_window import Ui_MainWindow
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
)
from .app_widgets import AppWidgetMixin
from .app_adjacent_framing import AdjacentFramingMixin
from .app_star_pair_table import StarPairTableMixin
from .app_alignment import AlignmentMixin
from .app_star_pair_io import StarPairIOMixin
from .app_image_group import ImageGroupMixin
from .app_image import ImageMixin
from .app_meteor_selection import MeteorSelectionMixin
from .app_sequence import SequenceBatchMixin
from .app_sequence_refinement import SequenceRefinementMixin
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
from .preferences_dialog import PreferencesDialog, PreferencesLauncher
from .about_dialog import AboutDialog
from .image_group_assistant_dialog import ImageGroupAssistantDialog
from .image_preview_dialog import ImagePreviewDialog
from .star_pair_assistant_dialog import StarPairAssistantDialog

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
    AdjacentFramingMixin,
    StarPairTableMixin,
    AlignmentMixin,
    StarPairIOMixin,
    SequenceBatchMixin,
    SequenceRefinementMixin,
    ImageGroupMixin,
    ImageMixin,
    MeteorSelectionMixin,
    AutoMatchMixin,
    MosaicProjectionMixin,
    MosaicBatchMixin,
    RenderingMixin,
    ViewControlsMixin,
):
    """HoshinoPanoAssistant 主窗口。

    采用 Mixin 多重继承模式将不同功能拆分到独立模块：
    - AppWidgetMixin: UI 辅助（标签、字体）
    - AdjacentFramingMixin: 参考图像粗略取景
    - StarPairTableMixin: 星对表格管理
    - AlignmentMixin: 天球配准与残差
    - StarPairIOMixin: JSON 导入导出
    - ImageGroupMixin: 多图列表与图像切换
    - ImageMixin: 图像与蒙版导入
    - MeteorSelectionMixin: 流星框选
    - AutoMatchMixin: 自动匹配场星
    - RenderingMixin: 渲染与模拟
    - ViewControlsMixin: 视图缩放与事件处理
    """

    def _apply_preferences(self, ui_config: StarMapUiConfig) -> None:
        """热更新可立即生效的参数，同时保留当前任务中的用户输入。"""

        previous_config = getattr(self, "ui_config", None)
        self.ui_config = ui_config
        self.renderer.ui_config = ui_config
        self._apply_ui_font_config(ui_config)

        if hasattr(self, "_star_pick_circle_diameter_px"):
            self._star_pick_circle_diameter_px = min(
                max(self._star_pick_circle_diameter_px, ui_config.star_pick_circle_min_diameter_px),
                ui_config.star_pick_circle_max_diameter_px,
            )

        for item_name in ("star_map_item", "reference_star_map_item", "real_reference_overlay_item"):
            item = getattr(self, item_name, None)
            if item is not None:
                item.update()

        # 已存在的匹配文字不是实时渲染项，需要直接同步字号。
        for _ellipse_item, label_item in getattr(self, "_star_pair_annotations", {}).values():
            label_font = label_item.font()
            label_font.setPointSize(ui_config.star_name_font_size_pt)
            label_item.setFont(label_font)
        if (
            previous_config is not None
            and previous_config.star_pair_psf_outer_diameter_multiplier
            != ui_config.star_pair_psf_outer_diameter_multiplier
            and hasattr(self, "_restore_star_pair_annotations_from_table")
        ):
            self._restore_star_pair_annotations_from_table()

        if hasattr(self, "_mosaic_min_render_interval_ms"):
            self._mosaic_min_render_interval_ms = max(
                1,
                int(round(1000.0 / self._mosaic_render_fps_limit())),
            )
        if hasattr(self, "render_timer"):
            self.schedule_render(delay_ms=0)
        if hasattr(self, "schedule_mosaic_render"):
            self.schedule_mosaic_render(delay_ms=0)
        self.ui.statusbar.showMessage("软件参数已应用到当前会话，未写入 preference.json。", 5000)

    def _notify_preferences_saved(self, _ui_config: StarMapUiConfig) -> None:
        """提示配置已经写入磁盘。"""

        self.ui.statusbar.showMessage("软件参数已保存到 preference.json。", 5000)

    def _show_preferences_dialog(self) -> None:
        """显示单实例非模态软件选项窗口，并将其提到前台。"""

        self.preferences_dialog.show()
        self.preferences_dialog.raise_()
        self.preferences_dialog.activateWindow()

    def _show_about_dialog(self) -> None:
        """显示单实例非模态关于窗口，并将其提到前台。"""

        self.about_dialog.show()
        self.about_dialog.raise_()
        self.about_dialog.activateWindow()

    def _show_star_pair_assistant(self) -> None:
        """显示单实例非模态星点匹配助手，并将已有窗口提到前台。"""

        self.star_pair_assistant.show()
        self.star_pair_assistant.raise_()
        self.star_pair_assistant.activateWindow()

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
        # 匹配控件实际位于独立窗口；同名接口仍挂在主 UI 上，供各业务模块共享。
        self.star_pair_assistant = StarPairAssistantDialog()
        self.star_pair_assistant.bind_controls_to(self.ui)
        self.image_preview_dialog = ImagePreviewDialog()
        self.image_group_assistant = ImageGroupAssistantDialog(self.image_preview_dialog)
        self.ui_config = load_star_map_ui_config()
        self.star_pair_assistant.set_always_on_top(
            self.ui_config.star_pair_assistant_always_on_top
        )
        self._apply_ui_font_config(self.ui_config)

        self.catalog = load_default_catalog(mag_limit=None)
        self.constellation_catalog = load_constellation_catalog()
        self.milky_way_catalog: MilkyWayCatalog = load_milky_way()
        self.renderer = StarMapRenderer(self.ui_config)
        # 选项窗口不设置主窗口父级，避免操作系统把它作为始终压在主窗口上方的 transient 窗口。
        self.preferences_dialog = PreferencesDialog()
        self.preferences_page = self.preferences_dialog.preferences_page
        self.preferences_page.preferences_applied.connect(self._apply_preferences)
        self.preferences_page.preferences_saved.connect(self._notify_preferences_saved)
        self.preferences_launcher = PreferencesLauncher(self.ui.tabWidgetMain)
        self.preferences_launcher.clicked.connect(self._show_preferences_dialog)
        self.about_dialog = AboutDialog(self)
        self.preferences_launcher.about_clicked.connect(self._show_about_dialog)
        self.ui.tabWidgetMain.setCornerWidget(self.preferences_launcher, Qt.TopRightCorner)
        self.preferences_shortcut = QShortcut(QKeySequence(QKeySequence.Preferences), self)
        self.preferences_shortcut.activated.connect(self._show_preferences_dialog)
        self._install_value_control_wheel_filters()
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
        self._init_meteor_selection_page()

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render_now)
        self.drag_start: QPoint | None = None
        self.last_drag_pos: QPoint | None = None
        self._horizontal_cache_key: tuple[object, ...] | None = None
        self._horizontal_cache: HorizontalStarCatalog | None = None
        self._milky_way_cache_key: tuple[object, ...] | None = None
        self._milky_way_cache: HorizontalMilkyWayCatalog | None = None
        self._constellation_cache_key: tuple[object, ...] | None = None
        self._constellation_cache: HorizontalConstellationCatalog | None = None
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
        self._adjacent_framing_thread: object | None = None
        self._adjacent_framing_worker: AdjacentFramingWorker | None = None
        self._adjacent_framing_progress: QProgressDialog | None = None
        self._adjacent_image_path: Path | None = None
        self._adjacent_model_json_path: Path | None = None
        self._adjacent_framing_result = None
        self._rough_alignment_transform = None
        self._rough_source_astrometric_model = None
        self._star_pair_session_import_switch_to_reference = True
        self._star_pair_session_import_clear_input_name = "新的匹配 JSON"
        self._star_pair_session_import_restore_observation_time = True
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
        self._hidden_star_pair_annotation_ids: set[str] = set()
        self._focused_star_annotations: list[QGraphicsItem] = []
        self._current_star_map: ProjectedStarMap | None = None
        self._current_reference_star_map: ProjectedStarMap | None = None
        self._current_reference_stars: tuple[ReferenceStar, ...] = ()
        self._sky_alignment_transform: SkyAlignmentTransform | None = None
        self._source_astrometric_model: SourceAstrometricModel | None = None
        self._restored_alignment_initial_rotation_matrix: np.ndarray | None = None
        self._imported_camera_calibration_profile: CameraCalibrationProfile | None = None
        self._imported_camera_calibration_profile_path: Path | None = None
        self._imported_camera_calibration_image_name = ""
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
        self._sequence_refinement_active = False
        self._preserve_sequence_on_next_image_load = False
        self._image_group_paths: tuple[Path, ...] = ()
        self._preserve_image_group_on_next_image_load = False

        self._init_defaults()
        self._connect_inputs()
        self._handle_tab_changed()
        self._connect_mosaic_projection_inputs()
        self._connect_mosaic_batch_inputs()
        self._connect_meteor_selection_inputs()
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
        self.ui.comboBoxSkyAlignmentModel.setCurrentIndex(1)
        self._init_adjacent_framing_defaults()
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
        self.ui.spinBoxAutoMatchRadius.setValue(self.ui_config.auto_match_default_search_radius_px)
        self.ui.spinBoxSequencePsfSearchRadius.setValue(self.ui_config.sequence_psf_search_radius_px)
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
        self.ui.pushButtonImportImages.clicked.connect(self.import_images)
        if hasattr(self.ui, "pushButtonImportAdjacentImage"):
            self.ui.pushButtonImportAdjacentImage.clicked.connect(self.import_adjacent_image)
        if hasattr(self.ui, "pushButtonPreviewAdjacentImage"):
            self.ui.pushButtonPreviewAdjacentImage.clicked.connect(self.show_adjacent_image_preview)
        if hasattr(self.ui, "pushButtonCalculateAdjacentFraming"):
            self.ui.pushButtonCalculateAdjacentFraming.clicked.connect(self.calculate_adjacent_rough_framing)
        if hasattr(self.ui, "toolButtonAdjacentAlignmentSettings"):
            self.ui.toolButtonAdjacentAlignmentSettings.clicked.connect(self.open_adjacent_alignment_settings)
        if hasattr(self.ui, "comboBoxAdjacentAlignmentMode"):
            self.ui.comboBoxAdjacentAlignmentMode.currentIndexChanged.connect(
                self._handle_adjacent_alignment_mode_changed
            )
        self.ui.pushButtonImportImageSequence.clicked.connect(self.import_image_sequence)
        self.ui.pushButtonProcessImageSequence.clicked.connect(self.process_image_sequence)
        if hasattr(self.ui, "pushButtonContinueImageSequence"):
            self.ui.pushButtonContinueImageSequence.clicked.connect(self.continue_image_sequence)
        if hasattr(self.ui, "pushButtonRefineSequenceFrames"):
            self.ui.pushButtonRefineSequenceFrames.clicked.connect(self.refine_sequence_frames)
        if hasattr(self.ui, "pushButtonImportImageSequenceSkyMask"):
            self.ui.pushButtonImportImageSequenceSkyMask.clicked.connect(self.import_image_sequence_sky_mask)
        if hasattr(self.ui, "pushButtonClearImageSequenceSkyMask"):
            self.ui.pushButtonClearImageSequenceSkyMask.clicked.connect(self.clear_image_sequence_sky_mask)
        if hasattr(self.ui, "checkBoxShowImageSequenceMask"):
            self.ui.checkBoxShowImageSequenceMask.toggled.connect(self._handle_image_sequence_mask_toggled)
        self.ui.pushButtonExportStarPairs.clicked.connect(self.export_star_pair_session)
        self.ui.pushButtonImportStarPairs.clicked.connect(self.import_star_pair_session)
        self.ui.pushButtonDeleteStarPairs.clicked.connect(self.delete_all_star_pair_rows)
        self.ui.pushButtonClearStarPairs.clicked.connect(self.clear_all_star_pair_positions)
        self.ui.pushButtonOpenImageGroupAssistant.clicked.connect(self._show_image_group_assistant)
        self.image_group_assistant.image_activated.connect(self._handle_image_group_image_activated)
        self.image_group_assistant.ui.checkBoxAutoSelectReference.toggled.connect(
            self._handle_automatic_image_group_reference_toggled
        )
        self.image_group_assistant.reference_selection_requested.connect(
            self._handle_image_group_reference_requested
        )
        self.ui.pushButtonOpenStarPairAssistant.clicked.connect(self._show_star_pair_assistant)
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
            self.ui.tableWidgetImageSequence.customContextMenuRequested.connect(
                self._show_image_sequence_context_menu
            )
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
        if getattr(self, "_mosaic_batch_thread", None) is not None:
            QMessageBox.information(self, "正在导出全景图", "全景图批处理仍在后台导出，请先取消或等待完成后再关闭窗口。")
            event.ignore()
            return
        if getattr(self, "_meteor_mask_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入流星蒙版", "流星检测蒙版仍在导入，请等待完成后再关闭窗口。")
            event.ignore()
            return
        ViewControlsMixin.closeEvent(self, event)
        if event.isAccepted():
            self.preferences_dialog.close()
            self.about_dialog.close()
            self.star_pair_assistant.close()
            self.image_group_assistant.close()
            self.image_preview_dialog.close()
            self._shutdown_meteor_detection_worker()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        ViewControlsMixin.resizeEvent(self, event)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if self._handle_mosaic_batch_event_filter(watched, event):
            return True
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

    app.setWindowIcon(QIcon(str(runtime_icon_path())))

    if not ensure_catalogs_ready_or_handle():
        return 0

    window = MainWindow()
    window.show()
    return int(app.exec_())
