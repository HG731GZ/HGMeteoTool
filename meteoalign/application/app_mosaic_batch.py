from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import monotonic

from PyQt5.QtCore import QEvent, QObject, QRectF, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QHeaderView,
    QMessageBox,
    QProgressDialog,
    QTableWidgetItem,
)

from .app_constants import SOURCE_MODEL_JSON_FILTER
from .app_graphics_items import GraphicsImageItem
from .app_mosaic import MOSAIC_PREVIEW_MAG_LIMIT
from ..alignment.constants import SKY_KNOWN_PROJECTION_DISPLAY_NAMES
from ..catalog import project_root
from ..image_preview import DEFAULT_PREVIEW_LONG_SIDE_PX, load_image_preview
from ..mosaic.render_types import MosaicRenderRequest
from ..mosaic_export import (
    MOSAIC_EXPORT_TIFF_FILTER,
    load_mosaic_export_source_image,
    mosaic_export_available,
    write_mosaic_reprojection_tiff,
)
from ..mosaic.export.geometry import MosaicExportGeometry
from ..mosaic.export.target_transform import target_icrs_to_pixel_transform_payload_matches
from ..mosaic.framing import MOSAIC_FRAMING_SCHEMA
from ..mosaic.model_io import MosaicSourceModel, _load_mosaic_source_model
from ..meteor_selection import MeteorBox, load_meteor_selection, meteor_json_path
from ..qt_tasks import start_qt_worker_task
from ..simulator import CameraSettings, ObserverSettings, RECTILINEAR_LENS_MODEL, ViewSettings
from ..sky_scene_service import SkySceneData


MOSAIC_BATCH_FRAMING_JSON_FILTER = "自由投影取景 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)"
MOSAIC_BATCH_MODE_SKY_INDEX = 0
MOSAIC_BATCH_IMAGE_NAME_COLUMN = 0
MOSAIC_BATCH_METEOR_COLUMN = 1
MOSAIC_BATCH_STATUS_COLUMN = 2


@dataclass(frozen=True)
class MosaicBatchFraming:
    json_path: Path
    payload: dict[str, object]
    geometry: MosaicExportGeometry
    target_icrs_to_pixel_payload: dict[str, object]
    observer: ObserverSettings
    camera: CameraSettings
    view: ViewSettings


@dataclass
class MosaicBatchImageItem:
    source_model: MosaicSourceModel
    meteor_boxes: tuple[MeteorBox, ...] = ()
    meteor_selection_path: Path | None = None
    meteor_selection_error: str = ""
    status: str = "待处理"
    output_path: Path | None = None
    error_message: str = ""


@dataclass(frozen=True)
class MosaicBatchExportTask:
    """后台批处理所需的只读参数快照。"""

    row: int
    item: MosaicBatchImageItem
    geometry: MosaicExportGeometry
    output_path: Path
    framing: MosaicBatchFraming | None
    base_model: MosaicSourceModel | None
    block_rows: int
    map_tile_size_px: int
    exact_remap_repair: bool
    tiff_lzw_compression: bool
    source_pixel_regions: tuple[tuple[int, int, int, int], ...] | None


def _write_mosaic_batch_export_task(
    task: MosaicBatchExportTask,
    update_export_progress,
) -> None:  # type: ignore[no-untyped-def]
    """执行单张批处理导出；此函数只在后台线程中调用。"""

    source_model = task.item.source_model
    if source_model.source_image_path is None:
        raise ValueError("源图模型 JSON 未记录原图路径。")
    output_path = task.output_path
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix or '.tif'}")
    try:
        update_export_progress("正在读取源图...", 0, 0)
        source_image = load_mosaic_export_source_image(source_model.source_image_path)
        if source_image.width_px != source_model.image_width_px or source_image.height_px != source_model.image_height_px:
            raise ValueError(
                "原图尺寸与源图模型不一致："
                f"原图 {source_image.width_px} x {source_image.height_px} px，"
                f"模型 {source_model.image_width_px} x {source_model.image_height_px} px。"
            )
        if task.framing is None and task.base_model is None:
            raise ValueError("当前批处理模式缺少目标取景或底图模型。")
        write_mosaic_reprojection_tiff(
            output_path=temp_path,
            source_model=source_model.model,
            source_image=source_image,
            camera=None if task.framing is None else task.framing.camera,
            view=None if task.framing is None else task.framing.view,
            observer=None if task.framing is None else task.framing.observer,
            geometry=task.geometry,
            framing_payload=None if task.framing is None else task.framing.payload,
            block_rows=task.block_rows,
            export_progress_callback=update_export_progress,
            target_icrs_to_pixel_payload=(
                None if task.framing is None else task.framing.target_icrs_to_pixel_payload
            ),
            target_model=None if task.base_model is None else task.base_model.model,
            map_tile_size_px=task.map_tile_size_px,
            exact_remap_repair=task.exact_remap_repair,
            tiff_lzw_compression=task.tiff_lzw_compression,
            source_pixel_regions=task.source_pixel_regions,
        )
        update_export_progress("正在完成文件写入...", 0, 0)
        temp_path.replace(output_path)
        update_export_progress("导出完成。", 1, 1)
    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise


class MosaicBatchExportWorker(QObject):
    """在独立线程内串行导出全景图，避免阻塞 Qt 主事件循环。"""

    progress = pyqtSignal(int, int, str, int, int)
    item_started = pyqtSignal(int)
    item_succeeded = pyqtSignal(int, object)
    item_failed = pyqtSignal(int, str)
    completed = pyqtSignal(int, int, bool)
    failed = pyqtSignal(str)

    def __init__(self, tasks: tuple[MosaicBatchExportTask, ...]) -> None:
        super().__init__()
        self.tasks = tasks
        self._cancel_requested = Event()
        self._last_progress_signature: tuple[str, int] | None = None
        self._last_progress_emitted_at = 0.0

    def request_cancel(self) -> None:
        """线程安全地请求在下一个可中断进度点停止。"""

        self._cancel_requested.set()

    def run(self) -> None:
        success_count = 0
        failed_count = 0
        try:
            total_count = len(self.tasks)
            for current_index, task in enumerate(self.tasks):
                if self._cancel_requested.is_set():
                    self.completed.emit(success_count, failed_count, True)
                    return
                self._last_progress_signature = None
                self._last_progress_emitted_at = 0.0
                self.item_started.emit(task.row)

                def update_export_progress(label: str, value: int, maximum: int) -> None:
                    # 文件已经原子替换完成后必须登记成功，不能因最后一刻的取消把成品误报为未完成。
                    if self._cancel_requested.is_set() and label != "导出完成。":
                        raise InterruptedError("用户取消了批处理。")
                    self._emit_throttled_progress(
                        current_index,
                        total_count,
                        label,
                        value,
                        maximum,
                    )

                try:
                    _write_mosaic_batch_export_task(task, update_export_progress)
                except InterruptedError:
                    self.completed.emit(success_count, failed_count, True)
                    return
                except Exception as exc:  # noqa: BLE001 - 单张失败后继续处理其余任务。
                    failed_count += 1
                    self.item_failed.emit(task.row, str(exc))
                    continue
                success_count += 1
                self.item_succeeded.emit(task.row, task.output_path)
            self.completed.emit(success_count, failed_count, False)
        except Exception as exc:  # noqa: BLE001 - 后台线程异常必须回传主线程。
            self.failed.emit(str(exc))

    def _emit_throttled_progress(
        self,
        current_index: int,
        total_count: int,
        label: str,
        value: int,
        maximum: int,
    ) -> None:
        """限制高频进度信号，避免大量小网格任务淹没主事件队列。"""

        safe_value = int(value)
        safe_maximum = int(maximum)
        signature = (str(label), safe_maximum)
        now = monotonic()
        phase_changed = signature != self._last_progress_signature
        phase_finished = safe_maximum > 0 and safe_value >= safe_maximum
        if not phase_changed and not phase_finished and now - self._last_progress_emitted_at < 0.05:
            return
        self._last_progress_signature = signature
        self._last_progress_emitted_at = now
        self.progress.emit(
            int(current_index),
            int(total_count),
            str(label),
            safe_value,
            safe_maximum,
        )


class MosaicBatchMixin:
    """全景图批处理页面 Mixin。"""

    ui: object
    ui_config: object

    def _init_mosaic_batch_page(self) -> None:
        if not hasattr(self.ui, "comboBoxMosaicBatchMode"):
            return
        self._mosaic_batch_framing: MosaicBatchFraming | None = None
        self._mosaic_batch_base_model: MosaicSourceModel | None = None
        self._mosaic_batch_items: list[MosaicBatchImageItem] = []
        self._mosaic_batch_thread: object | None = None
        self._mosaic_batch_worker: MosaicBatchExportWorker | None = None
        self._mosaic_batch_progress: QProgressDialog | None = None
        self._mosaic_batch_cancel_requested = False
        self._mosaic_batch_terminal_handled = False
        self._mosaic_batch_base_preview_path: Path | None = None
        self._mosaic_batch_base_preview_image: QImage | None = None
        self._init_mosaic_batch_preview()
        self.ui.comboBoxMosaicBatchMode.setCurrentIndex(MOSAIC_BATCH_MODE_SKY_INDEX)
        if hasattr(self.ui, "doubleSpinBoxMosaicBatchMapTileSize"):
            self.ui.doubleSpinBoxMosaicBatchMapTileSize.setValue(float(self._mosaic_batch_config_map_tile_size_px()))
        self._configure_mosaic_batch_table()
        for label_name in (
            "labelMosaicBatchFramingPath",
            "labelMosaicBatchImageCount",
        ):
            if hasattr(self.ui, label_name):
                getattr(self.ui, label_name).installEventFilter(self)
        self._update_mosaic_batch_controls()

    def _init_mosaic_batch_preview(self) -> None:
        """初始化批处理页的只读模拟星空预览。"""

        if not hasattr(self.ui, "mosaicBatchPreviewView"):
            return
        self.mosaic_batch_preview_scene = QGraphicsScene(self)
        self.mosaic_batch_preview_image_item = GraphicsImageItem()
        self.mosaic_batch_preview_scene.addItem(self.mosaic_batch_preview_image_item)
        self.ui.mosaicBatchPreviewView.setScene(self.mosaic_batch_preview_scene)
        self.ui.mosaicBatchPreviewView.installEventFilter(self)
        self.ui.mosaicBatchPreviewView.viewport().installEventFilter(self)

        self.mosaic_batch_preview_timer = QTimer(self)
        self.mosaic_batch_preview_timer.setSingleShot(True)
        self.mosaic_batch_preview_timer.timeout.connect(self.render_mosaic_batch_preview_now)
        self._set_mosaic_batch_preview_placeholder("请先导入取景 JSON")

    def _connect_mosaic_batch_inputs(self) -> None:
        if not hasattr(self.ui, "comboBoxMosaicBatchMode"):
            return
        self.ui.comboBoxMosaicBatchMode.currentIndexChanged.connect(self._update_mosaic_batch_controls)
        self.ui.pushButtonImportMosaicBatchFramingJson.clicked.connect(self.import_mosaic_batch_framing_json)
        self.ui.pushButtonImportMosaicBatchImageJson.clicked.connect(self.import_mosaic_batch_image_json)
        self.ui.pushButtonClearMosaicBatchImports.clicked.connect(self.clear_mosaic_batch_imports)
        self.ui.pushButtonStartMosaicBatch.clicked.connect(self.start_mosaic_batch_processing)
        if hasattr(self.ui, "checkBoxMosaicBatchMeteorOnly"):
            self.ui.checkBoxMosaicBatchMeteorOnly.toggled.connect(self._update_mosaic_batch_controls)

    def _configure_mosaic_batch_table(self) -> None:
        table = self.ui.tableWidgetMosaicBatchImages
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(MOSAIC_BATCH_IMAGE_NAME_COLUMN, QHeaderView.Stretch)
        header.setSectionResizeMode(MOSAIC_BATCH_METEOR_COLUMN, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(MOSAIC_BATCH_STATUS_COLUMN, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def _mosaic_batch_is_sky_mode(self) -> bool:
        if not hasattr(self.ui, "comboBoxMosaicBatchMode"):
            return True
        return self.ui.comboBoxMosaicBatchMode.currentIndex() == MOSAIC_BATCH_MODE_SKY_INDEX

    def _mosaic_batch_has_imports(self) -> bool:
        return (
            self._mosaic_batch_framing is not None
            or self._mosaic_batch_base_model is not None
            or bool(self._mosaic_batch_items)
        )

    def _mosaic_batch_target_geometry(self) -> MosaicExportGeometry | None:
        """返回当前模式的目标画布几何。"""

        if self._mosaic_batch_is_sky_mode():
            return None if self._mosaic_batch_framing is None else self._mosaic_batch_framing.geometry
        base_model = self._mosaic_batch_base_model
        if base_model is None:
            return None
        return MosaicExportGeometry(
            boundary_width_px=int(base_model.image_width_px),
            boundary_height_px=int(base_model.image_height_px),
            crop_left_px=0,
            crop_top_px=0,
            output_width_px=int(base_model.image_width_px),
            output_height_px=int(base_model.image_height_px),
        )

    def _update_mosaic_batch_controls(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self.ui, "comboBoxMosaicBatchMode"):
            return
        has_imports = self._mosaic_batch_has_imports()
        sky_mode = self._mosaic_batch_is_sky_mode()
        batch_active = getattr(self, "_mosaic_batch_thread", None) is not None
        self.ui.comboBoxMosaicBatchMode.setEnabled(not has_imports and not batch_active)
        target_name = "取景" if sky_mode else "底图"
        self.ui.labelMosaicBatchFramingPathTitle.setText(target_name)
        self.ui.pushButtonImportMosaicBatchFramingJson.setText(f"导入{target_name}JSON")
        self.ui.pushButtonImportMosaicBatchFramingJson.setToolTip(
            "导入自由投影页面导出的取景 JSON。" if sky_mode else "导入底图的 model.json，作为输出像素坐标参考。"
        )
        self.ui.pushButtonImportMosaicBatchFramingJson.setEnabled(not batch_active)
        self.ui.pushButtonImportMosaicBatchImageJson.setEnabled(not batch_active)
        self.ui.pushButtonClearMosaicBatchImports.setEnabled(has_imports and not batch_active)
        target_geometry = self._mosaic_batch_target_geometry()
        can_start = target_geometry is not None and bool(self._mosaic_batch_processable_rows())
        self.ui.pushButtonStartMosaicBatch.setEnabled(can_start and not batch_active)
        for control_name in (
            "checkBoxMosaicBatchMeteorOnly",
            "doubleSpinBoxMosaicBatchMapTileSize",
            "checkBoxMosaicBatchExactRemapRepair",
        ):
            if hasattr(self.ui, control_name):
                getattr(self.ui, control_name).setEnabled(not batch_active)
        if can_start:
            start_tooltip = ""
        elif target_geometry is None:
            start_tooltip = f"请先导入{target_name} JSON。"
        elif self._mosaic_batch_meteor_only_enabled():
            start_tooltip = "需要至少导入一张带流星框选的图片。"
        else:
            start_tooltip = "请先导入图像模型 JSON。"
        self.ui.pushButtonStartMosaicBatch.setToolTip(start_tooltip)
        self._update_mosaic_batch_summary_labels()
        self._update_mosaic_batch_preview_info()
        self.schedule_mosaic_batch_preview()

    def _mosaic_batch_meteor_only_enabled(self) -> bool:
        return bool(
            hasattr(self.ui, "checkBoxMosaicBatchMeteorOnly")
            and self.ui.checkBoxMosaicBatchMeteorOnly.isChecked()
        )

    def _mosaic_batch_processable_rows(self) -> list[int]:
        if not self._mosaic_batch_meteor_only_enabled():
            return list(range(len(self._mosaic_batch_items)))
        return [row for row, item in enumerate(self._mosaic_batch_items) if item.meteor_boxes]

    def _update_mosaic_batch_summary_labels(self) -> None:
        target = self._mosaic_batch_framing if self._mosaic_batch_is_sky_mode() else self._mosaic_batch_base_model
        if target is None:
            self._set_elided_label_text(self.ui.labelMosaicBatchFramingPath, "未导入")
        else:
            self._set_elided_label_text(
                self.ui.labelMosaicBatchFramingPath,
                target.json_path.name,
                str(target.json_path),
            )
        count = len(self._mosaic_batch_items)
        self.ui.labelMosaicBatchImageCount.setText(f"{count} 张" if count else "未导入")

    @staticmethod
    def _mosaic_batch_pixel_text(width_px: int, height_px: int) -> str:
        """把输出尺寸和总像素数压缩为适合标题栏显示的文本。"""

        width = max(0, int(width_px))
        height = max(0, int(height_px))
        if width <= 0 or height <= 0:
            return "-"
        megapixels = width * height / 1_000_000.0
        return f"{width} × {height} px（{megapixels:.2f} MP）"

    def _mosaic_batch_projection_display_name(self) -> str:
        """返回当前批处理目标的投影名称。"""

        if self._mosaic_batch_is_sky_mode():
            framing = self._mosaic_batch_framing
            if framing is None:
                return "-"
            projection_payload = framing.payload.get("projection")
            if isinstance(projection_payload, dict):
                display_name = str(projection_payload.get("display_name") or "").strip()
                projection_model = str(projection_payload.get("model") or framing.camera.lens_model)
                return display_name or SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(projection_model, projection_model)
            projection_model = str(framing.camera.lens_model)
            return SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(projection_model, projection_model)

        base_model = self._mosaic_batch_base_model
        if base_model is None:
            return "-"
        projection_model = str(base_model.model.camera_calibration_profile.base_projection_type)
        return SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(projection_model, projection_model)

    def _update_mosaic_batch_preview_info(self) -> None:
        """刷新预览标题栏中的投影和最终输出尺寸。"""

        if not hasattr(self.ui, "labelMosaicBatchPreviewInfo"):
            return
        geometry = self._mosaic_batch_target_geometry()
        if geometry is None:
            self.ui.labelMosaicBatchPreviewInfo.setText("投影：-  |  输出尺寸：-")
            return
        projection_name = self._mosaic_batch_projection_display_name()
        output_text = self._mosaic_batch_pixel_text(
            geometry.output_width_px,
            geometry.output_height_px,
        )
        self.ui.labelMosaicBatchPreviewInfo.setText(
            f"投影：{projection_name}  |  输出尺寸：{output_text}"
        )

    def schedule_mosaic_batch_preview(self, *unused, delay_ms: int = 30) -> None:  # type: ignore[no-untyped-def]
        """合并连续刷新请求，避免导入和布局变化时重复渲染。"""

        timer = getattr(self, "mosaic_batch_preview_timer", None)
        if timer is None:
            return
        timer.start(max(0, int(delay_ms)))

    def _mosaic_batch_preview_render_size(self, camera: CameraSettings) -> tuple[int, int]:
        """在视图可用区域内按完整输出画布比例计算预览尺寸。"""

        viewport_size = self.ui.mosaicBatchPreviewView.viewport().size()
        available_width = max(64, int(viewport_size.width()))
        available_height = max(64, int(viewport_size.height()))
        aspect = max(1.0, float(camera.image_width_px)) / max(1.0, float(camera.image_height_px))
        width = available_width
        height = max(1, int(round(width / aspect)))
        if height > available_height:
            height = available_height
            width = max(1, int(round(height * aspect)))
        return max(1, width), max(1, height)

    @staticmethod
    def _mosaic_batch_preview_camera(
        camera: CameraSettings,
        width_px: int,
        height_px: int,
    ) -> CameraSettings:
        """缩小相机输出像素数，同时保留原取景的投影几何。"""

        return CameraSettings(
            sensor_width_mm=float(camera.sensor_width_mm),
            sensor_height_mm=float(camera.sensor_height_mm),
            image_width_px=int(width_px),
            image_height_px=int(height_px),
            focal_length_mm=float(camera.focal_length_mm),
            lens_model=str(camera.lens_model),
            fisheye_fov_deg=float(camera.fisheye_fov_deg),
        )

    def _paint_mosaic_batch_crop_rect(
        self,
        image: QImage,
        geometry: MosaicExportGeometry,
    ) -> None:
        """按完整画布比例在模拟星空上绘制红色裁剪框。"""

        if image.isNull() or geometry.boundary_width_px <= 0 or geometry.boundary_height_px <= 0:
            return
        scale_x = image.width() / float(geometry.boundary_width_px)
        scale_y = image.height() / float(geometry.boundary_height_px)
        rect = QRectF(
            float(geometry.crop_left_px) * scale_x,
            float(geometry.crop_top_px) * scale_y,
            float(geometry.output_width_px) * scale_x,
            float(geometry.output_height_px) * scale_y,
        )
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(255, 32, 32, 230))
        pen.setWidthF(max(1.5, min(float(image.width()), float(image.height())) / 320.0))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        inset = pen.widthF() * 0.5
        painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
        painter.end()

    def _set_mosaic_batch_preview_placeholder(self, message: str) -> None:
        """在没有可渲染取景时显示简洁的占位提示。"""

        if not hasattr(self.ui, "mosaicBatchPreviewView"):
            return
        width = max(320, int(self.ui.mosaicBatchPreviewView.viewport().width()))
        height = max(180, int(self.ui.mosaicBatchPreviewView.viewport().height()))
        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(16, 18, 22))
        painter = QPainter(image)
        painter.setPen(QColor(168, 174, 184))
        painter.drawText(image.rect(), Qt.AlignCenter, str(message))
        painter.end()
        self._show_mosaic_batch_preview_image(image)

    def _show_mosaic_batch_preview_image(self, image: QImage) -> None:
        """把图像放入批处理预览场景并保持宽高比。"""

        if not hasattr(self, "mosaic_batch_preview_image_item"):
            return
        width = max(1, int(image.width()))
        height = max(1, int(image.height()))
        self.mosaic_batch_preview_image_item.set_image(image)
        self.mosaic_batch_preview_scene.setSceneRect(0.0, 0.0, float(width), float(height))
        self.ui.mosaicBatchPreviewView.fitInView(
            self.mosaic_batch_preview_scene.sceneRect(),
            Qt.KeepAspectRatio,
        )

    def _load_mosaic_batch_base_preview(self) -> QImage:
        """读取并缓存底图的 8-bit 屏幕预览。"""

        base_model = self._mosaic_batch_base_model
        if base_model is None:
            raise ValueError("请先导入底图 JSON。")
        image_path = base_model.source_image_path
        if image_path is None:
            raise ValueError("底图模型 JSON 未记录底图路径。")
        resolved_path = image_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"底图不存在：{resolved_path}")
        cached_image = self._mosaic_batch_base_preview_image
        if self._mosaic_batch_base_preview_path == resolved_path and cached_image is not None and not cached_image.isNull():
            return cached_image

        preview = load_image_preview(
            resolved_path,
            max_long_side_px=DEFAULT_PREVIEW_LONG_SIDE_PX,
        )
        image = preview.image.convertToFormat(QImage.Format_RGB888)
        if image.isNull():
            raise ValueError("无法生成底图的 8-bit 预览。")
        self._mosaic_batch_base_preview_path = resolved_path
        self._mosaic_batch_base_preview_image = image
        return image

    def render_mosaic_batch_preview_now(self) -> None:
        """按模式显示模拟星空与裁剪框，或底图的 8-bit 预览。"""

        if not hasattr(self.ui, "mosaicBatchPreviewView"):
            return
        if not self._mosaic_batch_is_sky_mode():
            if self._mosaic_batch_base_model is None:
                self._set_mosaic_batch_preview_placeholder("请先导入底图 JSON")
                return
            try:
                self._show_mosaic_batch_preview_image(self._load_mosaic_batch_base_preview())
            except Exception as exc:  # noqa: BLE001 - 底图预览错误不能阻断模型导入和批处理。
                self._set_mosaic_batch_preview_placeholder("底图 8-bit 预览加载失败")
                self.ui.statusbar.showMessage(f"底图 8-bit 预览加载失败: {exc}")
            return

        framing = self._mosaic_batch_framing
        if framing is None:
            self._set_mosaic_batch_preview_placeholder("请先导入取景 JSON")
            return
        try:
            width, height = self._mosaic_batch_preview_render_size(framing.camera)
            camera = self._mosaic_batch_preview_camera(framing.camera, width, height)
            observer = framing.observer
            scene = SkySceneData(
                horizontal_catalog=self._get_horizontal_catalog(observer, MOSAIC_PREVIEW_MAG_LIMIT),
                horizontal_milky_way=self._get_horizontal_milky_way(observer),
                horizontal_constellations=self._get_horizontal_constellations(observer),
                horizontal_solar_system=self._get_horizontal_solar_system(observer),
            )
            request = MosaicRenderRequest(
                camera=camera,
                view=framing.view,
                observer=observer,
                scene=scene,
                visible_mag_limit=MOSAIC_PREVIEW_MAG_LIMIT,
                sky_style=self._mosaic_sky_preview_style(),
                sources=(),
                overlay_enabled=False,
            )
            result = self._mosaic_render_coordinator.render(request)
            self._paint_mosaic_batch_crop_rect(result.image, framing.geometry)
            self._show_mosaic_batch_preview_image(result.image)
        except Exception as exc:  # noqa: BLE001 - 预览错误需要反馈到界面但不能阻断批处理导入。
            self._set_mosaic_batch_preview_placeholder("模拟星空预览渲染失败")
            self.ui.statusbar.showMessage(f"全景图批处理预览渲染失败: {exc}")

    def _handle_mosaic_batch_event_filter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        """视图尺寸变化后按新宽高重绘批处理预览。"""

        preview_view = getattr(self.ui, "mosaicBatchPreviewView", None)
        if preview_view is None:
            return False
        if watched in (preview_view, preview_view.viewport()) and event.type() in (QEvent.Resize, QEvent.Show):
            self.schedule_mosaic_batch_preview()
        return False

    def import_mosaic_batch_framing_json(self) -> None:
        if not self._mosaic_batch_is_sky_mode():
            self._import_mosaic_batch_base_json()
            return
        default_dir = self._import_dialog_directory(self._mosaic_batch_default_dir())
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入批处理取景 JSON",
            str(default_dir),
            MOSAIC_BATCH_FRAMING_JSON_FILTER,
        )
        if not file_path:
            return
        self._remember_import_path(file_path)
        try:
            framing = self._load_mosaic_batch_framing(Path(file_path).expanduser())
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON 错误直接反馈到界面。
            QMessageBox.critical(self, "导入取景 JSON 失败", str(exc))
            self.ui.statusbar.showMessage(f"导入批处理取景 JSON 失败: {exc}")
            return
        self._mosaic_batch_framing = framing
        self._mosaic_batch_base_model = None
        self._update_mosaic_batch_controls()
        self.ui.statusbar.showMessage(f"已导入批处理取景 JSON: {framing.json_path.name}")

    def _import_mosaic_batch_base_json(self) -> None:
        default_dir = self._import_dialog_directory(self._mosaic_batch_default_dir())
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入批处理底图模型 JSON",
            str(default_dir),
            SOURCE_MODEL_JSON_FILTER,
        )
        if not file_path:
            return
        self._remember_import_path(file_path)
        try:
            base_model = _load_mosaic_source_model(Path(file_path).expanduser())
            if base_model.image_width_px <= 0 or base_model.image_height_px <= 0:
                raise ValueError("底图模型缺少有效图像尺寸。")
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON 错误直接反馈到界面。
            QMessageBox.critical(self, "导入底图 JSON 失败", str(exc))
            self.ui.statusbar.showMessage(f"导入批处理底图 JSON 失败: {exc}")
            return
        self._mosaic_batch_base_model = base_model
        self._mosaic_batch_framing = None
        self._mosaic_batch_base_preview_path = None
        self._mosaic_batch_base_preview_image = None
        self._update_mosaic_batch_controls()
        self.ui.statusbar.showMessage(f"已导入批处理底图 JSON: {base_model.json_path.name}")

    def _load_mosaic_batch_framing(self, json_path: Path) -> MosaicBatchFraming:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("取景 JSON 根对象必须是对象。")
        if str(payload.get("schema") or "") != MOSAIC_FRAMING_SCHEMA:
            raise ValueError("这不是 HoshinoPanoAssistant 自由投影取景 JSON。")
        geometry = self._mosaic_geometry_from_framing_payload(payload)
        if geometry.output_width_px <= 0 or geometry.output_height_px <= 0:
            raise ValueError("裁剪后的导出尺寸无效。")
        target_payload = payload.get("target_icrs_to_pixel_transform")
        if not isinstance(target_payload, dict):
            raise ValueError("取景 JSON 缺少 target_icrs_to_pixel_transform，请重新导出取景。")
        if not target_icrs_to_pixel_transform_payload_matches(target_payload, geometry=geometry):
            raise ValueError("取景 JSON 中的 ICRS 到全景图像素变换与输出几何不匹配。")
        observer_payload = payload.get("observer")
        if not isinstance(observer_payload, dict):
            raise ValueError("取景 JSON 缺少 observer 对象。")
        observer, _utc_offset_hours = self._observer_from_mosaic_framing_payload(observer_payload)
        view_payload = payload.get("view")
        if not isinstance(view_payload, dict):
            raise ValueError("取景 JSON 缺少 view 对象。")
        return MosaicBatchFraming(
            json_path=json_path,
            payload=payload,
            geometry=geometry,
            target_icrs_to_pixel_payload=target_payload,
            observer=observer,
            camera=self._mosaic_batch_camera_from_target_payload(target_payload),
            view=ViewSettings(
                center_az_deg=float(view_payload.get("center_az_deg", 0.0)) % 360.0,
                center_alt_deg=max(-90.0, min(90.0, float(view_payload.get("center_alt_deg", 0.0)))),
                roll_deg=float(view_payload.get("roll_deg", 0.0)),
            ),
        )

    def _mosaic_batch_camera_from_target_payload(self, payload: dict[str, object]) -> CameraSettings:
        camera_payload = payload.get("camera")
        if not isinstance(camera_payload, dict):
            raise ValueError("取景 JSON 缺少 target_icrs_to_pixel.camera。")
        return CameraSettings(
            sensor_width_mm=float(camera_payload.get("sensor_width_mm", 36.0)),
            sensor_height_mm=float(camera_payload.get("sensor_height_mm", 24.0)),
            image_width_px=int(camera_payload.get("image_width_px", payload.get("boundary_width_px", 0))),
            image_height_px=int(camera_payload.get("image_height_px", payload.get("boundary_height_px", 0))),
            focal_length_mm=float(camera_payload.get("focal_length_mm", 24.0)),
            lens_model=str(camera_payload.get("lens_model", RECTILINEAR_LENS_MODEL)),
            fisheye_fov_deg=float(camera_payload.get("fisheye_fov_deg", 180.0)),
        )

    def import_mosaic_batch_image_json(self) -> None:
        default_dir = self._import_dialog_directory(self._mosaic_batch_default_dir())
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "导入批处理图像模型 JSON",
            str(default_dir),
            SOURCE_MODEL_JSON_FILTER,
        )
        if not file_paths:
            return
        self._remember_import_path(file_paths)
        imported_count = 0
        errors: list[str] = []
        existing_paths = {item.source_model.json_path.resolve() for item in self._mosaic_batch_items}
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            for file_path in file_paths:
                json_path = Path(file_path).expanduser()
                resolved_path = json_path.resolve()
                if resolved_path in existing_paths:
                    continue
                try:
                    source_model = _load_mosaic_source_model(json_path)
                except Exception as exc:  # noqa: BLE001 - 单个模型坏了不阻断其他模型导入。
                    errors.append(f"{json_path.name}: {exc}")
                    continue
                self._mosaic_batch_items.append(self._mosaic_batch_item_from_source_model(source_model))
                existing_paths.add(resolved_path)
                imported_count += 1
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_mosaic_batch_table()
        self._update_mosaic_batch_controls()
        if errors:
            QMessageBox.warning(
                self,
                "部分图像 JSON 导入失败",
                "\n".join(errors[:12]) + ("\n..." if len(errors) > 12 else ""),
            )
        self.ui.statusbar.showMessage(f"已导入 {imported_count} 个批处理图像模型 JSON")

    @staticmethod
    def _mosaic_batch_item_from_source_model(source_model: MosaicSourceModel) -> MosaicBatchImageItem:
        """自动读取源图同目录的流星框选 JSON；读取失败不阻断模型导入。"""

        image_path = source_model.source_image_path
        if image_path is None:
            return MosaicBatchImageItem(source_model=source_model)
        selection_path = meteor_json_path(image_path)
        if not selection_path.exists():
            return MosaicBatchImageItem(source_model=source_model)
        try:
            meteor_boxes = tuple(load_meteor_selection(image_path))
        except ValueError as exc:
            return MosaicBatchImageItem(
                source_model=source_model,
                meteor_selection_path=selection_path,
                meteor_selection_error=str(exc),
            )
        return MosaicBatchImageItem(
            source_model=source_model,
            meteor_boxes=meteor_boxes,
            meteor_selection_path=selection_path,
        )

    def clear_mosaic_batch_imports(self) -> None:
        self._mosaic_batch_framing = None
        self._mosaic_batch_base_model = None
        self._mosaic_batch_base_preview_path = None
        self._mosaic_batch_base_preview_image = None
        self._mosaic_batch_items = []
        self.ui.tableWidgetMosaicBatchImages.setRowCount(0)
        self._update_mosaic_batch_controls()
        self.ui.statusbar.showMessage("已清除全景图批处理导入。")

    def _refresh_mosaic_batch_table(self) -> None:
        table = self.ui.tableWidgetMosaicBatchImages
        table.setRowCount(len(self._mosaic_batch_items))
        for row, item in enumerate(self._mosaic_batch_items):
            name_item = self._read_only_mosaic_batch_item(self._mosaic_batch_display_name(item.source_model))
            name_item.setToolTip(str(item.source_model.source_image_path or item.source_model.json_path))
            meteor_text, meteor_tooltip = self._mosaic_batch_meteor_selection_text(item)
            meteor_item = self._read_only_mosaic_batch_item(meteor_text)
            meteor_item.setToolTip(meteor_tooltip)
            status_item = self._read_only_mosaic_batch_item(item.status)
            if item.output_path is not None:
                status_item.setToolTip(str(item.output_path))
            elif item.error_message:
                status_item.setToolTip(item.error_message)
            table.setItem(row, MOSAIC_BATCH_IMAGE_NAME_COLUMN, name_item)
            table.setItem(row, MOSAIC_BATCH_METEOR_COLUMN, meteor_item)
            table.setItem(row, MOSAIC_BATCH_STATUS_COLUMN, status_item)
            self._set_mosaic_batch_row_state(row, self._mosaic_batch_state_from_status(item.status))

    def _read_only_mosaic_batch_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _mosaic_batch_meteor_selection_text(self, item: MosaicBatchImageItem) -> tuple[str, str]:
        """生成表格中流星框选状态的文本与说明。"""

        if item.meteor_selection_error:
            return "异常", item.meteor_selection_error
        if item.meteor_selection_path is None:
            return "无", "未找到同名流星框选 JSON。"
        return (
            f"有（{len(item.meteor_boxes)}）",
            f"流星框选：{item.meteor_selection_path}\n框选数：{len(item.meteor_boxes)}",
        )

    def _mosaic_batch_display_name(self, source_model: MosaicSourceModel) -> str:
        if source_model.source_image_path is not None:
            return source_model.source_image_path.name
        if source_model.source_image_text:
            return source_model.source_image_text
        return source_model.json_path.name

    def _mosaic_batch_state_from_status(self, status: str) -> str:
        if status.startswith("已完成"):
            return "done"
        if status.startswith("失败") or status.startswith("跳过"):
            return "failed"
        if status == "处理中":
            return "running"
        return "pending"

    def _set_mosaic_batch_item_status(
        self,
        row: int,
        status: str,
        *,
        output_path: Path | None = None,
        error_message: str = "",
    ) -> None:
        if row < 0 or row >= len(self._mosaic_batch_items):
            return
        item = self._mosaic_batch_items[row]
        item.status = status
        item.output_path = output_path
        item.error_message = error_message
        status_item = self.ui.tableWidgetMosaicBatchImages.item(row, MOSAIC_BATCH_STATUS_COLUMN)
        if status_item is not None:
            status_item.setText(status)
            status_item.setToolTip(str(output_path) if output_path is not None else error_message)
        self._set_mosaic_batch_row_state(row, self._mosaic_batch_state_from_status(status))

    def _set_mosaic_batch_row_state(self, row: int, state: str) -> None:
        colors = {
            "pending": QColor(255, 255, 255),
            "running": QColor(255, 246, 205),
            "done": QColor(210, 244, 214),
            "failed": QColor(255, 220, 220),
        }
        brush = QBrush(colors.get(state, colors["pending"]))
        for column in range(self.ui.tableWidgetMosaicBatchImages.columnCount()):
            item = self.ui.tableWidgetMosaicBatchImages.item(row, column)
            if item is not None:
                item.setBackground(brush)

    def start_mosaic_batch_processing(self) -> None:
        if getattr(self, "_mosaic_batch_thread", None) is not None:
            return
        geometry = self._mosaic_batch_target_geometry()
        if geometry is None:
            target_name = "取景" if self._mosaic_batch_is_sky_mode() else "底图"
            QMessageBox.warning(self, "批处理失败", f"请先导入{target_name} JSON。")
            return
        processable_rows = self._mosaic_batch_processable_rows()
        if not processable_rows:
            message = "请先导入图像模型 JSON。" if not self._mosaic_batch_items else "当前没有可导出的流星框选。"
            QMessageBox.warning(self, "批处理失败", message)
            return
        if not mosaic_export_available():
            QMessageBox.critical(self, "批处理失败", "当前环境缺少 OpenCV 或 tifffile，无法写入 TIFF。")
            return
        output_dir_text = QFileDialog.getExistingDirectory(
            self,
            "选择批处理 TIFF 输出目录",
            str(self._mosaic_batch_default_dir()),
        )
        if not output_dir_text:
            return
        output_dir = Path(output_dir_text).expanduser()
        total_count = len(processable_rows)
        meteor_only_text = "\n仅导出已框选的流星区域，其他区域将透明。\n" if self._mosaic_batch_meteor_only_enabled() else ""
        if QMessageBox.question(
            self,
            "确认开始批处理",
            (
                f"将处理 {total_count} 张图像。\n"
                f"{meteor_only_text}\n"
                f"输出目录：\n{output_dir}\n\n"
                f"尺寸：{geometry.output_width_px} x {geometry.output_height_px} px\n"
                f"格式：{MOSAIC_EXPORT_TIFF_FILTER}"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        self._run_mosaic_batch(output_dir)

    def _run_mosaic_batch(self, output_dir: Path) -> None:
        geometry = self._mosaic_batch_target_geometry()
        if geometry is None:
            return
        used_output_paths: set[Path] = set()
        processable_rows = self._mosaic_batch_processable_rows()
        framing = self._mosaic_batch_framing if self._mosaic_batch_is_sky_mode() else None
        base_model = self._mosaic_batch_base_model if not self._mosaic_batch_is_sky_mode() else None
        block_rows = self._mosaic_export_block_rows()
        map_tile_size_px = self._mosaic_batch_map_tile_size_px()
        exact_remap_repair = self._mosaic_batch_exact_remap_repair_enabled()
        tiff_lzw_compression = self._mosaic_tiff_lzw_compression_enabled()
        tasks: list[MosaicBatchExportTask] = []
        for row, item in enumerate(self._mosaic_batch_items):
            if row not in processable_rows:
                self._set_mosaic_batch_item_status(row, "跳过：无流星框选")
                continue
            self._set_mosaic_batch_item_status(row, "待处理")
            tasks.append(
                MosaicBatchExportTask(
                    row=row,
                    item=item,
                    geometry=geometry,
                    output_path=self._mosaic_batch_unique_output_path(
                        output_dir,
                        item.source_model,
                        geometry,
                        used_output_paths,
                    ),
                    framing=framing,
                    base_model=base_model,
                    block_rows=block_rows,
                    map_tile_size_px=map_tile_size_px,
                    exact_remap_repair=exact_remap_repair,
                    tiff_lzw_compression=tiff_lzw_compression,
                    source_pixel_regions=self._mosaic_batch_item_source_regions(item),
                )
            )

        progress = QProgressDialog("正在准备批处理...", "取消", 0, 0, self)
        progress.setWindowTitle("全景图批处理")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        self._mosaic_batch_cancel_requested = False
        self._mosaic_batch_terminal_handled = False
        worker = MosaicBatchExportWorker(tuple(tasks))
        worker.item_started.connect(self._handle_mosaic_batch_item_started)
        worker.item_succeeded.connect(self._handle_mosaic_batch_item_succeeded)
        worker.item_failed.connect(self._handle_mosaic_batch_item_failed)
        progress.canceled.connect(self._cancel_mosaic_batch_processing)
        task_handle = start_qt_worker_task(
            parent=self,
            worker=worker,
            finished_signal=worker.completed,
            failed_signal=worker.failed,
            on_finished=self._handle_mosaic_batch_completed,
            on_failed=self._handle_mosaic_batch_worker_failed,
            progress_signal=worker.progress,
            on_progress=self._handle_mosaic_batch_progress,
            on_cleanup=self._cleanup_mosaic_batch_worker,
            progress_dialog=progress,
            start_delay_ms=1,
        )
        self._mosaic_batch_thread = task_handle.thread
        self._mosaic_batch_worker = worker
        self._mosaic_batch_progress = progress
        self._update_mosaic_batch_controls()
        self.ui.statusbar.showMessage(f"全景图批处理已在后台开始，共 {len(tasks)} 张。")

    def _handle_mosaic_batch_item_started(self, row: int) -> None:
        self._set_mosaic_batch_item_status(row, "处理中")

    def _handle_mosaic_batch_progress(
        self,
        current_index: int,
        total_count: int,
        label: str,
        value: int,
        maximum: int,
    ) -> None:
        """在主线程更新批处理进度弹窗。"""

        progress = self._mosaic_batch_progress
        if progress is None:
            return
        if not progress.isVisible():
            # 某些平台在不定进度与定量进度切换时会意外隐藏对话框，继续显示可保证任务入口始终可见。
            progress.show()
        progress.setLabelText(f"[{current_index + 1}/{total_count}] {label}")
        if maximum <= 0:
            progress.setRange(0, 0)
            return
        total_maximum = max(1, int(total_count) * 1000)
        file_progress = int(round(1000.0 * max(0, min(int(value), int(maximum))) / max(1, int(maximum))))
        progress.setRange(0, total_maximum)
        progress.setValue(int(current_index) * 1000 + file_progress)

    def _handle_mosaic_batch_item_succeeded(self, row: int, output_path: object) -> None:
        self._set_mosaic_batch_item_status(row, "已完成", output_path=Path(output_path))

    def _handle_mosaic_batch_item_failed(self, row: int, error_message: str) -> None:
        self._set_mosaic_batch_item_status(row, "失败", error_message=error_message)
        item = self._mosaic_batch_items[row]
        self.ui.statusbar.showMessage(f"批处理失败: {item.source_model.json_path.name}: {error_message}")

    def _cancel_mosaic_batch_processing(self) -> None:
        """请求后台任务安全停止，并保持进度窗口直至线程真正结束。"""

        worker = self._mosaic_batch_worker
        progress = self._mosaic_batch_progress
        if (
            worker is None
            or self._mosaic_batch_cancel_requested
            or self._mosaic_batch_terminal_handled
        ):
            return
        self._mosaic_batch_cancel_requested = True
        worker.request_cancel()
        if progress is not None:
            progress.setCancelButton(None)
            progress.setLabelText("正在取消批处理，请等待当前计算步骤结束...")
        self.ui.statusbar.showMessage("正在取消全景图批处理...")

    def _handle_mosaic_batch_completed(self, success_count: int, failed_count: int, canceled: bool) -> None:
        self._mosaic_batch_terminal_handled = True
        progress = self._mosaic_batch_progress
        if progress is not None:
            self._disconnect_mosaic_batch_cancel_signal(progress)
            if not canceled:
                total = max(1, success_count + failed_count)
                progress.setRange(0, total)
                progress.setValue(total)
            progress.close()
        if canceled:
            for row, item in enumerate(self._mosaic_batch_items):
                if item.status in {"待处理", "处理中"}:
                    self._set_mosaic_batch_item_status(row, "已取消")
            self.ui.statusbar.showMessage(f"全景图批处理已取消：已成功导出 {success_count} 张。")
            return
        self.ui.statusbar.showMessage(f"全景图批处理完成：成功 {success_count} 张，失败 {failed_count} 张。")
        if failed_count:
            QMessageBox.warning(self, "批处理完成", f"成功 {success_count} 张，失败 {failed_count} 张。")
        else:
            QMessageBox.information(self, "批处理完成", f"成功导出 {success_count} 张 TIFF。")

    def _handle_mosaic_batch_worker_failed(self, error_message: str) -> None:
        self._mosaic_batch_terminal_handled = True
        if self._mosaic_batch_progress is not None:
            self._disconnect_mosaic_batch_cancel_signal(self._mosaic_batch_progress)
            self._mosaic_batch_progress.close()
        for row, item in enumerate(self._mosaic_batch_items):
            if item.status == "处理中":
                self._set_mosaic_batch_item_status(row, "失败", error_message=error_message)
        self.ui.statusbar.showMessage(f"全景图批处理异常终止: {error_message}")
        QMessageBox.critical(self, "批处理失败", error_message)

    def _cleanup_mosaic_batch_worker(self) -> None:
        progress = self._mosaic_batch_progress
        if progress is not None:
            self._disconnect_mosaic_batch_cancel_signal(progress)
            progress.close()
            progress.deleteLater()
        self._mosaic_batch_thread = None
        self._mosaic_batch_worker = None
        self._mosaic_batch_progress = None
        self._mosaic_batch_cancel_requested = False
        self._mosaic_batch_terminal_handled = False
        self._update_mosaic_batch_controls()

    def _disconnect_mosaic_batch_cancel_signal(self, progress: QProgressDialog) -> None:
        """任务进入终态后断开取消信号，避免关闭弹窗再次覆盖状态栏。"""

        try:
            progress.canceled.disconnect(self._cancel_mosaic_batch_processing)
        except (TypeError, RuntimeError):
            # 信号可能已断开，或者 Qt 对象正处于销毁阶段。
            pass

    def _mosaic_batch_map_tile_size_px(self) -> int:
        if hasattr(self.ui, "doubleSpinBoxMosaicBatchMapTileSize"):
            configured = self.ui.doubleSpinBoxMosaicBatchMapTileSize.value()
            try:
                return max(1, min(512, int(configured)))
            except (TypeError, ValueError):
                return self._mosaic_batch_config_map_tile_size_px()
        method = getattr(self, "_mosaic_map_tile_size_px", None)
        if callable(method):
            return int(method())
        return 4

    def _mosaic_batch_exact_remap_repair_enabled(self) -> bool:
        if hasattr(self.ui, "checkBoxMosaicBatchExactRemapRepair"):
            return bool(self.ui.checkBoxMosaicBatchExactRemapRepair.isChecked())
        method = getattr(self, "_mosaic_exact_remap_repair_enabled", None)
        if callable(method):
            return bool(method())
        return False

    def _mosaic_batch_config_map_tile_size_px(self) -> int:
        configured = getattr(self.ui_config, "mosaic_map_tile_size_px", 4)
        try:
            value = int(configured)
        except (TypeError, ValueError):
            value = 4
        return max(1, min(512, value))

    def _mosaic_batch_item_source_regions(self, item: MosaicBatchImageItem) -> tuple[tuple[int, int, int, int], ...] | None:
        """在流星区域模式下返回需导出的源图像素矩形。"""

        if not self._mosaic_batch_meteor_only_enabled():
            return None
        return tuple(
            (
                int(round(box.left)),
                int(round(box.top)),
                int(round(box.right)),
                int(round(box.bottom)),
            )
            for box in item.meteor_boxes
        )

    def _mosaic_batch_unique_output_path(
        self,
        output_dir: Path,
        source_model: MosaicSourceModel,
        geometry: MosaicExportGeometry,
        used_output_paths: set[Path],
    ) -> Path:
        base_name = f"{source_model.json_path.stem}_mosaic_{geometry.output_width_px}x{geometry.output_height_px}"
        output_path = output_dir / f"{base_name}.tif"
        counter = 2
        while output_path.exists() or output_path in used_output_paths:
            output_path = output_dir / f"{base_name}_{counter}.tif"
            counter += 1
        used_output_paths.add(output_path)
        return output_path

    def _mosaic_batch_default_dir(self) -> Path:
        if self._mosaic_batch_items:
            parent = self._mosaic_batch_items[-1].source_model.json_path.parent
            if parent.exists():
                return parent
        if self._mosaic_batch_framing is not None and self._mosaic_batch_framing.json_path.parent.exists():
            return self._mosaic_batch_framing.json_path.parent
        if self._mosaic_batch_base_model is not None and self._mosaic_batch_base_model.json_path.parent.exists():
            return self._mosaic_batch_base_model.json_path.parent
        output_dir = project_root() / "outputs"
        return output_dir if output_dir.exists() else project_root()


__all__ = ["MosaicBatchExportTask", "MosaicBatchExportWorker", "MosaicBatchMixin"]
