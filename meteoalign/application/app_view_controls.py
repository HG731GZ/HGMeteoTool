from __future__ import annotations

import math
from collections.abc import Callable

from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication, QGraphicsView, QMainWindow, QMessageBox, QSizePolicy

from .app_constants import (
    IMAGE_VIEW_ZOOM_IN_FACTOR,
    IMAGE_VIEW_ZOOM_OUT_FACTOR,
    STAR_PICK_CIRCLE_STEP_PX,
    STAR_PICK_TOUCHPAD_STEPS_PER_ZOOM_UNIT,
    TOUCHPAD_ZOOM_MAX_FACTOR,
    TOUCHPAD_ZOOM_MIN_FACTOR,
    TOUCHPAD_ZOOM_SENSITIVITY,
)
from .app_graphics_items import LiveStarMapGraphicsItem
from ..simulator import horizontal_fov_deg, vertical_fov_deg
from ..view_gestures import (
    ViewZoomPolicy,
    native_gesture_zoom_factor,
    native_gesture_zoom_value,
    roll_after_drag,
    sky_center_after_drag,
    wheel_zoom_factor,
)


class ViewControlsMixin:
    """视图控制 Mixin：缩放、fit、拖拽、eventFilter、键盘、触控板手势。"""

    ui: object
    scene: object
    reference_scene: object
    real_image_scene: object
    star_map_item: object
    reference_star_map_item: object
    real_reference_overlay_item: object
    real_image_item: object
    image_sequence_item: object
    ui_config: object
    _syncing_reference_preview_splitter: bool
    _syncing_reference_real_views: bool
    _active_star_pair_row: int | None
    _star_pick_circle_diameter_px: int
    _star_pick_native_zoom_remainder: float
    _real_image_zoom_max_scale: float
    _image_import_thread: object | None
    _json_import_thread: object | None
    _mask_import_thread: object | None
    drag_start: QPoint | None
    last_drag_pos: QPoint | None
    render_timer: QTimer
    _reference_pick_press_pos: QPoint | None

    def _native_zoom_centers(self) -> dict[int, QPointF]:
        centers = getattr(self, "_graphics_view_native_zoom_centers", None)
        if centers is None:
            centers = {}
            self._graphics_view_native_zoom_centers = centers
        return centers

    def _clear_graphics_view_native_zoom_center(self, view: QGraphicsView | None = None) -> None:
        centers = self._native_zoom_centers()
        if view is None:
            centers.clear()
            return
        centers.pop(id(view), None)

    def _graphics_view_native_zoom_center(self, view: QGraphicsView) -> QPointF:
        centers = self._native_zoom_centers()
        key = id(view)
        center = centers.get(key)
        if center is None:
            center = view.mapToScene(view.viewport().rect().center())
            centers[key] = center
        return center

    def _configure_reference_preview_splitter(self) -> None:
        splitter = self.ui.splitterReferenceAndRealImage
        for preview_group in (self.ui.groupBoxReferencePreview, self.ui.groupBoxRealImagePreview):
            # 参考图标题栏控件较多；忽略其内容最小宽度，避免挤占真实图像的显示区域。
            preview_group.setMinimumWidth(0)
            preview_group.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        splitter.setChildrenCollapsible(False)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.installEventFilter(self)
        splitter.splitterMoved.connect(lambda _pos, _index: self._set_equal_reference_preview_sizes())
        QTimer.singleShot(0, self._set_equal_reference_preview_sizes)

    def _synchronize_reference_preview_header_heights(self) -> None:
        """补齐较短标题栏下方的空白，使两侧图像视口上下边缘严格对齐。"""

        reference_layout = self.ui.verticalLayoutReferencePreview
        real_layout = self.ui.verticalLayoutRealImagePreview
        reference_spacer = reference_layout.itemAt(1).spacerItem()
        real_spacer = real_layout.itemAt(1).spacerItem()
        if reference_spacer is None or real_spacer is None:
            return

        reference_header_height = max(0, reference_layout.itemAt(0).sizeHint().height())
        real_header_height = max(0, real_layout.itemAt(0).sizeHint().height())
        common_header_height = max(reference_header_height, real_header_height)
        reference_spacer.changeSize(
            0,
            common_header_height - reference_header_height,
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )
        real_spacer.changeSize(
            0,
            common_header_height - real_header_height,
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )
        reference_layout.invalidate()
        real_layout.invalidate()

    def _set_equal_reference_preview_sizes(self) -> None:
        if self._syncing_reference_preview_splitter:
            return
        self._synchronize_reference_preview_header_heights()
        splitter = self.ui.splitterReferenceAndRealImage
        total_width = sum(splitter.sizes())
        if total_width <= 0:
            total_width = max(0, splitter.width() - splitter.handleWidth())
        if total_width <= 0:
            return

        # 两个预览区需要并排对照星点，始终把分割器恢复成左右等宽。
        left_width = total_width // 2
        right_width = total_width - left_width
        self._syncing_reference_preview_splitter = True
        try:
            splitter.setSizes([left_width, right_width])
        finally:
            self._syncing_reference_preview_splitter = False

    def fit_star_map(self) -> None:
        if not self.scene.sceneRect().isEmpty():
            self.ui.starMapView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._update_live_star_map_zoom_scale(self.ui.starMapView)

    def fit_reference_map(self) -> None:
        if not self.reference_scene.sceneRect().isEmpty():
            self.ui.referenceImageView.fitInView(self.reference_scene.sceneRect(), Qt.KeepAspectRatio)
            self._update_live_star_map_zoom_scale(self.ui.referenceImageView)

    def fit_real_image(self) -> None:
        if not self.real_image_item.isNull():
            self.ui.realImageView.fitInView(self.real_image_scene.sceneRect(), Qt.KeepAspectRatio)
            self._cap_graphics_view_to_max_scale(self.ui.realImageView)
            self._update_live_star_map_zoom_scale(self.ui.realImageView)

    def fit_image_sequence_preview(self) -> None:
        if not hasattr(self.ui, "imageSequenceView"):
            return
        if self.image_sequence_item.isNull():
            return
        scene = self.ui.imageSequenceView.scene()
        if scene is None or scene.sceneRect().isEmpty():
            return
        self.ui.imageSequenceView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        self._cap_graphics_view_to_max_scale(self.ui.imageSequenceView)

    def fit_all_graphics_views(self) -> None:
        self._set_equal_reference_preview_sizes()
        self.fit_star_map()
        self.fit_image_sequence_preview()
        if self._can_sync_reference_real_views():
            self.fit_real_image()
            self._sync_reference_real_view_from(self.ui.realImageView, force=True)
        else:
            self.fit_reference_map()
            self.fit_real_image()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._image_import_thread is not None:
            QMessageBox.information(self, "正在导入图像", "图像预览仍在生成，请等待导入完成后再关闭窗口。")
            event.ignore()
            return
        if getattr(self, "_sequence_import_thread", None) is not None:
            QMessageBox.information(self, "正在导入序列", "图像序列仍在读取 EXIF，请等待导入完成后再关闭窗口。")
            event.ignore()
            return
        if self._json_import_thread is not None:
            QMessageBox.information(self, "正在导入 JSON", "JSON 仍在导入，请等待完成后再关闭窗口。")
            event.ignore()
            return
        if self._mask_import_thread is not None:
            QMessageBox.information(self, "正在导入蒙版", "蒙版仍在导入，请等待完成后再关闭窗口。")
            event.ignore()
            return
        QMainWindow.closeEvent(self, event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        QMainWindow.resizeEvent(self, event)
        QTimer.singleShot(0, self._set_equal_reference_preview_sizes)
        QTimer.singleShot(0, self._refresh_all_elided_labels)
        QTimer.singleShot(0, self.fit_all_graphics_views)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if event.type() == QEvent.Wheel and self._is_wheel_value_control(watched):
            return self._forward_value_control_wheel_to_scroll_area(watched, event)
        if watched is self.ui.splitterReferenceAndRealImage:
            if event.type() in (QEvent.Resize, QEvent.Show):
                QTimer.singleShot(0, self._set_equal_reference_preview_sizes)
            return False
        if watched in (
            self.ui.labelImportedImagePath,
            self.ui.labelSkyMaskStatus,
            self.ui.labelAlignmentTransformStatus,
            getattr(self.ui, "labelImageSequenceStatus", None),
            getattr(self.ui, "labelImageSequenceSummary", None),
            getattr(self.ui, "labelImageSequencePreviewTitle", None),
        ):
            if event.type() in (QEvent.Resize, QEvent.Show):
                QTimer.singleShot(0, lambda label=watched: self._refresh_elided_label(label))
            return False
        star_pair_table = getattr(self.ui, "tableWidgetStarPairs", None)
        if watched is star_pair_table:
            if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                return self._handle_star_pair_delete_key()
            if event.type() == QEvent.Wheel:
                return self._handle_star_pair_table_wheel(event)
        try:
            star_pair_viewport = star_pair_table.viewport() if star_pair_table is not None else None
        except RuntimeError:
            # 独立匹配窗口关闭时，Qt 可能先销毁表格，再向主窗口事件过滤器投递清理事件。
            star_pair_viewport = None
        if watched is star_pair_viewport and event.type() == QEvent.Wheel:
            return self._handle_star_pair_table_wheel(event)
        if hasattr(self.ui, "tableWidgetImageSequence") and watched is self.ui.tableWidgetImageSequence:
            if event.type() == QEvent.Wheel:
                return self._handle_table_wheel(self.ui.tableWidgetImageSequence, event)
        if (
            hasattr(self.ui, "tableWidgetImageSequence")
            and watched is self.ui.tableWidgetImageSequence.viewport()
            and event.type() == QEvent.Wheel
        ):
            return self._handle_table_wheel(self.ui.tableWidgetImageSequence, event)
        if watched is self.ui.starMapView.viewport():
            if bool(getattr(self, "_simulator_controls_locked", False)):
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self.drag_start = None
                    self.last_drag_pos = None
                    self.ui.statusbar.showMessage("已有 4 对或更多有效匹配，星空模拟参数已锁定；清除匹配后可继续调整。")
                    return True
                if event.type() in (QEvent.MouseMove, QEvent.MouseButtonRelease):
                    self.drag_start = None
                    self.last_drag_pos = None
                    self.ui.starMapView.viewport().unsetCursor()
                    return True
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.drag_start = event.pos()
                self.last_drag_pos = event.pos()
                self.render_timer.stop()
                self.ui.starMapView.viewport().setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self.last_drag_pos is not None:
                dx = event.pos().x() - self.last_drag_pos.x()
                dy = event.pos().y() - self.last_drag_pos.y()
                self.last_drag_pos = event.pos()
                if event.modifiers() & Qt.ShiftModifier:
                    self._apply_roll_drag_delta(dx)
                else:
                    self._apply_drag_delta(dx, dy)
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.drag_start = None
                self.last_drag_pos = None
                self.ui.starMapView.viewport().unsetCursor()
                self.render_now()
                return True
        if watched is self.ui.referenceImageView:
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.referenceImageView,
                event,
            ):
                return True
        if watched is self.ui.realImageView:
            if event.type() == QEvent.NativeGesture:
                if (
                    self._active_star_pair_row is not None
                    and event.modifiers() & Qt.ControlModifier
                    and self._handle_star_pick_native_zoom(event)
                ):
                    return True
                if self._handle_graphics_view_native_zoom(self.ui.realImageView, event):
                    return True
        if hasattr(self.ui, "imageSequenceView") and watched is self.ui.imageSequenceView:
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.imageSequenceView,
                event,
            ):
                return True
        if watched is self.ui.referenceImageView.viewport():
            if event.type() in (QEvent.Enter, QEvent.MouseMove, QEvent.KeyPress, QEvent.KeyRelease):
                self._update_reference_map_cursor(self._event_ctrl_pressed(event))
            if event.type() == QEvent.Leave:
                self._reference_pick_press_pos = None
                self.ui.referenceImageView.viewport().unsetCursor()
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._clear_graphics_view_native_zoom_center(self.ui.referenceImageView)
                if self._event_ctrl_pressed(event):
                    self._update_reference_map_cursor(True)
                    self._reference_pick_press_pos = event.pos()
                    return True
                self._update_reference_map_cursor(False)
                self._reference_pick_press_pos = None
                return False
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                press_pos = self._reference_pick_press_pos
                self._reference_pick_press_pos = None
                if press_pos is not None:
                    move_distance = (event.pos() - press_pos).manhattanLength()
                    if move_distance <= QApplication.startDragDistance():
                        self._handle_reference_map_click(event.pos())
                    return True
            if event.type() == QEvent.Wheel:
                self._clear_graphics_view_native_zoom_center(self.ui.referenceImageView)
                return self._handle_graphics_view_wheel_zoom(self.ui.referenceImageView, event)
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.referenceImageView,
                event,
            ):
                return True
        if watched is self.ui.realImageView.viewport():
            if self._active_star_pair_row is not None:
                if event.type() in (QEvent.Enter, QEvent.MouseMove, QEvent.KeyPress, QEvent.KeyRelease):
                    self._update_real_image_pick_cursor(self._event_ctrl_pressed(event))
                    self._show_star_pick_status_hint(self._active_star_pair_row)
                if event.type() == QEvent.Leave:
                    self.ui.realImageView.viewport().unsetCursor()
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._clear_graphics_view_native_zoom_center(self.ui.realImageView)
                    if self._event_ctrl_pressed(event):
                        self._handle_real_image_pick_click(event.pos())
                        return True
                    return False
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                    self._leave_star_pick_mode()
                    self.ui.statusbar.showMessage("已取消当前星点位置点选。")
                    return True
                if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
                    if not self._wheel_zoom_enabled():
                        return False
                    wheel_delta = event.angleDelta().y()
                    if wheel_delta == 0:
                        return True
                    wheel_steps = int(wheel_delta / 120)
                    if wheel_steps == 0:
                        wheel_steps = 1 if wheel_delta > 0 else -1
                    self._adjust_star_pick_circle_diameter(wheel_steps)
                    return True
                if (
                    event.type() == QEvent.NativeGesture
                    and event.modifiers() & Qt.ControlModifier
                    and self._handle_star_pick_native_zoom(event)
                ):
                    return True
                if event.type() == QEvent.KeyPress and self._handle_star_pick_key_press(event):
                    return True
            if event.type() == QEvent.Wheel:
                self._clear_graphics_view_native_zoom_center(self.ui.realImageView)
                return self._handle_graphics_view_wheel_zoom(self.ui.realImageView, event)
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.realImageView,
                event,
            ):
                return True
        if hasattr(self.ui, "imageSequenceView") and watched is self.ui.imageSequenceView.viewport():
            if event.type() == QEvent.Wheel:
                self._clear_graphics_view_native_zoom_center(self.ui.imageSequenceView)
                return self._handle_graphics_view_wheel_zoom(self.ui.imageSequenceView, event)
            if event.type() == QEvent.NativeGesture and self._handle_graphics_view_native_zoom(
                self.ui.imageSequenceView,
                event,
            ):
                return True
        return QMainWindow.eventFilter(self, watched, event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._handle_star_pick_key_press(event):
            return
        QMainWindow.keyPressEvent(self, event)

    def _handle_star_pick_key_press(self, event) -> bool:  # type: ignore[no-untyped-def]
        if self._active_star_pair_row is None or not (event.modifiers() & Qt.ControlModifier):
            return False
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._adjust_star_pick_circle_diameter(1)
            return True
        if key in (Qt.Key_Minus, Qt.Key_Underscore):
            self._adjust_star_pick_circle_diameter(-1)
            return True
        return False

    def _forward_value_control_wheel_to_scroll_area(self, control, event: QWheelEvent) -> bool:  # type: ignore[no-untyped-def]
        """阻止参数控件吃掉滚轮，并将其交给最近的外层滚动区域。"""

        ancestor = control.parentWidget()
        while ancestor is not None:
            inherits = getattr(ancestor, "inherits", None)
            if callable(inherits) and inherits("QAbstractScrollArea"):
                # 不能把同一个滚轮事件再次投递到 viewport：Qt 会把事件沿父级
                # 重新分发，最终又回到当前控件，从而触发递归。直接调整滚动条即可。
                pixel_delta = event.pixelDelta()
                angle_delta = event.angleDelta()
                if pixel_delta.y() != 0 or angle_delta.y() != 0:
                    scrollbar = ancestor.verticalScrollBar()
                    delta = pixel_delta.y()
                    if delta == 0:
                        delta = round(
                            angle_delta.y()
                            / 120.0
                            * max(1, QApplication.wheelScrollLines())
                            * max(1, scrollbar.singleStep())
                        )
                else:
                    scrollbar = ancestor.horizontalScrollBar()
                    delta = pixel_delta.x()
                    if delta == 0:
                        delta = round(
                            angle_delta.x()
                            / 120.0
                            * max(1, QApplication.wheelScrollLines())
                            * max(1, scrollbar.singleStep())
                        )
                scrollbar.setValue(scrollbar.value() - delta)
                return True
            ancestor = ancestor.parentWidget()
        return True

    def _wheel_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "wheel_zoom_enabled", True))

    def _touchpad_pinch_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "touchpad_pinch_zoom_enabled", True))

    def _handle_graphics_view_wheel_zoom(self, view: QGraphicsView, event) -> bool:  # type: ignore[no-untyped-def]
        if not self._wheel_zoom_enabled():
            return False
        self._apply_graphics_view_zoom(view, event.angleDelta().y())
        return True

    def _apply_graphics_view_zoom(self, view: QGraphicsView, wheel_delta: int) -> None:
        factor = wheel_zoom_factor(wheel_delta, IMAGE_VIEW_ZOOM_IN_FACTOR, IMAGE_VIEW_ZOOM_OUT_FACTOR)
        if factor is None:
            return
        self._apply_graphics_view_zoom_factor(view, factor)

    def _scale_graphics_view(
        self,
        view: QGraphicsView,
        factor: float,
        *,
        center_scene: QPointF | None = None,
    ) -> QPointF | None:
        center_on_viewport = center_scene is not None
        if not center_on_viewport:
            view.scale(factor, factor)
            return None

        previous_anchor = view.transformationAnchor()
        view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        try:
            view.scale(factor, factor)
        finally:
            view.setTransformationAnchor(previous_anchor)
        view.centerOn(center_scene)
        return center_scene

    def _run_with_reference_real_sync_suspended(self, view: QGraphicsView, operation: Callable[[], None]) -> None:
        should_suspend = view in (self.ui.referenceImageView, self.ui.realImageView)
        if not should_suspend or self._syncing_reference_real_views:
            operation()
            return

        self._syncing_reference_real_views = True
        try:
            operation()
        finally:
            self._syncing_reference_real_views = False

    def _apply_graphics_view_zoom_factor(
        self,
        view: QGraphicsView,
        factor: float,
        *,
        center_scene: QPointF | None = None,
    ) -> None:
        if factor <= 0.0 or not math.isfinite(factor) or abs(factor - 1.0) <= 1e-4:
            return
        if factor < 1.0:
            min_scale = self._graphics_view_fit_scale(view)
            current_scale = self._graphics_view_current_scale(view)
            if current_scale * factor <= min_scale:
                self._run_with_reference_real_sync_suspended(view, lambda: self._fit_graphics_view_to_scene(view))
                self._sync_reference_real_view_from(view)
                return
        if factor > 1.0:
            max_scale = self._graphics_view_max_scale(view)
            current_scale = self._graphics_view_current_scale(view)
            if max_scale is not None and current_scale * factor >= max_scale:
                def apply_max_scale() -> None:
                    self._set_graphics_view_scale(view, max_scale)
                    self._update_live_star_map_zoom_scale(view)

                self._run_with_reference_real_sync_suspended(view, apply_max_scale)
                self._sync_reference_real_view_from(view)
                return

        applied_center: QPointF | None = None

        def apply_scale() -> None:
            nonlocal applied_center
            applied_center = self._scale_graphics_view(view, factor, center_scene=center_scene)
            self._update_live_star_map_zoom_scale(view)

        self._run_with_reference_real_sync_suspended(view, apply_scale)
        self._sync_reference_real_view_from(view, source_center=applied_center)

    def _native_gesture_zoom_value(self, event) -> float:  # type: ignore[no-untyped-def]
        value = native_gesture_zoom_value(event, self._view_zoom_policy())
        return 0.0 if value is None else value

    def _native_gesture_zoom_factor(self, event) -> float:  # type: ignore[no-untyped-def]
        factor = native_gesture_zoom_factor(event, self._view_zoom_policy())
        return 1.0 if factor is None else factor

    def _view_zoom_policy(self) -> ViewZoomPolicy:
        return ViewZoomPolicy(
            wheel_enabled=self._wheel_zoom_enabled(),
            pinch_enabled=self._touchpad_pinch_zoom_enabled(),
            sensitivity=TOUCHPAD_ZOOM_SENSITIVITY,
            min_factor=TOUCHPAD_ZOOM_MIN_FACTOR,
            max_factor=TOUCHPAD_ZOOM_MAX_FACTOR,
        )

    def _handle_graphics_view_native_zoom(self, view: QGraphicsView, event) -> bool:  # type: ignore[no-untyped-def]
        begin_gesture = getattr(Qt, "BeginNativeGesture", None)
        if begin_gesture is not None and event.gestureType() == begin_gesture:
            self._clear_graphics_view_native_zoom_center(view)
            self._graphics_view_native_zoom_center(view)
            return False

        end_gesture = getattr(Qt, "EndNativeGesture", None)
        if end_gesture is not None and event.gestureType() == end_gesture:
            self._clear_graphics_view_native_zoom_center(view)
            return False

        factor = self._native_gesture_zoom_factor(event)
        if abs(factor - 1.0) <= 1e-4:
            return False
        self._apply_graphics_view_zoom_factor(view, factor, center_scene=self._graphics_view_native_zoom_center(view))
        return True

    def _handle_star_pick_native_zoom(self, event) -> bool:  # type: ignore[no-untyped-def]
        value = self._native_gesture_zoom_value(event)
        if value == 0.0:
            return False

        self._star_pick_native_zoom_remainder += value * STAR_PICK_TOUCHPAD_STEPS_PER_ZOOM_UNIT
        wheel_steps = int(self._star_pick_native_zoom_remainder)
        if wheel_steps != 0:
            self._star_pick_native_zoom_remainder -= wheel_steps
            self._adjust_star_pick_circle_diameter(wheel_steps)
        return True

    def _handle_star_pair_table_wheel(self, event) -> bool:  # type: ignore[no-untyped-def]
        return self._handle_table_wheel(self.ui.tableWidgetStarPairs, event)

    def _handle_table_wheel(self, table, event) -> bool:  # type: ignore[no-untyped-def]
        """让表格自行消费滚轮，避免带动外层滚动区。"""
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()

        vertical_delta = float(pixel_delta.y())
        horizontal_delta = float(pixel_delta.x())
        if abs(vertical_delta) <= 0.0 and angle_delta.y() != 0:
            vertical_delta = (
                float(angle_delta.y())
                / 120.0
                * max(1, QApplication.wheelScrollLines())
                * max(1, table.verticalScrollBar().singleStep())
            )
        if abs(horizontal_delta) <= 0.0 and angle_delta.x() != 0:
            horizontal_delta = (
                float(angle_delta.x())
                / 120.0
                * max(1, QApplication.wheelScrollLines())
                * max(1, table.horizontalScrollBar().singleStep())
            )

        if event.modifiers() & Qt.ShiftModifier and abs(horizontal_delta) <= 0.0:
            horizontal_delta = vertical_delta
            vertical_delta = 0.0

        self._apply_scrollbar_wheel_delta(table.verticalScrollBar(), vertical_delta)
        self._apply_scrollbar_wheel_delta(table.horizontalScrollBar(), horizontal_delta)
        event.accept()
        return True

    def _apply_scrollbar_wheel_delta(self, scroll_bar, wheel_delta: float) -> None:  # type: ignore[no-untyped-def]
        if abs(wheel_delta) <= 1e-6:
            return
        current_value = int(scroll_bar.value())
        target_value = current_value - int(round(wheel_delta))
        target_value = min(max(target_value, int(scroll_bar.minimum())), int(scroll_bar.maximum()))
        scroll_bar.setValue(target_value)

    def _graphics_view_current_scale(self, view: QGraphicsView) -> float:
        transform = view.transform()
        return min(abs(transform.m11()), abs(transform.m22()))

    def _graphics_view_fit_scale(self, view: QGraphicsView) -> float:
        scene = view.scene()
        if scene is None:
            return 1.0
        scene_rect = scene.sceneRect()
        if scene_rect.width() <= 0.0 or scene_rect.height() <= 0.0:
            return 1.0
        viewport_size = view.viewport().size()
        width_scale = viewport_size.width() / scene_rect.width()
        height_scale = viewport_size.height() / scene_rect.height()
        fit_scale = max(min(width_scale, height_scale), 1e-6)
        max_scale = self._graphics_view_max_scale(view)
        if max_scale is not None:
            return min(fit_scale, max_scale)
        return fit_scale

    def _fit_graphics_view_to_scene(self, view: QGraphicsView) -> None:
        scene = view.scene()
        if scene is None or scene.sceneRect().isEmpty():
            return
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        self._cap_graphics_view_to_max_scale(view)
        self._update_live_star_map_zoom_scale(view)

    def _graphics_view_max_scale(self, view: QGraphicsView) -> float | None:
        if view is self.ui.realImageView:
            return self._real_image_zoom_max_scale
        if hasattr(self.ui, "imageSequenceView") and view is self.ui.imageSequenceView:
            return self._real_image_zoom_max_scale
        if view is self.ui.referenceImageView and self._can_sync_reference_real_views():
            return self._real_image_zoom_max_scale
        return None

    def _live_star_map_item_for_view(self, view: QGraphicsView) -> LiveStarMapGraphicsItem | None:
        if view is self.ui.starMapView:
            return self.star_map_item
        if view is self.ui.referenceImageView:
            return self.reference_star_map_item
        if view is self.ui.realImageView:
            return self.real_reference_overlay_item
        return None

    def _update_live_star_map_zoom_scale(self, view: QGraphicsView) -> None:
        item = self._live_star_map_item_for_view(view)
        if item is None:
            return
        fit_scale = max(self._graphics_view_fit_scale(view), 1e-6)
        current_scale = max(self._graphics_view_current_scale(view), 1e-6)
        item.set_view_zoom_scale(current_scale / fit_scale)

    def _cap_graphics_view_to_max_scale(self, view: QGraphicsView) -> None:
        max_scale = self._graphics_view_max_scale(view)
        if max_scale is None:
            return
        if self._graphics_view_current_scale(view) > max_scale:
            self._set_graphics_view_scale(view, max_scale)

    def _set_graphics_view_scale(self, view: QGraphicsView, target_scale: float) -> None:
        center = view.mapToScene(view.viewport().rect().center())
        view.resetTransform()
        view.scale(target_scale, target_scale)
        view.centerOn(center)

    def _apply_drag_delta(self, dx: int, dy: int) -> None:
        camera = self._preview_camera_settings()
        az, alt = sky_center_after_drag(
            center_az_deg=self.ui.doubleSpinBoxAz.value(),
            center_alt_deg=self.ui.doubleSpinBoxAlt.value(),
            dx_px=dx,
            dy_px=dy,
            horizontal_fov_deg=horizontal_fov_deg(camera),
            vertical_fov_deg=vertical_fov_deg(camera),
            viewport_width_px=self.ui.starMapView.viewport().width(),
            viewport_height_px=self.ui.starMapView.viewport().height(),
            min_degrees_per_pixel=0.01,
        )
        self.ui.doubleSpinBoxAz.blockSignals(True)
        self.ui.doubleSpinBoxAlt.blockSignals(True)
        self.ui.doubleSpinBoxAz.setValue(az)
        self.ui.doubleSpinBoxAlt.setValue(alt)
        self.ui.doubleSpinBoxAz.blockSignals(False)
        self.ui.doubleSpinBoxAlt.blockSignals(False)
        self.render_now()

    def _apply_roll_drag_delta(self, dx: int) -> None:
        roll = roll_after_drag(self.ui.doubleSpinBoxRoll.value(), dx, drag_sign=1.0)
        self.ui.doubleSpinBoxRoll.blockSignals(True)
        self.ui.doubleSpinBoxRoll.setValue(roll)
        self.ui.doubleSpinBoxRoll.blockSignals(False)
        self.render_now()
