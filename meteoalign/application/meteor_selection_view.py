"""流星框选专用图像视图。"""

from __future__ import annotations

from collections.abc import Iterable

from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QMenu

from ..meteor_selection import MeteorBox
from ..view_gestures import ViewZoomPolicy, native_gesture_zoom_factor


class MeteorSelectionView(QGraphicsView):
    """显示预览图，并在原始图像像素坐标中维护流星框。"""

    boxesChanged = pyqtSignal(list)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(0.0)
        self._scene.addItem(self._pixmap_item)
        self._box_items: list[QGraphicsRectItem] = []
        self._drawing_item: QGraphicsRectItem | None = None
        self._drawing_start: QPointF | None = None
        self._image_rect = QRectF()
        self._touchpad_pinch_zoom_enabled = True
        self._box_editing_enabled = True
        self._native_zoom_center: QPointF | None = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        # macOS 原生手势可能被投递给视图本身或其 viewport，两者都需要监听。
        self.installEventFilter(self)
        self.viewport().installEventFilter(self)

    def set_touchpad_pinch_zoom_enabled(self, enabled: bool) -> None:
        """设置是否响应触控板双指捏合缩放。"""

        self._touchpad_pinch_zoom_enabled = bool(enabled)

    def set_box_editing_enabled(self, enabled: bool) -> None:
        """设置是否允许通过 Ctrl 拖拽新增流星框。"""

        self._box_editing_enabled = bool(enabled)
        if not self._box_editing_enabled and self._drawing_item is not None:
            self._scene.removeItem(self._drawing_item)
            self._drawing_item = None
            self._drawing_start = None
            self.viewport().unsetCursor()

    def clear_image(self) -> None:
        """移除当前图像与所有框选。"""

        if self._drawing_item is not None:
            self._scene.removeItem(self._drawing_item)
        self._pixmap_item.setPixmap(QPixmap())
        self._pixmap_item.setTransform(QTransform())
        self._remove_box_items()
        self._drawing_item = None
        self._drawing_start = None
        self._native_zoom_center = None
        self._image_rect = QRectF()
        self._scene.setSceneRect(QRectF())
        self.resetTransform()

    def set_image(self, image: QImage, original_width: int, original_height: int) -> None:
        """设置预览图，并把场景单位映射为图像原始像素。"""

        self.clear_image()
        if image.isNull() or original_width <= 0 or original_height <= 0:
            return
        preview_width = image.width()
        preview_height = image.height()
        if preview_width <= 0 or preview_height <= 0:
            return

        self._image_rect = QRectF(0.0, 0.0, float(original_width), float(original_height))
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._pixmap_item.setTransform(
            QTransform().scale(float(original_width) / preview_width, float(original_height) / preview_height)
        )
        self._scene.setSceneRect(self._image_rect)
        QTimer.singleShot(0, self.fit_image)

    def replace_display_image(self, image: QImage) -> None:
        """只替换当前显示像素，保留缩放位置和已有流星框。"""

        if image.isNull() or self._image_rect.isEmpty():
            return
        preview_width = image.width()
        preview_height = image.height()
        if preview_width <= 0 or preview_height <= 0:
            return
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._pixmap_item.setTransform(
            QTransform().scale(
                self._image_rect.width() / preview_width,
                self._image_rect.height() / preview_height,
            )
        )
        self.viewport().update()

    def set_boxes(self, boxes: Iterable[MeteorBox]) -> None:
        """替换当前图像的所有框选，不触发修改信号。"""

        self._remove_box_items()
        if self._image_rect.isEmpty():
            return
        for box in boxes:
            clamped = self._clamp_box(box)
            if clamped.right <= clamped.left or clamped.bottom <= clamped.top:
                continue
            self._add_box_item(self._rect_from_box(clamped))

    def boxes(self) -> list[MeteorBox]:
        """返回当前框选，坐标均位于原始图像像素系。"""

        return [self._box_from_rect(item.rect()) for item in self._box_items]

    def clear_boxes(self) -> None:
        """删除当前图像的全部框选。"""

        if not self._box_items:
            return
        self._remove_box_items()
        self.boxesChanged.emit(self.boxes())

    def fit_image(self) -> None:
        """将完整图像适配到可视区域。"""

        if not self._image_rect.isEmpty():
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._image_rect.isEmpty():
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.25 if delta > 0 else 0.8
        self._apply_zoom_factor(factor)
        event.accept()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        """在视图与 viewport 上接收 macOS 触控板原生缩放手势。"""

        if watched in (self, self.viewport()) and event.type() == QEvent.NativeGesture:
            return self._handle_native_gesture(event)
        return super().eventFilter(watched, event)

    def _handle_native_gesture(self, event) -> bool:  # type: ignore[no-untyped-def]
        """使用真实图像预览相同的中心锚定逻辑处理双指捏合。"""

        if self._image_rect.isEmpty():
            return False
        begin_gesture = getattr(Qt, "BeginNativeGesture", None)
        if begin_gesture is not None and event.gestureType() == begin_gesture:
            self._native_zoom_center = self._viewport_center_scene_point()
            return False

        end_gesture = getattr(Qt, "EndNativeGesture", None)
        if end_gesture is not None and event.gestureType() == end_gesture:
            self._native_zoom_center = None
            return False

        factor = native_gesture_zoom_factor(
            event,
            ViewZoomPolicy(pinch_enabled=self._touchpad_pinch_zoom_enabled),
        )
        if factor is None:
            return False
        if self._native_zoom_center is None:
            self._native_zoom_center = self._viewport_center_scene_point()
        self._apply_zoom_factor(factor, center_scene=self._native_zoom_center)
        return True

    def _viewport_center_scene_point(self) -> QPointF:
        """返回当前预览窗口中心对应的场景坐标。"""

        return self.mapToScene(self.viewport().rect().center())

    def _apply_zoom_factor(self, factor: float, *, center_scene: QPointF | None = None) -> None:
        """缩放视图，并确保缩小时不会小于完整图像适配比例。"""

        current_scale = min(abs(self.transform().m11()), abs(self.transform().m22()))
        if factor < 1.0 and current_scale * factor < self._fit_scale():
            self.fit_image()
            return
        if center_scene is None:
            self.scale(factor, factor)
            return

        # 原生手势没有可靠的鼠标位置。临时以视口中心缩放并恢复场景中心，
        # 与星点匹配的真实图像预览保持一致，避免图像向右下或左上漂移。
        previous_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        try:
            self.scale(factor, factor)
        finally:
            self.setTransformationAnchor(previous_anchor)
        self.centerOn(center_scene)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            self._box_editing_enabled
            and event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ControlModifier
            and not self._image_rect.isEmpty()
        ):
            self.setFocus(Qt.MouseFocusReason)
            self._drawing_start = self._clamp_scene_point(self.mapToScene(event.pos()))
            self._drawing_item = self._new_box_item(QRectF(self._drawing_start, self._drawing_start))
            self._scene.addItem(self._drawing_item)
            self.viewport().setCursor(Qt.SizeAllCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drawing_item is not None and self._drawing_start is not None:
            endpoint = self._clamp_scene_point(self.mapToScene(event.pos()))
            self._drawing_item.setRect(QRectF(self._drawing_start, endpoint).normalized())
            event.accept()
            return
        self._update_cursor_for_modifiers(event.modifiers())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton and self._drawing_item is not None:
            drawing_item = self._drawing_item
            rect = drawing_item.rect().normalized()
            self._drawing_item = None
            self._drawing_start = None
            if rect.width() < 1.0 or rect.height() < 1.0:
                self._scene.removeItem(drawing_item)
            else:
                # 临时项已成为正式框选，加入列表后统一从 boxes() 导出。
                self._box_items.append(drawing_item)
                self.boxesChanged.emit(self.boxes())
            self._update_cursor_for_modifiers(event.modifiers())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """仅在右键命中流星框内部时提供单框删除操作。"""

        if not self._box_editing_enabled:
            super().contextMenuEvent(event)
            return
        scene_position = self.mapToScene(event.pos())
        box_item = next(
            (item for item in reversed(self._box_items) if item.rect().contains(scene_position)),
            None,
        )
        if box_item is None:
            super().contextMenuEvent(event)
            return

        menu = QMenu(self)
        delete_action = menu.addAction("删除该选框（需手动保存）")
        selected_action = menu.exec_(event.globalPos())
        if selected_action is delete_action:
            self._scene.removeItem(box_item)
            self._box_items.remove(box_item)
            self.boxesChanged.emit(self.boxes())
        event.accept()

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._update_cursor_for_modifiers(event.modifiers())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._update_cursor_for_modifiers(event.modifiers())
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drawing_item is None:
            self.viewport().unsetCursor()
        super().leaveEvent(event)

    def _new_box_item(self, rect: QRectF) -> QGraphicsRectItem:
        item = QGraphicsRectItem(rect)
        pen = QPen(QColor(52, 211, 153))
        pen.setWidth(2)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.NoBrush))
        item.setZValue(1.0)
        return item

    def _add_box_item(self, rect: QRectF) -> None:
        item = self._new_box_item(rect)
        self._scene.addItem(item)
        self._box_items.append(item)

    def _remove_box_items(self) -> None:
        for item in self._box_items:
            self._scene.removeItem(item)
        self._box_items.clear()

    def _rect_from_box(self, box: MeteorBox) -> QRectF:
        return QRectF(box.left, box.top, box.right - box.left, box.bottom - box.top)

    def _box_from_rect(self, rect: QRectF) -> MeteorBox:
        normalized = rect.normalized()
        return MeteorBox(normalized.left(), normalized.top(), normalized.right(), normalized.bottom())

    def _clamp_box(self, box: MeteorBox) -> MeteorBox:
        return box.clamped(round(self._image_rect.width()), round(self._image_rect.height()))

    def _clamp_scene_point(self, point: QPointF) -> QPointF:
        return QPointF(
            min(max(point.x(), self._image_rect.left()), self._image_rect.right()),
            min(max(point.y(), self._image_rect.top()), self._image_rect.bottom()),
        )

    def _fit_scale(self) -> float:
        if self._image_rect.isEmpty() or self.viewport().width() <= 0 or self.viewport().height() <= 0:
            return 0.0
        return min(
            self.viewport().width() / self._image_rect.width(),
            self.viewport().height() / self._image_rect.height(),
        )

    def _update_cursor_for_modifiers(self, modifiers: Qt.KeyboardModifiers) -> None:
        if self._drawing_item is not None:
            self.viewport().setCursor(Qt.SizeAllCursor)
        elif self._box_editing_enabled and modifiers & Qt.ControlModifier and not self._image_rect.isEmpty():
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()
