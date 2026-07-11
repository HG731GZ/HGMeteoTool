"""流星框选专用图像视图。"""

from __future__ import annotations

from collections.abc import Iterable

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from ..meteor_selection import MeteorBox


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
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def clear_image(self) -> None:
        """移除当前图像与所有框选。"""

        if self._drawing_item is not None:
            self._scene.removeItem(self._drawing_item)
        self._pixmap_item.setPixmap(QPixmap())
        self._pixmap_item.setTransform(QTransform())
        self._remove_box_items()
        self._drawing_item = None
        self._drawing_start = None
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
        current_scale = min(abs(self.transform().m11()), abs(self.transform().m22()))
        if factor < 1.0 and current_scale * factor < self._fit_scale():
            self.fit_image()
        else:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            event.button() == Qt.LeftButton
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
        elif modifiers & Qt.ControlModifier and not self._image_rect.isEmpty():
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()
