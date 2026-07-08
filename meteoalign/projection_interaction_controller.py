from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import QEvent, QObject, QPoint, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QGraphicsView

from .view_gestures import (
    ViewZoomPolicy,
    native_gesture_zoom_factor,
    wheel_zoom_factor,
)


class ProjectionInteractionController(QObject):
    """投影视图的统一交互控制器。

    将鼠标拖拽、滚轮缩放、触控板手势和窗口尺寸变化的处理集中到一处，
    供自由投影类预览页面共用。

    使用方式
    --------
    controller = ProjectionInteractionController(
        view=graphics_view,
        on_pan=self._handle_pan,
        on_roll=self._handle_roll,
        on_zoom=self._handle_zoom,
        on_resize=self._handle_resize,
        on_interaction_start=self._handle_interaction_start,
        on_interaction_end=self._handle_interaction_end,
        zoom_step_factor=1.18,
        zoom_policy=ViewZoomPolicy(),
    )
    # 控制器在构造时自动安装事件过滤器。
    # 销毁页面时调用 controller.detach() 移除过滤器。
    """

    def __init__(
        self,
        *,
        view: QGraphicsView,
        on_pan: Callable[[int, int], None] | None = None,
        on_roll: Callable[[int], None] | None = None,
        on_zoom: Callable[[float], None] | None = None,
        on_resize: Callable[[], None] | None = None,
        on_interaction_start: Callable[[], None] | None = None,
        on_interaction_update: Callable[[], None] | None = None,
        on_interaction_end: Callable[[], None] | None = None,
        zoom_step_factor: float = 1.18,
        zoom_policy: ViewZoomPolicy | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._on_pan = on_pan
        self._on_roll = on_roll
        self._on_zoom = on_zoom
        self._on_resize = on_resize
        self._on_interaction_start = on_interaction_start
        self._on_interaction_update = on_interaction_update
        self._on_interaction_end = on_interaction_end
        self._zoom_step_factor = float(zoom_step_factor)
        self._zoom_policy = zoom_policy or ViewZoomPolicy()

        self._last_drag_pos: QPoint | None = None
        self._interaction_active: bool = False

        self._install_filters()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def detach(self) -> None:
        """移除事件过滤器并清理状态。"""
        self._view.removeEventFilter(self)
        viewport = self._view.viewport()
        if viewport is not None:
            viewport.removeEventFilter(self)

    def set_zoom_policy(self, policy: ViewZoomPolicy) -> None:
        """运行时更新缩放策略。"""
        self._zoom_policy = policy

    def set_zoom_step_factor(self, factor: float) -> None:
        """运行时更新鼠标滚轮缩放步长。"""
        self._zoom_step_factor = float(factor)

    # ------------------------------------------------------------------
    # 事件过滤
    # ------------------------------------------------------------------

    def _install_filters(self) -> None:
        """在视图及其视口上安装事件过滤器。"""
        self._view.installEventFilter(self)
        viewport = self._view.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
            viewport.setMouseTracking(True)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        """统一处理视图交互事件。"""
        try:
            viewport = self._view.viewport()
        except RuntimeError:
            # 视图的 C++ 对象已被销毁（窗口关闭过程中），安全忽略
            return False

        # 原生手势（macOS 触控板）在视图或视口上都可能触发
        if watched in (self._view, viewport):
            if event.type() == QEvent.NativeGesture:
                return self._handle_native_gesture(event)

        # 以下事件仅在视口上处理
        if watched is not viewport:
            return super().eventFilter(watched, event)

        if event.type() == QEvent.Resize:
            return self._handle_resize()

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            return self._handle_mouse_press(event)

        if event.type() == QEvent.MouseMove:
            return self._handle_mouse_move(event)

        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            return self._handle_mouse_release()

        if event.type() == QEvent.Wheel:
            return self._handle_wheel(event)

        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # 交互生命周期
    # ------------------------------------------------------------------

    def _begin_interaction(self) -> None:
        """标记交互开始，设置抓取光标。"""
        if not self._interaction_active:
            self._interaction_active = True
            viewport = self._view.viewport()
            if viewport is not None:
                viewport.setCursor(Qt.ClosedHandCursor)
            if self._on_interaction_start is not None:
                self._on_interaction_start()

    def _update_interaction(self) -> None:
        """标记交互进行中。"""
        if self._on_interaction_update is not None:
            self._on_interaction_update()

    def _end_interaction(self) -> None:
        """标记交互结束，恢复光标。"""
        if self._interaction_active:
            self._interaction_active = False
            viewport = self._view.viewport()
            if viewport is not None:
                viewport.unsetCursor()
            if self._on_interaction_end is not None:
                self._on_interaction_end()

    # ------------------------------------------------------------------
    # 鼠标拖拽
    # ------------------------------------------------------------------

    def _handle_mouse_press(self, event: QEvent) -> bool:
        self._last_drag_pos = QPoint(event.pos())
        self._begin_interaction()
        return True

    def _handle_mouse_move(self, event: QEvent) -> bool:
        if self._last_drag_pos is None:
            return False
        dx = event.pos().x() - self._last_drag_pos.x()
        dy = event.pos().y() - self._last_drag_pos.y()
        self._last_drag_pos = QPoint(event.pos())
        if abs(dx) < 1 and abs(dy) < 1:
            return True

        if event.modifiers() & Qt.ControlModifier:
            if self._on_roll is not None:
                self._on_roll(dx)
        else:
            if self._on_pan is not None:
                self._on_pan(dx, dy)
        self._update_interaction()
        return True

    def _handle_mouse_release(self) -> bool:
        self._last_drag_pos = None
        self._end_interaction()
        return True

    # ------------------------------------------------------------------
    # 滚轮缩放
    # ------------------------------------------------------------------

    def _handle_wheel(self, event: QEvent) -> bool:
        if not self._zoom_policy.wheel_enabled:
            return False
        factor = wheel_zoom_factor(event.angleDelta().y(), self._zoom_step_factor)
        if factor is None:
            return False
        if self._on_zoom is not None:
            self._on_zoom(factor)
        return True

    # ------------------------------------------------------------------
    # 原生手势缩放（macOS 触控板）
    # ------------------------------------------------------------------

    def _handle_native_gesture(self, event: QEvent) -> bool:
        factor = native_gesture_zoom_factor(event, self._zoom_policy)
        if factor is None:
            return False
        if self._on_zoom is not None:
            self._on_zoom(factor)
        return True

    # ------------------------------------------------------------------
    # 窗口尺寸变化
    # ------------------------------------------------------------------

    def _handle_resize(self) -> bool:
        if self._on_resize is not None:
            self._on_resize()
        # 返回 False 让事件继续传递，因为 Qt 内部也需要处理 resize
        return False


__all__ = ["ProjectionInteractionController"]
