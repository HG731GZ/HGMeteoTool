from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt5.QtCore import QEvent, Qt

from .application.app_constants import (
    TOUCHPAD_ZOOM_MAX_FACTOR,
    TOUCHPAD_ZOOM_MIN_FACTOR,
    TOUCHPAD_ZOOM_SENSITIVITY,
)


@dataclass(frozen=True)
class ViewZoomPolicy:
    """视图缩放手势的统一数学参数。"""

    wheel_enabled: bool = True
    pinch_enabled: bool = True
    min_fov: float = 1.0
    max_fov: float = 360.0
    sensitivity: float = TOUCHPAD_ZOOM_SENSITIVITY
    min_factor: float = TOUCHPAD_ZOOM_MIN_FACTOR
    max_factor: float = TOUCHPAD_ZOOM_MAX_FACTOR


def clamp_fov(fov_deg: float, min_fov: float, max_fov: float) -> float:
    """把 FOV 限制在页面允许的范围内。"""

    return max(float(min_fov), min(float(max_fov), float(fov_deg)))


def wheel_zoom_factor(
    wheel_delta: int | float,
    step_factor: float,
    reverse_step_factor: float | None = None,
) -> float | None:
    """把鼠标滚轮增量转换成缩放倍率。"""

    delta = float(wheel_delta)
    if abs(delta) <= 1e-9:
        return None
    factor = float(step_factor)
    if not math.isfinite(factor) or factor <= 0.0 or abs(factor - 1.0) <= 1e-9:
        return None
    if delta > 0.0:
        return factor
    if reverse_step_factor is None:
        return 1.0 / factor
    reverse_factor = float(reverse_step_factor)
    if not math.isfinite(reverse_factor) or reverse_factor <= 0.0:
        return None
    return reverse_factor


def fov_after_zoom(current_fov_deg: float, zoom_factor: float, policy: ViewZoomPolicy) -> float:
    """按“放大时 FOV 变小”的预览习惯计算新 FOV。"""

    if not math.isfinite(zoom_factor) or zoom_factor <= 0.0:
        return clamp_fov(current_fov_deg, policy.min_fov, policy.max_fov)
    return clamp_fov(float(current_fov_deg) / float(zoom_factor), policy.min_fov, policy.max_fov)


def native_gesture_zoom_value(event, policy: ViewZoomPolicy) -> float | None:  # type: ignore[no-untyped-def]
    """解析 macOS 触控板原生缩放增量。"""

    if not policy.pinch_enabled or event.type() != QEvent.NativeGesture:
        return None
    zoom_gesture = getattr(Qt, "ZoomNativeGesture", None)
    if zoom_gesture is None or event.gestureType() != zoom_gesture:
        return None
    try:
        value = float(event.value())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or abs(value) <= 1e-6:
        return None
    return value


def native_gesture_zoom_factor(event, policy: ViewZoomPolicy) -> float | None:  # type: ignore[no-untyped-def]
    """把原生缩放增量转换成平滑、有限幅的缩放倍率。"""

    value = native_gesture_zoom_value(event, policy)
    if value is None:
        return None
    factor = math.exp(value * float(policy.sensitivity))
    return max(float(policy.min_factor), min(float(policy.max_factor), factor))


def sky_center_after_drag(
    *,
    center_az_deg: float,
    center_alt_deg: float,
    dx_px: int | float,
    dy_px: int | float,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    viewport_width_px: int,
    viewport_height_px: int,
    min_degrees_per_pixel: float = 0.005,
) -> tuple[float, float]:
    """把预览拖拽像素量转换为中心方位角和高度角。"""

    az_degrees_per_pixel = max(float(horizontal_fov_deg) / max(int(viewport_width_px), 1), float(min_degrees_per_pixel))
    alt_degrees_per_pixel = max(float(vertical_fov_deg) / max(int(viewport_height_px), 1), float(min_degrees_per_pixel))
    az = (float(center_az_deg) - float(dx_px) * az_degrees_per_pixel) % 360.0
    alt = max(-90.0, min(90.0, float(center_alt_deg) + float(dy_px) * alt_degrees_per_pixel))
    return az, alt


def roll_after_drag(
    roll_deg: float,
    dx_px: int | float,
    *,
    degrees_per_pixel: float = 0.25,
    drag_sign: float = -1.0,
) -> float:
    """把水平拖拽转换为 roll，并规范到 [-180, 180]。"""

    roll = float(roll_deg) + float(drag_sign) * float(dx_px) * float(degrees_per_pixel)
    while roll > 180.0:
        roll -= 360.0
    while roll < -180.0:
        roll += 360.0
    return roll
