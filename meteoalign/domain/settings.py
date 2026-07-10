"""观测者、相机与取景设置的数据对象。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ObserverSettings:
    """观测时刻、经纬度和海拔。"""

    observation_time_utc: datetime
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0


@dataclass(frozen=True)
class CameraSettings:
    """目标或源图相机的传感器、像素和投影参数。"""

    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    focal_length_mm: float
    lens_model: str = "rectilinear"
    fisheye_fov_deg: float = 180.0


@dataclass(frozen=True)
class ViewSettings:
    """本地地平坐标系中的取景中心和滚转角。"""

    center_az_deg: float
    center_alt_deg: float
    roll_deg: float = 0.0
