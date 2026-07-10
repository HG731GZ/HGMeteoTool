from __future__ import annotations

from dataclasses import dataclass

from ..domain.settings import CameraSettings, ViewSettings
from .camera_models import horizontal_fov_deg


@dataclass(frozen=True)
class ProjectionViewState:
    """独立于 Qt 控件的投影视图状态。"""

    center_az_deg: float
    center_alt_deg: float
    roll_deg: float
    fov_deg: float
    lens_model: str
    width: int
    height: int

    @classmethod
    def from_camera_and_view(cls, camera: CameraSettings, view: ViewSettings) -> "ProjectionViewState":
        return cls(
            center_az_deg=float(view.center_az_deg) % 360.0,
            center_alt_deg=max(-90.0, min(90.0, float(view.center_alt_deg))),
            roll_deg=float(view.roll_deg),
            fov_deg=float(horizontal_fov_deg(camera)),
            lens_model=str(camera.lens_model),
            width=int(camera.image_width_px),
            height=int(camera.image_height_px),
        )

    def to_view_settings(self) -> ViewSettings:
        return ViewSettings(
            center_az_deg=float(self.center_az_deg) % 360.0,
            center_alt_deg=max(-90.0, min(90.0, float(self.center_alt_deg))),
            roll_deg=float(self.roll_deg),
        )
