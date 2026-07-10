from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QImage, QPainter

from .projection.grid import project_altaz_grid_to_screen
from .simulator import CameraSettings, ViewSettings
from .texture_projection import (
    rgba_array_to_qimage,
    texture_projection_available,
    warp_grid_texture_to_rgba,
)


class ProjectedTextureRenderer:
    """把源图纹理网格投影到当前天空预览画布。"""

    def render_rgba(
        self,
        *,
        width: int,
        height: int,
        camera: CameraSettings,
        view: ViewSettings,
        source_rgb: np.ndarray,
        source_grid_x_px: np.ndarray,
        source_grid_y_px: np.ndarray,
        source_scale_x: float,
        source_scale_y: float,
        alt_deg: np.ndarray,
        az_deg: np.ndarray,
        valid_points: np.ndarray,
        opacity: float = 1.0,
        skip_cylindrical_seam: bool = True,
    ) -> np.ndarray:
        rgba = np.zeros((int(height), int(width), 4), dtype=np.uint8)
        if not texture_projection_available():
            return rgba

        screen_grid = project_altaz_grid_to_screen(
            alt_deg,
            az_deg,
            camera=camera,
            view=view,
            valid=valid_points,
            include_cylindrical_longitudes=skip_cylindrical_seam,
        )
        warp_grid_texture_to_rgba(
            rgba,
            source_rgb=source_rgb,
            source_grid_x_px=source_grid_x_px,
            source_grid_y_px=source_grid_y_px,
            source_scale_x=source_scale_x,
            source_scale_y=source_scale_y,
            screen_x_px=screen_grid.x_px,
            screen_y_px=screen_grid.y_px,
            valid_points=screen_grid.valid,
            opacity=opacity,
            screen_longitudes_rad=screen_grid.screen_longitudes_rad,
        )
        return rgba

    def render_qimage(self, **kwargs) -> QImage:  # type: ignore[no-untyped-def]
        return rgba_array_to_qimage(self.render_rgba(**kwargs))

    def paint_on_qimage(self, image: QImage, **kwargs) -> None:  # type: ignore[no-untyped-def]
        overlay = self.render_qimage(width=image.width(), height=image.height(), **kwargs)
        painter = QPainter(image)
        painter.drawImage(0, 0, overlay)
        painter.end()
