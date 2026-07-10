from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QDateTime, QEvent, QPoint, QPointF, QTimer, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPolygonF
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QMessageBox

from .alignment.constants import (
    SKY_KNOWN_PROJECTION_DISPLAY_NAMES,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .application.app_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT,
    SOURCE_MODEL_JSON_FILTER,
)
from .application.app_graphics_items import GraphicsImageItem
from .catalog import project_root
from .frame_astrometry import FrameAstrometricModel
from .geometry2d import cell_crosses_angle_break, expand_polygon_radially, grid_cell_quad
from .image_preview import load_image_preview
from .projected_texture_renderer import ProjectedTextureRenderer
from .projection.grid import (
    build_pixel_radec_grid,
    grid_shape_for_long_side,
    project_altaz_grid_to_screen,
    radec_grid_to_altaz,
)
from .projection.view_state import ProjectionViewState
from .sky_scene_service import SkyPreviewRenderService, SkyPreviewStyle, SkySceneData
from .texture_projection import (
    qimage_to_rgb_array,
)
from .simulator import (
    CameraSettings,
    ObserverSettings,
    ViewSettings,
    compute_altaz_from_radec,
    horizontal_fov_deg,
    local_vectors_from_altaz,
    vertical_fov_deg,
)
from .view_gestures import (
    ViewZoomPolicy,
    clamp_fov,
    native_gesture_zoom_factor,
    roll_after_drag,
    sky_center_after_drag,
    wheel_zoom_factor,
)


MOSAIC_PROJECTION_MODELS = (
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
)
MOSAIC_RENDER_MIN_SIZE_PX = 128
MOSAIC_COVERAGE_GRID_LONG_SIDE = 36
MOSAIC_GRID_MIN_PRECISION = 12
MOSAIC_GRID_MAX_PRECISION = 180
MOSAIC_ZOOM_FACTOR = 1.18
MOSAIC_MODEL_REFIT_MIN_PAIRS = 4
MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX = 1920
MOSAIC_OVERLAY_MODE_COVERAGE = "coverage"
MOSAIC_OVERLAY_MODE_SOURCE_IMAGE = "source_image"
MOSAIC_OVERLAY_MODES = (
    MOSAIC_OVERLAY_MODE_COVERAGE,
    MOSAIC_OVERLAY_MODE_SOURCE_IMAGE,
)



__all__ = [name for name in globals() if not name.startswith("__")]
