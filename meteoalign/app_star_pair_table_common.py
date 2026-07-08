from __future__ import annotations
from .app_constants import *

import math

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTableWidgetItem,
)

from .config import StarMapUiConfig
from .star_fitting import FittedStarPosition

from .app_constants import (
    STAR_PAIR_INDEX_COLUMN,
    STAR_PAIR_NAME_COLUMN,
    STAR_PAIR_POSITION_COLUMN,
    STAR_PAIR_RESIDUAL_COLUMN,
    STAR_PAIR_RESIDUAL_WIDTH_SAMPLE,
    STAR_PAIR_SORT_KEY_INDEX,
    STAR_PAIR_SORT_KEY_RESIDUAL,
    STAR_PAIR_SORTABLE_COLUMNS,
    STAR_PAIR_ROW_TYPE_ROLE,
    STAR_PAIR_FIT_ROLE,
    STAR_PAIR_CONSTRAINT_MODE_ROLE,
    STAR_PAIR_FIT_WEIGHT_ROLE,
    STAR_PAIR_POSITION_ROLE,
    STAR_PAIR_AUTO_GROUP_ROLE,
    STAR_PAIR_ROW_TYPE_MANUAL,
    STAR_PAIR_ROW_TYPE_MANUAL_GROUP,
    STAR_PAIR_ROW_TYPE_AUTO_GROUP,
    STAR_PAIR_ROW_TYPE_AUTO_MATCH,
    STAR_PAIR_MANUAL_GROUP_LABEL,
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_CONSTRAINT_MODES,
    STAR_PAIR_FOCUS_MIN_MATCHED_COUNT,
    STAR_PAIR_FOCUS_ZOOM_FIT_SCALE,
    STAR_PAIR_FOCUS_MARKER_RADIUS_PX,
    STAR_PICK_CIRCLE_STEP_PX,
    MIN_PSF_RADIUS_PX,
    STAR_ANNOTATION_PSF_SIGMA_SCALE,
    STAR_ANNOTATION_MIN_RADIUS_PX,
    STAR_ANNOTATION_FALLBACK_RADIUS_PX,
    STAR_ANNOTATION_MAX_RADIUS_PX,
    REFERENCE_STAR_PICK_SCREEN_RADIUS_PX,
)




__all__ = [name for name in globals() if not name.startswith("__")]
