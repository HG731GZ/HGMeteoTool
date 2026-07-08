from __future__ import annotations
from .app_constants import *

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic

import numpy as np
from PyQt5.QtCore import QDateTime, Qt, QThread, QTimer
from PyQt5.QtGui import QBrush, QColor, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QProgressDialog,
    QTableWidgetItem,
)

from .alignment import (
    MIN_ALIGNMENT_PAIRS,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .app_auto_match import AUTO_MATCH_MIN_ALTITUDE_DEG
from .app_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX,
    AUTO_MATCH_MIN_AMPLITUDE,
    AUTO_MATCH_SEARCH_MAG_LIMIT,
)
from .app_utils import _image_with_binary_mask, _relative_image_path_for_session
from .app_workers import ImageSequenceCollectWorker
from .fixed_camera_model import (
    FixedCameraModel,
    FixedCameraTimeFitResult,
    estimate_frame_time_correction,
    fit_fixed_camera_model,
)
from .image_preview import IMAGE_FILE_FILTER, ImagePreview, load_image_preview
from .image_sequence import (
    ImageSequenceItem,
    RejectedSequenceImage,
    sequence_item_local_datetime,
    sequence_item_observation_time_utc,
    sequence_item_time_delta_seconds,
)
from .reference import build_reference_payload
from .simulator import (
    FISHEYE_EQUISOLID,
    ObserverSettings,
    ProjectedStarMap,
    ReferenceStar,
    camera_basis_from_view,
    local_vectors_from_altaz,
    project_horizontal_catalog,
)
from .sequence_geometry import frame_astrometric_model_from_fixed_camera
from .source_model import SourceAstrometricModel
from .star_pair_model import PsfFit, StarPairRecord
from .star_fitting import FittedStarPosition, fit_star_position


SEQUENCE_MIN_PAIR_FRACTION = 0.80
SEQUENCE_FILL_GRID_COLUMNS = 5
SEQUENCE_FILL_GRID_ROWS = 4
SEQUENCE_SUPPLEMENTAL_FIT_WEIGHT = 0.5
SEQUENCE_SUPPLEMENTAL_PAIR_ORIGIN = "auto_match"
IMAGE_SEQUENCE_INDEX_COLUMN = 0
IMAGE_SEQUENCE_NAME_COLUMN = 1
IMAGE_SEQUENCE_RMS_COLUMN = 2
IMAGE_SEQUENCE_PATH_ROLE = Qt.UserRole + 20
IMAGE_SEQUENCE_INDEX_ROLE = Qt.UserRole + 21
IMAGE_SEQUENCE_RMS_ROLE = Qt.UserRole + 22
IMAGE_SEQUENCE_SORT_KEY_INDEX = "index"
IMAGE_SEQUENCE_SORT_KEY_RMS = "rms"
IMAGE_SEQUENCE_SORTABLE_COLUMNS = {
    IMAGE_SEQUENCE_INDEX_COLUMN: IMAGE_SEQUENCE_SORT_KEY_INDEX,
    IMAGE_SEQUENCE_RMS_COLUMN: IMAGE_SEQUENCE_SORT_KEY_RMS,
}
IMAGE_SEQUENCE_PREVIEW_CACHE_LIMIT = 12
IMAGE_SEQUENCE_MASK_CACHE_LIMIT = 24
IMAGE_SEQUENCE_IMPORT_PROGRESS_MIN_VISIBLE_MS = 500


@dataclass(frozen=True)
class _SequencePairTemplate:
    star_id: str
    reference_star: ReferenceStar
    fitted_position: FittedStarPosition
    fit_constraint_mode: str
    fit_weight: float
    pair_origin: str


@dataclass(frozen=True)
class _SequenceCandidate:
    reference_star: ReferenceStar
    predicted_x_px: float
    predicted_y_px: float


@dataclass(frozen=True)
class _SequenceFitPlan:
    candidate: _SequenceCandidate
    fit_weight: float
    pair_origin: str


@dataclass(frozen=True)
class _SequenceMatchedPair:
    reference_star: ReferenceStar
    fitted_position: FittedStarPosition
    fit_constraint_mode: str
    fit_weight: float
    pair_origin: str
    predicted_x_px: float | None = None
    predicted_y_px: float | None = None
    search_x_px: float | None = None
    search_y_px: float | None = None
    initial_predicted_x_px: float | None = None
    initial_predicted_y_px: float | None = None
    time_delta_seconds: float | None = None
    adaptive_offset_x_px: float = 0.0
    adaptive_offset_y_px: float = 0.0



__all__ = [name for name in globals() if not name.startswith("__")]
