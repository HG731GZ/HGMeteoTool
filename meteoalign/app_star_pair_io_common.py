from __future__ import annotations
from .app_constants import *

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QDateTime, Qt, QThread, QTimer
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QInputDialog, QMessageBox, QProgressDialog,
    QTableWidgetItem,
)

from .app_utils import _relative_image_path_for_session, _resolve_star_pair_session_real_image_path
from .app_workers import StarPairSessionImportWorker, ReferenceJsonImportWorker
from .catalog import project_root
from .fixed_camera_model import (
    FixedCameraModel,
    FixedCameraTimeFitResult,
    estimate_frame_time_correction,
)
from .image_preview import load_image_preview, ImagePreview
from .image_sequence import ImageSequenceItem, read_image_capture_time, sequence_item_observation_time_utc
from .mapping_validation import MappingValidationDialog
from .reference import build_reference_payload
from .simulator import ObserverSettings, ReferenceStar, project_horizontal_catalog
from .star_pair_model import (
    PAIR_ORIGIN_AUTO_MATCH,
    PAIR_ORIGIN_MANUAL,
    PsfFit,
    StarPairRecord,
    reference_star_from_pair_payload,
    star_pair_records_from_payloads,
)
from .alignment import MIN_ALIGNMENT_PAIRS

from .app_constants import (
    STAR_PAIR_SESSION_FORMAT, STAR_PAIR_SESSION_VERSION,
    STAR_PAIR_SESSION_JSON_FILTER,
    STAR_PAIR_INDEX_COLUMN, STAR_PAIR_FIT_ROLE, STAR_PAIR_POSITION_ROLE,
    STAR_PAIR_CONSTRAINT_MODE_ROLE, STAR_PAIR_FIT_WEIGHT_ROLE,
    STAR_PAIR_ROW_TYPE_AUTO_MATCH, STAR_PAIR_ROW_TYPE_MANUAL,
    AUTO_MATCH_CONSTRAINT_ANCHOR, AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_CONSTRAINT_MODES, REFERENCE_LABEL_MODE_FIXED_COUNT,
    REFERENCE_LABEL_MODES, RECTILINEAR_LENS_MODEL, LENS_MODELS,
)




__all__ = [name for name in globals() if not name.startswith("__")]
