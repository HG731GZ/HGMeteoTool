from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QProgressDialog

from .app_sequence_export import SequenceExportMixin
from .app_sequence_import import SequenceImportMixin
from .app_sequence_matching import SequenceMatchingMixin
from .app_sequence_processing import SequenceProcessingMixin
from .app_sequence_table_preview import SequenceTablePreviewMixin
from .app_workers import ImageSequenceCollectWorker
from ..image_preview import ImagePreview
from ..image_sequence import ImageSequenceItem
from ..simulator import ProjectedStarMap
from ..source_model import SourceAstrometricModel
from ..sequence_types import (
    _SequenceCandidate,
    _SequenceFitPlan,
    _SequenceMatchedPair,
    _SequencePairTemplate,
)


class SequenceBatchMixin(
    SequenceTablePreviewMixin,
    SequenceImportMixin,
    SequenceMatchingMixin,
    SequenceExportMixin,
    SequenceProcessingMixin,
):
    """图像序列批处理 Mixin：组合序列表格、导入、匹配求解和输出写入。"""


    ui: object
    _image_sequence_items: list[ImageSequenceItem]
    _image_sequence_current_index: int
    _image_sequence_sort_key: str | None
    _image_sequence_sort_descending: bool
    _sequence_processing_active: bool
    _preserve_sequence_on_next_image_load: bool
    _current_star_map: ProjectedStarMap | None
    _source_astrometric_model: SourceAstrometricModel | None
    current_image_preview: ImagePreview | None
    current_sky_mask: np.ndarray | None
    current_sky_mask_path: Path | None
    image_sequence_item: object
    image_sequence_scene: object
    _sequence_import_thread: object | None
    _sequence_import_worker: ImageSequenceCollectWorker | None
    _sequence_import_progress: QProgressDialog | None
    _image_sequence_preview_cache: OrderedDict[str, ImagePreview]
    _image_sequence_scaled_mask_cache: OrderedDict[tuple[int, int, int], np.ndarray]
    _image_sequence_masked_preview_cache: OrderedDict[tuple[str, int, int, int], QImage]


__all__ = [
    "SequenceBatchMixin",
    "_SequenceCandidate",
    "_SequenceFitPlan",
    "_SequenceMatchedPair",
    "_SequencePairTemplate",
]
