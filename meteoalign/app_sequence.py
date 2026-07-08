from __future__ import annotations

from .app_sequence_common import *  # noqa: F401, F403
from .app_sequence_export import SequenceExportMixin
from .app_sequence_import import SequenceImportMixin
from .app_sequence_matching import SequenceMatchingMixin
from .app_sequence_processing import SequenceProcessingMixin
from .app_sequence_table_preview import SequenceTablePreviewMixin


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
    _sequence_import_thread: QThread | None
    _sequence_import_worker: ImageSequenceCollectWorker | None
    _sequence_import_progress: QProgressDialog | None
    _sequence_import_progress_shown_at: float | None
    _image_sequence_preview_cache: OrderedDict[str, ImagePreview]
    _image_sequence_scaled_mask_cache: OrderedDict[tuple[int, int, int], np.ndarray]
    _image_sequence_masked_preview_cache: OrderedDict[tuple[str, int, int, int], QImage]


__all__ = [name for name in globals() if not name.startswith("__")]
