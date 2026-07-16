from __future__ import annotations

from .app_reference_json_io import ReferenceJsonIOMixin
from .app_source_model_export import SourceModelExportMixin
from .app_star_pair_json_task import StarPairJsonTaskMixin
from .app_star_pair_reference_payload import StarPairReferencePayloadMixin
from .app_star_pair_session import StarPairSessionMixin
from ..star_pair_store import StarPairStore


class StarPairIOMixin(
    StarPairJsonTaskMixin,
    StarPairReferencePayloadMixin,
    StarPairSessionMixin,
    SourceModelExportMixin,
    ReferenceJsonIOMixin,
):
    """星对数据导入导出 Mixin：组合会话、源模型和参考 JSON 流程。"""


    ui: object
    _star_pair_store: StarPairStore
    _json_import_thread: object | None
    _json_import_worker: object | None
    _json_import_progress: QProgressDialog | None
    _star_pair_session_import_switch_to_reference: bool
    _star_pair_session_import_clear_input_name: str
    _alignment_model: object  # 方法
    _set_alignment_model: object  # 方法
    _auto_match_constraint_for_star_id: object  # 方法
    _auto_match_group_label: object  # 方法
    _ensure_auto_match_group: object  # 方法
    _normalize_auto_match_groups: object  # 方法
    _normalized_auto_match_constraint: object  # 方法
    _auto_match_constraint_mode: object  # 方法
    _auto_match_soft_weight: object  # 方法
    _reference_label_mode: object  # 方法
    _observer_settings: object  # 方法
    _output_camera_settings: object  # 方法
    _camera_settings_for_image_size: object  # 方法
    _build_projected_star_map: object  # 方法
    _view_settings: object  # 方法
    _get_horizontal_catalog: object  # 方法
    _get_horizontal_milky_way: object  # 方法
    _get_horizontal_solar_system: object  # 方法
    _select_current_reference_stars: object  # 方法
    _reference_star_with_index: object  # 方法
    _lens_model: object  # 方法
    _update_lens_model_controls: object  # 方法
    _update_reference_label_controls: object  # 方法
    _is_auto_match_row: object  # 方法
    _is_star_pair_group_row: object  # 方法
    _star_pair_star_id: object  # 方法
    _star_pair_fit_constraint: object  # 方法
    _star_pair_alignment_residual: object  # 方法
    _parse_star_pair_position_text: object  # 方法
    _star_pair_position_count: object  # 方法
    _sequence_base_templates: object  # 方法
    _fit_sequence_fixed_camera_model: object  # 方法
    _first_frame_matched_pairs: object  # 方法
    _sequence_pair_fit_arrays: object  # 方法
    _apply_sequence_time_fit: object  # 方法
    _sequence_pair_records: object  # 方法
    _clear_star_pair_positions: object  # 方法
    reset_reference_star_list: object  # 方法
    _clear_star_pair_annotations: object  # 方法
    _refresh_star_pair_table_styles: object  # 方法
    _refresh_reference_stars_from_current_map: object  # 方法
    _update_reference_alignment_transform: object  # 方法
    _update_reference_alignment_controls: object  # 方法
    _update_star_pair_export_control: object  # 方法
    _set_json_import_controls_enabled: object  # 方法
    _row_auto_match_group_id: object  # 方法
    _auto_match_reference_star_ids: list
    _auto_match_group_order: list
    _auto_match_group_by_star_id: dict
    _auto_match_group_expanded_by_id: dict
    _auto_match_next_group_index: int
    _manual_reference_star_ids: list
    _imported_reference_star_by_id: dict
    _excluded_reference_star_ids: list
    _mask_excluded_reference_star_ids: set
    _current_star_map: object | None
    _sky_alignment_transform: object | None
    _source_astrometric_model: object | None
    _source_model_error_message: str
    _sky_alignment_error_message: str
    _reference_alignment_error_message: str
    current_image_preview: object | None
    current_sky_mask: np.ndarray | None
    current_sky_mask_path: Path | None


__all__ = ["StarPairIOMixin"]
