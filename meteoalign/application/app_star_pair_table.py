from __future__ import annotations

from .app_star_pair_actions import StarPairActionsMixin
from .app_star_pair_annotations import StarPairAnnotationsMixin
from .app_star_pair_reset import StarPairResetMixin
from .app_star_pair_table_groups import StarPairTableGroupsMixin
from ..config import StarMapUiConfig
from ..alignment.models import SkyAlignmentTransform
from ..star_pair_store import StarPairStore
from PyQt5.QtGui import QCursor


class StarPairTableMixin(
    StarPairTableGroupsMixin,
    StarPairAnnotationsMixin,
    StarPairResetMixin,
    StarPairActionsMixin,
):
    """星对表格管理 Mixin：组合表格、标注、拾取和删除动作。"""

    # 这些属性由 MainWindow 或其它 Mixin 提供。
    ui: object
    ui_config: StarMapUiConfig
    _current_reference_stars: tuple
    _active_star_pair_row: int | None
    _star_pick_cursor: QCursor | None
    _star_pick_circle_diameter_px: int
    _star_pick_previous_drag_mode: object
    _star_pair_annotations: dict
    _hidden_star_pair_annotation_ids: set
    _focused_star_annotations: list
    _manual_match_group_expanded: bool
    _auto_match_reference_star_ids: list
    _auto_match_group_order: list
    _auto_match_group_by_star_id: dict
    _auto_match_group_expanded_by_id: dict
    _auto_match_next_group_index: int
    _star_pair_sort_key: str | None
    _star_pair_sort_descending: bool
    _sky_alignment_transform: SkyAlignmentTransform | None
    _sky_alignment_error_message: str
    _reference_alignment_error_message: str
    _star_pair_store: StarPairStore
    _current_star_map: object | None
    _manual_reference_star_ids: list
    _excluded_reference_star_ids: list
    _mask_excluded_reference_star_ids: set
    current_image_preview: object | None
    real_image_scene: object
    reference_scene: object
    _syncing_reference_real_views: bool
    _show_real_image_annotations: object  # 方法
    _star_pair_alignment_residual: object  # 方法
    _residual_warning_thresholds: object  # 方法
    _update_reference_alignment_transform: object  # 方法
    _update_reference_alignment_controls: object  # 方法
    _update_reference_alignment_display: object  # 方法
    _graphics_view_current_scale: object  # 方法
    _graphics_view_fit_scale: object  # 方法
    _graphics_view_max_scale: object  # 方法
    _cap_graphics_view_to_max_scale: object  # 方法
    _update_live_star_map_zoom_scale: object  # 方法
    _reference_star_lookup: object  # 方法
    _reference_star_for_row: object  # 方法
    _refresh_reference_stars_from_current_map: object  # 方法
    _reference_star_with_index: object  # 方法
    _normalized_auto_match_constraint: object  # 方法
    _update_auto_match_group_row_text: object  # 方法
    _update_star_pair_table: object  # 方法
    _auto_pair_star: object  # 方法
    _refresh_star_pair_mode_cell: object  # 方法
    _refresh_star_pair_quality_cell: object  # 方法
    _make_star_pair_table_item: object  # 方法
    _sync_star_pair_annotations_to_table: object  # 方法
    _refresh_star_pair_table_styles: object  # 方法
    _restore_star_pair_annotations_from_table: object  # 方法
    _remove_star_pair_annotation: object  # 方法

__all__ = ["StarPairTableMixin"]
