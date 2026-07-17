from __future__ import annotations


class StarPairResetMixin:
    """统一重置参考星列表及其匹配派生状态。"""

    def reset_reference_star_list(
        self,
        *,
        use_native_reference_map: bool = False,
    ) -> tuple[int, int, int]:
        """清空匹配状态并按当前标注设置重建列表，不改动标注数量等界面参数。

        返回清除的有效匹配数、移除的参考星行数和重建后的参考星行数。
        """

        table = self.ui.tableWidgetStarPairs
        is_group_row = getattr(self, "_is_star_pair_group_row", None)
        removed_row_count = sum(
            1
            for row in range(table.rowCount())
            if not callable(is_group_row) or not is_group_row(row)
        )
        pair_count = self._star_pair_position_count()

        if getattr(self, "_active_star_pair_row", None) is not None:
            if hasattr(self, "_leave_star_pick_mode"):
                self._leave_star_pick_mode()
            else:
                self._active_star_pair_row = None

        store = getattr(self, "_star_pair_store", None)
        if store is not None:
            store.clear()
        if hasattr(self, "_clear_star_pair_annotations"):
            self._clear_star_pair_annotations()

        signals_were_blocked = table.blockSignals(True)
        try:
            table.setRowCount(0)
        finally:
            table.blockSignals(signals_were_blocked)

        self._manual_match_group_expanded = True
        self._manual_reference_star_ids = []
        self._imported_reference_star_by_id = {}
        self._auto_match_reference_star_ids = []
        self._auto_match_group_order = []
        self._auto_match_group_by_star_id = {}
        self._auto_match_group_expanded_by_id = {}
        self._auto_match_next_group_index = 0
        self._excluded_reference_star_ids = []
        self._mask_excluded_reference_star_ids = set()
        self._star_pair_sort_key = None
        self._star_pair_sort_descending = True
        self._hidden_star_pair_annotation_ids = set()
        self._current_reference_stars = ()

        self._sky_alignment_transform = None
        self._source_astrometric_model = None
        self._reference_alignment_error_message = ""
        self._sky_alignment_error_message = ""
        self._source_model_error_message = ""

        if use_native_reference_map:
            self._current_reference_star_map = getattr(self, "_current_star_map", None)
        reference_map = getattr(self, "_current_reference_star_map", None) or getattr(
            self,
            "_current_star_map",
            None,
        )
        if reference_map is not None and hasattr(self, "_refresh_reference_stars_from_current_map"):
            self._refresh_reference_stars_from_current_map()
        if hasattr(self, "_update_reference_alignment_transform"):
            self._update_reference_alignment_transform()

        rebuilt_row_count = sum(
            1
            for row in range(table.rowCount())
            if not callable(is_group_row) or not is_group_row(row)
        )
        return pair_count, removed_row_count, rebuilt_row_count


__all__ = ["StarPairResetMixin"]
