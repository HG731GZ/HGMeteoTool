from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meteoalign.alignment.constants import SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
from meteoalign.application.app_reference_json_io import ReferenceJsonIOMixin
from meteoalign.application.app_mosaic import (
    MOSAIC_PREVIEW_MAG_LIMIT,
    MOSAIC_PREVIEW_SHOW_GRID,
    MosaicProjectionMixin,
    MosaicSourceItem,
)
from meteoalign.mosaic.render_coordinator import MosaicRenderCoordinator
from meteoalign.mosaic_framing import MOSAIC_FRAMING_SCHEMA
from meteoalign.mosaic_common import (
    MOSAIC_PROJECTION_MODELS,
)
from meteoalign.mosaic_export import MosaicExportGeometry
from meteoalign.mosaic_model_io import MosaicCoverageCache
from meteoalign.simulator import ObserverSettings


class _Control:
    def __init__(self, value=0) -> None:  # type: ignore[no-untyped-def]
        self._value = value
        self._blocked = False

    def blockSignals(self, blocked: bool) -> bool:  # noqa: N802 - Qt 控件接口命名
        previous = self._blocked
        self._blocked = bool(blocked)
        return previous


class _ComboBox(_Control):
    def __init__(self, value=0) -> None:  # type: ignore[no-untyped-def]
        super().__init__(value)
        self._items: list[str] = []
        self._enabled = True

    def currentIndex(self) -> int:  # noqa: N802 - Qt 控件接口命名
        return int(self._value)

    def setCurrentIndex(self, value: int) -> None:  # noqa: N802 - Qt 控件接口命名
        self._value = int(value)

    def clear(self) -> None:
        self._items.clear()
        self._value = 0

    def addItem(self, text: str) -> None:  # noqa: N802 - Qt 控件接口命名
        self._items.append(str(text))

    def count(self) -> int:
        return len(self._items)

    def itemText(self, index: int) -> str:  # noqa: N802 - Qt 控件接口命名
        return self._items[index]

    def setEnabled(self, value: bool) -> None:  # noqa: N802 - Qt 控件接口命名
        self._enabled = bool(value)

    def isEnabled(self) -> bool:  # noqa: N802 - Qt 控件接口命名
        return bool(self._enabled)


class _SpinBox(_Control):
    def __init__(self, value=0, minimum=0.0, maximum=360.0) -> None:  # type: ignore[no-untyped-def]
        super().__init__(value)
        self._minimum = float(minimum)
        self._maximum = float(maximum)

    def value(self) -> float:
        return float(self._value)

    def setValue(self, value: float) -> None:  # noqa: N802 - Qt 控件接口命名
        self._value = float(value)

    def minimum(self) -> float:
        return float(self._minimum)

    def maximum(self) -> float:
        return float(self._maximum)

    def setMaximum(self, value: float) -> None:  # noqa: N802 - Qt 控件接口命名
        self._maximum = float(value)


class _CheckBox(_Control):
    def isChecked(self) -> bool:  # noqa: N802 - Qt 控件接口命名
        return bool(self._value)

    def setChecked(self, value: bool) -> None:  # noqa: N802 - Qt 控件接口命名
        self._value = bool(value)


class _Button(_Control):
    def __init__(self) -> None:
        super().__init__(False)

    def setEnabled(self, value: bool) -> None:  # noqa: N802 - Qt 控件接口命名
        self._value = bool(value)

    def isEnabled(self) -> bool:  # noqa: N802 - Qt 控件接口命名
        return bool(self._value)


def _mosaic_window() -> MosaicProjectionMixin:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(
        comboBoxMosaicProjection=_ComboBox(0),
        doubleSpinBoxMosaicOverlayOpacity=_SpinBox(100.0),
        checkBoxMosaicSkyOnly=_CheckBox(True),
    )
    window._update_mosaic_projection_controls = lambda: None  # type: ignore[attr-defined]
    return window


def _source_model(base_projection: str, source_image_path: Path | None = Path("source.jpg")) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            camera_calibration_profile=SimpleNamespace(base_projection_type=base_projection),
        ),
        source_image_path=source_image_path,
    )


def test_mosaic_model_defaults_apply_json_projection_and_source_image_overlay() -> None:
    """导入模型后应使用其投影，并固定恢复原图叠加显示。"""

    window = _mosaic_window()
    source_model = _source_model(SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT)

    applied = MosaicProjectionMixin._set_mosaic_projection_from_source_model(window, source_model)
    MosaicProjectionMixin._set_mosaic_overlay_defaults(window)

    assert applied
    assert window.ui.comboBoxMosaicProjection.currentIndex() == MOSAIC_PROJECTION_MODELS.index(
        SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
    )
    assert not hasattr(window.ui, "comboBoxMosaicOverlayMode")
    assert window.ui.doubleSpinBoxMosaicOverlayOpacity.value() == 50.0
    assert not window.ui.checkBoxMosaicSkyOnly.isChecked()
    assert MosaicProjectionMixin._mosaic_overlay_enabled(window)

    window.ui.checkBoxMosaicSkyOnly.setChecked(True)

    assert not MosaicProjectionMixin._mosaic_overlay_enabled(window)


def test_mosaic_model_defaults_keep_projection_for_free_anchor_models() -> None:
    window = _mosaic_window()
    unknown_projection = "azimuthal_equidistant_tangent"

    applied = MosaicProjectionMixin._set_mosaic_projection_from_source_model(
        window,
        _source_model(unknown_projection),
    )

    assert not applied
    assert window.ui.comboBoxMosaicProjection.currentIndex() == 0


def test_mosaic_sky_preview_uses_fixed_grid_and_mag_limit() -> None:
    """全景构图应固定显示网格并使用 6.5 等星等上限。"""

    window = SimpleNamespace(
        ui_config=SimpleNamespace(
            mosaic_font_size_multiplier=0.45,
            mosaic_star_marker_size_multiplier=0.6,
        ),
    )

    style = MosaicProjectionMixin._mosaic_sky_preview_style(window)

    assert style.font_scale == 0.45
    assert style.star_radius_scale == 0.6
    assert style.draw_grid
    assert style.draw_direction_labels
    assert MOSAIC_PREVIEW_SHOW_GRID is True
    assert MOSAIC_PREVIEW_MAG_LIMIT == 6.5


def test_mosaic_texture_long_side_uses_lower_limit_and_interaction_half() -> None:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui_config = SimpleNamespace(
        mosaic_texture_scale_percent=50.0,
        mosaic_texture_max_long_side_px=1800,
    )
    source_model = SimpleNamespace(image_width_px=6000, image_height_px=4000)

    still_long_side = MosaicProjectionMixin._mosaic_source_texture_long_side_px(
        window,
        source_model,
        interaction=False,
    )
    drag_long_side = MosaicProjectionMixin._mosaic_source_texture_long_side_px(
        window,
        source_model,
        interaction=True,
    )

    assert still_long_side == 1800
    assert drag_long_side == 900


def test_mosaic_interaction_grid_reduction_keeps_image_edges() -> None:
    values = np.arange(30, dtype=np.float64).reshape(6, 5)
    cache = MosaicCoverageCache(
        grid_rows=6,
        grid_columns=5,
        grid_x_px=values,
        grid_y_px=values + 100.0,
        ra_deg=values + 200.0,
        dec_deg=values + 300.0,
        valid=np.ones((6, 5), dtype=bool),
    )

    reduced = MosaicRenderCoordinator.reduced_coverage_cache(cache)

    assert reduced.grid_rows == 4
    assert reduced.grid_columns == 3
    assert reduced.grid_x_px[0, 0] == cache.grid_x_px[0, 0]
    assert reduced.grid_x_px[-1, -1] == cache.grid_x_px[-1, -1]


def _mosaic_source_item(
    name: str,
    observation_time_utc: datetime | None = None,
) -> MosaicSourceItem:
    observer = ObserverSettings(
        observation_time_utc=observation_time_utc or datetime(2025, 12, 14, 18, 11, 45, tzinfo=timezone.utc),
        latitude_deg=25.0,
        longitude_deg=102.0,
        elevation_m=200.0,
    )
    source_model = SimpleNamespace(
        json_path=Path(name),
        source_image_path=Path(name).with_suffix(".jpg"),
        source_image_text=Path(name).with_suffix(".jpg").name,
        observer=observer,
        utc_offset_hours=8.0,
    )
    return MosaicSourceItem(source_model=source_model, coverage_cache=None)  # type: ignore[arg-type]


def test_mosaic_display_model_combo_lists_all_and_single_items() -> None:
    first = _mosaic_source_item("first_model.json")
    second = _mosaic_source_item("second_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(comboBoxMosaicDisplayModel=_ComboBox(0))
    window._mosaic_source_items = [first, second]

    MosaicProjectionMixin._update_mosaic_display_model_combo(window)

    assert window.ui.comboBoxMosaicDisplayModel.count() == 3
    assert window.ui.comboBoxMosaicDisplayModel.itemText(0) == "显示全部"
    assert window.ui.comboBoxMosaicDisplayModel.itemText(1) == "1. first_model.jpg"
    assert window.ui.comboBoxMosaicDisplayModel.isEnabled()

    window.ui.comboBoxMosaicDisplayModel.setCurrentIndex(2)

    assert MosaicProjectionMixin._selected_mosaic_source_items(window) == [second]


def test_mosaic_multi_model_observer_uses_earliest_capture_time() -> None:
    later = _mosaic_source_item("later_model.json", datetime(2025, 8, 13, 20, 0, tzinfo=timezone.utc))
    earlier = _mosaic_source_item("earlier_model.json", datetime(2025, 8, 13, 18, 30, tzinfo=timezone.utc))
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)

    selected = MosaicProjectionMixin._earliest_mosaic_source_item(window, [later, earlier])
    MosaicProjectionMixin._set_mosaic_model_observer_from_item(window, selected)

    assert selected is earlier
    assert window._mosaic_model_observer.observation_time_utc == datetime(
        2025, 8, 13, 18, 30, tzinfo=timezone.utc
    )


def test_mosaic_multi_model_mode_disables_single_model_actions() -> None:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(
        pushButtonExportMosaicProjectedImage=_Button(),
        pushButtonResetMosaicView=_Button(),
        pushButtonCalculateMosaicResolution=_Button(),
    )
    window._mosaic_source_model = None
    window._mosaic_multi_model_mode = True
    target_transform_payload = {
        "version": 1,
        "type": "icrs_to_cropped_output_pixel",
        "boundary_width_px": 100,
        "boundary_height_px": 80,
        "crop_left_px": 0,
        "crop_top_px": 0,
        "output_width_px": 100,
        "output_height_px": 80,
        "camera": {},
        "icrs_camera_basis": {},
    }
    MosaicProjectionMixin._set_mosaic_imported_framing(window, Path("target_framing.json"), target_transform_payload)

    assert not window.ui.pushButtonExportMosaicProjectedImage.isEnabled()
    assert not window.ui.pushButtonResetMosaicView.isEnabled()
    assert not window.ui.pushButtonCalculateMosaicResolution.isEnabled()


def test_mosaic_single_and_multi_modes_replace_each_other_without_merging() -> None:
    first = _mosaic_source_item("first_model.json")
    second = _mosaic_source_item("second_model.json")
    single = _mosaic_source_item("single_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)

    MosaicProjectionMixin._activate_multi_mosaic_source_items(window, [first, second])

    assert window._mosaic_multi_model_mode
    assert window._mosaic_source_model is None
    assert MosaicProjectionMixin._mosaic_current_source_items(window) == [first, second]

    MosaicProjectionMixin._activate_single_mosaic_source_item(window, single)

    assert not window._mosaic_multi_model_mode
    assert window._mosaic_source_model is single.source_model
    assert MosaicProjectionMixin._mosaic_current_source_items(window) == [single]

    MosaicProjectionMixin._activate_multi_mosaic_source_items(window, [first, second])

    assert window._mosaic_multi_model_mode
    assert window._mosaic_source_model is None
    assert MosaicProjectionMixin._mosaic_current_source_items(window) == [first, second]


def test_mosaic_source_row_removal_preserves_remaining_order() -> None:
    """右键移除所调用的行删除逻辑应保留其他文件并重新交给统一状态更新。"""

    first = _mosaic_source_item("first_model.json")
    second = _mosaic_source_item("second_model.json")
    third = _mosaic_source_item("third_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(statusbar=SimpleNamespace(showMessage=lambda _message: None))
    MosaicProjectionMixin._activate_multi_mosaic_source_items(window, [first, second, third])
    applied: list[list[MosaicSourceItem]] = []
    window._apply_mosaic_source_items_after_removal = lambda items: applied.append(list(items))  # type: ignore[method-assign]

    MosaicProjectionMixin._remove_mosaic_source_row(window, 1)

    assert applied == [[first, third]]


def test_mosaic_source_removal_keeps_imported_framing_observer_controls() -> None:
    """已有取景时删除模型不得重写或禁用拍摄信息控件。"""

    first = _mosaic_source_item("first_model.json")
    second = _mosaic_source_item("second_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(comboBoxMosaicDisplayModel=_ComboBox(0))
    MosaicProjectionMixin._activate_multi_mosaic_source_items(window, [first, second])
    window._mosaic_imported_framing_ready = lambda *args: True  # type: ignore[method-assign]
    observer_updates: list[object] = []
    window._set_mosaic_observer_controls_from_source_model = observer_updates.append  # type: ignore[method-assign]
    window._set_mosaic_observer_controls_enabled = observer_updates.append  # type: ignore[method-assign]
    for method_name in (
        "_set_mosaic_grid_controls_enabled",
        "_update_mosaic_grid_precision_tooltip",
        "_update_mosaic_display_model_combo",
        "_update_mosaic_model_labels",
        "_update_mosaic_export_button_state",
        "schedule_mosaic_render",
    ):
        setattr(window, method_name, lambda *args, **kwargs: None)

    MosaicProjectionMixin._apply_mosaic_source_items_after_removal(window, [first])

    assert MosaicProjectionMixin._mosaic_current_source_items(window) == [first]
    assert not observer_updates


def test_clear_all_mosaic_models_requires_confirmation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """清除全部导入必须在用户确认后才提交空列表。"""

    first = _mosaic_source_item("first_model.json")
    second = _mosaic_source_item("second_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    messages: list[str] = []
    window.ui = SimpleNamespace(statusbar=SimpleNamespace(showMessage=messages.append))
    MosaicProjectionMixin._activate_multi_mosaic_source_items(window, [first, second])
    applied: list[list[MosaicSourceItem]] = []
    window._apply_mosaic_source_items_after_removal = lambda items: applied.append(list(items))  # type: ignore[method-assign]

    monkeypatch.setattr(
        "meteoalign.application.app_mosaic.QMessageBox.question",
        lambda *args, **kwargs: 65536,
    )
    MosaicProjectionMixin.clear_all_mosaic_models(window)
    assert not applied
    assert messages[-1] == "已取消清除所有源图模型。"

    monkeypatch.setattr(
        "meteoalign.application.app_mosaic.QMessageBox.question",
        lambda *args, **kwargs: 16384,
    )
    MosaicProjectionMixin.clear_all_mosaic_models(window)
    assert applied == [[]]


def test_mosaic_multi_import_with_one_file_uses_single_model_loader() -> None:
    """兼容调用只传一个文件时，应退化为单模型导入。"""

    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    loaded_paths: list[Path] = []

    def load_single_model(json_path: Path) -> bool:
        loaded_paths.append(json_path)
        return True

    window.load_mosaic_model_json = load_single_model

    assert MosaicProjectionMixin.load_mosaic_models_json(window, ["single_model.json"])
    assert loaded_paths == [Path("single_model.json")]


def test_quiet_mosaic_model_load_replaces_all_existing_models(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """星点匹配导出后的静默加载必须清空旧列表，只保留新模型。"""

    first = _mosaic_source_item("first_model.json")
    second = _mosaic_source_item("second_model.json")
    exported = _mosaic_source_item("exported_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(statusbar=SimpleNamespace(showMessage=lambda _message: None))
    MosaicProjectionMixin._activate_multi_mosaic_source_items(window, [first, second])
    window._build_mosaic_source_item = lambda _path: exported  # type: ignore[method-assign]
    window._mosaic_imported_framing_ready = lambda *args: True  # type: ignore[method-assign]
    observer_updates: list[object] = []
    window._set_mosaic_observer_controls_from_source_model = observer_updates.append  # type: ignore[method-assign]
    window._set_mosaic_observer_controls_enabled = observer_updates.append  # type: ignore[method-assign]
    for method_name in (
        "_set_mosaic_projection_from_source_model",
        "_set_mosaic_overlay_defaults",
        "_set_mosaic_grid_controls_enabled",
        "_update_mosaic_grid_precision_tooltip",
        "_reset_mosaic_center_from_model",
        "_update_mosaic_view_label",
        "_update_mosaic_display_model_combo",
        "_update_mosaic_model_labels",
        "_update_mosaic_export_button_state",
        "schedule_mosaic_render",
    ):
        setattr(window, method_name, lambda *args, **kwargs: None)
    monkeypatch.setattr("meteoalign.application.app_mosaic.QApplication.setOverrideCursor", lambda *_args: None)
    monkeypatch.setattr("meteoalign.application.app_mosaic.QApplication.restoreOverrideCursor", lambda: None)

    assert MosaicProjectionMixin.load_mosaic_model_json(window, "exported_model.json", quiet=True)
    assert MosaicProjectionMixin._mosaic_current_source_items(window) == [exported]
    assert not window._mosaic_multi_model_mode
    assert not observer_updates


def test_multi_model_import_filters_invalid_json_and_keeps_valid_items(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """批量选择混合 JSON 时只加入有效模型，全部无效时保留原列表。"""

    class _Progress:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def __getattr__(self, _name: str):  # type: ignore[no-untyped-def]
            return lambda *args, **kwargs: None

        def wasCanceled(self) -> bool:  # noqa: N802 - Qt 接口命名
            return False

    first = _mosaic_source_item("first_model.json")
    valid = _mosaic_source_item("valid_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    messages: list[str] = []
    warnings: list[tuple[object, ...]] = []
    window.ui = SimpleNamespace(statusbar=SimpleNamespace(showMessage=messages.append))
    MosaicProjectionMixin._activate_single_mosaic_source_item(window, first)

    def build_item(path: Path) -> MosaicSourceItem:
        if path.name == "valid_model.json":
            return valid
        raise ValueError("不是源图模型 JSON")

    window._build_mosaic_source_item = build_item  # type: ignore[method-assign]
    window._mosaic_imported_framing_ready = lambda *args: True  # type: ignore[method-assign]
    for method_name in (
        "_update_mosaic_view_label",
        "_set_mosaic_overlay_defaults_for_source_items",
        "_set_mosaic_grid_controls_enabled",
        "_update_mosaic_grid_precision_tooltip",
        "_update_mosaic_display_model_combo",
        "_update_mosaic_model_labels",
        "_update_mosaic_export_button_state",
        "schedule_mosaic_render",
    ):
        setattr(window, method_name, lambda *args, **kwargs: None)
    monkeypatch.setattr("meteoalign.application.app_mosaic.QProgressDialog", _Progress)
    monkeypatch.setattr("meteoalign.application.app_mosaic.QApplication.setOverrideCursor", lambda *_args: None)
    monkeypatch.setattr("meteoalign.application.app_mosaic.QApplication.restoreOverrideCursor", lambda: None)
    monkeypatch.setattr("meteoalign.application.app_mosaic.QApplication.processEvents", lambda: None)
    monkeypatch.setattr(
        "meteoalign.application.app_mosaic.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    assert MosaicProjectionMixin.load_mosaic_models_json(
        window,
        ["valid_model.json", "reference.json"],
        append=True,
    )
    assert MosaicProjectionMixin._mosaic_current_source_items(window) == [first, valid]
    assert warnings and "reference.json" in str(warnings[-1])
    assert "跳过 1 个非模型 JSON" in messages[-1]

    assert not MosaicProjectionMixin.load_mosaic_models_json(
        window,
        ["unrelated.json"],
        append=True,
    )
    assert MosaicProjectionMixin._mosaic_current_source_items(window) == [first, valid]
    assert "未导入模型" in messages[-1]


def test_mosaic_mixin_stores_business_state_in_session_state() -> None:
    """Mixin 的旧属性入口必须代理到唯一的拼图会话状态。"""

    item = _mosaic_source_item("single_model.json")
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)

    MosaicProjectionMixin._activate_single_mosaic_source_item(window, item)
    window._mosaic_center_az_deg = 123.0
    window._mosaic_center_alt_deg = 35.0
    window._mosaic_roll_deg = -12.0
    window._mosaic_output_boundary_width_px = 4096
    window._mosaic_output_boundary_height_px = 2048

    assert window.mosaic_state.sources == [item]
    assert window.mosaic_state.active_source is item
    assert not window.mosaic_state.multi_model_mode
    assert window.mosaic_state.view.center_az_deg == 123.0
    assert window.mosaic_state.view.center_alt_deg == 35.0
    assert window.mosaic_state.view.roll_deg == -12.0
    assert window.mosaic_state.view.output_boundary_width_px == 4096
    assert window.mosaic_state.view.output_boundary_height_px == 2048
    assert "_mosaic_source_items" not in window.__dict__
    assert "_mosaic_center_az_deg" not in window.__dict__


class _MosaicWithReferenceJsonMixin(ReferenceJsonIOMixin, MosaicProjectionMixin):
    pass


def test_mosaic_framing_import_uses_mosaic_payload_parser_despite_mro_collision() -> None:
    window = _MosaicWithReferenceJsonMixin.__new__(_MosaicWithReferenceJsonMixin)
    window.ui = SimpleNamespace(
        comboBoxMosaicProjection=_ComboBox(0),
        doubleSpinBoxMosaicFov=_SpinBox(120.0, minimum=5.0, maximum=360.0),
        doubleSpinBoxMosaicCropTop=_SpinBox(0.0, minimum=0.0, maximum=1_000_000.0),
        doubleSpinBoxMosaicCropBottom=_SpinBox(0.0, minimum=0.0, maximum=1_000_000.0),
        doubleSpinBoxMosaicCropLeft=_SpinBox(0.0, minimum=0.0, maximum=1_000_000.0),
        doubleSpinBoxMosaicCropRight=_SpinBox(0.0, minimum=0.0, maximum=1_000_000.0),
    )
    window._mosaic_source_model = None
    window._mosaic_center_az_deg = 0.0
    window._mosaic_center_alt_deg = 20.0
    window._mosaic_roll_deg = 0.0
    window._mosaic_output_boundary_width_px = 0
    window._mosaic_output_boundary_height_px = 0
    window._mosaic_resolution_estimate = None
    window._mosaic_framing_observer = None
    window._mosaic_framing_utc_offset_hours = 0.0

    payload = {
        "schema": MOSAIC_FRAMING_SCHEMA,
        "version": 1,
        "projection": {
            "model": SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
            "fov_deg": 200.0,
        },
        "view": {
            "center_az_deg": 199.0,
            "center_alt_deg": 88.0,
            "roll_deg": 3.0,
        },
        "observer": {
            "observation_time_utc": datetime(2025, 12, 14, 18, 11, 45, tzinfo=timezone.utc).isoformat(),
            "utc_offset_hours": 8.0,
            "latitude_deg": 25.0,
            "longitude_deg": 102.0,
            "elevation_m": 200.0,
        },
        "output": {
            "boundary_width_px": 5835,
            "boundary_height_px": 4540,
            "crop": {
                "top_px": 190.0,
                "bottom_px": 210.0,
                "left_px": 770.0,
                "right_px": 750.0,
            },
        },
    }

    MosaicProjectionMixin._apply_mosaic_framing_payload(window, payload)

    assert window.ui.comboBoxMosaicProjection.currentIndex() == MOSAIC_PROJECTION_MODELS.index(
        SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
    )
    assert window.ui.doubleSpinBoxMosaicFov.value() == 200.0
    assert window._mosaic_output_size() == (5835, 4540)
    assert window._mosaic_framing_observer.latitude_deg == 25.0
    assert window.ui.doubleSpinBoxMosaicCropTop.value() == 190.0


def test_mosaic_framing_payload_does_not_embed_source_model_or_map() -> None:
    observer = ObserverSettings(
        observation_time_utc=datetime(2025, 12, 14, 18, 11, 45, tzinfo=timezone.utc),
        latitude_deg=25.0,
        longitude_deg=102.0,
        elevation_m=200.0,
    )
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(
        comboBoxMosaicProjection=_ComboBox(0),
        doubleSpinBoxMosaicFov=_SpinBox(120.0, minimum=5.0, maximum=360.0),
    )
    window._mosaic_source_model = SimpleNamespace(
        json_path=Path("source_model.json"),
        observer=observer,
        utc_offset_hours=8.0,
        image_width_px=6000,
        image_height_px=4000,
        rms_px=0.25,
    )
    window._mosaic_center_az_deg = 199.0
    window._mosaic_center_alt_deg = 45.0
    window._mosaic_roll_deg = 3.0
    window._mosaic_output_boundary_width_px = 5835
    window._mosaic_output_boundary_height_px = 4540
    window._mosaic_resolution_estimate = None
    window._mosaic_render_size = lambda: (1200, 800)  # type: ignore[attr-defined]

    payload = MosaicProjectionMixin._mosaic_framing_payload(window)

    assert "source_model" not in payload
    assert "reprojection_map" not in payload
    assert payload["output"]["boundary_width_px"] == 5835


def test_mosaic_imported_framing_ready_requires_matching_target_icrs_to_pixel_transform() -> None:
    geometry = MosaicExportGeometry(
        boundary_width_px=100,
        boundary_height_px=80,
        crop_left_px=10,
        crop_top_px=8,
        output_width_px=60,
        output_height_px=40,
    )
    target_transform_payload = {
        "version": 1,
        "type": "icrs_to_cropped_output_pixel",
        "boundary_width_px": 100,
        "boundary_height_px": 80,
        "crop_left_px": 10,
        "crop_top_px": 8,
        "output_width_px": 60,
        "output_height_px": 40,
        "camera": {},
        "icrs_camera_basis": {},
    }
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace()

    MosaicProjectionMixin._set_mosaic_imported_framing(window, Path("target_framing.json"), target_transform_payload)

    assert MosaicProjectionMixin._mosaic_imported_framing_ready(window, geometry)
    assert not MosaicProjectionMixin._mosaic_imported_framing_ready(
        window,
        MosaicExportGeometry(
            boundary_width_px=100,
            boundary_height_px=80,
            crop_left_px=11,
            crop_top_px=8,
            output_width_px=60,
            output_height_px=40,
        ),
    )

    MosaicProjectionMixin._clear_mosaic_imported_framing(window)

    assert not MosaicProjectionMixin._mosaic_imported_framing_ready(window)


def test_mosaic_export_button_requires_source_model_and_imported_framing() -> None:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(pushButtonExportMosaicProjectedImage=_Button())
    window._mosaic_source_model = None
    target_transform_payload = {
        "version": 1,
        "type": "icrs_to_cropped_output_pixel",
        "boundary_width_px": 100,
        "boundary_height_px": 80,
        "crop_left_px": 0,
        "crop_top_px": 0,
        "output_width_px": 100,
        "output_height_px": 80,
        "camera": {},
        "icrs_camera_basis": {},
    }

    MosaicProjectionMixin._update_mosaic_export_button_state(window)

    assert not window.ui.pushButtonExportMosaicProjectedImage.isEnabled()

    MosaicProjectionMixin._set_mosaic_imported_framing(window, Path("target_framing.json"), target_transform_payload)

    assert not window.ui.pushButtonExportMosaicProjectedImage.isEnabled()

    window._mosaic_source_model = SimpleNamespace()
    MosaicProjectionMixin._update_mosaic_export_button_state(window)

    assert window.ui.pushButtonExportMosaicProjectedImage.isEnabled()


def test_imported_framing_disables_reset_and_clear_preserves_parameters() -> None:
    """清除取景只解除锁定，不得还原或重算当前参数。"""

    status_messages: list[str] = []
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace(
        pushButtonResetMosaicView=_Button(),
        pushButtonCalculateMosaicResolution=_Button(),
        pushButtonExportMosaicProjectedImage=_Button(),
        pushButtonClearMosaicFraming=_Button(),
        statusbar=SimpleNamespace(showMessage=status_messages.append),
    )
    window._mosaic_source_model = SimpleNamespace()
    window._mosaic_center_az_deg = 123.0
    window._mosaic_center_alt_deg = 45.0
    window._mosaic_roll_deg = -7.0
    window._mosaic_output_boundary_width_px = 6000
    window._mosaic_output_boundary_height_px = 4000
    observer = object()
    window._mosaic_framing_observer = observer  # type: ignore[assignment]
    target_transform_payload = {
        "type": "icrs_to_cropped_output_pixel",
        "boundary_width_px": 6000,
        "boundary_height_px": 4000,
    }

    MosaicProjectionMixin._set_mosaic_imported_framing(
        window,
        Path("target_framing.json"),
        target_transform_payload,
    )

    assert not window.ui.pushButtonResetMosaicView.isEnabled()
    assert window.ui.pushButtonCalculateMosaicResolution.isEnabled()
    assert window.ui.pushButtonClearMosaicFraming.isEnabled()
    assert window.ui.pushButtonExportMosaicProjectedImage.isEnabled()

    MosaicProjectionMixin.clear_mosaic_framing(window)

    assert not MosaicProjectionMixin._mosaic_imported_framing_ready(window)
    assert window.ui.pushButtonResetMosaicView.isEnabled()
    assert window.ui.pushButtonCalculateMosaicResolution.isEnabled()
    assert not window.ui.pushButtonClearMosaicFraming.isEnabled()
    assert not window.ui.pushButtonExportMosaicProjectedImage.isEnabled()
    assert window._mosaic_center_az_deg == 123.0
    assert window._mosaic_center_alt_deg == 45.0
    assert window._mosaic_roll_deg == -7.0
    assert window._mosaic_output_size() == (6000, 4000)
    assert window._mosaic_framing_observer is observer
    assert "均已保留" in status_messages[-1]
