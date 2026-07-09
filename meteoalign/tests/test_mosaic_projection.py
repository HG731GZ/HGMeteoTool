from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt5.QtGui import QColor, QImage

from meteoalign.alignment.constants import SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
from meteoalign.app_reference_json_io import ReferenceJsonIOMixin
from meteoalign.app_mosaic import MosaicProjectionMixin, MosaicSourceItem
from meteoalign.config import load_star_map_ui_config
from meteoalign.mosaic_framing import MOSAIC_FRAMING_SCHEMA
from meteoalign.mosaic_common import (
    MOSAIC_OVERLAY_MODE_SOURCE_IMAGE,
    MOSAIC_OVERLAY_MODES,
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
        comboBoxMosaicOverlayMode=_ComboBox(0),
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


def test_mosaic_model_defaults_apply_json_projection_and_source_overlay() -> None:
    window = _mosaic_window()
    source_model = _source_model(SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT)

    applied = MosaicProjectionMixin._set_mosaic_projection_from_source_model(window, source_model)
    MosaicProjectionMixin._set_mosaic_overlay_defaults_for_model(window, source_model)

    assert applied
    assert window.ui.comboBoxMosaicProjection.currentIndex() == MOSAIC_PROJECTION_MODELS.index(
        SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
    )
    assert window.ui.comboBoxMosaicOverlayMode.currentIndex() == MOSAIC_OVERLAY_MODES.index(
        MOSAIC_OVERLAY_MODE_SOURCE_IMAGE
    )
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


def test_mosaic_ui_config_reads_texture_and_grid_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "preference.json"
    config_path.write_text(
        """
        {
          // 自由拼图预览贴图缩放比例。
          "mosaic_texture_scale_percent": 50.0,
          "mosaic_texture_max_long_side_px": 1800,
          "mosaic_grid_precision_default": 30,
          "mosaic_render_fps_limit": 75,
          "mosaic_export_block_rows": 512,
          "mosaic_map_tile_size_px": 4,
          "mosaic_export_tiff_lzw_compression": false
        }
        """,
        encoding="utf-8",
    )

    config = load_star_map_ui_config(config_path)

    assert config.mosaic_texture_scale_percent == 50.0
    assert config.mosaic_texture_max_long_side_px == 1800
    assert config.mosaic_grid_precision_default == 30
    assert config.mosaic_render_fps_limit == 75
    assert config.mosaic_export_block_rows == 512
    assert config.mosaic_map_tile_size_px == 4
    assert not config.mosaic_export_tiff_lzw_compression


def test_mosaic_render_fps_limit_is_clamped() -> None:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui_config = SimpleNamespace(mosaic_render_fps_limit=500)

    assert MosaicProjectionMixin._mosaic_render_fps_limit(window) == 240


def test_mosaic_exact_remap_repair_defaults_to_off() -> None:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui = SimpleNamespace()

    assert not MosaicProjectionMixin._mosaic_exact_remap_repair_enabled(window)

    window.ui = SimpleNamespace(checkBoxMosaicExactRemapRepair=_CheckBox(True))

    assert MosaicProjectionMixin._mosaic_exact_remap_repair_enabled(window)


def test_mosaic_map_tile_size_prefers_ui_control() -> None:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui_config = SimpleNamespace(mosaic_map_tile_size_px=16)
    window.ui = SimpleNamespace(doubleSpinBoxMosaicMapTileSize=_SpinBox(5.0))

    assert MosaicProjectionMixin._mosaic_map_tile_size_px(window) == 5


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

    reduced = MosaicProjectionMixin._reduced_mosaic_coverage_cache(cache)

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


def test_mosaic_source_items_paint_later_items_on_top() -> None:
    first = object()
    second = object()
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    image = QImage(2, 1, QImage.Format_ARGB32)
    image.fill(QColor(0, 0, 0))

    def fake_overlay(item, **_kwargs):  # type: ignore[no-untyped-def]
        if item is first:
            return np.asarray([[[255, 0, 0, 255], [255, 0, 0, 255]]], dtype=np.uint8)
        return np.asarray([[[0, 255, 0, 255], [0, 0, 0, 0]]], dtype=np.uint8)

    window._mosaic_source_item_overlay_rgba = fake_overlay  # type: ignore[attr-defined]

    MosaicProjectionMixin._paint_mosaic_source_items(
        window,
        image,
        camera=SimpleNamespace(),
        view=SimpleNamespace(),
        items=[first, second],  # type: ignore[list-item]
        observer=SimpleNamespace(),  # type: ignore[arg-type]
    )

    assert image.pixelColor(0, 0).green() == 255
    assert image.pixelColor(1, 0).red() == 255


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
