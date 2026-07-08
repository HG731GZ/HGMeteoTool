from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meteoalign.alignment.constants import SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
from meteoalign.app_reference_json_io import ReferenceJsonIOMixin
from meteoalign.app_mosaic import MosaicProjectionMixin
from meteoalign.config import load_star_map_ui_config
from meteoalign.mosaic_framing import MOSAIC_FRAMING_SCHEMA
from meteoalign.mosaic_common import (
    MOSAIC_OVERLAY_MODE_SOURCE_IMAGE,
    MOSAIC_OVERLAY_MODES,
    MOSAIC_PROJECTION_MODELS,
)
from meteoalign.mosaic_model_io import MosaicCoverageCache


class _Control:
    def __init__(self, value=0) -> None:  # type: ignore[no-untyped-def]
        self._value = value
        self._blocked = False

    def blockSignals(self, blocked: bool) -> bool:  # noqa: N802 - Qt 控件接口命名
        previous = self._blocked
        self._blocked = bool(blocked)
        return previous


class _ComboBox(_Control):
    def currentIndex(self) -> int:  # noqa: N802 - Qt 控件接口命名
        return int(self._value)

    def setCurrentIndex(self, value: int) -> None:  # noqa: N802 - Qt 控件接口命名
        self._value = int(value)


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
    config_path = tmp_path / "meteoalign_ui_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mosaic_texture_scale_percent": 50.0,
                "mosaic_texture_max_long_side_px": 1800,
                "mosaic_grid_precision_default": 30,
                "mosaic_render_fps_limit": 75,
            }
        ),
        encoding="utf-8",
    )

    config = load_star_map_ui_config(config_path)

    assert config.mosaic_texture_scale_percent == 50.0
    assert config.mosaic_texture_max_long_side_px == 1800
    assert config.mosaic_grid_precision_default == 30
    assert config.mosaic_render_fps_limit == 75


def test_mosaic_render_fps_limit_is_clamped() -> None:
    window = MosaicProjectionMixin.__new__(MosaicProjectionMixin)
    window.ui_config = SimpleNamespace(mosaic_render_fps_limit=500)

    assert MosaicProjectionMixin._mosaic_render_fps_limit(window) == 240


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
