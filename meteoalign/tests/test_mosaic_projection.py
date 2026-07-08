from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from meteoalign.alignment.constants import SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT
from meteoalign.app_mosaic import MosaicProjectionMixin
from meteoalign.mosaic_common import (
    MOSAIC_OVERLAY_MODE_SOURCE_IMAGE,
    MOSAIC_OVERLAY_MODES,
    MOSAIC_PROJECTION_MODELS,
)


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
    def value(self) -> float:
        return float(self._value)

    def setValue(self, value: float) -> None:  # noqa: N802 - Qt 控件接口命名
        self._value = float(value)


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
