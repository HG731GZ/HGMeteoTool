"""去混叠、饱和兼容 PSF 和一对一分配测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QImage

import meteoalign.star_fitting as star_fitting_module
from meteoalign.application import app_auto_match
from meteoalign.application.app_auto_match import AutoMatchMixin
from meteoalign.psf import fitting as psf_fitting
from meteoalign.psf import measure_star_candidate_fast
from meteoalign.psf.matching import assign_predicted_sources
from meteoalign.psf.models import FittedStarPosition, StarSourceCandidate
from meteoalign.star_fitting import (
    StarFitError,
    detect_star_candidates_from_array,
    fit_star_position,
    fit_star_position_from_array,
    qimage_to_grayscale_array,
)


def _gaussian_scene(
    stars: list[tuple[float, float, float, float]],
    *,
    size: int = 81,
    saturation: float | None = None,
) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float64)
    image = np.full((size, size), 10.0, dtype=np.float64)
    for x, y, amplitude, sigma in stars:
        image += amplitude * np.exp(-0.5 * (((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2))
    if saturation is not None:
        image = np.minimum(image, saturation)
    return image


@pytest.mark.parametrize("search_radius", [8, 15, 24])
def test_psf_size_is_independent_from_search_radius_in_double_star(search_radius: int) -> None:
    """扩大搜索圆不得把两颗星拟合成一个宽 PSF。"""

    image = _gaussian_scene([(40.0, 40.0, 100.0, 1.8), (46.0, 40.0, 80.0, 1.8)])

    fitted = fit_star_position_from_array(
        image,
        40.0,
        40.0,
        search_radius,
        saturation_level=255.0,
    )

    assert fitted.x == pytest.approx(40.0, abs=0.15)
    assert fitted.y == pytest.approx(40.0, abs=0.15)
    assert fitted.fwhm_x == pytest.approx(4.24, abs=0.45)
    assert fitted.fwhm_y == pytest.approx(4.24, abs=0.45)


def test_saturated_star_uses_unsaturated_wings() -> None:
    """大面积平顶饱和核心应被容忍，并由外围星翼恢复中心。"""

    image = _gaussian_scene(
        [(40.0, 40.0, 650.0, 5.0), (61.0, 40.0, 35.0, 1.8)],
        saturation=255.0,
    )

    fitted = fit_star_position_from_array(
        image,
        40.0,
        40.0,
        22,
        max_fit_radius_px=36,
        saturation_level=255.0,
    )

    assert fitted.saturated
    assert fitted.x == pytest.approx(40.0, abs=0.25)
    assert fitted.y == pytest.approx(40.0, abs=0.25)
    assert 8.0 < fitted.fwhm_x < 13.0
    assert fitted.quality_score >= 0.75


def test_psf_fit_error_limit_can_be_tightened_for_manual_picking() -> None:
    """可配置拟合残差门槛应直接控制 poor_fit 拒绝条件。"""

    image = _gaussian_scene([(40.0, 40.0, 100.0, 2.0)])

    with pytest.raises(StarFitError) as error:
        fit_star_position_from_array(
            image,
            40.0,
            40.0,
            18,
            fit_error_limit=0.000001,
        )

    assert error.value.code == "poor_fit"


def test_center_shift_tolerance_multiplier_can_accept_displaced_fit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """提高中心偏移容限后，应能接受仍有稳定星形但中心被背景拉动的结果。"""

    image = _gaussian_scene([(40.0, 40.0, 100.0, 2.0)])
    detected = psf_fitting._detect_sources(image)
    shifted_candidate = replace(detected.candidates[0], x=detected.candidates[0].x - 2.3)
    shifted_detection = replace(detected, candidates=(shifted_candidate,))
    monkeypatch.setattr(psf_fitting, "_detect_sources", lambda *_args, **_kwargs: shifted_detection)

    with pytest.raises(StarFitError) as error:
        fit_star_position_from_array(image, 37.7, 40.0, 10)
    assert error.value.code == "center_unstable"

    fitted = fit_star_position_from_array(
        image,
        37.7,
        40.0,
        10,
        center_shift_tolerance_multiplier=1.2,
    )
    assert fitted.x == pytest.approx(40.0, abs=0.1)

    forced = fit_star_position_from_array(
        image,
        37.7,
        40.0,
        10,
        force_reliable_source=True,
    )
    assert forced.forced
    assert forced.x == pytest.approx(40.0, abs=0.1)
    assert forced.quality_score <= 0.55


def test_size_boundary_tolerance_multiplier_can_accept_broad_psf() -> None:
    """提高尺寸边界容限后，应能接受接近较小拟合窗口边界的圆星点。"""

    image = _gaussian_scene([(40.0, 40.0, 100.0, 3.5)])

    with pytest.raises(StarFitError) as error:
        fit_star_position_from_array(image, 40.0, 40.0, 18, max_fit_radius_px=5)
    assert error.value.code == "size_at_bound"

    fitted = fit_star_position_from_array(
        image,
        40.0,
        40.0,
        18,
        max_fit_radius_px=5,
        size_boundary_tolerance_multiplier=1.2,
    )
    assert fitted.fwhm_x > 6.75

    forced = fit_star_position_from_array(
        image,
        40.0,
        40.0,
        18,
        max_fit_radius_px=5,
        force_reliable_source=True,
    )
    assert forced.forced
    assert forced.x == pytest.approx(40.0, abs=0.2)


def test_forced_manual_measurement_falls_back_when_strict_quality_rejects() -> None:
    """严格残差门槛拒绝可靠星点后，可信手动模式仍应返回低质量兜底矩心。"""

    image = _gaussian_scene([(40.25, 39.75, 100.0, 2.0)])
    fitted = fit_star_position_from_array(
        image,
        40.0,
        40.0,
        12,
        fit_error_limit=0.000001,
        force_reliable_source=True,
    )

    assert fitted.forced
    assert fitted.x == pytest.approx(40.25, abs=0.25)
    assert fitted.y == pytest.approx(39.75, abs=0.25)
    assert fitted.quality_score == pytest.approx(0.35)


def test_qt_adapter_honors_fit_radius_above_old_internal_limit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Qt 适配层不得再把界面允许的大拟合半径静默截断为 64 px。"""

    captured: dict[str, object] = {}

    def fake_fit(luminance, click_x, click_y, radius_px, **kwargs):  # type: ignore[no-untyped-def]
        captured["shape"] = luminance.shape
        captured["max_fit_radius_px"] = kwargs["max_fit_radius_px"]
        return FittedStarPosition(
            x=click_x,
            y=click_y,
            amplitude=10.0,
            background=1.0,
            sigma_x=1.0,
            sigma_y=1.0,
        )

    monkeypatch.setattr(star_fitting_module, "fit_star_position_from_array", fake_fit)
    image = np.zeros((600, 600), dtype=np.float64)
    fitted = fit_star_position(image, 300.0, 300.0, 12, max_fit_radius_px=120)

    assert captured["max_fit_radius_px"] == 120
    assert captured["shape"] == (265, 265)
    assert fitted.x == 300.0


@pytest.mark.parametrize(
    ("code", "option_name"),
    [
        ("center_unstable", "中心偏移容限倍率"),
        ("size_at_bound", "尺寸边界容限倍率"),
    ],
)
def test_tunable_psf_failure_text_names_corresponding_option(code: str, option_name: str) -> None:
    """两种可放宽的拒绝原因都应在状态栏提示对应参数。"""

    text = app_auto_match._psf_fit_failure_text(StarFitError("拟合失败。", code=code))

    assert option_name in text


def test_uint16_array_preserves_native_saturation_level() -> None:
    """纯数值接口应识别 16-bit 上限，不把局部最大值误当作位深上限。"""

    image = _gaussian_scene([(40.0, 40.0, 100.0, 4.0)]) * 500.0
    image = np.clip(image, 0.0, 65535.0).astype(np.uint16)

    fitted = fit_star_position_from_array(image, 40.0, 40.0, 18)

    assert not fitted.saturated
    assert fitted.x == pytest.approx(40.0, abs=0.3)
    assert fitted.y == pytest.approx(40.0, abs=0.3)

    yy, xx = np.indices((81, 81), dtype=np.float64)
    saturated = 5000.0 + 120000.0 * (1.0 + ((xx - 40.0) / 4.0) ** 2 + ((yy - 40.0) / 4.0) ** 2) ** -2.5
    saturated = np.clip(saturated, 0.0, 65535.0).astype(np.uint16)
    saturated_fit = fit_star_position_from_array(saturated, 40.0, 40.0, 20)
    assert saturated_fit.saturated
    assert saturated_fit.x == pytest.approx(40.0, abs=0.3)


def test_psf_qt_adapter_keeps_uint8_and_uint16_signal_scales() -> None:
    """同一星点的 8/16 位原生数组不能在 Qt 适配层被强制成同一量化尺度。"""

    image_8bit = np.clip(_gaussian_scene([(40.0, 40.0, 180.0, 2.0)]), 0.0, 250.0).astype(np.uint8)
    image_16bit = image_8bit.astype(np.uint16) * 257

    fitted_8bit = fit_star_position(image_8bit, 40.0, 40.0, 16)
    fitted_16bit = fit_star_position(image_16bit, 40.0, 40.0, 16)

    assert fitted_8bit.x == pytest.approx(fitted_16bit.x, abs=0.05)
    assert fitted_8bit.y == pytest.approx(fitted_16bit.y, abs=0.05)
    assert fitted_16bit.amplitude / fitted_8bit.amplitude == pytest.approx(257.0, rel=0.03)


def test_fast_sequence_measurement_uses_sep_subpixel_moments() -> None:
    """序列快速测量应保留可靠的亚像素中心和近似星点尺寸。"""

    image = _gaussian_scene([(40.25, 39.70, 100.0, 1.8)])
    candidates = detect_star_candidates_from_array(
        image,
        40.0,
        40.0,
        12,
        saturation_level=255.0,
    )

    fitted = measure_star_candidate_fast(candidates[0])

    assert fitted.x == pytest.approx(40.25, abs=0.08)
    assert fitted.y == pytest.approx(39.70, abs=0.08)
    assert fitted.fwhm_x == pytest.approx(4.24, abs=0.55)
    assert fitted.fwhm_y == pytest.approx(4.24, abs=0.55)
    assert 0.0 <= fitted.quality_score <= 1.0


def test_sequence_grayscale_array_owns_its_memory() -> None:
    """整帧灰度数组不得引用函数返回后已销毁的临时 QImage。"""

    image = QImage(8, 4, QImage.Format_RGB888)
    image.fill(QColor(255, 0, 0))

    luminance = qimage_to_grayscale_array(image)

    assert luminance.shape == (4, 8)
    assert luminance.dtype == np.uint8
    assert luminance.flags.owndata
    assert np.all(luminance > 0)


def test_landscape_edge_is_rejected_as_non_stellar() -> None:
    """带纹理的地景亮边不能通过恒星径向轮廓检查。"""

    yy, xx = np.indices((81, 81), dtype=np.float64)
    image = 45.0 + 0.35 * xx + 8.0 * np.sin(xx * 0.33) * np.cos(yy * 0.19)
    image += np.where(xx > 41.0 + 4.0 * np.sin(yy * 0.12), 55.0, 0.0)

    with pytest.raises(StarFitError):
        fit_star_position_from_array(image, 40.0, 40.0, 18, saturation_level=255.0)


def _candidate(x: float, y: float, quality: float = 10.0) -> StarSourceCandidate:
    return StarSourceCandidate(
        x=x,
        y=y,
        major_axis=1.8,
        minor_axis=1.6,
        theta_rad=0.0,
        flux=100.0,
        peak=40.0,
        snr=quality,
        npix=12,
        label=1,
        quality_score=quality,
    )


def test_catalog_assignment_is_globally_one_to_one() -> None:
    """密集星场中一颗检测星源不能同时分给两颗星表星。"""

    result = assign_predicted_sources(
        {"A": (10.0, 10.0), "B": (13.0, 10.0), "C": (30.0, 30.0)},
        [_candidate(10.4, 10.0), _candidate(13.1, 10.0)],
        search_radius_px=6.0,
    )

    assert result.assignments["A"].x == pytest.approx(10.4)
    assert result.assignments["B"].x == pytest.approx(13.1)
    assert "C" not in result.assignments
    assert len({id(source) for source in result.assignments.values()}) == len(result.assignments)


def test_manual_pick_rejects_masked_landscape_before_psf(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """手动点在蒙版外时不得进入底层 PSF 检测，也不得显示弹窗。"""

    class _View:
        def mapToScene(self, _position) -> QPointF:  # type: ignore[no-untyped-def]
            return QPointF(8.0, 9.0)

    class _StatusBar:
        message = ""

        def showMessage(self, message: str) -> None:
            self.message = message

    harness = AutoMatchMixin()
    harness._active_star_pair_row = 0
    harness.current_image_preview = type("Preview", (), {"image": QImage(20, 20, QImage.Format_RGB888)})()
    harness.ui = type("Ui", (), {"realImageView": _View(), "statusbar": _StatusBar()})()
    harness._sky_mask_allows_point = lambda _x, _y: False  # type: ignore[attr-defined]
    monkeypatch.setattr(
        app_auto_match.QMessageBox,
        "warning",
        lambda *_args: pytest.fail("手动匹配失败时不应显示警告弹窗"),
    )
    monkeypatch.setattr(
        app_auto_match.QMessageBox,
        "information",
        lambda *_args: pytest.fail("手动匹配失败时不应显示信息弹窗"),
    )
    monkeypatch.setattr(
        app_auto_match,
        "fit_star_position",
        lambda *_args, **_kwargs: pytest.fail("蒙版外点击不应调用 PSF 拟合"),
    )

    harness._handle_real_image_pick_click(object())

    assert harness.ui.statusbar.message == "手动匹配失败：点击位置位于天空蒙版外，已拒绝把地景纹理作为星点。"
