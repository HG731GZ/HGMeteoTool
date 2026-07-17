from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsScene, QWidget

from meteoalign.application.app_auto_match import AutoMatchMixin
from meteoalign.application.app_constants import STAR_PAIR_SORT_KEY_QUALITY
from meteoalign.application.app_star_pair_annotations import StarPairAnnotationsMixin
from meteoalign.application.app_star_pair_table_groups import StarPairTableGroupsMixin
from meteoalign.config import StarMapUiConfig
from meteoalign.simulator import ReferenceStar
from meteoalign.star_fitting import FittedStarPosition


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    app = QApplication.instance() or QApplication([])
    _QT_APP = app
    return app


def _reference_star(star_id: str, index: int) -> ReferenceStar:
    return ReferenceStar(
        index=index,
        star_id=star_id,
        name=star_id,
        display_name=star_id,
        common_name=star_id,
        ra_deg=float(index),
        dec_deg=float(index),
        mag_v=1.0,
        sim_x=0.0,
        sim_y=0.0,
        alt_deg=45.0,
        az_deg=90.0,
    )


def _fail_matching_dialog(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
    """匹配失败路径出现弹窗时立即让测试失败。"""

    raise AssertionError("匹配失败时不应显示弹窗")


def test_right_click_auto_pair_search_radius_uses_independent_preferences() -> None:
    """单星自动匹配应使用独立 RMS 公式，而不依赖批量匹配搜索半径控件。"""

    host = SimpleNamespace(
        current_image_preview=SimpleNamespace(
            image=SimpleNamespace(width=lambda: 1000, height=lambda: 800)
        ),
        ui_config=StarMapUiConfig(
            auto_pair_search_rms_multiplier=2.0,
            auto_pair_search_base_radius_px=7,
            auto_pair_search_max_radius_px=24,
        ),
    )

    radius = AutoMatchMixin._auto_pair_search_radius_px(host, SimpleNamespace(rms_px=10.0))

    assert radius == 24


class _AnnotationHarness(QWidget, StarPairAnnotationsMixin):
    """提供真实图像场景和固定星号的最小标注宿主。"""

    def __init__(self) -> None:
        super().__init__()
        self.ui_config = StarMapUiConfig(star_pair_psf_outer_diameter_multiplier=1.5)
        self.real_image_scene = QGraphicsScene(self)
        self._star_pair_annotations = {}
        self._focused_star_annotations = []

    def _star_pair_star_id(self, _row: int) -> str:
        return "HR1"

    def _star_pair_label(self, _row: int) -> str:
        return "1. HR1"

    def _show_real_image_annotations(self) -> bool:
        return True


def test_psf_annotation_adds_green_outer_ellipse_at_configured_diameter() -> None:
    """绿色外圈应与黄色 FWHM 圈同心同角度，并按配置放大直径。"""

    _qapp()
    host = _AnnotationHarness()
    fitted = FittedStarPosition(
        x=50.0,
        y=60.0,
        amplitude=100.0,
        background=2.0,
        sigma_x=4.0,
        sigma_y=6.0,
        theta_rad=0.2,
        fwhm_x=20.0,
        fwhm_y=12.0,
        quality_score=0.86,
    )

    blue_ellipse = QGraphicsEllipseItem(0.0, 0.0, 10.0, 10.0)
    host.real_image_scene.addItem(blue_ellipse)
    host._focused_star_annotations.append(blue_ellipse)
    host._add_or_update_star_pair_annotation(
        0,
        fitted,
        preserve_focus_annotation=True,
    )

    yellow_ellipse, _label = host._star_pair_annotations["HR1"]
    outer_items = yellow_ellipse.childItems()
    assert len(outer_items) == 1
    green_ellipse = outer_items[0]
    assert yellow_ellipse.rect().width() == 20.0
    assert yellow_ellipse.rect().height() == 12.0
    assert green_ellipse.rect().width() == 30.0
    assert green_ellipse.rect().height() == 18.0
    assert green_ellipse.pen().color() == QColor(80, 230, 120)
    assert "质量 0.86" in green_ellipse.toolTip()
    assert blue_ellipse.scene() is host.real_image_scene
    assert host._focused_star_annotations == [blue_ellipse]
    host.close()


def test_focus_blue_circle_radius_is_twice_auto_pair_search_radius() -> None:
    """蓝圈半径必须严格等于自动匹配搜索半径的两倍。"""

    _qapp()
    host = _AnnotationHarness()
    marker_diameter_px = host._focus_marker_diameter_px(19.0)

    host._create_focus_annotation_items(
        host.real_image_scene,
        QPointF(40.0, 50.0),
        marker_diameter_px,
    )

    assert marker_diameter_px == 76.0
    assert len(host._focused_star_annotations) == 2
    assert all(item.rect().width() == 76.0 for item in host._focused_star_annotations)
    assert all(item.rect().height() == 76.0 for item in host._focused_star_annotations)
    host.close()


def test_double_click_link_runs_auto_pair_only_for_unmatched_row() -> None:
    """联动开关开启时，未匹配行双击应执行自动匹配。"""

    calls: list[int] = []
    host = SimpleNamespace(
        ui_config=StarMapUiConfig(double_click_focus_auto_pair_enabled=True),
        _is_star_pair_group_row=lambda _row: False,
        _star_pair_position_text=lambda _row: "",
        _auto_pair_star=calls.append,
        _focus_star_pair_theoretical_position=lambda _row: None,
    )

    StarPairTableGroupsMixin._handle_star_pair_cell_double_clicked(host, 4, 1)

    assert calls == [4]


def test_auto_pair_failure_keeps_blue_circle_and_uses_statusbar(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """自动匹配拟合失败时应保留搜索蓝圈，并且只在状态栏提示。"""

    _qapp()
    status_messages: list[str] = []
    focused: list[tuple[int, float, float, float]] = []
    image = SimpleNamespace(width=lambda: 400, height=lambda: 300)
    transform = SimpleNamespace(
        rms_px=5.0,
        transform_radec=lambda _ra, _dec: (120.0, 80.0),
    )
    host = SimpleNamespace(
        current_image_preview=SimpleNamespace(image=image),
        _sky_alignment_transform=transform,
        _sky_alignment_error_message="",
        _reference_alignment_error_message="",
        ui_config=StarMapUiConfig(
            auto_pair_search_rms_multiplier=2.0,
            auto_pair_search_base_radius_px=7,
            auto_pair_search_max_radius_px=120,
            star_pick_psf_fit_error_limit=0.73,
            star_pick_saturated_psf_fit_error_limit=0.88,
        ),
        ui=SimpleNamespace(
            statusbar=SimpleNamespace(showMessage=status_messages.append),
        ),
        _update_reference_alignment_transform=lambda: None,
        _reference_star_for_row=lambda _row: SimpleNamespace(ra_deg=10.0, dec_deg=20.0),
        _star_pair_label=lambda _row: "1. HR1",
        _focus_star_pair_image_point=lambda row, x, y, radius: focused.append((row, x, y, radius)),
        _auto_pair_search_radius_px=lambda current_transform: (
            AutoMatchMixin._auto_pair_search_radius_px(host, current_transform)
        ),
        _report_auto_pair_failure=lambda message, **kwargs: AutoMatchMixin._report_auto_pair_failure(
            host,
            message,
            **kwargs,
        ),
    )

    def fail_fit(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        assert _kwargs["fit_error_limit"] == 0.73
        assert _kwargs["saturated_fit_error_limit"] == 0.88
        raise ValueError("搜索范围内没有可靠星点")

    monkeypatch.setattr("meteoalign.application.app_auto_match.fit_star_position", fail_fit)

    result = AutoMatchMixin._auto_pair_star(host, 0)

    assert result is False
    assert focused == [(0, 120.0, 80.0, 17)]
    assert status_messages[-1] == "自动匹配失败：搜索范围内没有可靠星点"


def test_auto_pair_failure_reports_status_without_dialog(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """单星自动匹配失败应只写入状态栏。"""

    status_messages: list[str] = []
    host = SimpleNamespace(
        ui=SimpleNamespace(statusbar=SimpleNamespace(showMessage=status_messages.append)),
    )

    monkeypatch.setattr("meteoalign.application.app_auto_match.QMessageBox.warning", _fail_matching_dialog)
    monkeypatch.setattr("meteoalign.application.app_auto_match.QMessageBox.information", _fail_matching_dialog)
    result = AutoMatchMixin._report_auto_pair_failure(
        host,
        "搜索范围内没有可靠星点",
    )

    assert result is False
    assert status_messages[-1] == "自动匹配失败：搜索范围内没有可靠星点"


def test_manual_pair_failure_reports_status_without_dialog(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """手动匹配的 PSF 拟合失败应只写入状态栏。"""

    _qapp()
    status_messages: list[str] = []

    host = AutoMatchMixin()
    host._active_star_pair_row = 0
    host.current_image_preview = SimpleNamespace(
        image=QImage(100, 80, QImage.Format_RGB888)
    )
    host.ui = SimpleNamespace(
        realImageView=SimpleNamespace(mapToScene=lambda _position: QPointF(30.0, 20.0)),
        statusbar=SimpleNamespace(showMessage=status_messages.append),
    )
    host._sky_mask_allows_point = lambda _x, _y: True
    host._star_pick_search_radius_px = lambda _position: 12
    host._star_pick_psf_radius_px = lambda _position: 18
    host.ui_config = StarMapUiConfig(
        star_pick_psf_fit_error_limit=0.73,
        star_pick_saturated_psf_fit_error_limit=0.88,
    )

    def fail_fit(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        assert _kwargs["fit_error_limit"] == 0.73
        assert _kwargs["saturated_fit_error_limit"] == 0.88
        raise ValueError("搜索范围内没有可靠星点")

    monkeypatch.setattr("meteoalign.application.app_auto_match.fit_star_position", fail_fit)
    monkeypatch.setattr("meteoalign.application.app_auto_match.QMessageBox.warning", _fail_matching_dialog)
    monkeypatch.setattr("meteoalign.application.app_auto_match.QMessageBox.information", _fail_matching_dialog)

    host._handle_real_image_pick_click(object())

    assert status_messages[-1] == "手动匹配失败：PSF 拟合失败：搜索范围内没有可靠星点"


def test_auto_match_quality_sort_keeps_missing_values_last_in_both_directions() -> None:
    """自动匹配质量排序无论方向如何都应把缺少指标的星放到末尾。"""

    first = _reference_star("first", 1)
    second = _reference_star("second", 2)
    missing = _reference_star("missing", 3)
    entries = [
        (1, first, "A1", "auto_match", "A"),
        (2, second, "A2", "auto_match", "A"),
        (3, missing, "A3", "auto_match", "A"),
    ]
    saved_states = {
        "first": {"extra_fields": {"auto_match_quality_score": 0.25}},
        "second": {"extra_fields": {"auto_match_quality_score": 0.90}},
        "missing": {"extra_fields": {}},
    }
    host = SimpleNamespace(
        _star_pair_sort_key=STAR_PAIR_SORT_KEY_QUALITY,
        _star_pair_sort_descending=True,
    )

    descending = StarPairTableGroupsMixin._sort_star_pair_entries(host, entries, saved_states)
    host._star_pair_sort_descending = False
    ascending = StarPairTableGroupsMixin._sort_star_pair_entries(host, entries, saved_states)

    assert [entry[1].star_id for entry in descending] == ["second", "first", "missing"]
    assert [entry[1].star_id for entry in ascending] == ["first", "second", "missing"]
