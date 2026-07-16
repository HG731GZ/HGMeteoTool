from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from meteoalign.application.app_auto_match import AutoMatchMixin
from meteoalign.simulator import ReferenceStar


def _star(star_id: str, mag_v: float) -> ReferenceStar:
    return ReferenceStar(
        index=0,
        star_id=star_id,
        name=star_id,
        display_name=star_id,
        common_name="",
        ra_deg=10.0,
        dec_deg=20.0,
        mag_v=mag_v,
        sim_x=0.0,
        sim_y=0.0,
        alt_deg=40.0,
        az_deg=180.0,
    )


class _ValueControl:
    def __init__(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value


class _StatusBar:
    """记录自动扩展状态文案。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.messages.append(message)


class _FakeStarMap:
    def __init__(self) -> None:
        self.ra_deg = np.asarray([10.0, 20.0, 30.0])
        self.dec_deg = np.asarray([10.0, 20.0, 30.0])
        self.alt_deg = np.asarray([40.0, 40.0, 40.0])
        self.mag_v = np.asarray([0.1, 4.0, 2.0])
        self.star_ids = np.asarray(["bright-crowded", "faint-empty", "mid-empty"])

    def __len__(self) -> int:
        return len(self.star_ids)


class _CandidateHarness(AutoMatchMixin):
    def __init__(self) -> None:
        self.ui = SimpleNamespace(
            doubleSpinBoxMagLimit=_ValueControl(8.0),
            spinBoxAutoMatchCount=_ValueControl(2),
        )
        self.current_image_preview = SimpleNamespace(
            image=SimpleNamespace(width=lambda: 1000, height=lambda: 800),
        )
        self.current_sky_mask = None
        self._mask_excluded_reference_star_ids: set[str] = set()
        self._fake_star_map = _FakeStarMap()

    def _auto_match_reference_star_map(self, _transform: object, _mag_limit: float) -> _FakeStarMap:
        return self._fake_star_map

    def _auto_match_blocked_reference_star_ids(self) -> set[str]:
        return set()

    def _existing_matched_positions(self) -> list[tuple[float, float]]:
        return [(850.0, 100.0), (870.0, 120.0), (890.0, 140.0)]

    def _reference_star_from_star_map_index(
        self,
        star_map: _FakeStarMap,
        star_index: int,
        output_index: int,
    ) -> ReferenceStar:
        return _star(str(star_map.star_ids[star_index]), float(star_map.mag_v[star_index]))


class _Transform:
    def transform_radec_points(self, _points: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [860.0, 180.0],
                [120.0, 120.0],
                [520.0, 620.0],
            ],
            dtype=np.float64,
        )


class _BatchThresholdHarness(AutoMatchMixin):
    """只验证批量自动扩展的四对门槛。"""

    def __init__(self, matched_count: int) -> None:
        self.current_image_preview = object()
        self._matched_count = matched_count

    def _star_pair_position_count(self) -> int:
        return self._matched_count


class _NoCandidateMaskHarness(_CandidateHarness):
    """验证候选全被蒙版预筛时仍更新底部状态栏。"""

    def __init__(self) -> None:
        super().__init__()
        self.current_sky_mask = np.ones((800, 1000), dtype=bool)
        self.ui.statusbar = _StatusBar()
        self._sky_alignment_transform = _Transform()
        self._sky_alignment_error_message = ""
        self._reference_alignment_error_message = ""

    def _star_pair_position_count(self) -> int:
        return 4

    def _sky_mask_allows_point(self, _x_px: float, _y_px: float) -> bool:
        return False


def test_auto_match_candidates_prefer_sparse_cells_over_brightness() -> None:
    harness = AutoMatchMixin()
    candidates = [
        _star("bright-crowded", 0.1),
        _star("faint-empty", 4.0),
        _star("mid-empty", 2.0),
    ]
    predicted_by_id = {
        "bright-crowded": (860.0, 180.0),
        "faint-empty": (120.0, 120.0),
        "mid-empty": (520.0, 620.0),
    }

    ordered = harness._auto_match_order_spatial_candidates(
        candidates,
        predicted_by_id=predicted_by_id,
        accepted_positions=[(850.0, 100.0), (870.0, 120.0), (890.0, 140.0)],
        target_size=(1000, 800),
    )

    assert [candidate.star_id for candidate in ordered[:2]] == ["faint-empty", "mid-empty"]


def test_auto_match_field_stars_stays_locked_before_four_pairs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """两点预配准不得放开批量“自动扩展匹配”。"""

    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )
    harness = _BatchThresholdHarness(matched_count=2)

    harness.auto_match_field_stars()

    assert messages == [("无法自动扩展匹配", "当前只有 2 对星点；自动扩展匹配至少需要 4 对。")]


def test_auto_match_candidate_limit_is_applied_after_grid_balancing() -> None:
    harness = _CandidateHarness()

    candidates, predicted_by_id, mask_prefiltered_count = harness._auto_match_candidate_stars(_Transform())

    assert [candidate.star_id for candidate in candidates] == ["faint-empty", "mid-empty"]
    assert set(predicted_by_id) == {"faint-empty", "mid-empty"}
    assert mask_prefiltered_count == 0


def test_auto_match_candidate_count_includes_mask_prefilter() -> None:
    """状态栏统计应包含进入 PSF 循环前被蒙版排除的视场星点。"""

    harness = _CandidateHarness()
    harness.current_sky_mask = np.ones((800, 1000), dtype=bool)
    harness._sky_mask_allows_point = lambda x_px, _y_px: x_px < 500.0  # type: ignore[method-assign]

    candidates, predicted_by_id, mask_prefiltered_count = harness._auto_match_candidate_stars(_Transform())

    assert [candidate.star_id for candidate in candidates] == ["faint-empty"]
    assert set(predicted_by_id) == {"faint-empty"}
    assert mask_prefiltered_count == 2


def test_no_candidate_status_reports_mask_prefilter(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """蒙版筛完全部候选时，不应继续显示误导性的“蒙版跳过 0”。"""

    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )
    harness = _NoCandidateMaskHarness()

    harness.auto_match_field_stars()

    assert harness.ui.statusbar.messages[-1] == "自动扩展匹配：没有可新增星；蒙版预筛 3 个视场星点。"
    assert messages[-1] == (
        "没有可新增星",
        "当前视场、数量设置和蒙版下没有新的可匹配参考星。\n蒙版预筛 3 个视场星点。",
    )


def test_auto_match_candidates_prefer_brighter_star_inside_same_cell() -> None:
    harness = AutoMatchMixin()
    candidates = [
        _star("faint", 4.0),
        _star("bright", 0.5),
    ]

    ordered = harness._auto_match_order_spatial_candidates(
        candidates,
        predicted_by_id={"faint": (120.0, 120.0), "bright": (140.0, 130.0)},
        accepted_positions=[],
        target_size=(1000, 800),
    )

    assert [candidate.star_id for candidate in ordered] == ["bright", "faint"]


def test_auto_match_candidate_rounds_cover_each_available_cell() -> None:
    harness = AutoMatchMixin()
    candidates = [
        _star("left-bright", 0.1),
        _star("left-faint", 3.0),
        _star("right-bright", 0.2),
        _star("right-faint", 4.0),
    ]

    ordered = harness._auto_match_order_spatial_candidates(
        candidates,
        predicted_by_id={
            "left-bright": (100.0, 100.0),
            "left-faint": (110.0, 110.0),
            "right-bright": (900.0, 100.0),
            "right-faint": (910.0, 110.0),
        },
        accepted_positions=[],
        target_size=(1000, 800),
    )

    assert [candidate.star_id for candidate in ordered] == [
        "left-bright",
        "right-bright",
        "left-faint",
        "right-faint",
    ]
