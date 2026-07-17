"""图像序列 PSF 搜索半径独立设置测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from meteoalign.application.app_sequence_matching import SequenceMatchingMixin
from meteoalign.sequence_types import _SequenceCandidate, _SequenceFitPlan
from meteoalign.simulator import ReferenceStar


class _ValueControl:
    def __init__(self, value: int) -> None:
        self._value = value

    def value(self) -> int:
        return self._value


class _SequenceFastHost(SequenceMatchingMixin):
    """只提供快速匹配测试所需的天空蒙版接口。"""

    @staticmethod
    def _sky_mask_allows_point(_x: float, _y: float) -> bool:
        return True


def test_sequence_psf_search_radius_does_not_read_star_matching_control() -> None:
    """序列 PSF 半径必须只读取序列页控件。"""

    host = SimpleNamespace(
        ui=SimpleNamespace(
            spinBoxSequencePsfSearchRadius=_ValueControl(17),
            spinBoxAutoMatchRadius=_ValueControl(99),
        )
    )

    assert SequenceMatchingMixin._sequence_psf_search_radius_px(host) == 17


def test_sequence_fast_candidate_path_uses_luminance_bounds() -> None:
    """序列快速匹配的检测和接收阶段都必须使用灰度数组尺寸。"""

    yy, xx = np.indices((48, 64), dtype=np.float64)
    luminance = 10.0 + 100.0 * np.exp(
        -0.5 * (((xx - 30.25) / 1.8) ** 2 + ((yy - 22.70) / 1.8) ** 2)
    )
    reference_star = ReferenceStar(
        index=0,
        star_id="test-star",
        name="测试星",
        display_name="测试星",
        common_name="",
        ra_deg=0.0,
        dec_deg=0.0,
        mag_v=1.0,
        sim_x=30.0,
        sim_y=23.0,
        alt_deg=45.0,
        az_deg=180.0,
    )
    plans = [
        _SequenceFitPlan(
            candidate=_SequenceCandidate(reference_star, 30.0, 23.0),
            fit_weight=1.0,
            pair_origin="test",
        )
    ]
    stats = {
        "failed_psf": 0,
        "skipped_mask": 0,
        "skipped_duplicate": 0,
        "skipped_outside": 0,
        "missing_target": 0,
    }

    matched = _SequenceFastHost()._fit_sequence_candidate_plans(
        luminance,
        plans,
        1,
        8,
        set(),
        set(),
        [],
        [],
        stats,
    )

    assert len(matched) == 1
    assert matched[0].fitted_position.x == pytest.approx(30.25, abs=0.08)
    assert matched[0].fitted_position.y == pytest.approx(22.70, abs=0.08)
