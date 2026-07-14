"""图像序列 PSF 搜索半径独立设置测试。"""

from __future__ import annotations

from types import SimpleNamespace

from meteoalign.application.app_sequence_matching import SequenceMatchingMixin


class _ValueControl:
    def __init__(self, value: int) -> None:
        self._value = value

    def value(self) -> int:
        return self._value


def test_sequence_psf_search_radius_does_not_read_star_matching_control() -> None:
    """序列 PSF 半径必须只读取序列页控件。"""

    host = SimpleNamespace(
        ui=SimpleNamespace(
            spinBoxSequencePsfSearchRadius=_ValueControl(17),
            spinBoxAutoMatchRadius=_ValueControl(99),
        )
    )

    assert SequenceMatchingMixin._sequence_psf_search_radius_px(host) == 17
