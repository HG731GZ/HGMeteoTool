"""自动匹配与序列匹配共享常量的兼容性测试。"""

from __future__ import annotations

from meteoalign.app_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT as facade_constraint_soft,
    AUTO_MATCH_MIN_ALTITUDE_DEG as facade_min_altitude_deg,
    AUTO_MATCH_SEARCH_MAG_LIMIT as facade_search_mag_limit,
)
from meteoalign.matching_constants import (
    AUTO_MATCH_CONSTRAINT_SOFT,
    AUTO_MATCH_MIN_ALTITUDE_DEG,
    AUTO_MATCH_SEARCH_MAG_LIMIT,
)


def test_app_constants_keep_matching_constant_compatibility_exports() -> None:
    """旧 UI 常量入口应继续导出与业务模块相同的值。"""

    assert facade_constraint_soft == AUTO_MATCH_CONSTRAINT_SOFT
    assert facade_min_altitude_deg == AUTO_MATCH_MIN_ALTITUDE_DEG
    assert facade_search_mag_limit == AUTO_MATCH_SEARCH_MAG_LIMIT
