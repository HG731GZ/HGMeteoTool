"""自由投影拼图状态对象的单元测试。"""

from __future__ import annotations

import numpy as np

from meteoalign.mosaic.state import MosaicSessionState, MosaicSourceState, MosaicViewState


def test_mosaic_source_state_clears_only_rendered_overlay_cache() -> None:
    """清理渲染缓存不能影响可复用的覆盖范围缓存。"""

    coverage_cache = object()
    source = MosaicSourceState(
        source_model=object(),
        coverage_cache=coverage_cache,
        rendered_overlay_key=("视角",),
        rendered_overlay_rgba=np.zeros((2, 2, 4), dtype=np.uint8),
    )

    source.clear_render_cache()

    assert source.coverage_cache is coverage_cache
    assert source.rendered_overlay_key is None
    assert source.rendered_overlay_rgba is None


def test_mosaic_session_state_owns_sources_and_view_state() -> None:
    """会话替换源图时同步活动源，多模型状态与视图边界保持独立。"""

    first_source = MosaicSourceState(source_model=object())
    second_source = MosaicSourceState(source_model=object())
    session = MosaicSessionState()

    session.set_sources([first_source, second_source], multi_model_mode=True)
    session.view.set_output_boundary(-10, 320)

    assert session.active_source is first_source
    assert session.active_source_index == 0
    assert session.multi_model_mode
    assert session.view.output_boundary_width_px == 0
    assert session.view.output_boundary_height_px == 320

    session.set_sources([], multi_model_mode=False)

    assert session.active_source is None
    assert session.active_source_index is None
    assert not session.multi_model_mode


def test_mosaic_session_state_clears_all_source_render_caches() -> None:
    """会话级缓存清理应转发给每一个源图。"""

    sources = [
        MosaicSourceState(source_model=object(), rendered_overlay_key=("第一个",)),
        MosaicSourceState(source_model=object(), rendered_overlay_key=("第二个",)),
    ]
    session = MosaicSessionState(sources=sources)

    session.clear_render_caches()

    assert all(source.rendered_overlay_key is None for source in sources)
