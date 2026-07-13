"""自由投影拼图会话的 UI 无关状态对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .framing import MosaicResolutionEstimate
    from .model_io import MosaicCoverageCache, MosaicSourceModel, MosaicSourceTextureCache
    from ..meteor_selection import MeteorBox
    from ..simulator import ObserverSettings


@dataclass
class MosaicSourceState:
    """一张拼图源图及其可重建缓存。

    缓存属于源图而非 MainWindow，视角变化时由调用方按需清除渲染缓存。
    """

    source_model: MosaicSourceModel
    coverage_cache: MosaicCoverageCache | None = None
    source_texture_cache: MosaicSourceTextureCache | None = None
    interaction_source_texture_cache: MosaicSourceTextureCache | None = None
    interaction_coverage_cache: MosaicCoverageCache | None = None
    interaction_coverage_source_id: int | None = None
    rendered_overlay_key: tuple[object, ...] | None = None
    rendered_overlay_rgba: np.ndarray | None = None
    meteor_boxes: tuple[MeteorBox, ...] = ()
    meteor_selection_path: Path | None = None
    meteor_selection_error: str = ""

    def clear_render_cache(self) -> None:
        """清除仅依赖当前视角的叠加层缓存。"""

        self.rendered_overlay_key = None
        self.rendered_overlay_rgba = None


@dataclass
class MosaicViewState:
    """自由投影的取景姿态与输出边界。"""

    center_az_deg: float = 0.0
    center_alt_deg: float = 20.0
    roll_deg: float = 0.0
    output_boundary_width_px: int = 0
    output_boundary_height_px: int = 0
    resolution_estimate: MosaicResolutionEstimate | None = None

    def set_output_boundary(self, width_px: int, height_px: int) -> None:
        """设置非负的完整输出边界尺寸。"""

        self.output_boundary_width_px = max(0, int(width_px))
        self.output_boundary_height_px = max(0, int(height_px))


@dataclass
class MosaicSessionState:
    """一个自由投影会话的唯一业务状态来源。"""

    sources: list[MosaicSourceState] = field(default_factory=list)
    active_source_index: int | None = None
    multi_model_mode: bool = False
    view: MosaicViewState = field(default_factory=MosaicViewState)
    model_observer: ObserverSettings | None = None
    model_utc_offset_hours: float = 0.0
    framing_observer: ObserverSettings | None = None
    framing_utc_offset_hours: float = 0.0
    framing_json_path: Path | None = None
    target_icrs_to_pixel_payload: dict[str, object] | None = None

    @property
    def active_source(self) -> MosaicSourceState | None:
        """返回当前活动源图；索引失效时安全地返回空。"""

        if self.active_source_index is None:
            return None
        if 0 <= self.active_source_index < len(self.sources):
            return self.sources[self.active_source_index]
        return None

    def set_sources(self, sources: list[MosaicSourceState], *, multi_model_mode: bool) -> None:
        """原子替换源图列表并同步活动源与多模型标记。"""

        self.sources = list(sources)
        self.multi_model_mode = bool(multi_model_mode)
        self.active_source_index = 0 if self.sources else None

    def clear_render_caches(self) -> None:
        """清除全部源图的视角相关渲染缓存。"""

        for source in self.sources:
            source.clear_render_cache()
