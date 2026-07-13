"""自由投影拼图渲染请求与结果的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtGui import QImage

from ..simulator import CameraSettings, ObserverSettings, ViewSettings
from ..sky_scene_service import SkyPreviewStyle, SkySceneData
from .state import MosaicSourceState


@dataclass(frozen=True)
class MosaicRenderRequest:
    """一次自由投影预览所需的全部显式输入。"""

    camera: CameraSettings
    view: ViewSettings
    observer: ObserverSettings | None
    scene: SkySceneData | None
    visible_mag_limit: float
    sky_style: SkyPreviewStyle
    sources: tuple[MosaicSourceState, ...] = ()
    overlay_enabled: bool = False
    overlay_mode: str = "coverage"
    overlay_opacity: float = 1.0
    interaction_active: bool = False
    source_texture_long_sides_px: tuple[int, ...] = ()
    meteor_only: bool = False


@dataclass(frozen=True)
class MosaicRenderResult:
    """渲染协调器输出的合成图、诊断信息和缓存键。"""

    image: QImage
    rendered_coverage_count: int
    rendered_texture_count: int
    diagnostics: tuple[str, ...]
    cache_keys: tuple[tuple[object, ...], ...]
