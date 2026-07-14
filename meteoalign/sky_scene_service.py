from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PyQt5.QtGui import QImage

from .renderer import StarMapRenderer
from .simulator import (
    CameraSettings,
    HorizontalConstellationCatalog,
    HorizontalMilkyWayCatalog,
    HorizontalSolarSystemCatalog,
    HorizontalStarCatalog,
    ReferenceStar,
    ViewSettings,
    project_horizontal_catalog,
)


@dataclass(frozen=True)
class SkySceneData:
    """已经转换到地平坐标系的天空场景数据。"""

    horizontal_catalog: HorizontalStarCatalog
    horizontal_milky_way: HorizontalMilkyWayCatalog | None = None
    horizontal_constellations: HorizontalConstellationCatalog | None = None
    horizontal_solar_system: HorizontalSolarSystemCatalog | None = None


@dataclass(frozen=True)
class SkyPreviewStyle:
    """天空预览渲染选项，避免各页面重复拼 renderer 参数。"""

    reference_stars: tuple[ReferenceStar, ...] = ()
    element_scale: float = 1.0
    draw_common_names: bool = False
    number_reference_stars: bool = False
    draw_background: bool = True
    draw_horizon_shadow: bool = True
    draw_grid: bool = True
    draw_solar_system_labels: bool = True
    draw_direction_labels: bool = True
    force_opaque: bool = False
    clear_grid_lines: bool = False
    clear_direction_labels: bool = False
    clear_horizon_shadow: bool = False


class SkyPreviewRenderService:
    """天空投影与 StarMapRenderer 调用的公共服务。"""

    def __init__(self, renderer: StarMapRenderer) -> None:
        self.renderer = renderer

    def render(
        self,
        *,
        scene: SkySceneData,
        camera: CameraSettings,
        view: ViewSettings,
        visible_mag_limit: float,
        style: SkyPreviewStyle | None = None,
    ) -> QImage:
        render_style = style or SkyPreviewStyle()
        star_map = project_horizontal_catalog(
            horizontal_catalog=scene.horizontal_catalog,
            camera=camera,
            view=view,
            visible_mag_limit=visible_mag_limit,
            horizontal_milky_way=scene.horizontal_milky_way,
            horizontal_constellations=scene.horizontal_constellations,
            horizontal_solar_system=scene.horizontal_solar_system,
            star_color_mag_limit=self.renderer.ui_config.star_color_mag_limit,
        )
        if (
            render_style.force_opaque
            or render_style.clear_grid_lines
            or render_style.clear_direction_labels
            or render_style.clear_horizon_shadow
        ):
            star_map = replace(
                star_map,
                alpha=(
                    np.full_like(star_map.alpha, 255, dtype=np.uint8)
                    if render_style.force_opaque
                    else star_map.alpha
                ),
                grid_lines=() if render_style.clear_grid_lines else star_map.grid_lines,
                direction_labels=() if render_style.clear_direction_labels else star_map.direction_labels,
                horizon_shadow_rects=() if render_style.clear_horizon_shadow else star_map.horizon_shadow_rects,
                solar_system_objects=(
                    tuple(replace(item, alpha=255) for item in star_map.solar_system_objects)
                    if render_style.force_opaque
                    else star_map.solar_system_objects
                ),
            )
        return self.renderer.render(
            star_map,
            reference_stars=render_style.reference_stars,
            element_scale=render_style.element_scale,
            draw_common_names=render_style.draw_common_names,
            number_reference_stars=render_style.number_reference_stars,
            draw_background=render_style.draw_background,
            draw_horizon_shadow=render_style.draw_horizon_shadow,
            draw_grid=render_style.draw_grid,
            draw_solar_system_labels=render_style.draw_solar_system_labels,
            draw_direction_labels=render_style.draw_direction_labels,
        )
