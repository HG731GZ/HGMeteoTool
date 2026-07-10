"""自由投影拼图预览的无 MainWindow 渲染编排。"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QColor, QImage, QPainter

from .model_io import MosaicCoverageCache, MosaicSourceTextureCache
from .overlay_renderer import coverage_altaz, load_source_texture, paint_coverage_overlay
from ..projected_texture_renderer import ProjectedTextureRenderer
from ..sky_scene_service import SkyPreviewRenderService
from ..texture_projection import rgba_array_to_qimage
from .render_types import MosaicRenderRequest, MosaicRenderResult
from .state import MosaicSourceState


MOSAIC_OVERLAY_MODE_SOURCE_IMAGE = "source_image"


class MosaicRenderCoordinator:
    """按“天空背景、覆盖层、源图贴图”的顺序组合自由投影预览。"""

    def __init__(
        self,
        sky_preview_renderer: SkyPreviewRenderService,
        texture_renderer: ProjectedTextureRenderer | None = None,
    ) -> None:
        self._sky_preview_renderer = sky_preview_renderer
        self._texture_renderer = texture_renderer or ProjectedTextureRenderer()

    def render(self, request: MosaicRenderRequest) -> MosaicRenderResult:
        """执行一次渲染，不读取 UI、不修改场景，也不显示任何对话框。"""

        width = max(1, int(request.camera.image_width_px))
        height = max(1, int(request.camera.image_height_px))
        if request.observer is None or request.scene is None:
            image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
            image.fill(QColor(0, 0, 0))
            return MosaicRenderResult(
                image=image,
                rendered_coverage_count=0,
                rendered_texture_count=0,
                diagnostics=(),
                cache_keys=(),
            )

        image = self._sky_preview_renderer.render(
            scene=request.scene,
            camera=request.camera,
            view=request.view,
            visible_mag_limit=float(request.visible_mag_limit),
            style=request.sky_style,
        )
        if not request.overlay_enabled or not request.sources:
            return MosaicRenderResult(
                image=image,
                rendered_coverage_count=0,
                rendered_texture_count=0,
                diagnostics=(),
                cache_keys=(),
            )

        if request.overlay_mode == MOSAIC_OVERLAY_MODE_SOURCE_IMAGE:
            return self._render_source_texture_overlays(image, request)
        return self._render_coverage_overlays(image, request)

    def _render_coverage_overlays(
        self,
        image: QImage,
        request: MosaicRenderRequest,
    ) -> MosaicRenderResult:
        assert request.observer is not None
        rendered_count = 0
        for source in request.sources:
            cache = self.render_coverage_cache(source, interaction_active=request.interaction_active)
            if cache is None:
                continue
            paint_coverage_overlay(
                image,
                cache,
                request.camera,
                request.view,
                request.observer,
                request.overlay_opacity,
            )
            rendered_count += 1
        return MosaicRenderResult(
            image=image,
            rendered_coverage_count=rendered_count,
            rendered_texture_count=0,
            diagnostics=(),
            cache_keys=(),
        )

    def _render_source_texture_overlays(
        self,
        image: QImage,
        request: MosaicRenderRequest,
    ) -> MosaicRenderResult:
        assert request.observer is not None
        width = int(image.width())
        height = int(image.height())
        combined_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        diagnostics: list[str] = []
        cache_keys: list[tuple[object, ...]] = []
        rendered_count = 0

        for index, source in enumerate(request.sources):
            cache = self.render_coverage_cache(source, interaction_active=request.interaction_active)
            if cache is None:
                continue
            texture_long_side = self._texture_long_side_for_source(request, index)
            texture = self._source_texture_for_item(
                source,
                interaction_active=request.interaction_active,
                texture_long_side=texture_long_side,
            )
            if texture is None:
                diagnostics.append(self._source_texture_error_message(source))
                continue
            cache_key = self._source_overlay_key(source, cache, texture, request, width, height)
            cache_keys.append(cache_key)
            if source.rendered_overlay_key == cache_key and source.rendered_overlay_rgba is not None:
                overlay_rgba = source.rendered_overlay_rgba
            else:
                alt_deg, az_deg, valid_points = coverage_altaz(cache, request.observer)
                overlay_rgba = self._texture_renderer.render_rgba(
                    width=width,
                    height=height,
                    camera=request.camera,
                    view=request.view,
                    source_rgb=texture.source_rgb,
                    source_grid_x_px=cache.grid_x_px,
                    source_grid_y_px=cache.grid_y_px,
                    source_scale_x=texture.source_scale_x,
                    source_scale_y=texture.source_scale_y,
                    alt_deg=alt_deg,
                    az_deg=az_deg,
                    valid_points=valid_points,
                    opacity=request.overlay_opacity,
                )
                source.rendered_overlay_key = cache_key
                source.rendered_overlay_rgba = overlay_rgba
            visible = overlay_rgba[:, :, 3] > 0
            if not np.any(visible):
                continue
            combined_rgba[visible] = overlay_rgba[visible]
            rendered_count += 1

        if rendered_count:
            painter = QPainter(image)
            painter.drawImage(0, 0, rgba_array_to_qimage(combined_rgba))
            painter.end()
        return MosaicRenderResult(
            image=image,
            rendered_coverage_count=0,
            rendered_texture_count=rendered_count,
            diagnostics=tuple(diagnostics),
            cache_keys=tuple(cache_keys),
        )

    @staticmethod
    def render_coverage_cache(
        source: MosaicSourceState,
        *,
        interaction_active: bool,
    ) -> MosaicCoverageCache | None:
        """按交互状态返回完整或降采样的覆盖网格缓存。"""

        cache = source.coverage_cache
        if cache is None or not interaction_active:
            return cache
        source_id = id(cache)
        if source.interaction_coverage_cache is None or source.interaction_coverage_source_id != source_id:
            source.interaction_coverage_cache = MosaicRenderCoordinator.reduced_coverage_cache(cache)
            source.interaction_coverage_source_id = source_id
        return source.interaction_coverage_cache

    @staticmethod
    def reduced_coverage_cache(cache: MosaicCoverageCache) -> MosaicCoverageCache:
        """从完整覆盖网格抽取约一半行列，同时保留四周边界。"""

        row_indices = MosaicRenderCoordinator._interaction_grid_indices(cache.grid_rows)
        column_indices = MosaicRenderCoordinator._interaction_grid_indices(cache.grid_columns)
        if len(row_indices) == cache.grid_rows and len(column_indices) == cache.grid_columns:
            return cache
        grid_index = np.ix_(row_indices, column_indices)
        return MosaicCoverageCache(
            grid_rows=int(len(row_indices)),
            grid_columns=int(len(column_indices)),
            grid_x_px=np.asarray(cache.grid_x_px[grid_index], dtype=np.float64),
            grid_y_px=np.asarray(cache.grid_y_px[grid_index], dtype=np.float64),
            ra_deg=np.asarray(cache.ra_deg[grid_index], dtype=np.float64),
            dec_deg=np.asarray(cache.dec_deg[grid_index], dtype=np.float64),
            valid=np.asarray(cache.valid[grid_index], dtype=bool),
        )

    @staticmethod
    def _interaction_grid_indices(length: int) -> np.ndarray:
        safe_length = max(1, int(length))
        if safe_length <= 3:
            return np.arange(safe_length, dtype=np.int64)
        indices = np.arange(0, safe_length, 2, dtype=np.int64)
        if int(indices[-1]) != safe_length - 1:
            indices = np.append(indices, safe_length - 1)
        return np.unique(indices).astype(np.int64)

    @staticmethod
    def _texture_long_side_for_source(request: MosaicRenderRequest, index: int) -> int | None:
        if 0 <= index < len(request.source_texture_long_sides_px):
            return max(1, int(request.source_texture_long_sides_px[index]))
        return None

    @staticmethod
    def _source_texture_for_item(
        source: MosaicSourceState,
        *,
        interaction_active: bool,
        texture_long_side: int | None,
    ) -> MosaicSourceTextureCache | None:
        existing_cache = (
            source.interaction_source_texture_cache or source.source_texture_cache
            if interaction_active
            else source.source_texture_cache
        )
        texture = load_source_texture(
            source.source_model,
            existing_cache,
            max_long_side_px=texture_long_side,
        )
        if texture is None:
            return None
        if interaction_active:
            source.interaction_source_texture_cache = texture
        else:
            source.source_texture_cache = texture
        return texture

    @staticmethod
    def _source_overlay_key(
        source: MosaicSourceState,
        cache: MosaicCoverageCache,
        texture: MosaicSourceTextureCache,
        request: MosaicRenderRequest,
        width: int,
        height: int,
    ) -> tuple[object, ...]:
        assert request.observer is not None
        return (
            width,
            height,
            str(request.camera.lens_model),
            round(float(request.camera.focal_length_mm), 8),
            round(float(request.camera.fisheye_fov_deg), 8),
            round(float(request.view.center_az_deg), 8),
            round(float(request.view.center_alt_deg), 8),
            round(float(request.view.roll_deg), 8),
            request.observer.observation_time_utc.astimezone(timezone.utc).isoformat(),
            round(float(request.observer.latitude_deg), 8),
            round(float(request.observer.longitude_deg), 8),
            round(float(request.observer.elevation_m), 4),
            round(float(request.overlay_opacity), 4),
            bool(request.interaction_active),
            id(cache),
            id(texture),
            str(source.source_model.json_path),
        )

    @staticmethod
    def _source_texture_error_message(source: MosaicSourceState) -> str:
        image_path = source.source_model.source_image_path
        if image_path is None:
            return "源图模型 JSON 未记录真实图像路径，无法显示原图。"
        try:
            resolved_path = Path(image_path).expanduser().resolve()
        except OSError:
            resolved_path = Path(image_path).expanduser()
        if not resolved_path.exists():
            return f"源图不存在，无法显示原图: {resolved_path}"
        return f"源图读取失败，无法显示原图: {resolved_path}"
