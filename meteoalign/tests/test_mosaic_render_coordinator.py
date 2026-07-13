"""自由投影拼图渲染协调器的单元测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt5.QtGui import QColor, QImage

from meteoalign.mosaic.render_coordinator import MosaicRenderCoordinator
from meteoalign.mosaic.render_types import MosaicRenderRequest
from meteoalign.mosaic.state import MosaicSourceState
from meteoalign.mosaic_model_io import MosaicCoverageCache, MosaicSourceTextureCache
from meteoalign.meteor_selection import MeteorBox
from meteoalign.simulator import CameraSettings, ObserverSettings, RECTILINEAR_LENS_MODEL, ViewSettings
from meteoalign.sky_scene_service import SkyPreviewStyle


class _SkyRenderer:
    """返回固定底图，用于验证协调器的编排输入。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def render(self, **kwargs) -> QImage:  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        image = QImage(4, 2, QImage.Format_ARGB32)
        image.fill(QColor(0, 0, 0))
        return image


class _TextureRenderer:
    """按源图首像素返回不透明单色覆盖层。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def render_rgba(self, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        source_rgb = np.asarray(kwargs["source_rgb"], dtype=np.uint8)
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = source_rgb[0, 0]
        rgba[:, :, 3] = 255
        if source_rgb[0, 0, 1] > 0:
            rgba[:, 1:, 3] = 0
        return rgba


def _camera() -> CameraSettings:
    return CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=18.0,
        image_width_px=4,
        image_height_px=2,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=90.0,
    )


def _observer() -> ObserverSettings:
    return ObserverSettings(
        observation_time_utc=datetime(2025, 12, 14, 18, 11, 45, tzinfo=timezone.utc),
        latitude_deg=25.0,
        longitude_deg=102.0,
        elevation_m=200.0,
    )


def _coverage_cache() -> MosaicCoverageCache:
    values = np.zeros((2, 2), dtype=np.float64)
    return MosaicCoverageCache(
        grid_rows=2,
        grid_columns=2,
        grid_x_px=values,
        grid_y_px=values,
        ra_deg=values,
        dec_deg=values,
        valid=np.zeros((2, 2), dtype=bool),
    )


def _source(name: str, color: tuple[int, int, int]) -> MosaicSourceState:
    image_path = Path(name).with_suffix(".tif").resolve()
    model = SimpleNamespace(
        json_path=Path(name),
        source_image_path=image_path,
        image_width_px=100,
        image_height_px=80,
    )
    source = MosaicSourceState(source_model=model, coverage_cache=_coverage_cache())  # type: ignore[arg-type]
    source.source_texture_cache = MosaicSourceTextureCache(
        source_image_path=model.source_image_path,
        source_rgb=np.asarray([[color]], dtype=np.uint8),
        source_scale_x=1.0,
        source_scale_y=1.0,
        source_width_px=1,
        source_height_px=1,
        texture_max_long_side_px=1,
    )
    return source


def test_coordinator_returns_black_image_without_observer_or_scene() -> None:
    """缺少观察者时协调器返回可显示的黑色占位图，不调用天空服务。"""

    sky_renderer = _SkyRenderer()
    coordinator = MosaicRenderCoordinator(sky_renderer)  # type: ignore[arg-type]
    request = MosaicRenderRequest(
        camera=_camera(),
        view=ViewSettings(center_az_deg=180.0, center_alt_deg=45.0, roll_deg=0.0),
        observer=None,
        scene=None,
        visible_mag_limit=6.5,
        sky_style=SkyPreviewStyle(),
    )

    result = coordinator.render(request)

    assert result.image.size().width() == 4
    assert result.image.size().height() == 2
    assert not sky_renderer.calls
    assert result.rendered_coverage_count == 0
    assert result.rendered_texture_count == 0


def test_coordinator_composites_source_layers_in_request_order() -> None:
    """后面的源图覆盖前面的源图，并把缓存键和计数写入结果。"""

    sky_renderer = _SkyRenderer()
    coordinator = MosaicRenderCoordinator(sky_renderer, _TextureRenderer())  # type: ignore[arg-type]
    first = _source("first.json", (255, 0, 0))
    second = _source("second.json", (0, 255, 0))
    request = MosaicRenderRequest(
        camera=_camera(),
        view=ViewSettings(center_az_deg=180.0, center_alt_deg=45.0, roll_deg=0.0),
        observer=_observer(),
        scene=object(),  # type: ignore[arg-type]
        visible_mag_limit=6.5,
        sky_style=SkyPreviewStyle(),
        sources=(first, second),
        overlay_enabled=True,
        overlay_mode="source_image",
        source_texture_long_sides_px=(1, 1),
    )

    result = coordinator.render(request)

    assert len(sky_renderer.calls) == 1
    assert result.rendered_texture_count == 2
    assert len(result.cache_keys) == 2
    assert result.image.pixelColor(0, 0).green() == 255
    assert result.image.pixelColor(1, 0).red() == 255


def test_coordinator_passes_meteor_regions_only_for_sources_with_selection() -> None:
    """勾选流星区域后，有 JSON 的源图传矩形，无 JSON 的源图继续传整图。"""

    texture_renderer = _TextureRenderer()
    coordinator = MosaicRenderCoordinator(_SkyRenderer(), texture_renderer)  # type: ignore[arg-type]
    selected = _source("selected.json", (255, 0, 0))
    selected.meteor_boxes = (MeteorBox(10.0, 20.0, 30.0, 40.0),)
    unselected = _source("unselected.json", (0, 255, 0))
    request = MosaicRenderRequest(
        camera=_camera(),
        view=ViewSettings(center_az_deg=180.0, center_alt_deg=45.0, roll_deg=0.0),
        observer=_observer(),
        scene=object(),  # type: ignore[arg-type]
        visible_mag_limit=6.5,
        sky_style=SkyPreviewStyle(),
        sources=(selected, unselected),
        overlay_enabled=True,
        overlay_mode="source_image",
        source_texture_long_sides_px=(1, 1),
        meteor_only=True,
    )

    coordinator.render(request)

    assert texture_renderer.calls[0]["source_pixel_regions"] == ((10, 20, 30, 40),)
    assert texture_renderer.calls[1]["source_pixel_regions"] is None
