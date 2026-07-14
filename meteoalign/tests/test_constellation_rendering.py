from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from meteoalign.config import StarMapUiConfig
from meteoalign.renderer import StarMapRenderer
from meteoalign.simulator import (
    CameraSettings,
    CYLINDRICAL_EQUIDISTANT_LENS_MODEL,
    HorizontalConstellation,
    HorizontalConstellationCatalog,
    HorizontalConstellationLine,
    HorizontalStarCatalog,
    ProjectedConstellation,
    ProjectedConstellationSegment,
    ViewSettings,
    project_horizontal_catalog,
)


_QT_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _QT_APP
    _QT_APP = QApplication.instance() or QApplication([])
    return _QT_APP


def _empty_star_catalog() -> HorizontalStarCatalog:
    empty_float = np.asarray([], dtype=np.float64)
    empty_object = np.asarray([], dtype=object)
    return HorizontalStarCatalog(
        source_name="测试星表",
        star_ids=empty_object,
        display_names=empty_object,
        ra_deg=empty_float,
        dec_deg=empty_float,
        mag_v=empty_float,
        color_index_bv=empty_float,
        spectral_type=empty_object,
        common_names=empty_object,
        alt_deg=empty_float,
        az_deg=empty_float,
    )


def _camera(lens_model: str = "rectilinear", fov_deg: float = 180.0) -> CameraSettings:
    return CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=400,
        image_height_px=240,
        focal_length_mm=20.0,
        lens_model=lens_model,
        fisheye_fov_deg=fov_deg,
    )


def test_constellation_projection_keeps_sequential_segments_and_chinese_label() -> None:
    horizontal_constellations = HorizontalConstellationCatalog(
        source_name="测试星座",
        constellations=(
            HorizontalConstellation(
                abbreviation="UMa",
                chinese_name="大熊座",
                lines=(
                    HorizontalConstellationLine(
                        hip_ids=(1, 2, 3),
                        alt_deg=np.asarray([40.0, 42.0, 44.0]),
                        az_deg=np.asarray([355.0, 0.0, 5.0]),
                        mag_v=np.asarray([2.0, 3.0, 4.0]),
                    ),
                ),
            ),
        ),
    )

    star_map = project_horizontal_catalog(
        _empty_star_catalog(),
        _camera(),
        ViewSettings(center_az_deg=0.0, center_alt_deg=42.0),
        visible_mag_limit=6.5,
        horizontal_constellations=horizontal_constellations,
    )

    assert len(star_map.constellations) == 1
    projected = star_map.constellations[0]
    assert projected.chinese_name == "大熊座"
    assert len(projected.segments) == 2
    assert projected.segments[0].end == projected.segments[1].start


def test_constellation_projection_drops_cylindrical_seam_segment() -> None:
    horizontal_constellations = HorizontalConstellationCatalog(
        source_name="测试星座",
        constellations=(
            HorizontalConstellation(
                abbreviation="Test",
                chinese_name="测试座",
                lines=(
                    HorizontalConstellationLine(
                        hip_ids=(1, 2),
                        alt_deg=np.asarray([0.0, 0.0]),
                        az_deg=np.asarray([179.0, 181.0]),
                        mag_v=np.asarray([2.0, 2.0]),
                    ),
                ),
            ),
        ),
    )

    star_map = project_horizontal_catalog(
        _empty_star_catalog(),
        _camera(CYLINDRICAL_EQUIDISTANT_LENS_MODEL, 360.0),
        ViewSettings(center_az_deg=0.0, center_alt_deg=0.0),
        visible_mag_limit=6.5,
        horizontal_constellations=horizontal_constellations,
    )

    assert star_map.constellations == ()


def test_constellation_renderer_leaves_gap_around_endpoint_star_face() -> None:
    _qapp()
    base_map = project_horizontal_catalog(
        _empty_star_catalog(),
        _camera(),
        ViewSettings(center_az_deg=0.0, center_alt_deg=45.0),
        visible_mag_limit=6.5,
    )
    constellation = ProjectedConstellation(
        abbreviation="Test",
        chinese_name="",
        segments=(
            ProjectedConstellationSegment(
                start=(20.0, 20.0),
                end=(80.0, 20.0),
                start_radius_px=5.0,
                end_radius_px=5.0,
            ),
        ),
        label_x_px=50.0,
        label_y_px=20.0,
    )
    star_map = replace(base_map, constellations=(constellation,))
    renderer = StarMapRenderer(
        StarMapUiConfig(
            constellation_line_width_px=1.0,
            constellation_line_color_hex="#E6E6E6",
            constellation_line_opacity=0.9,
        )
    )

    image = renderer.render(
        star_map,
        draw_background=False,
        draw_horizon_shadow=False,
        draw_grid=False,
        draw_direction_labels=False,
    )

    assert image.pixelColor(24, 20).alpha() == 0
    assert image.pixelColor(30, 20).alpha() > 0
