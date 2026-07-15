from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication

import meteoalign.renderer as renderer_module
from meteoalign.application.meteor_shower_selection_dialog import MeteorShowerSelectionDialog
from meteoalign.config import StarMapUiConfig
from meteoalign.meteor_showers import (
    MAXIMUM_METEOR_COUNT_MULTIPLIER,
    METEOR_SHOWER_SPECS,
    ProjectedMeteor,
    ProjectedMeteorRadiant,
    _meteor_horizontal_pool_cached,
    _meteor_sky_pool_cached,
    projected_meteor_radiants,
    projected_meteors,
)
from meteoalign.renderer import StarMapRenderer
from meteoalign.simulator import (
    FISHEYE_EQUIDISTANT,
    CameraSettings,
    ObserverSettings,
    ViewSettings,
    compute_horizontal_catalog,
    project_horizontal_catalog,
)
from meteoalign.catalog import StarCatalog


def _all_sky_map():  # type: ignore[no-untyped-def]
    observer = ObserverSettings(
        observation_time_utc=datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc),
        latitude_deg=30.0,
        longitude_deg=110.0,
    )
    catalog = StarCatalog(
        source_name="流星渲染测试",
        star_ids=np.asarray(["test"], dtype=object),
        display_names=np.asarray(["test"], dtype=object),
        ra_deg=np.asarray([0.0], dtype=np.float64),
        dec_deg=np.asarray([0.0], dtype=np.float64),
        mag_v=np.asarray([1.0], dtype=np.float64),
        color_index_bv=np.asarray([0.5], dtype=np.float64),
        spectral_type=np.asarray(["G"], dtype=object),
        common_names=np.asarray([""], dtype=object),
    )
    horizontal = compute_horizontal_catalog(catalog, observer, 6.5)
    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=800,
        image_height_px=800,
        focal_length_mm=8.0,
        lens_model=FISHEYE_EQUIDISTANT,
        fisheye_fov_deg=180.0,
    )
    view = ViewSettings(center_az_deg=0.0, center_alt_deg=90.0)
    return project_horizontal_catalog(horizontal, camera, view)


def test_same_seed_reproduces_identical_projected_meteors() -> None:
    star_map = _all_sky_map()
    config = StarMapUiConfig(
        show_meteor_showers=True,
        selected_meteor_shower_ids=("SDA",),
        meteor_count_multiplier=0.5,
        meteor_random_seed=12345,
    )
    first = projected_meteors(star_map, config)
    second = projected_meteors(star_map, config)
    assert first
    assert first == second

    changed = projected_meteors(star_map, StarMapUiConfig(
        show_meteor_showers=True,
        selected_meteor_shower_ids=("SDA",),
        meteor_count_multiplier=0.5,
        meteor_random_seed=12346,
    ))
    assert changed
    assert first != changed


def test_ten_times_master_pool_and_lower_multiplier_use_stable_prefix() -> None:
    """主池固定生成 10×，提高倍率时既有流星的投影不能改变。"""

    assert MAXIMUM_METEOR_COUNT_MULTIPLIER == 10.0
    pool = _meteor_sky_pool_cached(("GEM",), 2.0, 35.0, 24680)
    assert len(pool) == 1500
    assert np.array_equal(pool.rank_in_shower, np.arange(1500, dtype=np.int32))

    star_map = _all_sky_map()
    one_times = projected_meteors(
        star_map,
        StarMapUiConfig(
            show_meteor_showers=True,
            selected_meteor_shower_ids=("GEM",),
            meteor_count_multiplier=1.0,
            meteor_random_seed=24680,
        ),
    )
    three_times = projected_meteors(
        star_map,
        StarMapUiConfig(
            show_meteor_showers=True,
            selected_meteor_shower_ids=("GEM",),
            meteor_count_multiplier=3.0,
            meteor_random_seed=24680,
        ),
    )
    assert one_times
    assert len(three_times) > len(one_times)
    assert three_times[: len(one_times)] == one_times


def test_view_changes_reuse_horizontal_pool_and_adaptive_sampling() -> None:
    """移动取景角度只重新投影；鱼眼轨迹使用少于49点的长度自适应采样。"""

    _meteor_horizontal_pool_cached.cache_clear()
    star_map = _all_sky_map()
    config = StarMapUiConfig(
        show_meteor_showers=True,
        selected_meteor_shower_ids=("GEM",),
        meteor_count_multiplier=1.0,
        meteor_random_seed=97531,
    )
    first = projected_meteors(star_map, config)
    moved_map = replace(star_map, view=ViewSettings(center_az_deg=30.0, center_alt_deg=75.0))
    second = projected_meteors(moved_map, config)
    cache_info = _meteor_horizontal_pool_cached.cache_info()

    assert first and second
    assert cache_info.misses == 1
    assert cache_info.hits >= 1
    point_counts = {len(meteor.points) for meteor in first + second}
    assert min(point_counts) >= 5
    assert max(point_counts) <= 25
    assert all((count - 1) % 4 == 0 for count in point_counts)


def test_zero_thickness_and_hidden_switch_are_valid_settings() -> None:
    star_map = _all_sky_map()
    hidden = StarMapUiConfig(show_meteor_showers=False, selected_meteor_shower_ids=("SDA",))
    assert projected_meteors(star_map, hidden) == ()

    line_config = StarMapUiConfig(
        show_meteor_showers=True,
        selected_meteor_shower_ids=("SDA",),
        meteor_count_multiplier=0.5,
        meteor_thickness_ratio=0.0,
    )
    assert projected_meteors(star_map, line_config)


def test_legacy_list_selection_is_normalized_before_cached_rendering() -> None:
    """旧会话传入列表时也必须在进入缓存前转换为可哈希元组。"""

    star_map = _all_sky_map()
    config = StarMapUiConfig(
        show_meteor_showers=True,
        selected_meteor_shower_ids=["SDA"],  # type: ignore[arg-type]
        meteor_count_multiplier=0.5,
    )
    assert projected_meteors(star_map, config)


def test_selection_dialog_lists_all_showers_and_defaults_to_unchecked() -> None:
    """选择窗口不按日期过滤，空默认值应让全年项目全部不勾选。"""

    app = QApplication.instance() or QApplication([])
    dialog = MeteorShowerSelectionDialog()
    assert dialog.ui.tableWidgetMeteorShowers.rowCount() == len(METEOR_SHOWER_SPECS)
    assert dialog.selected_ids() == ()
    dialog.close()
    app.processEvents()


def test_selected_visible_radiants_are_projected_into_the_image() -> None:
    """辐射点模式应只返回地平线以上且实际位于画面内的所选项目。"""

    star_map = _all_sky_map()
    selected_ids = tuple(spec.shower_id for spec in METEOR_SHOWER_SPECS)
    radiants = projected_meteor_radiants(
        star_map,
        StarMapUiConfig(show_meteor_showers=True, selected_meteor_shower_ids=selected_ids),
    )

    assert radiants
    assert {radiant.shower_id for radiant in radiants}.issubset(set(selected_ids))
    assert all(0.0 <= radiant.x_px <= star_map.width for radiant in radiants)
    assert all(0.0 <= radiant.y_px <= star_map.height for radiant in radiants)
    assert all(radiant.label for radiant in radiants)


def test_radiant_only_mode_draws_green_rice_symbol_without_meteor_tracks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """仅显示辐射点时必须跳过轨迹生成，并绘制绿色米字符号。"""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        renderer_module,
        "projected_meteors",
        lambda *_args: (_ for _ in ()).throw(AssertionError("辐射点模式不应生成流星轨迹")),
    )
    monkeypatch.setattr(
        renderer_module,
        "projected_meteor_radiants",
        lambda *_args: (ProjectedMeteorRadiant("TEST", "测试流星雨", 400.0, 400.0),),
    )
    renderer = StarMapRenderer(
        StarMapUiConfig(
            show_meteor_showers=True,
            meteor_radiant_only=True,
            meteor_radiant_label_font_size_pt=18,
            meteor_opacity=1.0,
            selected_meteor_shower_ids=("TEST",),
        )
    )
    image = renderer.render(
        _all_sky_map(),
        draw_background=False,
        draw_horizon_shadow=False,
        draw_grid=False,
        draw_common_names=False,
        draw_solar_system_labels=False,
        draw_direction_labels=False,
    )

    center_color = image.pixelColor(400, 400)
    assert center_color.alpha() > 0
    assert center_color.green() > center_color.red()
    assert center_color.green() > center_color.blue()
    app.processEvents()


def test_continuous_meteor_path_is_clipped_at_both_image_edges(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """跨出画面的连续梭形应在边缘截断，中心线上不能出现逐段空隙。"""

    app = QApplication.instance() or QApplication([])
    meteor = ProjectedMeteor(
        shower_id="TEST",
        color_hex="#FFFFFF",
        brightness=1.0,
        points=(
            (-80.0, 50.0, 0.10, True),
            (20.0, 50.0, 0.35, True),
            (100.0, 50.0, 0.75, True),
            (180.0, 50.0, 0.82, True),
            (280.0, 50.0, 0.90, True),
        ),
    )
    monkeypatch.setattr(renderer_module, "projected_meteors", lambda *_args: (meteor,))
    renderer = StarMapRenderer(
        StarMapUiConfig(
            show_meteor_showers=True,
            selected_meteor_shower_ids=("TEST",),
            meteor_thickness_ratio=0.04,
            meteor_opacity=1.0,
        )
    )
    image = QImage(200, 100, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    renderer._draw_meteor_showers(
        painter,
        SimpleNamespace(width=200, height=100, sky_circle_radius_px=None),  # type: ignore[arg-type]
    )
    painter.end()

    assert image.pixelColor(0, 50).alpha() > 0
    assert image.pixelColor(199, 50).alpha() > 0
    center_alphas = [image.pixelColor(x_value, 50).alpha() for x_value in range(5, 195)]
    assert min(center_alphas) > 0
    assert max(center_alphas) <= QColor(255, 255, 255, 255).alpha()
    app.processEvents()
