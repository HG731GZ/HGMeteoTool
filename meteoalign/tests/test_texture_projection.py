import numpy as np
import pytest

from meteoalign.projected_texture_renderer import ProjectedTextureRenderer
from meteoalign.simulator import CameraSettings, FISHEYE_EQUIDISTANT, ViewSettings
from meteoalign.texture_projection import texture_projection_available, warp_grid_texture_to_rgba


@pytest.mark.skipif(not texture_projection_available(), reason="OpenCV 不可用，跳过纹理投影测试。")
def test_warp_grid_texture_to_rgba_maps_single_cell_with_opacity() -> None:
    source_rgb = np.full((2, 2, 3), (20, 80, 160), dtype=np.uint8)
    source_grid_x = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    source_grid_y = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    screen_x = np.asarray([[0.0, 3.0], [0.0, 3.0]], dtype=np.float64)
    screen_y = np.asarray([[0.0, 0.0], [3.0, 3.0]], dtype=np.float64)
    valid = np.ones((2, 2), dtype=bool)
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)

    assert warp_grid_texture_to_rgba(
        rgba,
        source_rgb=source_rgb,
        source_grid_x_px=source_grid_x,
        source_grid_y_px=source_grid_y,
        source_scale_x=1.0,
        source_scale_y=1.0,
        screen_x_px=screen_x,
        screen_y_px=screen_y,
        valid_points=valid,
        opacity=0.5,
    )

    assert tuple(rgba[1, 1, :3]) == (20, 80, 160)
    assert 120 <= int(rgba[1, 1, 3]) <= 128


@pytest.mark.skipif(not texture_projection_available(), reason="OpenCV 不可用，跳过纹理投影测试。")
def test_warp_grid_texture_only_keeps_selected_source_region() -> None:
    """流星区域模式应把源图矩形之外的预览像素保持为透明。"""

    source_rgb = np.full((8, 8, 3), (20, 80, 160), dtype=np.uint8)
    source_grid_x = np.asarray([[0.0, 7.0], [0.0, 7.0]], dtype=np.float64)
    source_grid_y = np.asarray([[0.0, 0.0], [7.0, 7.0]], dtype=np.float64)
    screen_x = np.asarray([[0.0, 7.0], [0.0, 7.0]], dtype=np.float64)
    screen_y = np.asarray([[0.0, 0.0], [7.0, 7.0]], dtype=np.float64)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)

    assert warp_grid_texture_to_rgba(
        rgba,
        source_rgb=source_rgb,
        source_grid_x_px=source_grid_x,
        source_grid_y_px=source_grid_y,
        source_scale_x=1.0,
        source_scale_y=1.0,
        screen_x_px=screen_x,
        screen_y_px=screen_y,
        valid_points=np.ones((2, 2), dtype=bool),
        source_pixel_regions=((2, 2, 6, 6),),
        seam_padding_px=0.0,
    )

    assert rgba[3, 3, 3] == 255
    assert rgba[0, 0, 3] == 0
    assert rgba[7, 7, 3] == 0


@pytest.mark.skipif(not texture_projection_available(), reason="OpenCV 不可用，跳过纹理投影测试。")
def test_warp_grid_texture_keeps_partial_cell_at_viewport_boundary() -> None:
    """网格仅有部分进入视场时，进入部分仍应贴图而非整格丢弃。"""

    source_rgb = np.full((2, 2, 3), (40, 120, 200), dtype=np.uint8)
    source_grid_x = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    source_grid_y = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    screen_x = np.asarray([[1.0, 6.0], [1.0, 6.0]], dtype=np.float64)
    screen_y = np.asarray([[1.0, 1.0], [6.0, 6.0]], dtype=np.float64)
    source_valid = np.ones((2, 2), dtype=bool)
    projection_valid = np.asarray([[True, False], [True, True]], dtype=bool)
    target_valid = np.zeros((8, 8), dtype=bool)
    target_valid[:, :4] = True
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)

    assert warp_grid_texture_to_rgba(
        rgba,
        source_rgb=source_rgb,
        source_grid_x_px=source_grid_x,
        source_grid_y_px=source_grid_y,
        source_scale_x=1.0,
        source_scale_y=1.0,
        screen_x_px=screen_x,
        screen_y_px=screen_y,
        valid_points=source_valid,
        projection_valid_points=projection_valid,
        target_valid_mask=target_valid,
    )

    assert tuple(rgba[3, 2, :3]) == (40, 120, 200)
    assert rgba[3, 2, 3] == 255
    assert not np.any(rgba[:, 4:, 3])


@pytest.mark.skipif(not texture_projection_available(), reason="OpenCV 不可用，跳过纹理投影测试。")
def test_projected_texture_keeps_grid_fragment_inside_fisheye_boundary() -> None:
    """鱼眼边缘单元应保留圆形视场内的纹理片段。"""

    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=36.0,
        image_width_px=20,
        image_height_px=20,
        focal_length_mm=8.0,
        lens_model=FISHEYE_EQUIDISTANT,
        fisheye_fov_deg=180.0,
    )
    renderer = ProjectedTextureRenderer()
    rgba = renderer.render_rgba(
        width=20,
        height=20,
        camera=camera,
        view=ViewSettings(center_az_deg=0.0, center_alt_deg=0.0, roll_deg=0.0),
        source_rgb=np.full((2, 2, 3), (80, 150, 220), dtype=np.uint8),
        source_grid_x_px=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        source_grid_y_px=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        source_scale_x=1.0,
        source_scale_y=1.0,
        alt_deg=np.asarray([[0.0, 0.0], [-45.0, -45.0]], dtype=np.float64),
        az_deg=np.asarray([[0.0, 110.0], [0.0, 110.0]], dtype=np.float64),
        valid_points=np.ones((2, 2), dtype=bool),
    )

    viewport_mask = renderer._target_viewport_mask(camera, 20, 20)
    assert viewport_mask is not None
    assert np.any(rgba[:, :, 3] > 0)
    assert not np.any(rgba[:, :, 3][~viewport_mask])
