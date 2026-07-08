import numpy as np
import pytest

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
