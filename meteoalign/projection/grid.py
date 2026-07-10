from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.settings import CameraSettings, ObserverSettings, ViewSettings
from .camera_models import (
    CYLINDRICAL_LENS_MODELS,
    _camera_longitudes_from_altaz,
    _project_altaz_points,
    camera_basis_from_view,
)
from ..simulator import compute_altaz_from_radec


@dataclass(frozen=True)
class PixelGrid:
    """源图像上的规则像素网格。"""

    rows: int
    columns: int
    x_px: np.ndarray
    y_px: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.rows * self.columns)

    @property
    def points(self) -> np.ndarray:
        return np.column_stack((self.x_px.ravel(), self.y_px.ravel())).astype(np.float64)


@dataclass(frozen=True)
class PixelSkyGrid:
    """源图像像素网格对应的天球坐标网格。"""

    pixel_grid: PixelGrid
    first_deg: np.ndarray
    second_deg: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class ScreenGrid:
    """投影到当前预览画布后的屏幕网格。"""

    x_px: np.ndarray
    y_px: np.ndarray
    valid: np.ndarray
    screen_longitudes_rad: np.ndarray | None = None


def grid_shape_for_long_side(
    width: int,
    height: int,
    long_side_cells: int,
    *,
    min_minor_cells: int = 2,
) -> tuple[int, int]:
    """按图像长边点数计算保持宽高比的网格行列数。"""

    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    long_side = max(int(min_minor_cells), int(long_side_cells))
    min_minor = max(1, int(min_minor_cells))
    if safe_width >= safe_height:
        columns = long_side
        rows = max(min_minor, int(round(long_side * safe_height / float(safe_width))))
    else:
        rows = long_side
        columns = max(min_minor, int(round(long_side * safe_width / float(safe_height))))
    return rows, columns


def build_pixel_grid(width: int, height: int, rows: int, columns: int) -> PixelGrid:
    """构建覆盖源图像边界的规则像素网格。"""

    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    safe_rows = max(2, int(rows))
    safe_columns = max(2, int(columns))
    x_values = np.linspace(0.0, max(safe_width - 1, 0), safe_columns, dtype=np.float64)
    y_values = np.linspace(0.0, max(safe_height - 1, 0), safe_rows, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    return PixelGrid(
        rows=safe_rows,
        columns=safe_columns,
        x_px=grid_x.astype(np.float64),
        y_px=grid_y.astype(np.float64),
    )


def build_pixel_radec_grid(model, width: int, height: int, rows: int, columns: int) -> PixelSkyGrid:
    """把源图像像素网格反解为 RA/Dec 网格。"""

    pixel_grid = build_pixel_grid(width, height, rows, columns)
    radec = model.pixel_to_sky_points(pixel_grid.points)
    valid = np.all(np.isfinite(radec), axis=1)
    return PixelSkyGrid(
        pixel_grid=pixel_grid,
        first_deg=radec[:, 0].reshape((pixel_grid.rows, pixel_grid.columns)).astype(np.float64),
        second_deg=radec[:, 1].reshape((pixel_grid.rows, pixel_grid.columns)).astype(np.float64),
        valid=valid.reshape((pixel_grid.rows, pixel_grid.columns)).astype(bool),
    )


def radec_grid_to_altaz(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    valid: np.ndarray,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 RA/Dec 网格转换为当前观测者地平坐标网格。"""

    ra_grid = np.asarray(ra_deg, dtype=np.float64)
    dec_grid = np.asarray(dec_deg, dtype=np.float64)
    valid_grid = np.asarray(valid, dtype=bool)
    flat_ra = ra_grid.ravel()
    flat_dec = dec_grid.ravel()
    valid_input = valid_grid.ravel() & np.isfinite(flat_ra) & np.isfinite(flat_dec)
    alt_deg = np.full(flat_ra.shape, np.nan, dtype=np.float64)
    az_deg = np.full(flat_ra.shape, np.nan, dtype=np.float64)
    if np.any(valid_input):
        alt_values, az_values = compute_altaz_from_radec(flat_ra[valid_input], flat_dec[valid_input], observer)
        alt_deg[valid_input] = alt_values
        az_deg[valid_input] = az_values
    valid_altaz = valid_input & np.isfinite(alt_deg) & np.isfinite(az_deg)
    return (
        alt_deg.reshape(ra_grid.shape).astype(np.float64),
        az_deg.reshape(ra_grid.shape).astype(np.float64),
        valid_altaz.reshape(ra_grid.shape).astype(bool),
    )


def project_altaz_grid_to_screen(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    *,
    camera: CameraSettings,
    view: ViewSettings,
    valid: np.ndarray | None = None,
    include_cylindrical_longitudes: bool = True,
) -> ScreenGrid:
    """把地平坐标网格投影到当前相机画布。"""

    alt_grid = np.asarray(alt_deg, dtype=np.float64)
    az_grid = np.asarray(az_deg, dtype=np.float64)
    basis = camera_basis_from_view(view)
    x_px, y_px, valid_projection = _project_altaz_points(
        alt_grid.ravel(),
        az_grid.ravel(),
        camera=camera,
        basis=basis,
    )
    screen_x = x_px.reshape(alt_grid.shape).astype(np.float64)
    screen_y = y_px.reshape(alt_grid.shape).astype(np.float64)
    valid_screen = (
        valid_projection.reshape(alt_grid.shape)
        & np.isfinite(screen_x)
        & np.isfinite(screen_y)
    )
    if valid is not None:
        valid_screen &= np.asarray(valid, dtype=bool)

    screen_longitudes = None
    if include_cylindrical_longitudes and camera.lens_model in CYLINDRICAL_LENS_MODELS:
        screen_longitudes = _camera_longitudes_from_altaz(
            alt_grid.ravel(),
            az_grid.ravel(),
            basis,
        ).reshape(alt_grid.shape)

    return ScreenGrid(
        x_px=screen_x,
        y_px=screen_y,
        valid=valid_screen.astype(bool),
        screen_longitudes_rad=screen_longitudes,
    )
