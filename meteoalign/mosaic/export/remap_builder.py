"""目标像素到源图像素的重投影 map 构建。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from ...coordinates import unit_vectors_to_radec
from ...domain.settings import CameraSettings, ObserverSettings, ViewSettings
from .geometry import MosaicExportGeometry
from .target_transform import _icrs_camera_basis_from_view, target_image_points_to_icrs_vectors


@dataclass(frozen=True)
class MosaicReprojectionMap:
    """可传给重采样器的目标像素到源图像素映射。"""

    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, object]


def build_reprojection_map(
    *,
    source_model: object,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
    block_rows: int = 1024,
    progress_callback: Callable[[int], None] | None = None,
) -> MosaicReprojectionMap:
    """构建完整的裁剪输出像素到源图像素 map_x/map_y。"""

    output_height = int(geometry.output_height_px)
    blocks = list(
        iter_reprojection_map_blocks(
            source_model=source_model,
            camera=camera,
            view=view,
            observer=observer,
            geometry=geometry,
            block_rows=block_rows,
            progress_callback=progress_callback,
        )
    )
    if blocks:
        map_x = np.ascontiguousarray(np.vstack([block["map_x"] for block in blocks]), dtype=np.float32)
        map_y = np.ascontiguousarray(np.vstack([block["map_y"] for block in blocks]), dtype=np.float32)
    else:
        shape = (max(0, output_height), max(0, int(geometry.output_width_px)))
        map_x = np.full(shape, -1.0, dtype=np.float32)
        map_y = np.full(shape, -1.0, dtype=np.float32)
    valid_mask = np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= -0.5) & (map_y >= -0.5)
    return MosaicReprojectionMap(
        map_x=map_x,
        map_y=map_y,
        valid_mask=valid_mask.astype(bool),
        metadata={
            "pixel_convention": "0-based_pixel_center",
            "crop_left_px": int(geometry.crop_left_px),
            "crop_top_px": int(geometry.crop_top_px),
            "output_width_px": int(geometry.output_width_px),
            "output_height_px": int(geometry.output_height_px),
            "block_rows": max(1, int(block_rows)),
        },
    )


def iter_reprojection_map_blocks(
    *,
    source_model: object,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
    block_rows: int = 1024,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[dict[str, object]]:
    """逐块生成目标像素到源图像素的 map_x/map_y。"""

    output_width = int(geometry.output_width_px)
    output_height = int(geometry.output_height_px)
    safe_block_rows = max(1, int(block_rows))
    full_x = (geometry.crop_left_px + np.arange(output_width, dtype=np.float64)).astype(np.float64)
    basis = _icrs_camera_basis_from_view(view, observer)
    completed_rows = 0
    for row_start in range(0, output_height, safe_block_rows):
        rows = min(safe_block_rows, output_height - row_start)
        full_y = geometry.crop_top_px + row_start + np.arange(rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(full_x, full_y)
        map_x, map_y = _build_reprojection_map_block(
            source_model=source_model,
            x_px=grid_x.ravel(),
            y_px=grid_y.ravel(),
            rows=rows,
            columns=output_width,
            camera=camera,
            icrs_basis=basis,
        )
        completed_rows += rows
        if progress_callback is not None:
            progress_callback(completed_rows)
        yield {"row_start": int(row_start), "map_x": map_x, "map_y": map_y}


def _build_reprojection_map_block(
    *,
    source_model: object,
    x_px: np.ndarray,
    y_px: np.ndarray,
    rows: int,
    columns: int,
    camera: CameraSettings,
    icrs_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    icrs_vectors, valid_projection = target_image_points_to_icrs_vectors(
        x_px,
        y_px,
        camera=camera,
        icrs_basis=icrs_basis,
    )
    icrs_vectors[~valid_projection] = np.nan
    return _source_pixel_map_from_icrs_vectors(source_model, icrs_vectors, rows, columns)


def _source_pixel_map_from_icrs_vectors(
    source_model: object,
    icrs_vectors: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    vector_count = int(np.asarray(icrs_vectors).shape[0])
    map_x = np.full(vector_count, -1.0, dtype=np.float32)
    map_y = np.full(vector_count, -1.0, dtype=np.float32)
    source_pixels, valid = source_pixel_points_from_icrs_vectors(source_model, icrs_vectors)
    if np.any(valid):
        map_x[valid] = source_pixels[valid, 0].astype(np.float32)
        map_y[valid] = source_pixels[valid, 1].astype(np.float32)
    return map_x.reshape((rows, columns)), map_y.reshape((rows, columns))


def source_pixel_points_from_icrs_vectors(
    source_model: object,
    icrs_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将 ICRS 单位方向投影到源图，并返回有效像素掩码。"""

    vectors = np.asarray(icrs_vectors, dtype=np.float64)
    pixels = np.full((vectors.shape[0], 2), np.nan, dtype=np.float64)
    vector_valid = np.all(np.isfinite(vectors), axis=1)
    if not np.any(vector_valid):
        return pixels, np.zeros(vectors.shape[0], dtype=bool)
    projected = source_pixels_from_icrs_vectors(source_model, vectors[vector_valid])
    projected_valid = np.all(np.isfinite(projected), axis=1)
    source_width = getattr(source_model, "image_width_px", None)
    source_height = getattr(source_model, "image_height_px", None)
    if source_width is not None and source_height is not None:
        projected_valid &= (
            (projected[:, 0] >= -0.5)
            & (projected[:, 0] <= float(source_width) - 0.5)
            & (projected[:, 1] >= -0.5)
            & (projected[:, 1] <= float(source_height) - 0.5)
        )
    accepted = np.flatnonzero(vector_valid)[projected_valid]
    pixels[accepted] = projected[projected_valid]
    valid = np.zeros(vectors.shape[0], dtype=bool)
    valid[accepted] = True
    return pixels, valid


def source_pixels_from_icrs_vectors(source_model: object, icrs_vectors: np.ndarray) -> np.ndarray:
    """优先调用源模型的 ICRS 直接投影接口。"""

    direct_project = getattr(source_model, "icrs_vectors_to_pixel_points", None)
    if callable(direct_project):
        return np.asarray(direct_project(icrs_vectors), dtype=np.float64)
    radec = unit_vectors_to_radec(icrs_vectors)
    return np.asarray(source_model.sky_to_pixel_points(radec), dtype=np.float64)
