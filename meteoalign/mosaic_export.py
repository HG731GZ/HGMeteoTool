from __future__ import annotations

import copy
import json
import os
import tempfile
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image

from .coordinates import radec_to_unit_vectors, unit_vectors_to_radec
from .native_image import load_native_image_array
from .mosaic.export.geometry import MosaicExportGeometry, mosaic_export_cropped_geometry
from .mosaic.export.remap_builder import (
    MosaicReprojectionMap,
    build_reprojection_map,
    iter_reprojection_map_blocks,
    source_pixel_points_from_icrs_vectors as _source_pixel_points_from_icrs_vectors,
    source_pixels_from_icrs_vectors as _source_pixels_from_icrs_vectors,
)
from .mosaic.export.remap_repair import (
    MOSAIC_FORWARD_REMAP_LOW_WEIGHT_THRESHOLD,
    finalize_forward_inverse_map,
)
from .mosaic.export.target_transform import (
    MOSAIC_TARGET_ICRS_TO_PIXEL_VERSION,
    _icrs_camera_basis_from_view,
    _target_transform_basis,
    _target_transform_camera,
    build_target_icrs_to_pixel_transform_payload,
    target_camera_vectors_to_image_points,
    target_icrs_to_pixel_transform_payload_matches,
    target_icrs_vectors_to_output_pixel_points,
    target_image_points_to_camera_vectors,
    target_image_points_to_icrs_vectors,
)
from .simulator import (
    CameraSettings,
    ObserverSettings,
    ViewSettings,
)

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 是导出重采样依赖；兜底方便环境诊断。
    cv2 = None

try:
    import tifffile
except ImportError:  # pragma: no cover - tifffile 由环境声明，兜底用于给界面友好报错。
    tifffile = None

try:
    import imagecodecs
except ImportError:  # pragma: no cover - LZW 流式压缩依赖；环境诊断时保留友好错误。
    imagecodecs = None

try:
    import tifftools
except ImportError:  # pragma: no cover - tifftools 仅用于保留完整 EXIF IFD 树。
    tifftools = None


MOSAIC_EXPORT_TIFF_FILTER = "自适应位深 RGBA TIFF (*.tif *.tiff)"
MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS = 1024
MOSAIC_FORWARD_REMAP_EXACT_FILL_BATCH_PIXELS = 1_000_000
MosaicExportProgressCallback = Callable[[str, int, int], None]
SourcePixelRegion = tuple[int, int, int, int]

# 这些标签由导出图自身决定，继续复制原图值会造成尺寸、压缩或方向冲突。
_EXIF_EXCLUDED_TAGS = {
    256, 257, 258, 259, 262, 273, 274, 277, 278, 279, 282, 283, 284, 296,
    305, 317, 320, 322, 323, 324, 325, 338, 339, 33723, 34665, 34675, 34853,
    40962, 40963, 513, 514,
}

# 这些标签描述输出像素编码本身，不能从原图复用；其他元数据（含 EXIF/GPS 子 IFD）均应保留。
_TIFF_METADATA_COPY_EXCLUDED_TAGS = {
    256, 257, 258, 259, 262, 273, 274, 277, 278, 279, 284, 317, 320,
    322, 323, 324, 325, 338, 339, 40962, 40963, 513, 514,
}


@dataclass(frozen=True)
class MosaicExportSourceImage:
    """用于最终导出的全分辨率源图数据。"""

    path: Path
    rgb: np.ndarray
    exif_tags: tuple[tuple[int, str, int, object, bool], ...]
    icc_profile: bytes | None

    @property
    def bit_depth(self) -> int:
        return int(np.iinfo(self.rgb.dtype).bits)

    @property
    def width_px(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height_px(self) -> int:
        return int(self.rgb.shape[0])


def _emit_export_progress(
    callback: MosaicExportProgressCallback | None,
    label: str,
    value: int,
    maximum: int,
) -> None:
    if callback is not None:
        callback(label, int(value), int(maximum))


def mosaic_export_available() -> bool:
    return cv2 is not None and tifffile is not None


def mosaic_export_block_rows(config: object, default_value: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS) -> int:
    """从 UI 配置读取导出分块行数。"""

    configured = getattr(config, "mosaic_export_block_rows", default_value)
    try:
        value = int(configured)
    except (TypeError, ValueError):
        value = int(default_value)
    return max(8, min(4096, value))


def load_mosaic_export_source_image(image_path: str | Path) -> MosaicExportSourceImage:
    """读取源图全分辨率像素，保留 8 位或最高 16 位的 RGB 数组。"""

    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"源图不存在：{path}")
    path = path.resolve()

    if path.suffix.lower() in {".tif", ".tiff"}:
        # TIFF 的完整 IFD 树会在写入后复制，避免 Pillow 对超大全景触发像素数限制。
        exif_tags = ()
        icc_profile = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with Image.open(path) as image:
                exif = image.getexif()
                exif_tags = _safe_exif_tags(exif)
                icc_profile = image.info.get("icc_profile")

    rgb = _image_array_to_export_rgb(load_native_image_array(path))
    return MosaicExportSourceImage(
        path=path,
        rgb=np.ascontiguousarray(rgb),
        exif_tags=exif_tags,
        icc_profile=icc_profile if isinstance(icc_profile, bytes) else None,
    )


def _image_array_to_export_rgb(array: np.ndarray) -> np.ndarray:
    """统一通道数，并按源 dtype 选择 uint8 或 uint16 输出。"""

    values = np.asarray(array)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    elif values.ndim == 3 and values.shape[2] >= 3:
        values = values[:, :, :3]
    else:
        raise ValueError(f"源图像素形状不支持：{values.shape}")

    if values.dtype == np.uint8:
        return values.astype(np.uint8, copy=False)
    if values.dtype == np.uint16:
        return values.astype(np.uint16, copy=False)
    if values.dtype == np.bool_:
        return values.astype(np.uint8) * 255
    if np.issubdtype(values.dtype, np.integer):
        if values.dtype.itemsize <= 1:
            return np.clip(values, 0, 255).astype(np.uint8)
        return np.clip(values, 0, 65535).astype(np.uint16)
    if np.issubdtype(values.dtype, np.floating):
        finite = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.nanmax(finite)) <= 1.0:
            finite = finite * 65535.0
        return np.clip(finite, 0.0, 65535.0).astype(np.uint16)
    raise ValueError(f"源图像素类型不支持：{values.dtype}")


def _safe_exif_tags(exif: object) -> tuple[tuple[int, str, int, object, bool], ...]:
    tags: list[tuple[int, str, int, object, bool]] = []
    items = getattr(exif, "items", None)
    if not callable(items):
        return ()
    for tag, value in items():
        try:
            tag_code = int(tag)
        except (TypeError, ValueError):
            continue
        if tag_code in _EXIF_EXCLUDED_TAGS:
            continue
        converted = _exif_value_to_tiff_tag(tag_code, value)
        if converted is not None:
            tags.append(converted)
    return tuple(tags)


def _copy_full_tiff_exif_metadata(source_path: Path, output_path: Path) -> None:
    """把 TIFF 原图的完整 EXIF/GPS 子 IFD 树合并到已写出的 TIFF 中。"""

    if (
        tifftools is None
        or not source_path.exists()
        or source_path.suffix.lower() not in {".tif", ".tiff"}
    ):
        return
    try:
        source_info = tifftools.read_tiff(str(source_path))
        output_info = tifftools.read_tiff(str(output_path))
        source_ifds = source_info.get("ifds", [])
        output_ifds = output_info.get("ifds", [])
        if not source_ifds or not output_ifds:
            return

        source_tags = source_ifds[0].get("tags", {})
        output_tags = output_ifds[0].get("tags", {})
        copied_tag_count = 0
        for tag_code, tag_info in source_tags.items():
            if int(tag_code) in _TIFF_METADATA_COPY_EXCLUDED_TAGS:
                continue
            output_tags[int(tag_code)] = copy.deepcopy(tag_info)
            copied_tag_count += 1
        if copied_tag_count <= 0:
            return

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}_exif_",
            suffix=output_path.suffix,
            dir=str(output_path.parent),
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        try:
            tifftools.write_tiff(
                output_info,
                str(temporary_path),
                bigEndian=bool(output_info.get("bigEndian", False)),
                bigtiff=bool(output_info.get("bigtiff", False)),
            )
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    except Exception as exc:  # noqa: BLE001 - 元数据写入失败时不能静默产生不完整导出。
        raise RuntimeError(f"无法将原图完整 EXIF 写入导出 TIFF：{exc}") from exc


def _exif_value_to_tiff_tag(tag: int, value: object) -> tuple[int, str, int, object, bool] | None:
    if isinstance(value, str):
        text = _tiff_ascii(value.strip("\x00"))
        if not text:
            return None
        return (tag, "s", 0, text, True)
    if isinstance(value, bytes):
        if not value:
            return None
        return (tag, "B", len(value), value, True)
    if isinstance(value, int):
        dtype = "H" if 0 <= value <= 65535 else "I"
        return (tag, dtype, 1, int(value), True)

    rational = _rational_pair(value)
    if rational is not None:
        return (tag, "2I", 1, rational, True)

    if isinstance(value, (tuple, list)) and value:
        rational_values = [_rational_pair(item) for item in value]
        if all(item is not None for item in rational_values):
            return (tag, "2I", len(rational_values), tuple(rational_values), True)
        if all(isinstance(item, int) for item in value):
            dtype = "H" if all(0 <= int(item) <= 65535 for item in value) else "I"
            return (tag, dtype, len(value), tuple(int(item) for item in value), True)
    return None


def _rational_pair(value: object) -> tuple[int, int] | None:
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is None or denominator is None:
        return None
    try:
        num = int(numerator)
        den = int(denominator)
    except (TypeError, ValueError):
        return None
    if num < 0 or den <= 0:
        return None
    return num, den


def normalize_source_pixel_regions(
    regions: tuple[SourcePixelRegion, ...] | list[SourcePixelRegion],
    *,
    source_width_px: int,
    source_height_px: int,
) -> tuple[SourcePixelRegion, ...]:
    """将源图像素矩形规范为位于图像内的左上闭、右下开区域。"""

    width = max(0, int(source_width_px))
    height = max(0, int(source_height_px))
    normalized: list[SourcePixelRegion] = []
    for region in regions:
        if len(region) != 4:
            continue
        left, top, right, bottom = (int(value) for value in region)
        left, right = sorted((max(0, min(left, width)), max(0, min(right, width))))
        top, bottom = sorted((max(0, min(top, height)), max(0, min(bottom, height))))
        if right > left and bottom > top:
            normalized.append((left, top, right, bottom))
    return tuple(normalized)


def restrict_reprojection_map_to_source_regions(
    map_x: np.ndarray,
    map_y: np.ndarray,
    regions: tuple[SourcePixelRegion, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """将未落入任一源图区域的重投影采样点置为透明边界。"""

    allowed = np.zeros(map_x.shape, dtype=bool)
    for left, top, right, bottom in regions:
        allowed |= (
            (map_x >= float(left) - 0.5)
            & (map_x <= float(right) - 0.5)
            & (map_y >= float(top) - 0.5)
            & (map_y <= float(bottom) - 0.5)
        )
    restricted_x = np.asarray(map_x, dtype=np.float32).copy()
    restricted_y = np.asarray(map_y, dtype=np.float32).copy()
    restricted_x[~allowed] = -1.0
    restricted_y[~allowed] = -1.0
    return restricted_x, restricted_y


def mosaic_reprojection_blocks(
    *,
    source_model: object,
    source_rgb: np.ndarray,
    camera: CameraSettings | None,
    view: ViewSettings | None,
    observer: ObserverSettings | None,
    geometry: MosaicExportGeometry,
    block_rows: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS,
    progress_callback: Callable[[int], None] | None = None,
    export_progress_callback: MosaicExportProgressCallback | None = None,
    target_icrs_to_pixel_payload: dict[str, object] | None = None,
    target_model: object | None = None,
    map_tile_size_px: int = 4,
    exact_remap_repair: bool = False,
    source_pixel_regions: tuple[SourcePixelRegion, ...] | None = None,
) -> Iterator[np.ndarray]:
    """逐块生成与源图位深一致的 RGBA 像素。"""

    if cv2 is None:
        raise RuntimeError("当前环境缺少 OpenCV，无法执行重投影导出。")
    if target_icrs_to_pixel_payload is not None and target_model is not None:
        raise ValueError("目标 ICRS 变换和底图模型不能同时指定。")
    source = np.ascontiguousarray(source_rgb)
    if source.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError(f"重投影源图必须是 uint8 或 uint16，实际为 {source.dtype}。")
    normalized_regions = (
        None
        if source_pixel_regions is None
        else normalize_source_pixel_regions(
            source_pixel_regions,
            source_width_px=int(source.shape[1]),
            source_height_px=int(source.shape[0]),
        )
    )
    if normalized_regions is not None and not normalized_regions:
        safe_block_rows = max(1, int(block_rows))
        for row_start in range(0, int(geometry.output_height_px), safe_block_rows):
            rows = min(safe_block_rows, int(geometry.output_height_px) - row_start)
            yield np.zeros((rows, int(geometry.output_width_px), 4), dtype=source.dtype)
        return
    if normalized_regions is not None and target_icrs_to_pixel_payload is None and target_model is None:
        if camera is None or view is None or observer is None:
            raise ValueError("天球模式缺少目标相机、取景或观测者参数。")
        target_icrs_to_pixel_payload = build_target_icrs_to_pixel_transform_payload(
            camera=camera,
            view=view,
            observer=observer,
            geometry=geometry,
        )
    if target_icrs_to_pixel_payload is None and target_model is None:
        if camera is None or view is None or observer is None:
            raise ValueError("天球模式缺少目标相机、取景或观测者参数。")
        map_blocks = mosaic_reprojection_map_blocks(
            source_model=source_model,
            camera=camera,
            view=view,
            observer=observer,
            geometry=geometry,
            block_rows=block_rows,
            progress_callback=None,
        )
        completed_rows = 0
        for map_block in map_blocks:
            map_x = np.asarray(map_block["map_x"], dtype=np.float32)
            map_y = np.asarray(map_block["map_y"], dtype=np.float32)
            _emit_export_progress(export_progress_callback, "正在重采样源图...", 0, 0)
            block = _render_mosaic_reprojection_block_from_map(
                source_rgb=source,
                map_x=map_x,
                map_y=map_y,
            )
            completed_rows += int(map_x.shape[0])
            if progress_callback is not None:
                progress_callback(completed_rows)
            _emit_export_progress(
                export_progress_callback,
                "正在计算全景图到源图的坐标映射...",
                completed_rows,
                int(geometry.output_height_px),
            )
            _emit_export_progress(export_progress_callback, "正在写入 TIFF...", 0, 0)
            yield block
        return

    map_x, map_y = _build_mosaic_forward_remap_from_source_to_target(
        source_model=source_model,
        source_width_px=int(source.shape[1]),
        source_height_px=int(source.shape[0]),
        target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
        target_model=target_model,
        geometry=geometry,
        map_tile_size_px=map_tile_size_px,
        exact_remap_repair=exact_remap_repair,
        progress_callback=progress_callback,
        export_progress_callback=export_progress_callback,
        source_pixel_regions=normalized_regions,
    )
    safe_block_rows = max(1, int(block_rows))
    output_height = int(geometry.output_height_px)
    for row_start in range(0, output_height, safe_block_rows):
        row_end = min(output_height, row_start + safe_block_rows)
        block = _render_mosaic_reprojection_block_from_map(
            source_rgb=source,
            map_x=map_x[row_start:row_end],
            map_y=map_y[row_start:row_end],
        )
        _emit_export_progress(
            export_progress_callback,
            "正在重采样源图...",
            row_end,
            output_height,
        )
        yield block


def _build_mosaic_forward_remap_from_source_to_target(
    *,
    source_model: object,
    source_width_px: int,
    source_height_px: int,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry,
    map_tile_size_px: int,
    exact_remap_repair: bool,
    progress_callback: Callable[[int], None] | None = None,
    export_progress_callback: MosaicExportProgressCallback | None = None,
    source_pixel_regions: tuple[SourcePixelRegion, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按源图固定网格建立全景图到源图的近似 remap。"""

    output_width = int(geometry.output_width_px)
    output_height = int(geometry.output_height_px)
    source_width = int(source_width_px)
    source_height = int(source_height_px)
    tile_size = max(1, int(map_tile_size_px))
    accum_x = np.zeros((output_height, output_width), dtype=np.float32)
    accum_y = np.zeros((output_height, output_width), dtype=np.float32)
    weights = np.zeros((output_height, output_width), dtype=np.float32)
    if source_pixel_regions is not None:
        _accumulate_forward_source_regions(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            accum_x=accum_x,
            accum_y=accum_y,
            weights=weights,
            source_pixel_regions=source_pixel_regions,
            tile_size=tile_size,
            progress_callback=progress_callback,
            export_progress_callback=export_progress_callback,
        )
    elif tile_size <= 1:
        for row_start in range(0, source_height, max(1, int(MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS))):
            row_end = min(source_height, row_start + int(MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS))
            ys = np.arange(row_start, row_end, dtype=np.float64)
            xs = np.arange(source_width, dtype=np.float64)
            grid_x, grid_y = np.meshgrid(xs, ys)
            target_pixels, valid = _source_pixels_to_target_pixels(
                source_model,
                np.column_stack((grid_x.ravel(), grid_y.ravel())),
                target_icrs_to_pixel_payload,
                target_model=target_model,
                geometry=geometry,
            )
            _accumulate_source_pixels_to_inverse_map(
                accum_x,
                accum_y,
                weights,
                target_pixels,
                grid_x.ravel(),
                grid_y.ravel(),
                valid,
            )
            if progress_callback is not None:
                progress_callback(int(round(output_height * row_end / max(source_height, 1))))
            _emit_export_progress(
                export_progress_callback,
                "正在计算源图到全景图的坐标映射...",
                int(round(output_height * row_end / max(source_height, 1))),
                output_height,
            )
    else:
        for y0 in range(0, source_height, tile_size):
            y1 = min(source_height, y0 + tile_size)
            _accumulate_source_tile_row_to_inverse_map(
                source_model=source_model,
                target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
                target_model=target_model,
                geometry=geometry,
                accum_x=accum_x,
                accum_y=accum_y,
                weights=weights,
                y0=y0,
                y1=y1,
                tile_size=tile_size,
            )
            if progress_callback is not None:
                progress_callback(int(round(output_height * y1 / max(source_height, 1))))
            _emit_export_progress(
                export_progress_callback,
                "正在计算源图到全景图的坐标映射...",
                int(round(output_height * y1 / max(source_height, 1))),
                output_height,
            )
    if progress_callback is not None:
        progress_callback(output_height)
    _emit_export_progress(export_progress_callback, "正在整理全景图透明区域...", 0, 0)

    map_x, map_y = _finalize_forward_inverse_map(
        accum_x,
        accum_y,
        weights,
        tile_size,
        source_model=source_model,
        target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
        target_model=target_model,
        geometry=geometry,
        exact_remap_repair=exact_remap_repair,
        export_progress_callback=export_progress_callback,
    )
    if source_pixel_regions is not None:
        map_x, map_y = restrict_reprojection_map_to_source_regions(map_x, map_y, source_pixel_regions)
    return map_x, map_y


def _render_mosaic_forward_remap_from_source_to_target(
    *,
    source_model: object,
    source_rgb: np.ndarray,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry,
    map_tile_size_px: int,
    exact_remap_repair: bool,
    progress_callback: Callable[[int], None] | None = None,
    export_progress_callback: MosaicExportProgressCallback | None = None,
    source_pixel_regions: tuple[SourcePixelRegion, ...] | None = None,
) -> np.ndarray:
    """兼容旧调用：构建完整 remap 后一次性重采样整张图。"""

    map_x, map_y = _build_mosaic_forward_remap_from_source_to_target(
        source_model=source_model,
        source_width_px=int(source_rgb.shape[1]),
        source_height_px=int(source_rgb.shape[0]),
        target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
        target_model=target_model,
        geometry=geometry,
        map_tile_size_px=map_tile_size_px,
        exact_remap_repair=exact_remap_repair,
        progress_callback=progress_callback,
        export_progress_callback=export_progress_callback,
        source_pixel_regions=source_pixel_regions,
    )
    _emit_export_progress(export_progress_callback, "正在重采样源图...", 0, 0)
    return _render_mosaic_reprojection_block_from_map(
        source_rgb=source_rgb,
        map_x=map_x,
        map_y=map_y,
    )


def _accumulate_forward_source_regions(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry,
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    source_pixel_regions: tuple[SourcePixelRegion, ...],
    tile_size: int,
    progress_callback: Callable[[int], None] | None,
    export_progress_callback: MosaicExportProgressCallback | None,
) -> None:
    """只用选定源图区域建立正向 remap，减少无关区域的天球计算。"""

    total_rows = sum(bottom - top for _left, top, _right, bottom in source_pixel_regions)
    completed_rows = 0
    output_height = int(geometry.output_height_px)
    for left, top, right, bottom in source_pixel_regions:
        if tile_size <= 1:
            for row_start in range(top, bottom, max(1, int(MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS))):
                row_end = min(bottom, row_start + int(MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS))
                ys = np.arange(row_start, row_end, dtype=np.float64)
                xs = np.arange(left, right, dtype=np.float64)
                grid_x, grid_y = np.meshgrid(xs, ys)
                target_pixels, valid = _source_pixels_to_target_pixels(
                    source_model,
                    np.column_stack((grid_x.ravel(), grid_y.ravel())),
                    target_icrs_to_pixel_payload,
                    target_model=target_model,
                    geometry=geometry,
                )
                _accumulate_source_pixels_to_inverse_map(
                    accum_x,
                    accum_y,
                    weights,
                    target_pixels,
                    grid_x.ravel(),
                    grid_y.ravel(),
                    valid,
                )
                completed_rows += row_end - row_start
                _emit_region_remap_progress(
                    progress_callback,
                    export_progress_callback,
                    completed_rows,
                    total_rows,
                    output_height,
                )
            continue

        for y0 in range(top, bottom, tile_size):
            y1 = min(bottom, y0 + tile_size)
            _accumulate_source_region_tile_row(
                source_model=source_model,
                target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
                target_model=target_model,
                geometry=geometry,
                accum_x=accum_x,
                accum_y=accum_y,
                weights=weights,
                left=left,
                right=right,
                y0=y0,
                y1=y1,
                tile_size=tile_size,
            )
            completed_rows += y1 - y0
            _emit_region_remap_progress(
                progress_callback,
                export_progress_callback,
                completed_rows,
                total_rows,
                output_height,
            )


def _emit_region_remap_progress(
    progress_callback: Callable[[int], None] | None,
    export_progress_callback: MosaicExportProgressCallback | None,
    completed_rows: int,
    total_rows: int,
    output_height: int,
) -> None:
    progress_value = int(round(output_height * completed_rows / max(total_rows, 1)))
    if progress_callback is not None:
        progress_callback(progress_value)
    _emit_export_progress(
        export_progress_callback,
        "正在计算流星区域到全景图的坐标映射...",
        progress_value,
        output_height,
    )


def _accumulate_source_region_tile_row(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry,
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    left: int,
    right: int,
    y0: int,
    y1: int,
    tile_size: int,
) -> None:
    """累积单个流星框中一行固定尺寸网格。"""

    width = right - left
    full_tile_count = width // tile_size
    if full_tile_count:
        _accumulate_equal_width_source_tiles_to_inverse_map(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            accum_x=accum_x,
            accum_y=accum_y,
            weights=weights,
            x0_values=left + np.arange(full_tile_count, dtype=np.int64) * tile_size,
            y0=y0,
            y1=y1,
            tile_width=tile_size,
        )
    remainder = width - full_tile_count * tile_size
    if remainder:
        _accumulate_equal_width_source_tiles_to_inverse_map(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            accum_x=accum_x,
            accum_y=accum_y,
            weights=weights,
            x0_values=np.asarray([right - remainder], dtype=np.int64),
            y0=y0,
            y1=y1,
            tile_width=remainder,
        )


def _finalize_forward_inverse_map(
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    tile_size: int,
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None = None,
    geometry: MosaicExportGeometry | None = None,
    exact_remap_repair: bool,
    export_progress_callback: MosaicExportProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """兼容旧入口，空洞处理和快速修补已迁入 remap_repair。"""

    def exact_repair(map_x: np.ndarray, map_y: np.ndarray, exact_mask: np.ndarray) -> np.ndarray:
        return _fill_forward_inverse_map_exact(
            map_x,
            map_y,
            exact_mask,
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            progress_callback=lambda value, maximum: _emit_export_progress(
                export_progress_callback,
                "正在精确修复亮星小黑点...",
                value,
                maximum,
            ),
        )

    return finalize_forward_inverse_map(
        accum_x,
        accum_y,
        weights,
        tile_size,
        exact_remap_repair=exact_remap_repair,
        exact_repair=exact_repair,
        fast_progress_callback=lambda value, maximum: _emit_export_progress(
            export_progress_callback,
            "正在快速整理全景图透明区域...",
            value,
            maximum,
        ),
    )


def _fill_forward_inverse_map_exact(
    map_x: np.ndarray,
    map_y: np.ndarray,
    exact_mask: np.ndarray,
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry | None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """对缺失和低权重全景图像素精确反算源图坐标，避免亮星处坐标平均出黑点。"""

    exact_valid = np.zeros(exact_mask.shape, dtype=bool)
    if not np.any(exact_mask):
        return exact_valid

    y_indices, x_indices = np.nonzero(exact_mask)
    batch_size = int(MOSAIC_FORWARD_REMAP_EXACT_FILL_BATCH_PIXELS)
    total = int(x_indices.size)
    if progress_callback is not None:
        progress_callback(0, total)
    for start in range(0, int(x_indices.size), batch_size):
        end = min(int(x_indices.size), start + batch_size)
        x_batch = x_indices[start:end].astype(np.float64)
        y_batch = y_indices[start:end].astype(np.float64)
        source_pixels, valid = _source_pixel_points_from_target_output_pixels(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            output_x_px=x_batch,
            output_y_px=y_batch,
        )
        if np.any(valid):
            y_valid = y_indices[start:end][valid]
            x_valid = x_indices[start:end][valid]
            map_x[y_valid, x_valid] = source_pixels[valid, 0].astype(np.float32)
            map_y[y_valid, x_valid] = source_pixels[valid, 1].astype(np.float32)
            exact_valid[y_valid, x_valid] = True
        if progress_callback is not None:
            progress_callback(end, total)
    return exact_valid


def _source_pixel_points_from_target_output_pixels(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry | None,
    output_x_px: np.ndarray,
    output_y_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把裁剪后全景图像素精确反算到源图像素。"""

    crop_left_px, crop_top_px = _target_output_crop_offsets(target_icrs_to_pixel_payload, geometry)
    full_x = np.asarray(output_x_px, dtype=np.float64) + crop_left_px
    full_y = np.asarray(output_y_px, dtype=np.float64) + crop_top_px
    if target_model is not None:
        icrs_vectors, valid_projection = _model_pixel_points_to_icrs_vectors(
            target_model,
            np.column_stack((full_x, full_y)),
        )
    else:
        if target_icrs_to_pixel_payload is None:
            raise ValueError("缺少目标 ICRS 变换或底图模型。")
        icrs_vectors, valid_projection = target_image_points_to_icrs_vectors(
            full_x,
            full_y,
            camera=_target_transform_camera(target_icrs_to_pixel_payload),
            icrs_basis=_target_transform_basis(target_icrs_to_pixel_payload),
        )
    icrs_vectors[~valid_projection] = np.nan
    source_pixels, valid_source = _source_pixel_points_from_icrs_vectors(source_model, icrs_vectors)
    return source_pixels, valid_projection & valid_source


def _accumulate_source_tile_row_to_inverse_map(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry,
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    y0: int,
    y1: int,
    tile_size: int,
) -> None:
    source_width = int(getattr(source_model, "image_width_px", 0) or 0)
    if source_width <= 0:
        raise ValueError("源图模型缺少有效 image_width_px。")
    tile_height = int(y1 - y0)
    full_tile_count = source_width // tile_size
    if full_tile_count > 0:
        _accumulate_equal_width_source_tiles_to_inverse_map(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            accum_x=accum_x,
            accum_y=accum_y,
            weights=weights,
            x0_values=np.arange(full_tile_count, dtype=np.int64) * int(tile_size),
            y0=y0,
            y1=y1,
            tile_width=int(tile_size),
        )
    if full_tile_count * tile_size < source_width:
        x0 = int(full_tile_count * tile_size)
        _accumulate_equal_width_source_tiles_to_inverse_map(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            accum_x=accum_x,
            accum_y=accum_y,
            weights=weights,
            x0_values=np.asarray([x0], dtype=np.int64),
            y0=y0,
            y1=y1,
            tile_width=source_width - x0,
        )


def _accumulate_equal_width_source_tiles_to_inverse_map(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry,
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    x0_values: np.ndarray,
    y0: int,
    y1: int,
    tile_width: int,
) -> None:
    tile_count = int(x0_values.size)
    tile_height = int(y1 - y0)
    if tile_count <= 0 or tile_height <= 0:
        return
    if tile_width <= 1 or tile_height <= 1:
        exact_pixels = _source_tile_row_pixels_to_target_pixels_exact(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            target_model=target_model,
            geometry=geometry,
            x0_values=x0_values,
            y0=y0,
            width=tile_width,
            height=tile_height,
        )
        source_coords = _source_tile_coordinate_blocks(x0_values, y0, y1, tile_width).reshape((-1, 2))
        _accumulate_source_pixels_to_inverse_map(
            accum_x,
            accum_y,
            weights,
            exact_pixels[0],
            source_coords[:, 0],
            source_coords[:, 1],
            exact_pixels[1],
        )
        return

    x_last_values = x0_values + tile_width - 1
    y_last = int(y1 - 1)
    center_x_values = x0_values.astype(np.float64) + (tile_width - 1) * 0.5
    center_y = float(y0) + (tile_height - 1) * 0.5
    sample_points = np.empty((tile_count, 5, 2), dtype=np.float64)
    sample_points[:, 0, :] = np.column_stack((x0_values, np.full(tile_count, y0)))
    sample_points[:, 1, :] = np.column_stack((x_last_values, np.full(tile_count, y0)))
    sample_points[:, 2, :] = np.column_stack((x0_values, np.full(tile_count, y_last)))
    sample_points[:, 3, :] = np.column_stack((x_last_values, np.full(tile_count, y_last)))
    sample_points[:, 4, :] = np.column_stack((center_x_values, np.full(tile_count, center_y)))
    sample_target, sample_valid = _source_pixels_to_target_pixels(
        source_model,
        sample_points.reshape((-1, 2)),
        target_icrs_to_pixel_payload,
        target_model=target_model,
        geometry=geometry,
    )
    sample_target = sample_target.reshape((tile_count, 5, 2))
    sample_valid = sample_valid.reshape((tile_count, 5))
    valid_tiles = np.all(sample_valid, axis=1)

    if np.any(valid_tiles):
        local_x = np.arange(tile_width, dtype=np.float64)
        local_y = np.arange(tile_height, dtype=np.float64)
        u = local_x / max(float(tile_width - 1), 1.0)
        v = local_y / max(float(tile_height - 1), 1.0)
        corners = sample_target[valid_tiles, :4, :]
        center = sample_target[valid_tiles, 4, :]
        top = corners[:, 0, None, :] * (1.0 - u[None, :, None]) + corners[:, 1, None, :] * u[None, :, None]
        bottom = corners[:, 2, None, :] * (1.0 - u[None, :, None]) + corners[:, 3, None, :] * u[None, :, None]
        target_pixels = top[:, None, :, :] * (1.0 - v[None, :, None, None]) + bottom[:, None, :, :] * v[None, :, None, None]
        center_pred = (corners[:, 0, :] + corners[:, 1, :] + corners[:, 2, :] + corners[:, 3, :]) * 0.25
        residual = center - center_pred
        bump = 16.0 * u[None, None, :, None] * (1.0 - u[None, None, :, None]) * v[None, :, None, None] * (
            1.0 - v[None, :, None, None]
        )
        target_pixels += residual[:, None, None, :] * bump
        tile_indices = np.flatnonzero(valid_tiles)
        source_coords = _source_tile_coordinate_blocks(x0_values, y0, y1, tile_width)[tile_indices].reshape(
            (-1, 2)
        )
        _accumulate_source_pixels_to_inverse_map(
            accum_x,
            accum_y,
            weights,
            target_pixels.reshape((-1, 2)),
            source_coords[:, 0],
            source_coords[:, 1],
            np.ones(target_pixels.shape[:3], dtype=bool).reshape(-1),
        )

    # 固定 tile 模式不再对无效或剧烈变化区域逐像素兜底，避免边界区域拖慢批处理。


def _source_tile_coordinate_blocks(
    x0_values: np.ndarray,
    y0: int,
    y1: int,
    tile_width: int,
) -> np.ndarray:
    """按 tile 顺序生成源图像素坐标块，兼容最右侧不足完整 tile 的情况。"""

    local_x = np.arange(tile_width, dtype=np.float64)
    local_y = np.arange(y0, y1, dtype=np.float64)
    tile_x = x0_values[:, None, None].astype(np.float64) + local_x[None, None, :]
    tile_x = np.broadcast_to(tile_x, (x0_values.size, local_y.size, tile_width))
    tile_y = np.broadcast_to(local_y[None, :, None], (x0_values.size, local_y.size, tile_width))
    return np.stack((tile_x, tile_y), axis=3)


def _source_tile_row_pixels_to_target_pixels_exact(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    target_model: object | None,
    geometry: MosaicExportGeometry,
    x0_values: np.ndarray,
    y0: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_x = np.arange(width, dtype=np.float64)
    local_y = np.arange(height, dtype=np.float64)
    tile_x = np.broadcast_to(
        x0_values[:, None, None].astype(np.float64) + local_x[None, None, :],
        (x0_values.size, height, width),
    )
    tile_y = np.full((x0_values.size, height, width), float(y0), dtype=np.float64) + local_y[None, :, None]
    points = np.column_stack((tile_x.reshape(-1), tile_y.reshape(-1)))
    return _source_pixels_to_target_pixels(
        source_model,
        points,
        target_icrs_to_pixel_payload,
        target_model=target_model,
        geometry=geometry,
    )


def _source_pixels_to_target_pixels(
    source_model: object,
    source_pixels: np.ndarray,
    target_icrs_to_pixel_payload: dict[str, object] | None,
    *,
    target_model: object | None = None,
    geometry: MosaicExportGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(source_pixels, dtype=np.float64)
    radec = np.asarray(source_model.pixel_to_sky_points(pixels), dtype=np.float64)
    vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])
    if target_model is not None:
        target_pixels = _source_pixels_from_icrs_vectors(target_model, vectors)
        crop_left_px, crop_top_px = _target_output_crop_offsets(target_icrs_to_pixel_payload, geometry)
        target_pixels[:, 0] -= crop_left_px
        target_pixels[:, 1] -= crop_top_px
        valid = np.all(np.isfinite(radec), axis=1) & np.all(np.isfinite(target_pixels), axis=1)
        target_pixels[~valid] = np.nan
        return target_pixels.astype(np.float64), valid.astype(bool)
    if target_icrs_to_pixel_payload is None:
        raise ValueError("缺少目标 ICRS 变换或底图模型。")
    return target_icrs_vectors_to_output_pixel_points(vectors, target_icrs_to_pixel_payload)


def _target_output_crop_offsets(
    target_icrs_to_pixel_payload: dict[str, object] | None,
    geometry: MosaicExportGeometry | None,
) -> tuple[float, float]:
    """读取目标画布相对于完整图像左上角的裁剪偏移。"""

    if geometry is not None:
        return float(geometry.crop_left_px), float(geometry.crop_top_px)
    if target_icrs_to_pixel_payload is not None:
        return (
            float(target_icrs_to_pixel_payload.get("crop_left_px", 0.0)),
            float(target_icrs_to_pixel_payload.get("crop_top_px", 0.0)),
        )
    return 0.0, 0.0


def _model_pixel_points_to_icrs_vectors(
    model: object,
    pixel_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """通过 Pixel→ICRS 模型把图像像素转换为单位方向。"""

    pixels = np.asarray(pixel_points, dtype=np.float64)
    radec = np.asarray(model.pixel_to_sky_points(pixels), dtype=np.float64)
    if radec.shape != (pixels.shape[0], 2):
        raise ValueError(f"底图模型 Pixel→ICRS 输出形状异常：{radec.shape}")
    valid = np.all(np.isfinite(radec), axis=1)
    vectors = np.full((pixels.shape[0], 3), np.nan, dtype=np.float64)
    if np.any(valid):
        vectors[valid] = radec_to_unit_vectors(radec[valid, 0], radec[valid, 1])
    return vectors, valid.astype(bool)


def _accumulate_source_pixels_to_inverse_map(
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    target_pixels: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    valid: np.ndarray,
) -> None:
    target = np.asarray(target_pixels, dtype=np.float64)
    src_x = np.asarray(source_x, dtype=np.float32)
    src_y = np.asarray(source_y, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.all(np.isfinite(target), axis=1)
    if not np.any(mask):
        return
    x = target[mask, 0]
    y = target[mask, 1]
    src_x = src_x[mask]
    src_y = src_y[mask]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    dx = (x - x0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    for offset_y, wy in ((0, 1.0 - dy), (1, dy)):
        yi = y0 + offset_y
        for offset_x, wx in ((0, 1.0 - dx), (1, dx)):
            xi = x0 + offset_x
            contribution = (wx * wy).astype(np.float32)
            inside = (
                (contribution > 0.0)
                & (xi >= 0)
                & (xi < weights.shape[1])
                & (yi >= 0)
                & (yi < weights.shape[0])
            )
            if not np.any(inside):
                continue
            target_index = (yi[inside], xi[inside])
            np.add.at(weights, target_index, contribution[inside])
            np.add.at(accum_x, target_index, src_x[inside] * contribution[inside])
            np.add.at(accum_y, target_index, src_y[inside] * contribution[inside])


def mosaic_reprojection_map_blocks(
    *,
    source_model: object,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
    block_rows: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[dict[str, object]]:
    """兼容旧入口，实际 map 构建已迁入 remap_builder。"""

    yield from iter_reprojection_map_blocks(
        source_model=source_model,
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=block_rows,
        progress_callback=progress_callback,
    )


def _render_mosaic_reprojection_block_from_map(
    *,
    source_rgb: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    remapped = cv2.remap(
        source_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if remapped.ndim == 2:
        remapped = np.repeat(remapped[:, :, None], 3, axis=2)
    source_height, source_width = source_rgb.shape[:2]
    valid_alpha = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= -0.5)
        & (map_x <= float(source_width) - 0.5)
        & (map_y >= -0.5)
        & (map_y <= float(source_height) - 0.5)
    )
    output_dtype = source_rgb.dtype
    alpha_max = np.iinfo(output_dtype).max
    alpha = np.where(valid_alpha, alpha_max, 0).astype(output_dtype)
    remapped = np.asarray(remapped, dtype=output_dtype)
    remapped[~valid_alpha] = 0
    rgba = np.dstack((remapped, alpha))
    return np.ascontiguousarray(rgba, dtype=output_dtype)


def write_mosaic_reprojection_tiff(
    *,
    output_path: str | Path,
    source_model: object,
    source_image: MosaicExportSourceImage,
    camera: CameraSettings | None,
    view: ViewSettings | None,
    observer: ObserverSettings | None,
    geometry: MosaicExportGeometry,
    framing_payload: dict[str, object] | None = None,
    block_rows: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS,
    progress_callback: Callable[[int], None] | None = None,
    export_progress_callback: MosaicExportProgressCallback | None = None,
    target_icrs_to_pixel_payload: dict[str, object] | None = None,
    target_model: object | None = None,
    map_tile_size_px: int = 4,
    exact_remap_repair: bool = False,
    tiff_lzw_compression: bool = True,
    source_pixel_regions: tuple[SourcePixelRegion, ...] | None = None,
) -> None:
    """把重投影结果写成与原图一致、最高 16 位的 RGBA TIFF。"""

    if tifffile is None:
        raise RuntimeError("当前环境缺少 tifffile，无法写入 TIFF。")
    if geometry.output_width_px <= 0 or geometry.output_height_px <= 0:
        raise ValueError("裁剪后的导出尺寸无效。")
    if target_icrs_to_pixel_payload is not None and target_model is not None:
        raise ValueError("目标 ICRS 变换和底图模型不能同时指定。")
    if target_icrs_to_pixel_payload is not None and not target_icrs_to_pixel_transform_payload_matches(
        target_icrs_to_pixel_payload,
        geometry=geometry,
    ):
        raise ValueError("取景 JSON 中的 ICRS 到全景图像素变换与当前输出几何不匹配。")
    if target_model is not None:
        target_width = int(getattr(target_model, "image_width_px", 0) or 0)
        target_height = int(getattr(target_model, "image_height_px", 0) or 0)
        if target_width != int(geometry.boundary_width_px) or target_height != int(geometry.boundary_height_px):
            raise ValueError(
                "底图模型尺寸与输出边界不一致："
                f"模型 {target_width} x {target_height} px，"
                f"边界 {geometry.boundary_width_px} x {geometry.boundary_height_px} px。"
            )
    description = _mosaic_export_description(framing_payload)
    output_height = int(geometry.output_height_px)
    output_width = int(geometry.output_width_px)
    safe_block_rows = max(1, min(int(block_rows), output_height))
    output_dtype = np.dtype(source_image.rgb.dtype)
    compression = "lzw" if tiff_lzw_compression else None
    compression_label = "LZW 压缩" if tiff_lzw_compression else "无压缩"
    if tiff_lzw_compression and imagecodecs is None:
        raise RuntimeError("当前环境缺少 imagecodecs，无法流式写入 LZW TIFF。")

    blocks = mosaic_reprojection_blocks(
        source_model=source_model,
        source_rgb=source_image.rgb,
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=safe_block_rows,
        progress_callback=progress_callback,
        export_progress_callback=export_progress_callback,
        target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
        target_model=target_model,
        map_tile_size_px=map_tile_size_px,
        exact_remap_repair=exact_remap_repair,
        source_pixel_regions=source_pixel_regions,
    )

    def validated_blocks() -> Iterator[np.ndarray]:
        completed_rows = 0
        for block in blocks:
            values = np.ascontiguousarray(block, dtype=output_dtype)
            if values.ndim != 3 or values.shape[1:] != (output_width, 4):
                raise ValueError(f"重投影输出块尺寸异常：{values.shape}")
            if values.shape[0] <= 0 or values.shape[0] > safe_block_rows:
                raise ValueError(f"重投影输出块行数异常：{values.shape[0]}")
            completed_rows += int(values.shape[0])
            if completed_rows > output_height:
                raise ValueError("重投影输出行数超过目标高度。")
            yield values
        if completed_rows != output_height:
            raise ValueError(f"重投影输出行数异常：{completed_rows}，预期 {output_height}。")

    def emit_written_progress(completed_rows: int) -> None:
        _emit_export_progress(
            export_progress_callback,
            f"正在写入 {compression_label} TIFF...",
            completed_rows,
            output_height,
        )

    def encoded_strips() -> Iterator[bytes]:
        if not tiff_lzw_compression:
            completed_rows = 0
            for values in validated_blocks():
                completed_rows += int(values.shape[0])
                emit_written_progress(completed_rows)
                yield values.tobytes(order="C")
            return

        # LZW 编码由 imagecodecs 释放 GIL；少量有界线程可恢复 tifffile 原有并行压缩速度，
        # 同时只保留几个行块，避免重新持有整张 RGBA 图。
        cpu_count = max(1, int(os.cpu_count() or 1))
        worker_count = min(4, max(1, cpu_count // 2))
        pending: deque[tuple[int, Future[bytes]]] = deque()
        completed_rows = 0
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mosaic-lzw") as executor:
            for values in validated_blocks():
                pending.append((int(values.shape[0]), executor.submit(imagecodecs.lzw_encode, values)))
                if len(pending) < worker_count * 2:
                    continue
                rows, future = pending.popleft()
                encoded = future.result()
                completed_rows += rows
                emit_written_progress(completed_rows)
                yield encoded
            while pending:
                rows, future = pending.popleft()
                encoded = future.result()
                completed_rows += rows
                emit_written_progress(completed_rows)
                yield encoded

    _emit_export_progress(
        export_progress_callback,
        "正在准备分块 TIFF 写入...",
        0,
        0,
    )
    tifffile.imwrite(
        str(output_path),
        encoded_strips(),
        shape=(output_height, output_width, 4),
        dtype=output_dtype,
        photometric="rgb",
        planarconfig="contig",
        extrasamples=("unassalpha",),
        rowsperstrip=safe_block_rows,
        compression=compression,
        metadata=None,
        description=description,
        software="HoshinoPanoAssistant",
        iccprofile=source_image.icc_profile,
        extratags=source_image.exif_tags or None,
    )
    _copy_full_tiff_exif_metadata(source_image.path, output_path)


def _mosaic_export_description(framing_payload: dict[str, object] | None) -> str:
    if framing_payload is None:
        return "HoshinoPanoAssistant free projection mosaic export"
    try:
        return _tiff_ascii(json.dumps(
            {
                "software": "HoshinoPanoAssistant",
                "mosaic_framing": framing_payload,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ))
    except (TypeError, ValueError):
        return "HoshinoPanoAssistant free projection mosaic export"


def _tiff_ascii(text: str) -> str:
    """TIFF ASCII 标签只能写 7-bit 字符，非 ASCII 内容用转义形式保留。"""

    return text.encode("ascii", errors="backslashreplace").decode("ascii")


__all__ = [name for name in globals() if not name.startswith("__")]
