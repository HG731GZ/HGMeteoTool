from __future__ import annotations

import base64
import json
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from PIL import Image, ImageOps

from .coordinates import normalize_vector, radec_to_unit_vectors, unit_vectors_to_radec
from .simulator import (
    CYLINDRICAL_LENS_MODELS,
    FISHEYE_LENS_MODELS,
    CameraSettings,
    ObserverSettings,
    MERCATOR_LENS_MODEL,
    RECTILINEAR_LENS_MODEL,
    ViewSettings,
    _fisheye_theta_from_radius_ratio,
    _projection_horizontal_scale_px,
)

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV 是导出重采样依赖；兜底方便环境诊断。
    cv2 = None

try:
    import tifffile
except ImportError:  # pragma: no cover - tifffile 由环境声明，兜底用于给界面友好报错。
    tifffile = None


MOSAIC_EXPORT_TIFF_FILTER = "无压缩 16-bit TIFF (*.tif *.tiff)"
MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS = 1024
MOSAIC_EXPORT_WHITE_U16 = 65535
MOSAIC_TARGET_ICRS_MAP_VERSION = 1
MOSAIC_TARGET_ICRS_MAP_COMPRESSION_LEVEL = 6

# 这些标签由导出图自身决定，继续复制原图值会造成尺寸、压缩或方向冲突。
_EXIF_EXCLUDED_TAGS = {
    256, 257, 258, 259, 262, 273, 274, 277, 278, 279, 282, 283, 284, 296,
    305, 317, 320, 322, 323, 324, 325, 338, 339, 33723, 34665, 34675, 34853,
    40962, 40963, 513, 514,
}


@dataclass(frozen=True)
class MosaicExportSourceImage:
    """用于最终导出的全分辨率源图数据。"""

    path: Path
    rgb_u16: np.ndarray
    exif_tags: tuple[tuple[int, str, int, object, bool], ...]
    icc_profile: bytes | None

    @property
    def width_px(self) -> int:
        return int(self.rgb_u16.shape[1])

    @property
    def height_px(self) -> int:
        return int(self.rgb_u16.shape[0])


@dataclass(frozen=True)
class MosaicExportGeometry:
    """完整边界与裁剪后的实际导出区域。"""

    boundary_width_px: int
    boundary_height_px: int
    crop_left_px: int
    crop_top_px: int
    output_width_px: int
    output_height_px: int


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


def mosaic_export_cropped_geometry(
    *,
    boundary_width_px: int,
    boundary_height_px: int,
    crop: dict[str, object],
) -> MosaicExportGeometry:
    """根据完整输出边界和四边裁剪量计算实际导出区域。"""

    boundary_width = max(0, int(round(float(boundary_width_px))))
    boundary_height = max(0, int(round(float(boundary_height_px))))
    left = _crop_margin_px(crop.get("left_px"), boundary_width)
    right = _crop_margin_px(crop.get("right_px"), max(0, boundary_width - left))
    top = _crop_margin_px(crop.get("top_px"), boundary_height)
    bottom = _crop_margin_px(crop.get("bottom_px"), max(0, boundary_height - top))
    return MosaicExportGeometry(
        boundary_width_px=boundary_width,
        boundary_height_px=boundary_height,
        crop_left_px=left,
        crop_top_px=top,
        output_width_px=max(0, boundary_width - left - right),
        output_height_px=max(0, boundary_height - top - bottom),
    )


def _crop_margin_px(value: object, maximum: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    return max(0, min(int(round(numeric)), max(0, int(maximum))))


def build_target_icrs_map_payload(
    *,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
    block_rows: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, object]:
    """生成 A 像素到 ICRS 单位向量的分块压缩负载。"""

    blocks: list[dict[str, object]] = []
    for block in target_icrs_map_blocks(
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=block_rows,
        progress_callback=progress_callback,
    ):
        blocks.append(
            _encode_target_icrs_block(
                int(block["row_start"]),
                np.asarray(block["icrs_vectors"], dtype=np.float32),
            )
        )
    return {
        "version": MOSAIC_TARGET_ICRS_MAP_VERSION,
        "scope": "cropped_output",
        "dtype": "float32",
        "encoding": "zlib+base64",
        "vector_frame": "ICRS",
        "vector_components": ["x", "y", "z"],
        "pixel_convention": "0-based_pixel_center",
        "block_rows": int(max(1, block_rows)),
        "boundary_width_px": int(geometry.boundary_width_px),
        "boundary_height_px": int(geometry.boundary_height_px),
        "crop_left_px": int(geometry.crop_left_px),
        "crop_top_px": int(geometry.crop_top_px),
        "output_width_px": int(geometry.output_width_px),
        "output_height_px": int(geometry.output_height_px),
        "blocks": blocks,
    }


def target_icrs_map_payload_matches(
    payload: object,
    *,
    geometry: MosaicExportGeometry,
) -> bool:
    """检查 A 像素到 ICRS map 是否匹配当前裁剪输出几何。"""

    if not isinstance(payload, dict):
        return False
    if int(payload.get("version", 0) or 0) != MOSAIC_TARGET_ICRS_MAP_VERSION:
        return False
    if str(payload.get("scope") or "") != "cropped_output":
        return False
    if str(payload.get("vector_frame") or "") != "ICRS":
        return False
    expected = {
        "boundary_width_px": geometry.boundary_width_px,
        "boundary_height_px": geometry.boundary_height_px,
        "crop_left_px": geometry.crop_left_px,
        "crop_top_px": geometry.crop_top_px,
        "output_width_px": geometry.output_width_px,
        "output_height_px": geometry.output_height_px,
    }
    for key, expected_value in expected.items():
        try:
            actual_value = int(payload.get(key, -1))
        except (TypeError, ValueError):
            return False
        if actual_value != int(expected_value):
            return False
    return isinstance(payload.get("blocks"), list)


def target_icrs_map_blocks(
    *,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
    block_rows: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[dict[str, object]]:
    """逐块生成裁剪后 A 像素中心对应的 ICRS 单位向量。"""

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
        vectors, valid = target_image_points_to_icrs_vectors(
            grid_x.ravel(),
            grid_y.ravel(),
            camera=camera,
            icrs_basis=basis,
        )
        vectors[~valid] = np.nan
        completed_rows += rows
        if progress_callback is not None:
            progress_callback(completed_rows)
        yield {
            "row_start": int(row_start),
            "icrs_vectors": vectors.reshape((rows, output_width, 3)).astype(np.float32),
        }


def iter_target_icrs_map_payload_blocks(payload: dict[str, object]) -> Iterator[dict[str, object]]:
    """逐块解码取景 JSON 中保存的 A 像素到 ICRS map。"""

    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("取景 JSON 中的 target_icrs_map.blocks 必须是数组。")
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("取景 JSON 中的 target_icrs_map block 必须是对象。")
        rows = int(block.get("rows", 0) or 0)
        columns = int(block.get("columns", 0) or 0)
        row_start = int(block.get("row_start", 0) or 0)
        if rows <= 0 or columns <= 0:
            raise ValueError("取景 JSON 中的 target_icrs_map block 尺寸无效。")
        yield {
            "row_start": row_start,
            "icrs_vectors": _decode_target_icrs_vectors(str(block.get("icrs_vectors") or ""), rows, columns),
        }


def _encode_target_icrs_block(row_start: int, icrs_vectors: np.ndarray) -> dict[str, object]:
    rows, columns, components = icrs_vectors.shape
    if components != 3:
        raise ValueError("ICRS map block 必须是 rows x columns x 3。")
    return {
        "row_start": int(row_start),
        "rows": int(rows),
        "columns": int(columns),
        "icrs_vectors": _encode_float32_array(icrs_vectors),
    }


def _encode_float32_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype=np.float32)
    compressed = zlib.compress(values.tobytes(order="C"), level=MOSAIC_TARGET_ICRS_MAP_COMPRESSION_LEVEL)
    return base64.b64encode(compressed).decode("ascii")


def _decode_target_icrs_vectors(encoded: str, rows: int, columns: int) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(encoded.encode("ascii")))
    values = np.frombuffer(raw, dtype=np.float32)
    expected_size = int(rows) * int(columns) * 3
    if values.size != expected_size:
        raise ValueError("取景 JSON 中的 ICRS map 数组长度与尺寸不一致。")
    return values.reshape((int(rows), int(columns), 3)).copy()


def load_mosaic_export_source_image(image_path: str | Path) -> MosaicExportSourceImage:
    """读取源图全分辨率像素，输出统一的 RGB uint16 数组。"""

    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"源图不存在：{path}")
    path = path.resolve()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with Image.open(path) as image:
            exif = image.getexif()
            transformed = ImageOps.exif_transpose(image)
            array = np.asarray(transformed)
            exif_tags = _safe_exif_tags(exif)
            icc_profile = image.info.get("icc_profile")

    rgb_u16 = _image_array_to_rgb_u16(array)
    return MosaicExportSourceImage(
        path=path,
        rgb_u16=np.ascontiguousarray(rgb_u16, dtype=np.uint16),
        exif_tags=exif_tags,
        icc_profile=icc_profile if isinstance(icc_profile, bytes) else None,
    )


def _image_array_to_rgb_u16(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    elif values.ndim == 3 and values.shape[2] >= 3:
        values = values[:, :, :3]
    else:
        raise ValueError(f"源图像素形状不支持：{values.shape}")

    if values.dtype == np.uint16:
        return values.astype(np.uint16, copy=False)
    if values.dtype == np.uint8:
        return (values.astype(np.uint16) * 257).astype(np.uint16)
    if np.issubdtype(values.dtype, np.integer):
        return np.clip(values, 0, MOSAIC_EXPORT_WHITE_U16).astype(np.uint16)
    if np.issubdtype(values.dtype, np.floating):
        finite = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.nanmax(finite)) <= 1.0:
            finite = finite * MOSAIC_EXPORT_WHITE_U16
        return np.clip(finite, 0.0, float(MOSAIC_EXPORT_WHITE_U16)).astype(np.uint16)
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


def mosaic_reprojection_blocks(
    *,
    source_model: object,
    source_rgb_u16: np.ndarray,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
    block_rows: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS,
    progress_callback: Callable[[int], None] | None = None,
    target_icrs_map_payload: dict[str, object] | None = None,
) -> Iterator[np.ndarray]:
    """逐块生成裁剪后目标图像的 RGB uint16 像素。"""

    if cv2 is None:
        raise RuntimeError("当前环境缺少 OpenCV，无法执行重投影导出。")
    source = np.ascontiguousarray(source_rgb_u16, dtype=np.uint16)
    if target_icrs_map_payload is None:
        map_blocks = mosaic_reprojection_map_blocks(
            source_model=source_model,
            camera=camera,
            view=view,
            observer=observer,
            geometry=geometry,
            block_rows=block_rows,
            progress_callback=None,
        )
    else:
        map_blocks = _mosaic_reprojection_map_blocks_from_target_icrs_map(
            source_model=source_model,
            target_icrs_map_payload=target_icrs_map_payload,
        )
    completed_rows = 0

    for map_block in map_blocks:
        map_x = np.asarray(map_block["map_x"], dtype=np.float32)
        map_y = np.asarray(map_block["map_y"], dtype=np.float32)
        block = _render_mosaic_reprojection_block_from_map(
            source_rgb_u16=source,
            map_x=map_x,
            map_y=map_y,
        )
        completed_rows += int(map_x.shape[0])
        if progress_callback is not None:
            progress_callback(completed_rows)
        yield block


def _mosaic_reprojection_map_blocks_from_target_icrs_map(
    *,
    source_model: object,
    target_icrs_map_payload: dict[str, object],
) -> Iterator[dict[str, object]]:
    """把已导入的 A 像素到 ICRS map 转换成当前源图 B 的 remap 坐标。"""

    for block in iter_target_icrs_map_payload_blocks(target_icrs_map_payload):
        icrs_vectors = np.asarray(block["icrs_vectors"], dtype=np.float64)
        rows, columns, components = icrs_vectors.shape
        if components != 3:
            raise ValueError("取景 JSON 中的 ICRS map block 形状无效。")
        map_x, map_y = _source_pixel_map_from_icrs_vectors(
            source_model=source_model,
            icrs_vectors=icrs_vectors.reshape((-1, 3)),
            rows=rows,
            columns=columns,
        )
        yield {
            "row_start": int(block["row_start"]),
            "map_x": map_x,
            "map_y": map_y,
        }


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
        map_x, map_y = _build_mosaic_reprojection_map_block(
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
        yield {
            "row_start": int(row_start),
            "map_x": map_x,
            "map_y": map_y,
        }


def _build_mosaic_reprojection_map_block(
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
    return _source_pixel_map_from_icrs_vectors(
        source_model=source_model,
        icrs_vectors=icrs_vectors,
        rows=rows,
        columns=columns,
    )


def _source_pixel_map_from_icrs_vectors(
    *,
    source_model: object,
    icrs_vectors: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把 ICRS 单位向量投到源图 B 像素，生成 OpenCV remap 坐标。"""

    vector_count = int(np.asarray(icrs_vectors).shape[0])
    map_x = np.full(vector_count, -1.0, dtype=np.float32)
    map_y = np.full(vector_count, -1.0, dtype=np.float32)
    valid = np.all(np.isfinite(icrs_vectors), axis=1)
    if np.any(valid):
        source_pixels = _source_pixels_from_icrs_vectors(source_model, icrs_vectors[valid])
        mapped_valid = (
            np.all(np.isfinite(source_pixels), axis=1)
        )
        source_width = getattr(source_model, "image_width_px", None)
        source_height = getattr(source_model, "image_height_px", None)
        if source_width is not None and source_height is not None:
            mapped_valid &= (
                (source_pixels[:, 0] >= -0.5)
                & (source_pixels[:, 0] <= float(source_width) - 0.5)
                & (source_pixels[:, 1] >= -0.5)
                & (source_pixels[:, 1] <= float(source_height) - 0.5)
            )
        valid_indices = np.flatnonzero(valid)
        accepted = valid_indices[mapped_valid]
        map_x[accepted] = source_pixels[mapped_valid, 0].astype(np.float32)
        map_y[accepted] = source_pixels[mapped_valid, 1].astype(np.float32)
    return map_x.reshape((rows, columns)), map_y.reshape((rows, columns))


def _source_pixels_from_icrs_vectors(source_model: object, icrs_vectors: np.ndarray) -> np.ndarray:
    direct_project = getattr(source_model, "icrs_vectors_to_pixel_points", None)
    if callable(direct_project):
        return np.asarray(direct_project(icrs_vectors), dtype=np.float64)
    radec = unit_vectors_to_radec(icrs_vectors)
    return np.asarray(source_model.sky_to_pixel_points(radec), dtype=np.float64)


def _render_mosaic_reprojection_block_from_map(
    *,
    source_rgb_u16: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    rows, columns = map_x.shape

    remapped = cv2.remap(
        source_rgb_u16,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(MOSAIC_EXPORT_WHITE_U16, MOSAIC_EXPORT_WHITE_U16, MOSAIC_EXPORT_WHITE_U16),
    )
    if remapped.ndim == 2:
        remapped = np.repeat(remapped[:, :, None], 3, axis=2)
    return np.ascontiguousarray(remapped, dtype=np.uint16)


def target_image_points_to_icrs_vectors(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    camera: CameraSettings,
    icrs_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """把目标画布像素直接反解为 ICRS 单位方向。"""

    camera_vectors, valid = target_image_points_to_camera_vectors(x_px, y_px, camera)
    right, up, forward = icrs_basis
    icrs_x = camera_vectors[:, 0, None] * right[None, :]
    icrs_y = camera_vectors[:, 1, None] * up[None, :]
    icrs_z = camera_vectors[:, 2, None] * forward[None, :]
    vectors = icrs_x + icrs_y + icrs_z
    norm = np.linalg.norm(vectors, axis=1)
    valid &= np.isfinite(norm) & (norm > 1e-12)
    vectors[valid] /= norm[valid, None]
    vectors[~valid] = np.nan
    return vectors.astype(np.float64), valid.astype(bool)


def target_image_points_to_camera_vectors(
    x_px: np.ndarray,
    y_px: np.ndarray,
    camera: CameraSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """把目标画布像素反解为目标相机坐标系下的方向。"""

    x_values = np.asarray(x_px, dtype=np.float64)
    y_values = np.asarray(y_px, dtype=np.float64)
    if x_values.shape != y_values.shape:
        raise ValueError("目标像素 x/y 数组形状必须一致。")

    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        x_mm = (x_values - camera.image_width_px * 0.5) * camera.sensor_width_mm / camera.image_width_px
        y_mm = (camera.image_height_px * 0.5 - y_values) * camera.sensor_height_mm / camera.image_height_px
        cam_x = x_mm / camera.focal_length_mm
        cam_y = y_mm / camera.focal_length_mm
        cam_z = np.ones_like(cam_x, dtype=np.float64)
        valid = np.isfinite(cam_x) & np.isfinite(cam_y)
    elif camera.lens_model in FISHEYE_LENS_MODELS:
        center_x = camera.image_width_px * 0.5
        center_y = camera.image_height_px * 0.5
        screen_x = x_values - center_x
        screen_y = center_y - y_values
        r_px = np.hypot(screen_x, screen_y)
        r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
        rho = np.divide(r_px, r_limit, out=np.full_like(r_px, np.inf), where=r_limit > 1e-12)
        theta_max = np.deg2rad(camera.fisheye_fov_deg * 0.5)
        theta = _fisheye_theta_from_radius_ratio(np.clip(rho, 0.0, 1.0), theta_max, camera.lens_model)
        plane_norm = np.sin(theta)
        unit_x = np.divide(screen_x, r_px, out=np.zeros_like(screen_x), where=r_px > 1e-12)
        unit_y = np.divide(screen_y, r_px, out=np.zeros_like(screen_y), where=r_px > 1e-12)
        cam_x = unit_x * plane_norm
        cam_y = unit_y * plane_norm
        cam_z = np.cos(theta)
        valid = (rho <= 1.0 + 1e-9) & np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z)
    elif camera.lens_model in CYLINDRICAL_LENS_MODELS:
        center_x = camera.image_width_px * 0.5
        center_y = camera.image_height_px * 0.5
        scale_px = _projection_horizontal_scale_px(camera)
        longitude = (x_values - center_x) / max(scale_px, 1e-12)
        plane_y = (center_y - y_values) / max(scale_px, 1e-12)
        if camera.lens_model == MERCATOR_LENS_MODEL:
            latitude = np.arcsin(np.clip(np.tanh(plane_y), -1.0, 1.0))
            valid = np.isfinite(longitude) & np.isfinite(latitude)
        else:
            latitude = plane_y
            valid = np.isfinite(longitude) & np.isfinite(latitude) & (np.abs(latitude) <= np.pi * 0.5 + 1e-8)
        cos_lat = np.cos(latitude)
        cam_x = cos_lat * np.sin(longitude)
        cam_y = np.sin(latitude)
        cam_z = cos_lat * np.cos(longitude)
    else:
        raise ValueError(f"不支持的目标投影模型：{camera.lens_model}")

    vectors = np.column_stack((cam_x, cam_y, cam_z)).astype(np.float64)
    norm = np.linalg.norm(vectors, axis=1)
    valid &= np.isfinite(norm) & (norm > 1e-12)
    vectors[valid] /= norm[valid, None]
    vectors[~valid] = np.nan
    return vectors.astype(np.float64), valid.astype(bool)


def _icrs_camera_basis_from_view(
    view: ViewSettings,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, zenith, north = _reference_icrs_vectors_from_view(view, observer)
    right = np.cross(center, zenith)
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.cross(center, north)
    right = normalize_vector(right)
    up = normalize_vector(np.cross(right, center))

    roll = np.deg2rad(view.roll_deg)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    rolled_right = right * cos_roll + up * sin_roll
    rolled_up = -right * sin_roll + up * cos_roll
    return (
        normalize_vector(rolled_right),
        normalize_vector(rolled_up),
        normalize_vector(center),
    )


def _reference_icrs_vectors_from_view(
    view: ViewSettings,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alt_deg = np.asarray([view.center_alt_deg, 90.0, 0.0], dtype=np.float64)
    az_deg = np.asarray([view.center_az_deg, view.center_az_deg, 0.0], dtype=np.float64)
    location = EarthLocation.from_geodetic(
        lon=observer.longitude_deg * u.deg,
        lat=observer.latitude_deg * u.deg,
        height=observer.elevation_m * u.m,
    )
    altaz = SkyCoord(
        alt=alt_deg * u.deg,
        az=az_deg * u.deg,
        frame=AltAz(obstime=Time(observer.observation_time_utc), location=location),
    )
    icrs = altaz.icrs
    vectors = radec_to_unit_vectors(icrs.ra.degree, icrs.dec.degree)
    return normalize_vector(vectors[0]), normalize_vector(vectors[1]), normalize_vector(vectors[2])


def write_mosaic_reprojection_tiff(
    *,
    output_path: str | Path,
    source_model: object,
    source_image: MosaicExportSourceImage,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
    framing_payload: dict[str, object] | None = None,
    block_rows: int = MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS,
    progress_callback: Callable[[int], None] | None = None,
    target_icrs_map_payload: dict[str, object] | None = None,
) -> None:
    """把重投影结果写成无压缩 16-bit RGB TIFF。"""

    if tifffile is None:
        raise RuntimeError("当前环境缺少 tifffile，无法写入 16-bit TIFF。")
    if geometry.output_width_px <= 0 or geometry.output_height_px <= 0:
        raise ValueError("裁剪后的导出尺寸无效。")
    if target_icrs_map_payload is not None and not target_icrs_map_payload_matches(
        target_icrs_map_payload,
        geometry=geometry,
    ):
        raise ValueError("取景 JSON 中的 A 像素到 ICRS map 与当前输出几何不匹配。")
    description = _mosaic_export_description(framing_payload)
    blocks = mosaic_reprojection_blocks(
        source_model=source_model,
        source_rgb_u16=source_image.rgb_u16,
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=block_rows,
        progress_callback=progress_callback,
        target_icrs_map_payload=target_icrs_map_payload,
    )
    tifffile.imwrite(
        str(output_path),
        blocks,
        shape=(int(geometry.output_height_px), int(geometry.output_width_px), 3),
        dtype=np.uint16,
        photometric="rgb",
        planarconfig="contig",
        compression=None,
        metadata=None,
        description=description,
        software="MeteoAlign",
        iccprofile=source_image.icc_profile,
        extratags=source_image.exif_tags or None,
    )


def _mosaic_export_description(framing_payload: dict[str, object] | None) -> str:
    if framing_payload is None:
        return "MeteoAlign free projection mosaic export"
    try:
        return _tiff_ascii(json.dumps(
            {
                "software": "MeteoAlign",
                "mosaic_framing": framing_payload,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ))
    except (TypeError, ValueError):
        return "MeteoAlign free projection mosaic export"


def _tiff_ascii(text: str) -> str:
    """TIFF ASCII 标签只能写 7-bit 字符，非 ASCII 内容用转义形式保留。"""

    return text.encode("ascii", errors="backslashreplace").decode("ascii")


__all__ = [name for name in globals() if not name.startswith("__")]
