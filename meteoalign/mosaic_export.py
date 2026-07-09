from __future__ import annotations

import json
import warnings
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
    _fisheye_radius_ratio,
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
MOSAIC_TARGET_ICRS_TO_PIXEL_VERSION = 1
MOSAIC_FORWARD_REMAP_LOW_WEIGHT_THRESHOLD = 0.25
MOSAIC_FORWARD_REMAP_EXACT_FILL_BATCH_PIXELS = 1_000_000

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


def build_target_icrs_to_pixel_transform_payload(
    *,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
) -> dict[str, object]:
    """导出源图无关的 ICRS 到 A 像素变换。"""

    right, up, forward = _icrs_camera_basis_from_view(view, observer)
    return {
        "version": MOSAIC_TARGET_ICRS_TO_PIXEL_VERSION,
        "type": "icrs_to_cropped_output_pixel",
        "pixel_convention": "0-based_pixel_center",
        "boundary_width_px": int(geometry.boundary_width_px),
        "boundary_height_px": int(geometry.boundary_height_px),
        "crop_left_px": int(geometry.crop_left_px),
        "crop_top_px": int(geometry.crop_top_px),
        "output_width_px": int(geometry.output_width_px),
        "output_height_px": int(geometry.output_height_px),
        "camera": {
            "lens_model": str(camera.lens_model),
            "sensor_width_mm": float(camera.sensor_width_mm),
            "sensor_height_mm": float(camera.sensor_height_mm),
            "image_width_px": int(camera.image_width_px),
            "image_height_px": int(camera.image_height_px),
            "focal_length_mm": float(camera.focal_length_mm),
            "fisheye_fov_deg": float(camera.fisheye_fov_deg),
        },
        "icrs_camera_basis": {
            "right": [float(value) for value in right],
            "up": [float(value) for value in up],
            "forward": [float(value) for value in forward],
        },
    }


def target_icrs_to_pixel_transform_payload_matches(
    payload: object,
    *,
    geometry: MosaicExportGeometry,
) -> bool:
    """检查 ICRS 到 A 像素变换是否匹配当前裁剪输出几何。"""

    if not isinstance(payload, dict):
        return False
    if int(payload.get("version", 0) or 0) != MOSAIC_TARGET_ICRS_TO_PIXEL_VERSION:
        return False
    if str(payload.get("type") or "") != "icrs_to_cropped_output_pixel":
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
    return isinstance(payload.get("camera"), dict) and isinstance(payload.get("icrs_camera_basis"), dict)


def _target_transform_camera(payload: dict[str, object]) -> CameraSettings:
    camera_payload = payload.get("camera")
    if not isinstance(camera_payload, dict):
        raise ValueError("取景 JSON 缺少 target_icrs_to_pixel.camera。")
    return CameraSettings(
        sensor_width_mm=float(camera_payload.get("sensor_width_mm", 36.0)),
        sensor_height_mm=float(camera_payload.get("sensor_height_mm", 24.0)),
        image_width_px=int(camera_payload.get("image_width_px", payload.get("boundary_width_px", 0))),
        image_height_px=int(camera_payload.get("image_height_px", payload.get("boundary_height_px", 0))),
        focal_length_mm=float(camera_payload.get("focal_length_mm", 24.0)),
        lens_model=str(camera_payload.get("lens_model", RECTILINEAR_LENS_MODEL)),
        fisheye_fov_deg=float(camera_payload.get("fisheye_fov_deg", 180.0)),
    )


def _target_transform_basis(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_payload = payload.get("icrs_camera_basis")
    if not isinstance(basis_payload, dict):
        raise ValueError("取景 JSON 缺少 target_icrs_to_pixel.icrs_camera_basis。")
    return (
        _payload_vector3(basis_payload.get("right"), "target_icrs_to_pixel.icrs_camera_basis.right"),
        _payload_vector3(basis_payload.get("up"), "target_icrs_to_pixel.icrs_camera_basis.up"),
        _payload_vector3(basis_payload.get("forward"), "target_icrs_to_pixel.icrs_camera_basis.forward"),
    )


def _payload_vector3(value: object, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"取景 JSON 字段 {field_name} 必须是 3 个有限数值。")
    return vector.astype(np.float64)


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
    target_icrs_to_pixel_payload: dict[str, object] | None = None,
    map_tile_size_px: int = 4,
    exact_remap_repair: bool = False,
) -> Iterator[np.ndarray]:
    """逐块生成裁剪后目标图像的 RGB uint16 像素。"""

    if cv2 is None:
        raise RuntimeError("当前环境缺少 OpenCV，无法执行重投影导出。")
    source = np.ascontiguousarray(source_rgb_u16, dtype=np.uint16)
    if target_icrs_to_pixel_payload is None:
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
            block = _render_mosaic_reprojection_block_from_map(
                source_rgb_u16=source,
                map_x=map_x,
                map_y=map_y,
            )
            completed_rows += int(map_x.shape[0])
            if progress_callback is not None:
                progress_callback(completed_rows)
            yield block
        return

    yield _render_mosaic_forward_remap_from_source_to_target(
        source_model=source_model,
        source_rgb_u16=source,
        target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
        geometry=geometry,
        map_tile_size_px=map_tile_size_px,
        exact_remap_repair=exact_remap_repair,
        progress_callback=progress_callback,
    )


def _render_mosaic_forward_remap_from_source_to_target(
    *,
    source_model: object,
    source_rgb_u16: np.ndarray,
    target_icrs_to_pixel_payload: dict[str, object],
    geometry: MosaicExportGeometry,
    map_tile_size_px: int,
    exact_remap_repair: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    """按 B 的固定 tile 建立 A 到 B 的近似 remap，再由 OpenCV 反向采样。"""

    output_width = int(geometry.output_width_px)
    output_height = int(geometry.output_height_px)
    source_height, source_width = source_rgb_u16.shape[:2]
    tile_size = max(1, int(map_tile_size_px))
    accum_x = np.zeros((output_height, output_width), dtype=np.float32)
    accum_y = np.zeros((output_height, output_width), dtype=np.float32)
    weights = np.zeros((output_height, output_width), dtype=np.float32)
    if tile_size <= 1:
        for row_start in range(0, source_height, max(1, int(MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS))):
            row_end = min(source_height, row_start + int(MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS))
            ys = np.arange(row_start, row_end, dtype=np.float64)
            xs = np.arange(source_width, dtype=np.float64)
            grid_x, grid_y = np.meshgrid(xs, ys)
            target_pixels, valid = _source_pixels_to_target_pixels(
                source_model,
                np.column_stack((grid_x.ravel(), grid_y.ravel())),
                target_icrs_to_pixel_payload,
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
    else:
        for y0 in range(0, source_height, tile_size):
            y1 = min(source_height, y0 + tile_size)
            _accumulate_source_tile_row_to_inverse_map(
                source_model=source_model,
                target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
                accum_x=accum_x,
                accum_y=accum_y,
                weights=weights,
                y0=y0,
                y1=y1,
                tile_size=tile_size,
            )
            if progress_callback is not None:
                progress_callback(int(round(output_height * y1 / max(source_height, 1))))
    if progress_callback is not None:
        progress_callback(output_height)

    map_x, map_y = _finalize_forward_inverse_map(
        accum_x,
        accum_y,
        weights,
        tile_size,
        source_model=source_model,
        target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
        exact_remap_repair=exact_remap_repair,
    )
    return _render_mosaic_reprojection_block_from_map(
        source_rgb_u16=source_rgb_u16,
        map_x=map_x,
        map_y=map_y,
    )


def _finalize_forward_inverse_map(
    accum_x: np.ndarray,
    accum_y: np.ndarray,
    weights: np.ndarray,
    tile_size: int,
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object],
    exact_remap_repair: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """把正向累积的源坐标整理成 cv2.remap 可用的 A 到 B map。"""

    map_x = np.full(weights.shape, -1.0, dtype=np.float32)
    map_y = np.full(weights.shape, -1.0, dtype=np.float32)
    covered = weights > 1e-6
    if np.any(covered):
        map_x[covered] = accum_x[covered] / weights[covered]
        map_y[covered] = accum_y[covered] / weights[covered]
        target_mask = _forward_inverse_map_target_mask(covered, tile_size)
        if exact_remap_repair:
            low_weight = covered & (weights < MOSAIC_FORWARD_REMAP_LOW_WEIGHT_THRESHOLD)
            exact_mask = target_mask & (~covered | low_weight)
            exact_valid = _fill_forward_inverse_map_exact(
                map_x,
                map_y,
                exact_mask,
                source_model=source_model,
                target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            )
            covered = covered | exact_valid
        else:
            covered = _fill_forward_inverse_map_holes_fast(map_x, map_y, covered, target_mask)
    map_x[~covered] = -1.0
    map_y[~covered] = -1.0
    return np.ascontiguousarray(map_x, dtype=np.float32), np.ascontiguousarray(map_y, dtype=np.float32)


def _forward_inverse_map_target_mask(
    covered: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    """用正向覆盖区域估计源图在 A 上的内部投影范围。"""

    if cv2 is None or not np.any(covered):
        return covered
    radius = max(1, min(32, int(tile_size) * 2))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)


def _fill_forward_inverse_map_holes_fast(
    map_x: np.ndarray,
    map_y: np.ndarray,
    covered: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """快速用邻域坐标补小缝隙，不做额外精确反算。"""

    if cv2 is None:
        return covered
    missing = target_mask & ~covered
    if not np.any(missing):
        return covered

    filled = covered.copy()
    neighbor_kernel = np.ones((3, 3), dtype=np.float32)
    max_iterations = 32
    for _index in range(max_iterations):
        missing = target_mask & ~filled
        if not np.any(missing):
            break
        valid_float = filled.astype(np.float32)
        count = cv2.filter2D(valid_float, cv2.CV_32F, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
        sum_x = cv2.filter2D(
            np.where(filled, map_x, 0.0).astype(np.float32),
            cv2.CV_32F,
            neighbor_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        sum_y = cv2.filter2D(
            np.where(filled, map_y, 0.0).astype(np.float32),
            cv2.CV_32F,
            neighbor_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        fillable = missing & (count > 0.0)
        if not np.any(fillable):
            break
        map_x[fillable] = sum_x[fillable] / count[fillable]
        map_y[fillable] = sum_y[fillable] / count[fillable]
        filled[fillable] = True
    return filled


def _fill_forward_inverse_map_exact(
    map_x: np.ndarray,
    map_y: np.ndarray,
    exact_mask: np.ndarray,
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object],
) -> np.ndarray:
    """对缺失和低权重 A 像素精确反算 B 坐标，避免亮星处坐标平均出黑点。"""

    exact_valid = np.zeros(exact_mask.shape, dtype=bool)
    if not np.any(exact_mask):
        return exact_valid

    y_indices, x_indices = np.nonzero(exact_mask)
    batch_size = int(MOSAIC_FORWARD_REMAP_EXACT_FILL_BATCH_PIXELS)
    for start in range(0, int(x_indices.size), batch_size):
        end = min(int(x_indices.size), start + batch_size)
        x_batch = x_indices[start:end].astype(np.float64)
        y_batch = y_indices[start:end].astype(np.float64)
        source_pixels, valid = _source_pixel_points_from_target_output_pixels(
            source_model=source_model,
            target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
            output_x_px=x_batch,
            output_y_px=y_batch,
        )
        if not np.any(valid):
            continue
        y_valid = y_indices[start:end][valid]
        x_valid = x_indices[start:end][valid]
        map_x[y_valid, x_valid] = source_pixels[valid, 0].astype(np.float32)
        map_y[y_valid, x_valid] = source_pixels[valid, 1].astype(np.float32)
        exact_valid[y_valid, x_valid] = True
    return exact_valid


def _source_pixel_points_from_target_output_pixels(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object],
    output_x_px: np.ndarray,
    output_y_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把裁剪后 A 像素精确反算到 B 像素。"""

    full_x = np.asarray(output_x_px, dtype=np.float64) + float(target_icrs_to_pixel_payload.get("crop_left_px", 0.0))
    full_y = np.asarray(output_y_px, dtype=np.float64) + float(target_icrs_to_pixel_payload.get("crop_top_px", 0.0))
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
    target_icrs_to_pixel_payload: dict[str, object],
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
    target_icrs_to_pixel_payload: dict[str, object],
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
    """按 tile 顺序生成 B 图像素坐标块，兼容最右侧不足完整 tile 的情况。"""

    local_x = np.arange(tile_width, dtype=np.float64)
    local_y = np.arange(y0, y1, dtype=np.float64)
    tile_x = x0_values[:, None, None].astype(np.float64) + local_x[None, None, :]
    tile_x = np.broadcast_to(tile_x, (x0_values.size, local_y.size, tile_width))
    tile_y = np.broadcast_to(local_y[None, :, None], (x0_values.size, local_y.size, tile_width))
    return np.stack((tile_x, tile_y), axis=3)


def _source_tile_row_pixels_to_target_pixels_exact(
    *,
    source_model: object,
    target_icrs_to_pixel_payload: dict[str, object],
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
    return _source_pixels_to_target_pixels(source_model, points, target_icrs_to_pixel_payload)


def _source_pixels_to_target_pixels(
    source_model: object,
    source_pixels: np.ndarray,
    target_icrs_to_pixel_payload: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(source_pixels, dtype=np.float64)
    radec = np.asarray(source_model.pixel_to_sky_points(pixels), dtype=np.float64)
    vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])
    return target_icrs_vectors_to_output_pixel_points(vectors, target_icrs_to_pixel_payload)


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


def target_icrs_vectors_to_output_pixel_points(
    vectors: np.ndarray,
    target_icrs_to_pixel_payload: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """把 ICRS 单位方向投到裁剪后 A 像素坐标。"""

    vector_array = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(vector_array, axis=1)
    valid = np.all(np.isfinite(vector_array), axis=1) & np.isfinite(norm) & (norm > 1e-12)
    normalized = np.full_like(vector_array, np.nan, dtype=np.float64)
    normalized[valid] = vector_array[valid] / norm[valid, None]
    right, up, forward = _target_transform_basis(target_icrs_to_pixel_payload)
    camera_vectors = np.column_stack((normalized @ right, normalized @ up, normalized @ forward))
    camera = _target_transform_camera(target_icrs_to_pixel_payload)
    full_pixels, projection_valid = target_camera_vectors_to_image_points(camera_vectors, camera)
    output_pixels = full_pixels.copy()
    output_pixels[:, 0] -= float(target_icrs_to_pixel_payload.get("crop_left_px", 0.0))
    output_pixels[:, 1] -= float(target_icrs_to_pixel_payload.get("crop_top_px", 0.0))
    valid &= projection_valid & np.all(np.isfinite(output_pixels), axis=1)
    output_pixels[~valid] = np.nan
    return output_pixels.astype(np.float64), valid.astype(bool)


def target_camera_vectors_to_image_points(
    camera_vectors: np.ndarray,
    camera: CameraSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """把目标相机坐标系方向投影到完整 A 边界像素。"""

    vectors = np.asarray(camera_vectors, dtype=np.float64)
    cam_x = vectors[:, 0]
    cam_y = vectors[:, 1]
    cam_z = vectors[:, 2]
    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        valid = np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z) & (cam_z > 1e-6)
        x_px = np.full_like(cam_z, np.nan, dtype=np.float64)
        y_px = np.full_like(cam_z, np.nan, dtype=np.float64)
        if np.any(valid):
            x_mm = camera.focal_length_mm * cam_x[valid] / cam_z[valid]
            y_mm = camera.focal_length_mm * cam_y[valid] / cam_z[valid]
            x_px[valid] = camera.image_width_px * 0.5 + (x_mm / camera.sensor_width_mm) * camera.image_width_px
            y_px[valid] = camera.image_height_px * 0.5 - (y_mm / camera.sensor_height_mm) * camera.image_height_px
    elif camera.lens_model in FISHEYE_LENS_MODELS:
        norm = np.linalg.norm(vectors, axis=1)
        unit_z = np.divide(cam_z, norm, out=np.full_like(cam_z, np.nan), where=norm > 1e-12)
        theta = np.arccos(np.clip(unit_z, -1.0, 1.0))
        theta_max = np.deg2rad(camera.fisheye_fov_deg * 0.5)
        rho = _fisheye_radius_ratio(theta, theta_max, camera.lens_model)
        r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
        r_px = r_limit * rho
        plane_norm = np.hypot(cam_x, cam_y)
        unit_x = np.divide(cam_x, plane_norm, out=np.zeros_like(cam_x), where=plane_norm > 1e-12)
        unit_y = np.divide(cam_y, plane_norm, out=np.zeros_like(cam_y), where=plane_norm > 1e-12)
        x_px = camera.image_width_px * 0.5 + unit_x * r_px
        y_px = camera.image_height_px * 0.5 - unit_y * r_px
        valid = (theta <= theta_max + 1e-9) & np.isfinite(x_px) & np.isfinite(y_px)
    elif camera.lens_model in CYLINDRICAL_LENS_MODELS:
        norm = np.linalg.norm(vectors, axis=1)
        unit_y = np.divide(cam_y, norm, out=np.zeros_like(cam_y), where=norm > 1e-12)
        longitude = np.arctan2(cam_x, cam_z)
        latitude = np.arcsin(np.clip(unit_y, -1.0, 1.0))
        scale_px = _projection_horizontal_scale_px(camera)
        if camera.lens_model == MERCATOR_LENS_MODEL:
            valid = np.isfinite(longitude) & np.isfinite(latitude) & (np.abs(latitude) < (np.pi * 0.5 - 1e-8))
            plane_y = np.arctanh(np.clip(np.sin(latitude), -1.0 + 1e-12, 1.0 - 1e-12))
        else:
            valid = np.isfinite(longitude) & np.isfinite(latitude)
            plane_y = latitude
        x_px = camera.image_width_px * 0.5 + scale_px * longitude
        y_px = camera.image_height_px * 0.5 - scale_px * plane_y
    else:
        raise ValueError(f"不支持的目标投影模型：{camera.lens_model}")
    pixels = np.column_stack((x_px, y_px)).astype(np.float64)
    valid &= np.all(np.isfinite(pixels), axis=1)
    pixels[~valid] = np.nan
    return pixels, valid.astype(bool)


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
    source_pixels, valid = _source_pixel_points_from_icrs_vectors(source_model, icrs_vectors)
    if np.any(valid):
        map_x[valid] = source_pixels[valid, 0].astype(np.float32)
        map_y[valid] = source_pixels[valid, 1].astype(np.float32)
    return map_x.reshape((rows, columns)), map_y.reshape((rows, columns))


def _source_pixel_points_from_icrs_vectors(source_model: object, icrs_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 ICRS 单位向量投到源图 B，返回像素点和有效掩码。"""

    vectors = np.asarray(icrs_vectors, dtype=np.float64)
    pixels = np.full((vectors.shape[0], 2), np.nan, dtype=np.float64)
    vector_valid = np.all(np.isfinite(vectors), axis=1)
    if not np.any(vector_valid):
        return pixels, np.zeros(vectors.shape[0], dtype=bool)

    projected = _source_pixels_from_icrs_vectors(source_model, vectors[vector_valid])
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
    vector_indices = np.flatnonzero(vector_valid)
    accepted = vector_indices[projected_valid]
    pixels[accepted] = projected[projected_valid]
    valid = np.zeros(vectors.shape[0], dtype=bool)
    valid[accepted] = True
    return pixels, valid


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
    target_icrs_to_pixel_payload: dict[str, object] | None = None,
    map_tile_size_px: int = 4,
    exact_remap_repair: bool = False,
) -> None:
    """把重投影结果写成无压缩 16-bit RGB TIFF。"""

    if tifffile is None:
        raise RuntimeError("当前环境缺少 tifffile，无法写入 16-bit TIFF。")
    if geometry.output_width_px <= 0 or geometry.output_height_px <= 0:
        raise ValueError("裁剪后的导出尺寸无效。")
    if target_icrs_to_pixel_payload is not None and not target_icrs_to_pixel_transform_payload_matches(
        target_icrs_to_pixel_payload,
        geometry=geometry,
    ):
        raise ValueError("取景 JSON 中的 ICRS 到 A 像素变换与当前输出几何不匹配。")
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
        target_icrs_to_pixel_payload=target_icrs_to_pixel_payload,
        map_tile_size_px=map_tile_size_px,
        exact_remap_repair=exact_remap_repair,
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
