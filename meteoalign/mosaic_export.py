from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image, ImageOps

from .coordinates import unit_vectors_to_radec
from .sequence_geometry import icrs_to_enu_rotation_matrix
from .simulator import (
    CameraSettings,
    ObserverSettings,
    ViewSettings,
    camera_basis_from_view,
    image_points_to_local_vectors,
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
MOSAIC_EXPORT_DEFAULT_BLOCK_ROWS = 32
MOSAIC_EXPORT_WHITE_U16 = 65535

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
) -> Iterator[np.ndarray]:
    """逐块生成裁剪后目标图像的 RGB uint16 像素。"""

    if cv2 is None:
        raise RuntimeError("当前环境缺少 OpenCV，无法执行重投影导出。")
    source = np.ascontiguousarray(source_rgb_u16, dtype=np.uint16)
    basis = camera_basis_from_view(view)
    enu_to_icrs = icrs_to_enu_rotation_matrix(observer)
    output_width = int(geometry.output_width_px)
    output_height = int(geometry.output_height_px)
    safe_block_rows = max(1, int(block_rows))
    full_x = (geometry.crop_left_px + np.arange(output_width, dtype=np.float64)).astype(np.float64)
    completed_rows = 0

    for row_start in range(0, output_height, safe_block_rows):
        rows = min(safe_block_rows, output_height - row_start)
        full_y = geometry.crop_top_px + row_start + np.arange(rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(full_x, full_y)
        block = _render_mosaic_reprojection_block(
            source_model=source_model,
            source_rgb_u16=source,
            x_px=grid_x.ravel(),
            y_px=grid_y.ravel(),
            rows=rows,
            columns=output_width,
            camera=camera,
            basis=basis,
            enu_to_icrs=enu_to_icrs,
        )
        completed_rows += rows
        if progress_callback is not None:
            progress_callback(completed_rows)
        yield block


def _render_mosaic_reprojection_block(
    *,
    source_model: object,
    source_rgb_u16: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    rows: int,
    columns: int,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    enu_to_icrs: np.ndarray,
) -> np.ndarray:
    vectors, valid_projection = image_points_to_local_vectors(x_px, y_px, camera, basis)
    map_x = np.full(x_px.shape, -1.0, dtype=np.float32)
    map_y = np.full(y_px.shape, -1.0, dtype=np.float32)
    valid = valid_projection & np.all(np.isfinite(vectors), axis=1)
    if np.any(valid):
        icrs_vectors = vectors[valid] @ enu_to_icrs
        radec = unit_vectors_to_radec(icrs_vectors)
        source_pixels = np.asarray(source_model.sky_to_pixel_points(radec), dtype=np.float64)
        mapped_valid = (
            np.all(np.isfinite(source_pixels), axis=1)
            & (source_pixels[:, 0] >= -0.5)
            & (source_pixels[:, 0] <= source_rgb_u16.shape[1] - 0.5)
            & (source_pixels[:, 1] >= -0.5)
            & (source_pixels[:, 1] <= source_rgb_u16.shape[0] - 0.5)
        )
        valid_indices = np.flatnonzero(valid)
        accepted = valid_indices[mapped_valid]
        map_x[accepted] = source_pixels[mapped_valid, 0].astype(np.float32)
        map_y[accepted] = source_pixels[mapped_valid, 1].astype(np.float32)

    remapped = cv2.remap(
        source_rgb_u16,
        map_x.reshape((rows, columns)),
        map_y.reshape((rows, columns)),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(MOSAIC_EXPORT_WHITE_U16, MOSAIC_EXPORT_WHITE_U16, MOSAIC_EXPORT_WHITE_U16),
    )
    if remapped.ndim == 2:
        remapped = np.repeat(remapped[:, :, None], 3, axis=2)
    return np.ascontiguousarray(remapped, dtype=np.uint16)


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
) -> None:
    """把重投影结果写成无压缩 16-bit RGB TIFF。"""

    if tifffile is None:
        raise RuntimeError("当前环境缺少 tifffile，无法写入 16-bit TIFF。")
    if geometry.output_width_px <= 0 or geometry.output_height_px <= 0:
        raise ValueError("裁剪后的导出尺寸无效。")
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
