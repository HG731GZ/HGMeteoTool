"""自由投影拼图导出的输出边界与裁剪几何。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MosaicExportGeometry:
    """完整边界与裁剪后的实际导出区域。"""

    boundary_width_px: int
    boundary_height_px: int
    crop_left_px: int
    crop_top_px: int
    output_width_px: int
    output_height_px: int


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
    """将任意裁剪输入规整为不超过剩余边界的整数像素。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    return max(0, min(int(round(numeric)), max(0, int(maximum))))
