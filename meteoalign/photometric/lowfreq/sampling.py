from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...domain.settings import CameraSettings
from ...mosaic.export.target_transform import target_image_points_to_icrs_vectors


@dataclass(frozen=True)
class TargetSampleGrid:
    vectors_icrs: np.ndarray
    output_points_xy: np.ndarray
    valid: np.ndarray
    sample_width: int
    sample_height: int
    output_width_px: int
    output_height_px: int


def load_framing_transform(path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    framing_path = Path(path)
    payload = json.loads(framing_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("取景 JSON 根对象必须是对象。")
    transform = payload.get("target_icrs_to_pixel_transform")
    if not isinstance(transform, dict):
        raise ValueError("取景 JSON 缺少 target_icrs_to_pixel_transform。")
    if str(transform.get("type") or "") != "icrs_to_cropped_output_pixel":
        raise ValueError("取景 JSON 的目标变换类型不受支持。")
    return payload, transform


def build_target_sample_grid(
    framing_path: str | Path,
    *,
    long_side_px: int,
) -> TargetSampleGrid:
    _payload, transform = load_framing_transform(framing_path)
    output_width = int(transform.get("output_width_px", 0) or 0)
    output_height = int(transform.get("output_height_px", 0) or 0)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("取景 JSON 的输出尺寸无效。")
    scale = float(long_side_px) / max(output_width, output_height)
    sample_width = max(2, int(round(output_width * scale)))
    sample_height = max(2, int(round(output_height * scale)))
    x_output = np.linspace(0.5, output_width - 0.5, sample_width, dtype=np.float64)
    y_output = np.linspace(0.5, output_height - 0.5, sample_height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_output, y_output)

    camera_payload = transform.get("camera")
    basis_payload = transform.get("icrs_camera_basis")
    if not isinstance(camera_payload, dict) or not isinstance(basis_payload, dict):
        raise ValueError("取景 JSON 缺少 camera 或 icrs_camera_basis。")
    camera = CameraSettings(
        sensor_width_mm=float(camera_payload.get("sensor_width_mm", 36.0)),
        sensor_height_mm=float(camera_payload.get("sensor_height_mm", 24.0)),
        image_width_px=int(camera_payload.get("image_width_px", 0)),
        image_height_px=int(camera_payload.get("image_height_px", 0)),
        focal_length_mm=float(camera_payload.get("focal_length_mm", 24.0)),
        lens_model=str(camera_payload.get("lens_model", "")),
        fisheye_fov_deg=float(camera_payload.get("fisheye_fov_deg", 180.0)),
    )
    basis = tuple(
        np.asarray(basis_payload.get(name), dtype=np.float64)
        for name in ("right", "up", "forward")
    )
    if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in basis):
        raise ValueError("取景 JSON 的 ICRS 相机基向量无效。")

    full_x = grid_x.ravel() + float(transform.get("crop_left_px", 0.0))
    full_y = grid_y.ravel() + float(transform.get("crop_top_px", 0.0))
    vectors, valid = target_image_points_to_icrs_vectors(
        full_x,
        full_y,
        camera=camera,
        icrs_basis=basis,  # type: ignore[arg-type]
    )
    return TargetSampleGrid(
        vectors_icrs=vectors,
        output_points_xy=np.column_stack((grid_x.ravel(), grid_y.ravel())),
        valid=valid,
        sample_width=sample_width,
        sample_height=sample_height,
        output_width_px=output_width,
        output_height_px=output_height,
    )

