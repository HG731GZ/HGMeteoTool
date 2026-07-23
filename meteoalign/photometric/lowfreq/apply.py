from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .bspline import rasterize_correction
from .models import brightness_interpolation_indices, corrected_code_values
from .types import CancelCallback, PhotometricFrame, PhotometricSolution, ProgressCallback


def _tag_value(page: tifffile.TiffPage, name: str):
    tag = page.tags.get(name)
    return None if tag is None else tag.value


def rasterize_solution_parameter_block(
    source_block: np.ndarray,
    frame: PhotometricFrame,
    solution: PhotometricSolution,
    *,
    y_start: int,
    y_stop: int,
) -> np.ndarray:
    """Evaluate V3–V5 shared field, frame offset, and optional frame plane."""

    config = solution.solver_config
    coefficient_layers = solution.coefficient_layers_rgb
    layer_count = coefficient_layers.shape[1]
    parameter = np.zeros(source_block.shape, dtype=np.float32)
    if layer_count == 1:
        parameter[:] = rasterize_correction(
            coefficient_layers[:, 0, :],
            columns=solution.grid_columns,
            rows=solution.grid_rows,
            width_px=frame.width_px,
            height_px=frame.height_px,
            y_start_px=y_start,
            y_stop_px=y_stop,
        )
    else:
        lower, upper, fraction = brightness_interpolation_indices(
            source_block,
            config,
            solution.brightness_knot_values,
        )
        for layer in range(layer_count):
            field = rasterize_correction(
                coefficient_layers[:, layer, :],
                columns=solution.grid_columns,
                rows=solution.grid_rows,
                width_px=frame.width_px,
                height_px=frame.height_px,
                y_start_px=y_start,
                y_stop_px=y_stop,
            )
            weight = (
                (lower == layer) * (1.0 - fraction)
                + (upper == layer) * fraction
            )
            parameter += (field * weight).astype(np.float32)
    parameter += solution.frame_offsets_rgb[frame.index][None, None, :]
    if config.enable_frame_plane:
        gradient = solution.frame_gradient_values_rgb[frame.index].astype(np.float32)
        x_coordinate = np.linspace(-0.5, 0.5, frame.width_px, dtype=np.float32)
        y_all = np.linspace(-0.5, 0.5, frame.height_px, dtype=np.float32)
        parameter += x_coordinate[None, :, None] * gradient[:, 0][None, None, :]
        parameter += y_all[y_start:y_stop, None, None] * gradient[:, 1][None, None, :]
    return parameter


def apply_solution_to_frame(
    frame: PhotometricFrame,
    solution: PhotometricSolution,
    output_path: str | Path,
    *,
    block_rows: int = 256,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> Path:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"为保护已有结果，不会覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.tif")
    if temporary.exists():
        temporary.unlink()

    with tifffile.TiffFile(frame.image_path) as source_tiff:
        page = source_tiff.pages[0]
        if page.shape != (frame.height_px, frame.width_px, 3):
            raise ValueError(f"校正仅支持 H×W×3 RGB TIFF：{frame.image_path}")
        if page.dtype != np.dtype(np.uint16):
            raise ValueError(f"校正输出要求输入为 uint16 TIFF：{frame.image_path}")
        icc_profile = _tag_value(page, "InterColorProfile")
        x_resolution = _tag_value(page, "XResolution")
        y_resolution = _tag_value(page, "YResolution")
        resolution_unit = _tag_value(page, "ResolutionUnit")
        software = _tag_value(page, "Software")

    try:
        source = tifffile.memmap(frame.image_path, mode="r")
    except ValueError:
        # 压缩 TIFF 不能直接内存映射；V1 仍允许读取后按块写出。
        source = tifffile.imread(frame.image_path)
    try:
        output = tifffile.memmap(
            temporary,
            shape=source.shape,
            dtype=np.uint16,
            photometric="rgb",
            metadata=None,
            iccprofile=icc_profile,
            resolution=(
                x_resolution if x_resolution is not None else 72.0,
                y_resolution if y_resolution is not None else 72.0,
            ),
            resolutionunit=resolution_unit,
            software=software or "HGMeteoTool photometric correction V5",
        )
        try:
            for y_start in range(0, frame.height_px, max(1, int(block_rows))):
                if cancel_callback is not None and cancel_callback():
                    raise InterruptedError("用户已取消周边梯度优化。")
                y_stop = min(frame.height_px, y_start + max(1, int(block_rows)))
                source_block = source[y_start:y_stop].astype(np.float32)
                parameter = rasterize_solution_parameter_block(
                    source_block,
                    frame,
                    solution,
                    y_start=y_start,
                    y_stop=y_stop,
                )
                corrected = corrected_code_values(source_block, parameter, solution.solver_config)
                output[y_start:y_stop] = np.clip(np.rint(corrected), 0, 65535).astype(np.uint16)
                if progress_callback is not None:
                    progress_callback(
                        f"应用校正：{frame.image_path.name}",
                        y_stop,
                        frame.height_px,
                    )
            output.flush()
        finally:
            del output
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        del source
    return destination
