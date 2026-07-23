from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .bspline import rasterize_correction
from .types import CancelCallback, PhotometricFrame, PhotometricSolution, ProgressCallback


def _tag_value(page: tifffile.TiffPage, name: str):
    tag = page.tags.get(name)
    return None if tag is None else tag.value


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
            software=software or "HGMeteoTool low-frequency correction V1",
        )
        try:
            for y_start in range(0, frame.height_px, max(1, int(block_rows))):
                if cancel_callback is not None and cancel_callback():
                    raise InterruptedError("用户已取消周边梯度优化。")
                y_stop = min(frame.height_px, y_start + max(1, int(block_rows)))
                correction = rasterize_correction(
                    solution.coefficients_rgb,
                    columns=solution.grid_columns,
                    rows=solution.grid_rows,
                    width_px=frame.width_px,
                    height_px=frame.height_px,
                    y_start_px=y_start,
                    y_stop_px=y_stop,
                )
                correction += solution.frame_offsets_rgb[frame.index][None, None, :]
                corrected = source[y_start:y_stop].astype(np.float32) - correction
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
