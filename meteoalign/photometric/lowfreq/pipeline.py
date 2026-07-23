from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tifffile

from ...frame_astrometry import FrameAstrometricModel
from ...image_path_resolution import companion_sky_mask_path, is_reserved_mask_path
from .apply import apply_solution_to_frame
from .diagnostics import export_diagnostics
from .masking import read_binary_mask
from .observations import generate_observations
from .sampling import build_model_coverage_sample_grid
from .solution_io import write_solution
from .solver import solve_photometric_model
from .types import (
    CancelCallback,
    LowFrequencyRunResult,
    PhotometricFrame,
    ProgressCallback,
    SolverConfig,
)


_FRAME_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _frame_number(path: Path) -> int | None:
    match = _FRAME_NUMBER_PATTERN.search(path.stem.replace("_model", ""))
    return None if match is None else int(match.group(1))


def _same_stem_json(directory: Path, stem: str) -> Path | None:
    stem_key = stem.casefold()
    candidates = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".json"
            and path.stem.casefold() == stem_key
        ),
        key=lambda path: path.name.casefold(),
    )
    return candidates[0] if candidates else None


def photometric_frame_from_image(
    image_path: str | Path,
    *,
    index: int,
    mask_path: str | Path | None = None,
    auto_mask: bool = True,
) -> PhotometricFrame:
    """Load one selected TIFF and its sibling ``*_model.json``."""

    source_path = Path(image_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"图像不存在：{source_path}")
    if source_path.suffix.casefold() not in {".tif", ".tiff"}:
        raise ValueError(f"只支持 16-bit RGB TIFF：{source_path.name}")
    if is_reserved_mask_path(source_path):
        raise ValueError(f"蒙版不能作为原图导入：{source_path.name}")
    model_path = _same_stem_json(source_path.parent, f"{source_path.stem}_model")
    if model_path is None:
        raise FileNotFoundError(f"未找到同路径同名模型：{source_path.stem}_model.json")
    raw_bytes = model_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"模型 JSON 根对象必须是对象：{model_path.name}")
    model = FrameAstrometricModel.from_json_payload(payload)
    resolved_mask = (
        Path(mask_path).expanduser().resolve()
        if mask_path is not None
        else companion_sky_mask_path(source_path) if auto_mask else None
    )
    return PhotometricFrame(
        index=int(index),
        model_path=model_path,
        image_path=source_path,
        mask_path=resolved_mask,
        model=model,
        source_payload=payload,
        model_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def discover_photometric_frames(image_directory: str | Path) -> tuple[PhotometricFrame, ...]:
    directory = Path(image_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"图像目录不存在：{directory}")
    image_paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() in {".tif", ".tiff"}
            and not is_reserved_mask_path(path)
        ),
        key=lambda path: (
            _frame_number(path) is None,
            _frame_number(path) or 0,
            path.name.casefold(),
        ),
    )
    frames: list[PhotometricFrame] = []
    for image_path in image_paths:
        try:
            frame = photometric_frame_from_image(image_path, index=len(frames))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        frames.append(frame)
    if len(frames) < 2:
        raise ValueError("至少需要两张带同名 model.json 的 16-bit RGB TIFF。")
    return tuple(frames)


def validate_photometric_frame(
    frame: PhotometricFrame,
    *,
    expected_size: tuple[int, int] | None = None,
) -> None:
    model_size = (frame.width_px, frame.height_px)
    if expected_size is not None and model_size != expected_size:
        raise ValueError(
            "图像模型尺寸与已导入图像不一致："
            f"{frame.image_path.name} {model_size[0]}×{model_size[1]}"
        )
    with tifffile.TiffFile(frame.image_path) as tiff:
        if not tiff.pages:
            raise ValueError(f"TIFF 没有图像页面：{frame.image_path.name}")
        page = tiff.pages[0]
        actual_size = (int(page.imagewidth), int(page.imagelength))
        if actual_size != model_size:
            raise ValueError(
                f"源图尺寸与模型不一致：{frame.image_path.name} "
                f"{actual_size[0]}×{actual_size[1]} vs {model_size[0]}×{model_size[1]}"
            )
        if page.dtype != np.dtype(np.uint16) or page.samplesperpixel != 3:
            raise ValueError(
                f"只接受 16-bit RGB TIFF：{frame.image_path.name}"
            )
    if frame.mask_path is not None:
        mask = read_binary_mask(frame.mask_path)
        mask_size = (int(mask.shape[1]), int(mask.shape[0]))
        if mask_size != model_size:
            raise ValueError(
                f"蒙版尺寸与源图不一致：{frame.mask_path.name} "
                f"{mask_size[0]}×{mask_size[1]}"
            )
        if not np.any(mask):
            raise ValueError(f"蒙版中没有非零有效区域：{frame.mask_path.name}")


def validate_low_frequency_inputs(
    frames: tuple[PhotometricFrame, ...],
) -> tuple[PhotometricFrame, ...]:
    if len(frames) < 2:
        raise ValueError("至少需要两张有效图像。")
    if tuple(frame.index for frame in frames) != tuple(range(len(frames))):
        raise ValueError("图像序号必须从 0 连续排列。")
    expected_size = (frames[0].width_px, frames[0].height_px)
    for frame in frames:
        validate_photometric_frame(frame, expected_size=expected_size)
    return frames


def _frame_records(frames: tuple[PhotometricFrame, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "index": frame.index,
            "source_model": str(frame.model_path),
            "source_model_sha256": frame.model_sha256,
            "source_image": str(frame.image_path),
            "source_image_size_bytes": int(frame.image_path.stat().st_size),
            "source_image_mtime_ns": int(frame.image_path.stat().st_mtime_ns),
            "mask": None if frame.mask_path is None else str(frame.mask_path),
        }
        for frame in frames
    )


def run_low_frequency_correction(
    *,
    frames: tuple[PhotometricFrame, ...],
    config: SolverConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> LowFrequencyRunResult:
    solver_config = (config or SolverConfig()).validated()
    if not frames:
        raise ValueError("尚未导入图像。")
    output_root = frames[0].image_path.parent
    solution_path = output_root / "photometric_solution.json"
    diagnostics_directory = output_root / "gradient_diagnostics"
    corrected_directory = output_root / "gradient_corrected"

    if progress_callback is not None:
        progress_callback("验证 TIFF、模型和蒙版…", 0, 1)
    frames = validate_low_frequency_inputs(frames)
    if cancel_callback is not None and cancel_callback():
        raise InterruptedError("用户已取消周边梯度优化。")
    if progress_callback is not None:
        masked_count = sum(frame.mask_path is not None for frame in frames)
        progress_callback(
            f"输入验证完成：{len(frames)} 帧，{masked_count} 帧绑定外部天空蒙版。",
            1,
            1,
        )

    sample_grid = build_model_coverage_sample_grid(
        frames,
        long_side_px=solver_config.sample_long_side_px,
    )
    if progress_callback is not None:
        progress_callback(
            "model.json ICRS 覆盖采样："
            f"每帧 {sample_grid.sample_width}×{sample_grid.sample_height}，"
            f"{sample_grid.candidate_count:,} 个候选中保留 "
            f"{sample_grid.sample_count:,} 个跨帧 ICRS 方向。",
            1,
            1,
        )
    observations = generate_observations(
        frames,
        sample_grid,
        config=solver_config,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    solution = solve_photometric_model(
        observations,
        frame_records=_frame_records(frames),
        config=solver_config,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    write_solution(solution_path, solution)
    export_diagnostics(diagnostics_directory, solution, observations)
    if progress_callback is not None:
        before = ", ".join(f"{value:.2f}" for value in solution.diagnostics.rms_before_rgb)
        after = ", ".join(f"{value:.2f}" for value in solution.diagnostics.rms_after_rgb)
        progress_callback(f"重叠残差 RMS RGB：[{before}] → [{after}]", 1, 1)
        fit_before = ", ".join(
            f"{value:.6g}" for value in solution.diagnostics.fit_rms_before_rgb
        )
        fit_after = ", ".join(
            f"{value:.6g}" for value in solution.diagnostics.fit_rms_after_rgb
        )
        progress_callback(
            f"拟合域 RMS（{solution.diagnostics.fit_domain}）："
            f"[{fit_before}] → [{fit_after}]",
            1,
            1,
        )
        if solver_config.enable_brightness_nonlinearity:
            knot_text = ", ".join(
                f"{value:.4f}" for value in solution.brightness_knot_values
            )
            progress_callback(f"自适应对数亮度节点：[{knot_text}]", 1, 1)

    corrected_paths: list[Path] = []
    if solver_config.apply_correction:
        corrected_directory.mkdir(parents=True, exist_ok=True)
        for frame_position, frame in enumerate(frames):
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError("用户已取消周边梯度优化。")
            destination = corrected_directory / frame.image_path.name
            corrected_paths.append(
                apply_solution_to_frame(
                    frame,
                    solution,
                    destination,
                    block_rows=solver_config.output_block_rows,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                    overwrite=True,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    f"已输出 {destination.name}",
                    frame_position + 1,
                    len(frames),
                )
    return LowFrequencyRunResult(
        solution_path=solution_path,
        diagnostics_directory=diagnostics_directory,
        corrected_paths=tuple(corrected_paths),
        solution=solution,
    )
