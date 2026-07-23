from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tifffile

from ...frame_astrometry import FrameAstrometricModel
from .apply import apply_solution_to_frame
from .diagnostics import export_diagnostics
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
_RANGE_MASK_PATTERN = re.compile(r"^mask_(\d+)_(\d+)$", re.IGNORECASE)


def _frame_number(path: Path) -> int | None:
    match = _FRAME_NUMBER_PATTERN.search(path.stem.replace("_model", ""))
    return None if match is None else int(match.group(1))


def _same_stem_tiff(directory: Path, stem: str) -> Path | None:
    stem_key = stem.casefold()
    candidates = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() in {".tif", ".tiff"}
            and path.stem.casefold() == stem_key
        ),
        key=lambda path: (path.suffix.casefold() != ".tif", path.name.casefold()),
    )
    return candidates[0] if candidates else None


def _source_stem(payload: dict[str, object], model_path: Path) -> str:
    source_image = payload.get("source_image")
    if isinstance(source_image, dict):
        for key in ("file_stem", "file_name", "relative_path", "path"):
            value = source_image.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value.strip().replace("\\", "/")).stem
    stem = model_path.stem
    return stem[:-6] if stem.casefold().endswith("_model") else stem


def _discover_mask(directory: Path, image_path: Path, frame_number: int | None) -> Path | None:
    companion = _same_stem_tiff(directory, f"{image_path.stem}_Mask")
    if companion is not None:
        return companion
    if frame_number is None:
        return None
    matches: list[tuple[int, Path]] = []
    for candidate in directory.iterdir():
        if not candidate.is_file() or candidate.suffix.casefold() not in {".tif", ".tiff"}:
            continue
        range_match = _RANGE_MASK_PATTERN.match(candidate.stem)
        if range_match is None:
            continue
        start, stop = (int(value) for value in range_match.groups())
        if start <= frame_number <= stop:
            matches.append((stop - start, candidate))
    if not matches:
        return None
    return min(matches, key=lambda item: (item[0], item[1].name.casefold()))[1]


def discover_photometric_frames(input_directory: str | Path) -> tuple[PhotometricFrame, ...]:
    directory = Path(input_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"输入目录不存在：{directory}")
    model_paths = sorted(
        directory.glob("*_model.json"),
        key=lambda path: (_frame_number(path) is None, _frame_number(path) or 0, path.name.casefold()),
    )
    if len(model_paths) < 2:
        raise ValueError("输入目录至少需要两个 *_model.json。")
    frames: list[PhotometricFrame] = []
    for model_path in model_paths:
        raw_bytes = model_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"模型 JSON 根对象必须是对象：{model_path.name}")
        model = FrameAstrometricModel.from_json_payload(payload)
        image_stem = _source_stem(payload, model_path)
        image_path = _same_stem_tiff(directory, image_stem)
        if image_path is None:
            raise FileNotFoundError(f"找不到模型对应的 TIFF：{model_path.name} → {image_stem}")
        frame_number = _frame_number(image_path)
        frames.append(
            PhotometricFrame(
                index=len(frames),
                model_path=model_path,
                image_path=image_path,
                mask_path=_discover_mask(directory, image_path, frame_number),
                model=model,
                source_payload=payload,
                model_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            )
        )
    return tuple(frames)


def validate_low_frequency_inputs(
    input_directory: str | Path,
) -> tuple[PhotometricFrame, ...]:
    frames = discover_photometric_frames(input_directory)
    expected_size = (frames[0].width_px, frames[0].height_px)
    for frame in frames:
        if (frame.width_px, frame.height_px) != expected_size:
            raise ValueError("周边梯度优化要求所有源图模型具有相同尺寸。")
        with tifffile.TiffFile(frame.image_path) as tiff:
            if not tiff.pages:
                raise ValueError(f"TIFF 没有图像页面：{frame.image_path.name}")
            page = tiff.pages[0]
            actual_size = (int(page.imagewidth), int(page.imagelength))
            if actual_size != expected_size:
                raise ValueError(
                    f"源图尺寸与模型不一致：{frame.image_path.name} "
                    f"{actual_size[0]}×{actual_size[1]} vs {expected_size[0]}×{expected_size[1]}"
                )
            if page.dtype != np.dtype(np.uint16) or page.samplesperpixel != 3:
                raise ValueError(
                    f"周边梯度优化只接受 16-bit RGB TIFF：{frame.image_path.name}"
                )
        if frame.mask_path is not None:
            with tifffile.TiffFile(frame.mask_path) as mask_tiff:
                mask_page = mask_tiff.pages[0]
                mask_size = (int(mask_page.imagewidth), int(mask_page.imagelength))
                if mask_size != expected_size:
                    raise ValueError(
                        f"蒙版尺寸与源图不一致：{frame.mask_path.name} "
                        f"{mask_size[0]}×{mask_size[1]}"
                    )
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
    input_directory: str | Path,
    output_directory: str | Path,
    config: SolverConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> LowFrequencyRunResult:
    solver_config = (config or SolverConfig()).validated()
    output_root = Path(output_directory).expanduser().resolve()
    solution_path = output_root / "photometric_solution.json"
    diagnostics_directory = output_root / "diagnostics"
    corrected_directory = output_root / "corrected"
    protected_outputs = [solution_path, diagnostics_directory]
    if solver_config.apply_correction:
        protected_outputs.append(corrected_directory)
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "输出位置已有本功能的结果；为避免覆盖，请选择空目录："
            + "、".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback("验证 TIFF、模型和蒙版…", 0, 1)
    frames = validate_low_frequency_inputs(input_directory)
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
            "V6 model.json 直采样："
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
            progress_callback(f"V5 自适应对数亮度节点：[{knot_text}]", 1, 1)

    corrected_paths: list[Path] = []
    if solver_config.apply_correction:
        corrected_directory.mkdir(parents=True, exist_ok=False)
        for frame_position, frame in enumerate(frames):
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError("用户已取消周边梯度优化。")
            destination = corrected_directory / f"{frame.image_path.stem}_lowfreq_corrected.tif"
            corrected_paths.append(
                apply_solution_to_frame(
                    frame,
                    solution,
                    destination,
                    block_rows=solver_config.output_block_rows,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
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
