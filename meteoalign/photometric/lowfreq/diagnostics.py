from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from .bspline import rasterize_correction
from .types import ObservationSet, PhotometricSolution


def _json_safe_array(values: np.ndarray) -> list[object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return [None if not np.isfinite(value) else float(value) for value in array]
    return [_json_safe_array(row) for row in array]


def export_diagnostics(
    output_directory: str | Path,
    solution: PhotometricSolution,
    observations: ObservationSet,
) -> Path:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "correction_*.tif",
        "observability*.tif",
        "overlap_residuals.csv",
        "residual_before.csv",
        "residual_after.csv",
        "summary.json",
    ):
        for existing_path in directory.glob(pattern):
            if existing_path.is_file():
                existing_path.unlink()
    coefficient_layers = solution.coefficient_layers_rgb
    preview_layers = np.stack(
        [
            rasterize_correction(
                coefficient_layers[:, layer, :],
                columns=solution.grid_columns,
                rows=solution.grid_rows,
                width_px=512,
                height_px=512,
            )
            for layer in range(coefficient_layers.shape[1])
        ],
        axis=0,
    )
    representative_layer = coefficient_layers.shape[1] // 2
    for channel, channel_name in enumerate(("R", "G", "B")):
        tifffile.imwrite(
            directory / f"correction_{channel_name}.tif",
            preview_layers[representative_layer, :, :, channel].astype(np.float32),
            photometric="minisblack",
            metadata=None,
        )
        if coefficient_layers.shape[1] > 1:
            for layer in range(coefficient_layers.shape[1]):
                tifffile.imwrite(
                    directory / f"correction_{channel_name}_k{layer:02d}.tif",
                    preview_layers[layer, :, :, channel].astype(np.float32),
                    photometric="minisblack",
                    metadata=None,
                )
    tifffile.imwrite(
        directory / "observability.tif",
        solution.diagnostics.observability.astype(np.float32),
        photometric="minisblack",
        metadata=None,
    )
    if solution.diagnostics.observability.ndim == 3:
        for layer, layer_map in enumerate(solution.diagnostics.observability):
            tifffile.imwrite(
                directory / f"observability_k{layer:02d}.tif",
                layer_map.astype(np.float32),
                photometric="minisblack",
                metadata=None,
            )

    csv_path = directory / "overlap_residuals.csv"
    before = solution.diagnostics.residual_before_rgb
    after = solution.diagnostics.residual_after_rgb
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            (
                "frame_i",
                "frame_j",
                "target_sample_index",
                "before_R",
                "before_G",
                "before_B",
                "after_R",
                "after_G",
                "after_B",
            )
        )
        for index in range(observations.count):
            writer.writerow(
                (
                    int(observations.frame_i[index]),
                    int(observations.frame_j[index]),
                    int(observations.target_sample_index[index]),
                    *[float(value) for value in before[index]],
                    *[float(value) for value in after[index]],
                )
            )
    for file_name, residual in (
        ("residual_before.csv", before),
        ("residual_after.csv", after),
    ):
        with (directory / file_name).open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                (
                    "frame_i",
                    "frame_j",
                    "target_sample_index",
                    "residual_R",
                    "residual_G",
                    "residual_B",
                )
            )
            for index in range(observations.count):
                writer.writerow(
                    (
                        int(observations.frame_i[index]),
                        int(observations.frame_j[index]),
                        int(observations.target_sample_index[index]),
                        *[float(value) for value in residual[index]],
                    )
                )
    summary = {
        "observation_count": solution.diagnostics.observation_count,
        "rms_before_rgb": solution.diagnostics.rms_before_rgb.tolist(),
        "rms_after_rgb": solution.diagnostics.rms_after_rgb.tolist(),
        "mad_before_rgb": solution.diagnostics.mad_before_rgb.tolist(),
        "mad_after_rgb": solution.diagnostics.mad_after_rgb.tolist(),
        "correction_min_rgb": solution.diagnostics.correction_min_rgb.tolist(),
        "correction_max_rgb": solution.diagnostics.correction_max_rgb.tolist(),
        "frame_observation_counts": solution.diagnostics.frame_observation_counts.tolist(),
        "overlap_observation_counts": solution.diagnostics.overlap_observation_counts,
        "frame_offsets_rgb": solution.frame_offsets_rgb.tolist(),
        "frame_gradients_rgb": solution.frame_gradient_values_rgb.tolist(),
        "correction_model": solution.solver_config.correction_model,
        "brightness_knots_log_coordinate": solution.brightness_knot_values.tolist(),
        "fit_domain": solution.diagnostics.fit_domain,
        "fit_rms_before_rgb": solution.diagnostics.fit_rms_before_rgb.tolist(),
        "fit_rms_after_rgb": solution.diagnostics.fit_rms_after_rgb.tolist(),
        "brightness_bin_edges_code": solution.diagnostics.brightness_bin_edges_code.tolist(),
        "brightness_bin_counts_rgb": solution.diagnostics.brightness_bin_counts_rgb.tolist(),
        "brightness_bin_rms_before_rgb": _json_safe_array(
            solution.diagnostics.brightness_bin_rms_before_rgb
        ),
        "brightness_bin_rms_after_rgb": _json_safe_array(
            solution.diagnostics.brightness_bin_rms_after_rgb
        ),
        "per_frame_rms_before_rgb": solution.diagnostics.per_frame_rms_before_rgb.tolist(),
        "per_frame_rms_after_rgb": solution.diagnostics.per_frame_rms_after_rgb.tolist(),
        "per_overlap_rms_before_rgb": solution.diagnostics.per_overlap_rms_before_rgb,
        "per_overlap_rms_after_rgb": solution.diagnostics.per_overlap_rms_after_rgb,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return directory
