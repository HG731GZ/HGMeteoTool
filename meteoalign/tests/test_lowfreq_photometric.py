from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from PyQt5.QtWidgets import QApplication, QMainWindow

import meteoalign.application.app_lowfreq_gradient as app_lowfreq_gradient
from meteoalign.application.app_lowfreq_gradient import LowFrequencyGradientMixin
from meteoalign.photometric.lowfreq.apply import (
    apply_solution_to_frame,
    rasterize_solution_parameter_block,
)
from meteoalign.photometric.lowfreq.bspline import (
    mean_field_weights,
    one_dimensional_basis,
    rasterize_correction,
    tensor_product_basis,
)
from meteoalign.photometric.lowfreq.models import (
    brightness_basis,
    derive_brightness_knots,
)
from meteoalign.photometric.lowfreq.sampling import build_model_coverage_sample_grid
from meteoalign.photometric.lowfreq.solver import solve_photometric_model
from meteoalign.photometric.lowfreq.solution_io import read_solution, write_solution
from meteoalign.photometric.lowfreq.types import (
    DiagnosticsResult,
    ObservationSet,
    PhotometricFrame,
    PhotometricSolution,
    SolverConfig,
)
from meteoalign.ui.ui_main_window import Ui_MainWindow


def _empty_diagnostics(coefficient_count: int) -> DiagnosticsResult:
    return DiagnosticsResult(
        observation_count=0,
        frame_observation_counts=np.zeros(1, dtype=np.int64),
        overlap_observation_counts={},
        rms_before_rgb=np.zeros(3),
        rms_after_rgb=np.zeros(3),
        mad_before_rgb=np.zeros(3),
        mad_after_rgb=np.zeros(3),
        correction_min_rgb=np.zeros(3),
        correction_max_rgb=np.zeros(3),
        observability=np.zeros((1, coefficient_count)),
        per_frame_rms_before_rgb=np.zeros((1, 3)),
        per_frame_rms_after_rgb=np.zeros((1, 3)),
        per_overlap_rms_before_rgb={},
        per_overlap_rms_after_rgb={},
        residual_before_rgb=np.empty((0, 3)),
        residual_after_rgb=np.empty((0, 3)),
    )


def test_cubic_bspline_basis_partitions_unity_and_rasterizes_constant() -> None:
    values = np.linspace(0.0, 1.0, 101)
    basis = one_dimensional_basis(values, 8)
    assert np.allclose(np.sum(basis, axis=1), 1.0)
    points = np.column_stack((values, values[::-1]))
    tensor = tensor_product_basis(points, columns=8, rows=6)
    assert np.allclose(np.asarray(tensor.sum(axis=1)).ravel(), 1.0)

    field = rasterize_correction(
        np.full(8 * 6, 123.5),
        columns=8,
        rows=6,
        width_px=73,
        height_px=41,
    )
    assert field.shape == (41, 73)
    assert np.allclose(field, 123.5, atol=1e-4)


def test_sparse_solver_recovers_synthetic_overlap_differences() -> None:
    rng = np.random.default_rng(20260723)
    config = SolverConfig(
        grid_columns=6,
        grid_rows=4,
        smooth_lambda=0.0,
        frame_offset_lambda=0.0,
        robust_loss="none",
        irls_iterations=1,
        apply_correction=False,
    )
    coefficient_count = config.grid_columns * config.grid_rows
    frame_count = 5
    observation_count = 2500
    point_i = rng.uniform(0.0, 1.0, (observation_count, 2))
    point_j = rng.uniform(0.0, 1.0, (observation_count, 2))
    frame_i = rng.integers(0, frame_count, observation_count)
    frame_j = (frame_i + rng.integers(1, frame_count, observation_count)) % frame_count
    true_coefficients = rng.normal(0.0, 300.0, (3, coefficient_count))
    field_mean = mean_field_weights(config.grid_columns, config.grid_rows)
    true_coefficients -= (true_coefficients @ field_mean)[:, None]
    true_offsets = rng.normal(0.0, 50.0, (frame_count, 3))
    true_offsets -= np.mean(true_offsets, axis=0, keepdims=True)
    basis_i = tensor_product_basis(
        point_i,
        columns=config.grid_columns,
        rows=config.grid_rows,
    )
    basis_j = tensor_product_basis(
        point_j,
        columns=config.grid_columns,
        rows=config.grid_rows,
    )
    difference = np.column_stack(
        [
            (basis_i - basis_j) @ true_coefficients[channel]
            + true_offsets[frame_i, channel]
            - true_offsets[frame_j, channel]
            for channel in range(3)
        ]
    )
    observations = ObservationSet(
        frame_i=frame_i.astype(np.int32),
        frame_j=frame_j.astype(np.int32),
        point_i_xy=point_i * np.array([999.0, 799.0]),
        point_j_xy=point_j * np.array([999.0, 799.0]),
        difference_rgb=difference,
        target_sample_index=np.arange(observation_count),
        frame_count=frame_count,
        image_width_px=1000,
        image_height_px=800,
    )
    solution = solve_photometric_model(
        observations,
        frame_records=tuple({"index": index} for index in range(frame_count)),
        config=config,
    )
    assert np.max(solution.diagnostics.rms_after_rgb) < 1e-4
    assert np.allclose(solution.coefficients_rgb, true_coefficients, atol=2e-3)
    assert np.allclose(solution.frame_offsets_rgb, true_offsets, atol=2e-3)


def test_v3_log_gain_and_v4_frame_planes_recover_synthetic_data() -> None:
    rng = np.random.default_rng(30504)
    config = SolverConfig(
        grid_columns=6,
        grid_rows=4,
        correction_model="log_gain",
        smooth_lambda=0.0,
        frame_offset_lambda=0.0,
        enable_frame_plane=True,
        frame_plane_lambda=0.0,
        robust_loss="none",
        apply_correction=False,
    )
    coefficient_count = 24
    frame_count = 5
    count = 3500
    point_i = rng.uniform(0.0, 1.0, (count, 2))
    point_j = rng.uniform(0.0, 1.0, (count, 2))
    frame_i = rng.integers(0, frame_count, count)
    frame_j = (frame_i + rng.integers(1, frame_count, count)) % frame_count
    coefficients = rng.normal(0.0, 0.025, (3, coefficient_count))
    field_mean = mean_field_weights(6, 4)
    coefficients -= (coefficients @ field_mean)[:, None]
    offsets = rng.normal(0.0, 0.008, (frame_count, 3))
    offsets -= np.mean(offsets, axis=0, keepdims=True)
    gradients = rng.normal(0.0, 0.01, (frame_count, 3, 2))
    gradients -= np.mean(gradients, axis=0, keepdims=True)
    basis_i = tensor_product_basis(point_i, columns=6, rows=4)
    basis_j = tensor_product_basis(point_j, columns=6, rows=4)
    parameter_i = np.column_stack(
        [
            basis_i @ coefficients[channel]
            + offsets[frame_i, channel]
            + gradients[frame_i, channel, 0] * (point_i[:, 0] - 0.5)
            + gradients[frame_i, channel, 1] * (point_i[:, 1] - 0.5)
            for channel in range(3)
        ]
    )
    parameter_j = np.column_stack(
        [
            basis_j @ coefficients[channel]
            + offsets[frame_j, channel]
            + gradients[frame_j, channel, 0] * (point_j[:, 0] - 0.5)
            + gradients[frame_j, channel, 1] * (point_j[:, 1] - 0.5)
            for channel in range(3)
        ]
    )
    sky = rng.uniform(500.0, 50000.0, (count, 3))
    floor = config.intensity_floor_code
    measurement_i = (sky + floor) * np.exp(parameter_i) - floor
    measurement_j = (sky + floor) * np.exp(parameter_j) - floor
    observations = ObservationSet(
        frame_i=frame_i.astype(np.int32),
        frame_j=frame_j.astype(np.int32),
        point_i_xy=point_i * np.array([999.0, 799.0]),
        point_j_xy=point_j * np.array([999.0, 799.0]),
        difference_rgb=measurement_i - measurement_j,
        target_sample_index=np.arange(count),
        frame_count=frame_count,
        image_width_px=1000,
        image_height_px=800,
        measurement_i_rgb=measurement_i,
        measurement_j_rgb=measurement_j,
    )
    solution = solve_photometric_model(
        observations,
        frame_records=tuple({"index": index} for index in range(frame_count)),
        config=config,
    )

    assert np.max(solution.diagnostics.rms_after_rgb) < 0.02
    assert np.allclose(solution.coefficients_rgb, coefficients, atol=2e-5)
    assert np.allclose(solution.frame_offsets_rgb, offsets, atol=2e-5)
    assert np.allclose(solution.frame_gradient_values_rgb, gradients, atol=2e-5)


def test_v5_brightness_layers_recover_nonlinear_additive_field() -> None:
    rng = np.random.default_rng(50505)
    config = SolverConfig(
        grid_columns=6,
        grid_rows=4,
        correction_model="additive",
        smooth_lambda=0.0,
        frame_offset_lambda=0.0,
        enable_brightness_nonlinearity=True,
        brightness_knot_count=4,
        brightness_smooth_lambda=0.0,
        robust_loss="none",
        apply_correction=False,
    )
    count = 5000
    point_i = rng.uniform(0.0, 1.0, (count, 2))
    point_j = rng.uniform(0.0, 1.0, (count, 2))
    frame_i = rng.integers(0, 3, count)
    frame_j = (frame_i + rng.integers(1, 3, count)) % 3
    coefficients = rng.normal(0.0, 5.0, (3, 4, 24))
    field_mean = mean_field_weights(6, 4)
    coefficients -= np.einsum("ckn,n->ck", coefficients, field_mean)[:, :, None]
    basis_i = tensor_product_basis(point_i, columns=6, rows=4)
    basis_j = tensor_product_basis(point_j, columns=6, rows=4)
    field_i = np.stack(
        [
            np.column_stack(
                [basis_i @ coefficients[channel, layer] for layer in range(4)]
            )
            for channel in range(3)
        ],
        axis=1,
    )
    field_j = np.stack(
        [
            np.column_stack(
                [basis_j @ coefficients[channel, layer] for layer in range(4)]
            )
            for channel in range(3)
        ],
        axis=1,
    )
    log_coordinate = rng.uniform(0.08, 0.98, (count, 3))
    floor = config.intensity_floor_code
    sky = floor * (
        np.exp(log_coordinate * np.log1p(65535.0 / floor))
        - 1.0
    )
    preliminary = ObservationSet(
        frame_i=frame_i.astype(np.int32),
        frame_j=frame_j.astype(np.int32),
        point_i_xy=point_i * np.array([999.0, 799.0]),
        point_j_xy=point_j * np.array([999.0, 799.0]),
        difference_rgb=np.zeros((count, 3)),
        target_sample_index=np.arange(count),
        frame_count=3,
        image_width_px=1000,
        image_height_px=800,
        measurement_i_rgb=sky,
        measurement_j_rgb=sky,
    )
    adaptive_knots = derive_brightness_knots(preliminary, config)

    def fixed_point_measurement(field_values: np.ndarray) -> np.ndarray:
        measurement = sky.copy()
        for _iteration in range(12):
            for channel in range(3):
                weights = brightness_basis(
                    measurement[:, channel],
                    config,
                    adaptive_knots,
                )
                measurement[:, channel] = sky[:, channel] + np.sum(
                    weights * field_values[:, channel, :],
                    axis=1,
                )
        return measurement

    measurement_i = fixed_point_measurement(field_i)
    measurement_j = fixed_point_measurement(field_j)
    observations = ObservationSet(
        frame_i=frame_i.astype(np.int32),
        frame_j=frame_j.astype(np.int32),
        point_i_xy=point_i * np.array([999.0, 799.0]),
        point_j_xy=point_j * np.array([999.0, 799.0]),
        difference_rgb=measurement_i - measurement_j,
        target_sample_index=np.arange(count),
        frame_count=3,
        image_width_px=1000,
        image_height_px=800,
        measurement_i_rgb=measurement_i,
        measurement_j_rgb=measurement_j,
    )
    solution = solve_photometric_model(
        observations,
        frame_records=tuple({"index": index} for index in range(3)),
        config=config,
    )

    assert solution.coefficient_layers_rgb.shape == (3, 4, 24)
    assert solution.diagnostics.observability.shape == (4, 4, 6)
    assert np.max(solution.diagnostics.rms_after_rgb) < 0.02
    assert np.allclose(solution.coefficient_layers_rgb, coefficients, atol=0.08)


def test_apply_solution_writes_uint16_rgb_without_touching_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source.tif"
    output_path = tmp_path / "corrected.tif"
    source = np.full((12, 16, 3), 1000, dtype=np.uint16)
    tifffile.imwrite(source_path, source, photometric="rgb", metadata=None)
    model = SimpleNamespace(image_width_px=16, image_height_px=12)
    frame = PhotometricFrame(
        index=0,
        model_path=tmp_path / "source_model.json",
        image_path=source_path,
        mask_path=None,
        model=model,  # type: ignore[arg-type]
        source_payload={},
    )
    coefficients = np.vstack(
        (
            np.full(24, 100.0),
            np.full(24, 200.0),
            np.full(24, 300.0),
        )
    )
    config = SolverConfig(
        grid_columns=6,
        grid_rows=4,
        apply_correction=True,
        output_block_rows=5,
    )
    solution = PhotometricSolution(
        grid_columns=6,
        grid_rows=4,
        coefficients_rgb=coefficients,
        frame_offsets_rgb=np.array([[10.0, 20.0, 30.0]]),
        frame_records=({"index": 0},),
        solver_config=config,
        diagnostics=_empty_diagnostics(24),
    )
    apply_solution_to_frame(frame, solution, output_path, block_rows=5)
    assert np.array_equal(tifffile.imread(source_path), source)
    corrected = tifffile.imread(output_path)
    assert corrected.dtype == np.uint16
    assert corrected.shape == source.shape
    assert np.all(corrected == np.array([890, 780, 670], dtype=np.uint16))

    replacement_solution = PhotometricSolution(
        grid_columns=6,
        grid_rows=4,
        coefficients_rgb=np.zeros((3, 24), dtype=np.float64),
        frame_offsets_rgb=np.zeros((1, 3), dtype=np.float64),
        frame_records=({"index": 0},),
        solver_config=config,
        diagnostics=_empty_diagnostics(24),
    )
    apply_solution_to_frame(
        frame,
        replacement_solution,
        output_path,
        block_rows=5,
        overwrite=True,
    )
    assert np.array_equal(tifffile.imread(output_path), source)


def test_v3_multiplicative_and_v5_layered_application_are_pixelwise() -> None:
    source = np.array(
        [
            [[1000.0, 5000.0, 10000.0], [20000.0, 40000.0, 60000.0]],
            [[3000.0, 8000.0, 16000.0], [24000.0, 48000.0, 65000.0]],
        ],
        dtype=np.float32,
    )
    model = SimpleNamespace(image_width_px=2, image_height_px=2)
    frame = PhotometricFrame(
        index=0,
        model_path=Path("model.json"),
        image_path=Path("source.tif"),
        mask_path=None,
        model=model,  # type: ignore[arg-type]
        source_payload={},
    )
    coefficients = np.zeros((3, 3, 24), dtype=np.float64)
    coefficients[:, 0, :] = 0.05
    coefficients[:, 1, :] = 0.15
    coefficients[:, 2, :] = 0.30
    config = SolverConfig(
        grid_columns=6,
        grid_rows=4,
        correction_model="multiplicative",
        enable_brightness_nonlinearity=True,
        brightness_knot_count=3,
    )
    solution = PhotometricSolution(
        grid_columns=6,
        grid_rows=4,
        coefficients_rgb=coefficients,
        frame_offsets_rgb=np.zeros((1, 3)),
        frame_records=({"index": 0},),
        solver_config=config,
        diagnostics=_empty_diagnostics(24),
        brightness_knots=np.linspace(0.0, 1.0, 3),
    )
    parameter = rasterize_solution_parameter_block(
        source,
        frame,
        solution,
        y_start=0,
        y_stop=2,
    )
    corrected = source / (1.0 + parameter)

    assert np.all(parameter[1, 1] > parameter[0, 0])
    assert np.all(corrected < source)


def test_solution_json_round_trip_keeps_model_and_offsets(tmp_path: Path) -> None:
    coefficients = np.arange(72, dtype=np.float64).reshape(3, 24)
    offsets = np.arange(9, dtype=np.float64).reshape(3, 3)
    solution = PhotometricSolution(
        grid_columns=6,
        grid_rows=4,
        coefficients_rgb=coefficients,
        frame_offsets_rgb=offsets,
        frame_records=tuple({"index": index, "source_image": f"{index}.tif"} for index in range(3)),
        solver_config=SolverConfig(grid_columns=6, grid_rows=4),
        diagnostics=DiagnosticsResult(
            observation_count=123,
            frame_observation_counts=np.array([80, 90, 76]),
            overlap_observation_counts={"0-1": 45, "1-2": 40},
            rms_before_rgb=np.array([10.0, 11.0, 12.0]),
            rms_after_rgb=np.array([2.0, 2.5, 3.0]),
            mad_before_rgb=np.array([5.0, 6.0, 7.0]),
            mad_after_rgb=np.array([1.0, 1.2, 1.4]),
            correction_min_rgb=np.array([-20.0, -30.0, -40.0]),
            correction_max_rgb=np.array([20.0, 30.0, 40.0]),
            observability=np.arange(24, dtype=np.float64).reshape(4, 6),
            per_frame_rms_before_rgb=np.ones((3, 3)),
            per_frame_rms_after_rgb=np.full((3, 3), 0.25),
            per_overlap_rms_before_rgb={"0-1": [10.0, 11.0, 12.0]},
            per_overlap_rms_after_rgb={"0-1": [2.0, 2.5, 3.0]},
            residual_before_rgb=np.empty((0, 3)),
            residual_after_rgb=np.empty((0, 3)),
        ),
    )
    path = write_solution(tmp_path / "photometric_solution.json", solution)
    restored = read_solution(path)

    assert np.array_equal(restored.coefficients_rgb, coefficients)
    assert np.array_equal(restored.frame_offsets_rgb, offsets)
    assert restored.diagnostics.observation_count == 123
    assert restored.diagnostics.observability.shape == (4, 6)
    assert restored.frame_records[2]["source_image"] == "2.tif"


def test_v5_solution_round_trip_keeps_layers_planes_and_model(tmp_path: Path) -> None:
    config = SolverConfig(
        grid_columns=6,
        grid_rows=4,
        correction_model="log_gain",
        enable_frame_plane=True,
        enable_brightness_nonlinearity=True,
        brightness_knot_count=4,
    )
    solution = PhotometricSolution(
        grid_columns=6,
        grid_rows=4,
        coefficients_rgb=np.arange(288, dtype=np.float64).reshape(3, 4, 24),
        frame_offsets_rgb=np.arange(6, dtype=np.float64).reshape(2, 3),
        frame_records=({"index": 0}, {"index": 1}),
        solver_config=config,
        diagnostics=_empty_diagnostics(24),
        frame_gradients_rgb=np.arange(12, dtype=np.float64).reshape(2, 3, 2),
        brightness_knots=np.linspace(0.0, 1.0, 4),
    )
    restored = read_solution(
        write_solution(tmp_path / "v5_solution.json", solution)
    )

    assert restored.solver_config.correction_model == "log_gain"
    assert restored.solver_config.enable_frame_plane
    assert restored.solver_config.enable_brightness_nonlinearity
    assert np.array_equal(restored.coefficient_layers_rgb, solution.coefficients_rgb)
    assert np.array_equal(
        restored.frame_gradient_values_rgb,
        solution.frame_gradients_rgb,
    )


def test_version_one_solution_remains_readable(tmp_path: Path) -> None:
    solution = PhotometricSolution(
        grid_columns=6,
        grid_rows=4,
        coefficients_rgb=np.arange(72, dtype=np.float64).reshape(3, 24),
        frame_offsets_rgb=np.zeros((2, 3)),
        frame_records=({"index": 0}, {"index": 1}),
        solver_config=SolverConfig(grid_columns=6, grid_rows=4),
        diagnostics=_empty_diagnostics(24),
    )
    path = write_solution(tmp_path / "legacy.json", solution)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 1
    payload["model_type"] = "additive_bspline"
    payload["grid"].pop("brightness_knots_log_coordinate", None)
    for channel in payload["channels"].values():
        channel.pop("coefficient_layers", None)
    for frame in payload["frames"]:
        frame.pop("gradient_x_rgb", None)
        frame.pop("gradient_y_rgb", None)
    for key in (
        "correction_model",
        "enable_frame_plane",
        "enable_brightness_nonlinearity",
        "brightness_knot_count",
    ):
        payload["solver"].pop(key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = read_solution(path)
    assert restored.solver_config.correction_model == "additive"
    assert restored.coefficient_layers_rgb.shape == (3, 1, 24)
    assert np.all(restored.frame_gradient_values_rgb == 0.0)


def test_v6_builds_overlap_samples_directly_from_frame_models() -> None:
    class LinearSkyModel:
        image_width_px = 120
        image_height_px = 80

        def __init__(self, ra_start_deg: float) -> None:
            self.ra_start_deg = ra_start_deg

        def pixel_to_sky_points(self, pixels: np.ndarray) -> np.ndarray:
            values = np.asarray(pixels, dtype=np.float64)
            return np.column_stack(
                (
                    self.ra_start_deg + values[:, 0] / 100.0,
                    -0.4 + values[:, 1] / 100.0,
                )
            )

        def icrs_vectors_to_pixel_points(self, vectors: np.ndarray) -> np.ndarray:
            values = np.asarray(vectors, dtype=np.float64)
            ra = np.rad2deg(np.arctan2(values[:, 1], values[:, 0])) % 360.0
            dec = np.rad2deg(np.arcsin(np.clip(values[:, 2], -1.0, 1.0)))
            return np.column_stack(
                (
                    (ra - self.ra_start_deg) * 100.0,
                    (dec + 0.4) * 100.0,
                )
            )

    frames = tuple(
        PhotometricFrame(
            index=index,
            model_path=Path(f"frame_{index}_model.json"),
            image_path=Path(f"frame_{index}.tif"),
            mask_path=None,
            model=LinearSkyModel(0.5 * index),  # type: ignore[arg-type]
            source_payload={},
        )
        for index in range(3)
    )
    grid = build_model_coverage_sample_grid(frames, long_side_px=90)

    assert grid.sampling_mode == "model_icrs_coverage"
    assert grid.source_frame_count == 3
    assert grid.sample_count > 0
    assert grid.sample_count < grid.candidate_count
    coverage_count = np.zeros(grid.sample_count, dtype=np.int8)
    for frame in frames:
        pixels = frame.model.icrs_vectors_to_pixel_points(grid.vectors_icrs)
        coverage_count += (
            np.all(np.isfinite(pixels), axis=1)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= frame.width_px - 1.0)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= frame.height_px - 1.0)
        )
    assert np.all(coverage_count >= 2)


def test_low_frequency_image_import_filters_masks_missing_models_and_invalid_tiffs(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    app = QApplication.instance() or QApplication([])

    class Window(QMainWindow, LowFrequencyGradientMixin):
        pass

    window = Window()
    window.ui = Ui_MainWindow()
    window.ui.setupUi(window)
    window._init_low_frequency_gradient_page()
    paths = [
        tmp_path / "good.tif",
        tmp_path / "missing_model.tif",
        tmp_path / "invalid_depth.tif",
        tmp_path / "good_Mask.tif",
    ]
    for path in paths:
        path.touch()

    def fake_frame(image_path, *, index, **_kwargs):  # type: ignore[no-untyped-def]
        path = Path(image_path)
        if path.name == "missing_model.tif":
            raise FileNotFoundError("未找到同路径同名模型")
        return PhotometricFrame(
            index=index,
            model_path=tmp_path / f"{path.stem}_model.json",
            image_path=path,
            mask_path=None,
            model=SimpleNamespace(image_width_px=10, image_height_px=8),  # type: ignore[arg-type]
            source_payload={},
        )

    def fake_validate(frame, *, expected_size=None):  # type: ignore[no-untyped-def]
        if frame.image_path.name == "invalid_depth.tif":
            raise ValueError("只接受 16-bit RGB TIFF")

    monkeypatch.setattr(
        app_lowfreq_gradient,
        "photometric_frame_from_image",
        fake_frame,
    )
    monkeypatch.setattr(
        app_lowfreq_gradient,
        "validate_photometric_frame",
        fake_validate,
    )

    imported_count, issues = window._add_low_frequency_image_paths(
        [str(path) for path in paths]
    )
    window._refresh_low_frequency_image_table()

    assert imported_count == 1
    assert [frame.image_path.name for frame in window._low_frequency_frames] == [
        "good.tif"
    ]
    assert len(issues) == 3
    assert window.ui.tableWidgetLowFrequencyImages.item(0, 1).text() == "good.tif"
    window.close()


def test_low_frequency_page_has_controls_on_left_and_debug_log_on_right() -> None:
    app = QApplication.instance() or QApplication([])

    class Window(QMainWindow, LowFrequencyGradientMixin):
        pass

    window = Window()
    window.ui = Ui_MainWindow()
    window.ui.setupUi(window)
    window._init_low_frequency_gradient_page()
    page_layout = window.ui.horizontalLayoutLowFrequencyGradient

    page_index = window.ui.tabWidgetMain.indexOf(window.ui.tabLowFrequencyGradient)
    assert window.ui.tabWidgetMain.tabText(page_index) == "周边梯度优化"
    assert page_layout.itemAt(0).widget() is window.ui.scrollAreaLowFrequencyControls
    assert page_layout.itemAt(1).widget() is not None
    assert window.ui.plainTextEditLowFrequencyLog.isReadOnly()
    assert window.ui.tableWidgetLowFrequencyImages.columnCount() == 3
    assert [
        window.ui.tableWidgetLowFrequencyImages.horizontalHeaderItem(index).text()
        for index in range(3)
    ] == ["序号", "文件名", "蒙版"]
    assert window.ui.pushButtonStartLowFrequencyCorrection.text() == "开始处理"
    assert window.ui.pushButtonLowFrequencyTuningGuide.text() == "调参指南"
    assert not hasattr(window.ui, "lineEditLowFrequencyInputDirectory")
    assert not hasattr(window.ui, "lineEditLowFrequencyOutputDirectory")
    assert not window.ui.checkBoxLowFrequencyExportCorrected.isChecked()
    assert window.ui.comboBoxLowFrequencyCorrectionModel.count() == 3
    assert "V1" not in window.ui.comboBoxLowFrequencyCorrectionModel.itemText(0)
    assert not window.ui.doubleSpinBoxLowFrequencyFramePlaneLambda.isEnabled()
    assert not window.ui.spinBoxLowFrequencyBrightnessKnots.isEnabled()
    window._show_low_frequency_tuning_guide()
    assert window._low_frequency_guide_dialog is not None
    guide_text = window._low_frequency_guide_dialog.guideBrowser.toPlainText()
    assert "V1" not in guide_text
    assert "V5" not in guide_text
    assert "对数增益" in guide_text
    assert "correction field" in guide_text
    assert (
        window.ui.tabWidgetMain.widget(window.ui.tabWidgetMain.count() - 1)
        is window.ui.tabMosaicBatch
    )
    window.close()
