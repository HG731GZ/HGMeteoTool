from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from PyQt5.QtWidgets import QApplication, QMainWindow

from meteoalign.application.app_lowfreq_gradient import LowFrequencyGradientMixin
from meteoalign.photometric.lowfreq.apply import apply_solution_to_frame
from meteoalign.photometric.lowfreq.bspline import (
    mean_field_weights,
    one_dimensional_basis,
    rasterize_correction,
    tensor_product_basis,
)
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
    assert page_layout.itemAt(1).widget().title() == "运行日志（Debug）"
    assert window.ui.plainTextEditLowFrequencyLog.isReadOnly()
    assert window.ui.pushButtonStartLowFrequencyCorrection.text() == "开始处理"
    window.close()
