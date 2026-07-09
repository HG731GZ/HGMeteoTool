from __future__ import annotations

import numpy as np

import meteoalign.alignment.interpolation as interpolation_module
from meteoalign.alignment.interpolation import fit_anchor_interpolation


def test_anchor_interpolation_evaluate_points_reuses_tps_kernel_for_xy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = np.asarray(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 80.0],
            [100.0, 80.0],
            [45.0, 30.0],
        ],
        dtype=np.float64,
    )
    target = np.column_stack(
        (
            source[:, 0] + 0.03 * source[:, 1] + 2.0,
            source[:, 1] - 0.02 * source[:, 0] - 1.5,
        )
    )
    model = fit_anchor_interpolation(source, target)
    probe = np.asarray(
        [
            [12.0, 9.0],
            [50.0, 40.0],
            [92.0, 70.0],
        ],
        dtype=np.float64,
    )
    normalized = model.normalized_points(probe)
    expected = np.column_stack(
        (
            interpolation_module._evaluate_thin_plate_spline(
                normalized,
                model.anchor_points,
                model.tps_weights_x,
                model.tps_affine_x,
            ),
            interpolation_module._evaluate_thin_plate_spline(
                normalized,
                model.anchor_points,
                model.tps_weights_y,
                model.tps_affine_y,
            ),
        )
    )
    original_kernel = interpolation_module._thin_plate_spline_kernel
    kernel_calls = 0

    def counted_kernel(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        nonlocal kernel_calls
        kernel_calls += 1
        return original_kernel(points, anchors)

    monkeypatch.setattr(interpolation_module, "_thin_plate_spline_kernel", counted_kernel)

    actual = model.evaluate_points(probe)

    assert kernel_calls == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
