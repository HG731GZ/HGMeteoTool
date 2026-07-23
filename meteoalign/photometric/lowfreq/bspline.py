from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline


BSPLINE_DEGREE = 3


def open_uniform_knots(control_count: int, degree: int = BSPLINE_DEGREE) -> np.ndarray:
    if control_count <= degree:
        raise ValueError("控制点数量必须大于 B-spline 次数。")
    interior_count = control_count - degree - 1
    interior = (
        np.linspace(0.0, 1.0, interior_count + 2, dtype=np.float64)[1:-1]
        if interior_count > 0
        else np.empty(0, dtype=np.float64)
    )
    return np.concatenate(
        (
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        )
    )


def one_dimensional_basis(values: np.ndarray, control_count: int) -> np.ndarray:
    coordinates = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    knots = open_uniform_knots(control_count)
    return BSpline.design_matrix(
        coordinates,
        knots,
        BSPLINE_DEGREE,
        extrapolate=False,
    ).toarray()


def tensor_product_basis(
    points_xy: np.ndarray,
    *,
    columns: int,
    rows: int,
) -> sparse.csr_matrix:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("B-spline 采样点必须是 N×2 数组。")
    basis_x = one_dimensional_basis(points[:, 0], columns)
    basis_y = one_dimensional_basis(points[:, 1], rows)
    dense = np.einsum("ni,nj->nij", basis_y, basis_x, optimize=True)
    return sparse.csr_matrix(dense.reshape(points.shape[0], rows * columns))


def smoothness_regularization(columns: int, rows: int) -> sparse.csr_matrix:
    """Second differences along both source-image axes."""

    coefficient_count = rows * columns
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    equation = 0

    def append_difference(indices: tuple[int, int, int]) -> None:
        nonlocal equation
        for coefficient_index, value in zip(indices, (1.0, -2.0, 1.0)):
            row_indices.append(equation)
            column_indices.append(coefficient_index)
            values.append(value)
        equation += 1

    for y_index in range(rows):
        for x_index in range(1, columns - 1):
            base = y_index * columns
            append_difference((base + x_index - 1, base + x_index, base + x_index + 1))
    for y_index in range(1, rows - 1):
        for x_index in range(columns):
            append_difference(
                (
                    (y_index - 1) * columns + x_index,
                    y_index * columns + x_index,
                    (y_index + 1) * columns + x_index,
                )
            )
    return sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(equation, coefficient_count),
    )


def mean_field_weights(columns: int, rows: int) -> np.ndarray:
    """Numerically integrate the B-spline field for the mean(V)=0 gauge."""

    sample_x = np.linspace(0.0, 1.0, max(32, columns * 8), dtype=np.float64)
    sample_y = np.linspace(0.0, 1.0, max(32, rows * 8), dtype=np.float64)
    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
    basis = tensor_product_basis(
        np.column_stack((grid_x.ravel(), grid_y.ravel())),
        columns=columns,
        rows=rows,
    )
    return np.asarray(basis.mean(axis=0), dtype=np.float64).ravel()


def rasterize_correction(
    coefficients: np.ndarray,
    *,
    columns: int,
    rows: int,
    width_px: int,
    height_px: int,
    y_start_px: int = 0,
    y_stop_px: int | None = None,
) -> np.ndarray:
    """Rasterize one or three coefficient channels in source-image coordinates."""

    stop = height_px if y_stop_px is None else int(y_stop_px)
    start = int(y_start_px)
    if start < 0 or stop > height_px or stop <= start:
        raise ValueError("无效的 B-spline 栅格化行范围。")
    x_values = np.linspace(0.0, 1.0, width_px, dtype=np.float64)
    y_all = np.linspace(0.0, 1.0, height_px, dtype=np.float64)
    basis_x = one_dimensional_basis(x_values, columns)
    basis_y = one_dimensional_basis(y_all[start:stop], rows)
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if coefficient_array.ndim == 1:
        control_grid = coefficient_array.reshape(rows, columns)
        return (basis_y @ control_grid @ basis_x.T).astype(np.float32)
    if coefficient_array.shape != (3, rows * columns):
        raise ValueError("RGB B-spline 系数尺寸无效。")
    fields = [
        basis_y @ coefficient_array[channel].reshape(rows, columns) @ basis_x.T
        for channel in range(3)
    ]
    return np.stack(fields, axis=-1).astype(np.float32)

