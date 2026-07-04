from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_ALIGNMENT_PAIRS = 4
QUADRATIC_ALIGNMENT_PAIRS = 6
MAX_ALIGNMENT_CONDITION_NUMBER = 1e10
MAX_TRANSFORM_ABS_PX = 1e9


@dataclass(frozen=True)
class ReferenceAlignmentTransform:
    degree: int
    pair_count: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    coeff_x: np.ndarray
    coeff_y: np.ndarray
    rms_px: float

    @property
    def display_name(self) -> str:
        if self.degree >= 2:
            return "二阶多项式"
        return "一次仿射"

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, 2)
        if point_array.shape[1] != 2:
            raise ValueError("参考图配准点必须是 Nx2 数组。")

        if not _alignment_coefficients_are_valid(self.coeff_x, self.coeff_y):
            return np.full((point_array.shape[0], 2), np.nan, dtype=np.float64)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            terms = _polynomial_terms(point_array[:, 0], point_array[:, 1], self.degree)
            x_values = terms @ self.coeff_x
            y_values = terms @ self.coeff_y
            transformed = np.column_stack((x_values, y_values)).astype(np.float64)
        transformed[~np.all(np.isfinite(transformed), axis=1)] = np.nan
        return transformed

    def transform_point(self, x_value: float, y_value: float) -> tuple[float, float]:
        transformed = self.transform_points(np.asarray([[x_value, y_value]], dtype=np.float64))[0]
        return float(transformed[0]), float(transformed[1])


@dataclass(frozen=True)
class SkyAlignmentTransform:
    degree: int
    pair_count: int
    center_vector: np.ndarray
    east_vector: np.ndarray
    north_vector: np.ndarray
    coeff_x: np.ndarray
    coeff_y: np.ndarray
    rms_px: float

    @property
    def display_name(self) -> str:
        if self.degree >= 2:
            return "二阶多项式"
        return "一次仿射"

    def transform_radec_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
        if ra_dec_array.ndim == 1:
            ra_dec_array = ra_dec_array.reshape(1, 2)
        if ra_dec_array.shape[1] != 2:
            raise ValueError("天球配准点必须是 Nx2 的 RA/Dec 数组。")

        if not _sky_transform_components_are_valid(self):
            return np.full((ra_dec_array.shape[0], 2), np.nan, dtype=np.float64)

        sky_points = _project_radec_to_sky_plane(
            ra_dec_array[:, 0],
            ra_dec_array[:, 1],
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            terms = _polynomial_terms(sky_points[:, 0], sky_points[:, 1], self.degree)
            x_values = terms @ self.coeff_x
            y_values = terms @ self.coeff_y
            transformed = np.column_stack((x_values, y_values)).astype(np.float64)
        transformed[~np.all(np.isfinite(transformed), axis=1)] = np.nan
        return transformed

    def transform_radec(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        transformed = self.transform_radec_points(np.asarray([[ra_deg, dec_deg]], dtype=np.float64))[0]
        return float(transformed[0]), float(transformed[1])


def _polynomial_terms(x_values: np.ndarray, y_values: np.ndarray, degree: int) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if degree >= 2:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.column_stack(
                (
                    np.ones_like(x_values),
                    x_values,
                    y_values,
                    x_values * x_values,
                    x_values * y_values,
                    y_values * y_values,
                )
            )
    return np.column_stack((np.ones_like(x_values), x_values, y_values))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("无法归一化零长度向量。")
    return vector / norm


def _array_is_finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def _alignment_coefficients_are_valid(coeff_x: np.ndarray, coeff_y: np.ndarray) -> bool:
    return _array_is_finite(coeff_x) and _array_is_finite(coeff_y)


def _sky_transform_components_are_valid(transform: SkyAlignmentTransform) -> bool:
    return (
        _array_is_finite(transform.center_vector)
        and _array_is_finite(transform.east_vector)
        and _array_is_finite(transform.north_vector)
        and _alignment_coefficients_are_valid(transform.coeff_x, transform.coeff_y)
    )


def _radec_to_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra_rad = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec_rad = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec_rad)
    return np.column_stack((cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad))).astype(np.float64)


def _sky_plane_basis(ra_dec_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _radec_to_unit_vectors(ra_dec_points[:, 0], ra_dec_points[:, 1])
    center = vectors.mean(axis=0)
    if float(np.linalg.norm(center)) <= 1e-8:
        center = vectors[0]
    center = _normalize_vector(center)

    celestial_north = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(celestial_north, center)
    if float(np.linalg.norm(east)) <= 1e-8:
        east = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    east = _normalize_vector(east)
    north = _normalize_vector(np.cross(center, east))
    return center, east, north


def _project_radec_to_sky_plane(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_vector: np.ndarray,
    east_vector: np.ndarray,
    north_vector: np.ndarray,
) -> np.ndarray:
    ra_array = np.asarray(ra_deg, dtype=np.float64)
    dec_array = np.asarray(dec_deg, dtype=np.float64)
    if not (_array_is_finite(center_vector) and _array_is_finite(east_vector) and _array_is_finite(north_vector)):
        return np.full((ra_array.size, 2), np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        vectors = _radec_to_unit_vectors(ra_array, dec_array)
        center_component = np.clip(vectors @ center_vector, -1.0, 1.0)
        east_component = vectors @ east_vector
        north_component = vectors @ north_vector
        theta = np.arccos(center_component)
        sin_theta = np.sin(theta)
        scale = np.divide(theta, sin_theta, out=np.ones_like(theta), where=np.abs(sin_theta) > 1e-12)
        radians_to_degrees = 180.0 / np.pi
        projected = np.column_stack(
            (east_component * scale * radians_to_degrees, north_component * scale * radians_to_degrees)
        )
    projected[~np.all(np.isfinite(projected), axis=1)] = np.nan
    return projected


def _fit_polynomial_coefficients(
    source_points: np.ndarray,
    target_values: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, int, int]:
    design = _polynomial_terms(source_points[:, 0], source_points[:, 1], degree)
    if not _array_is_finite(design) or not _array_is_finite(target_values):
        raise ValueError("配准点包含无效数值，无法稳定求解配准。")
    try:
        condition_number = float(np.linalg.cond(design))
    except np.linalg.LinAlgError as exc:
        raise ValueError("配准矩阵无法稳定求解，请检查配对星位置。") from exc
    if not np.isfinite(condition_number) or condition_number > MAX_ALIGNMENT_CONDITION_NUMBER:
        raise ValueError("配对星几何分布过于集中，无法稳定求解配准。")
    try:
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(design, target_values, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("配准矩阵无法稳定求解，请检查配对星位置。") from exc
    if not _array_is_finite(coefficients):
        raise ValueError("配准结果包含无效系数，请检查配对星位置。")
    return coefficients.astype(np.float64), int(rank), int(design.shape[1])


def _fit_image_polynomial(
    source_points: np.ndarray,
    target_points: np.ndarray,
    preferred_degree: int,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    candidate_degrees = (preferred_degree, 1) if preferred_degree > 1 else (1,)
    last_error: ValueError | None = None
    for degree in candidate_degrees:
        try:
            coeff_x, rank_x, term_count = _fit_polynomial_coefficients(source_points, target_points[:, 0], degree)
            coeff_y, rank_y, _term_count_y = _fit_polynomial_coefficients(source_points, target_points[:, 1], degree)
            if rank_x < term_count or rank_y < term_count:
                raise ValueError("配对星几何分布过于集中，无法稳定求解配准。")
        except ValueError as exc:
            last_error = exc
            continue

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            predicted = _polynomial_terms(source_points[:, 0], source_points[:, 1], degree) @ np.column_stack(
                (coeff_x, coeff_y)
            )
        if not _array_is_finite(predicted) or float(np.nanmax(np.abs(predicted))) > MAX_TRANSFORM_ABS_PX:
            last_error = ValueError("配准结果数值过大，请检查配对星是否匹配正确。")
            continue
        residual = predicted - target_points
        rms_px = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if not np.isfinite(rms_px):
            last_error = ValueError("配准残差包含无效数值，请检查配对星位置。")
            continue
        return degree, coeff_x, coeff_y, rms_px

    if last_error is not None:
        raise last_error
    raise ValueError("无法计算配准。")


def fit_reference_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> ReferenceAlignmentTransform:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("参考图配准需要 Nx2 的源点与目标点。")
    if source.shape[0] != target.shape[0]:
        raise ValueError("参考图配准的源点与目标点数量不一致。")

    finite_mask = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source = source[finite_mask]
    target = target[finite_mask]
    pair_count = int(source.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对参考星才能计算自动配准。")

    preferred_degree = 2 if pair_count >= QUADRATIC_ALIGNMENT_PAIRS else 1
    degree, coeff_x, coeff_y, rms_px = _fit_image_polynomial(source, target, preferred_degree)
    return ReferenceAlignmentTransform(
        degree=degree,
        pair_count=pair_count,
        source_width=int(source_size[0]),
        source_height=int(source_size[1]),
        target_width=int(target_size[0]),
        target_height=int(target_size[1]),
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        rms_px=rms_px,
    )


def fit_sky_alignment(
    ra_dec_points: np.ndarray,
    target_points: np.ndarray,
) -> SkyAlignmentTransform:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if sky_radec.ndim != 2 or target.ndim != 2 or sky_radec.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("天球配准需要 Nx2 的 RA/Dec 点与目标点。")
    if sky_radec.shape[0] != target.shape[0]:
        raise ValueError("天球配准的 RA/Dec 点与目标点数量不一致。")

    finite_mask = np.all(np.isfinite(sky_radec), axis=1) & np.all(np.isfinite(target), axis=1)
    sky_radec = sky_radec[finite_mask]
    target = target[finite_mask]
    pair_count = int(sky_radec.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对参考星才能计算天球残差。")

    center, east, north = _sky_plane_basis(sky_radec)
    sky_points = _project_radec_to_sky_plane(sky_radec[:, 0], sky_radec[:, 1], center, east, north)
    if not _array_is_finite(sky_points):
        raise ValueError("天球投影包含无效数值，请检查配对星坐标。")
    preferred_degree = 2 if pair_count >= QUADRATIC_ALIGNMENT_PAIRS else 1
    degree, coeff_x, coeff_y, rms_px = _fit_image_polynomial(sky_points, target, preferred_degree)
    return SkyAlignmentTransform(
        degree=degree,
        pair_count=pair_count,
        center_vector=center,
        east_vector=east,
        north_vector=north,
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        rms_px=rms_px,
    )
