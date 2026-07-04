from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .alignment import MAX_ALIGNMENT_CONDITION_NUMBER, MIN_ALIGNMENT_PAIRS
from .coordinates import project_radec_to_sky_plane, sky_plane_basis, sky_plane_to_radec, unit_vectors_to_radec


SOURCE_MODEL_FORMAT = "meteoalign_source_astrometric_model"
SOURCE_MODEL_VERSION = 1


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def _finite_point_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=bool)
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.all(np.isfinite(array), axis=1)
    return mask


def polynomial_terms(x_values: np.ndarray, y_values: np.ndarray, degree: int) -> np.ndarray:
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


def evaluate_polynomial(points: np.ndarray, coeff: np.ndarray, degree: int) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim == 1:
        point_array = point_array.reshape(1, 2)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("多项式求值需要 Nx2 点数组。")

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        values = polynomial_terms(point_array[:, 0], point_array[:, 1], degree) @ np.asarray(coeff, dtype=np.float64)
    values = values.astype(np.float64)
    values[~np.isfinite(values)] = np.nan
    return values


def _fit_polynomial_coefficients(source_points: np.ndarray, target_values: np.ndarray, degree: int) -> np.ndarray:
    design = polynomial_terms(source_points[:, 0], source_points[:, 1], degree)
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target_values)):
        raise ValueError("源图模型拟合点包含无效数值。")
    try:
        condition_number = float(np.linalg.cond(design))
    except np.linalg.LinAlgError as exc:
        raise ValueError("源图模型矩阵无法稳定求解，请检查配对星分布。") from exc
    if not np.isfinite(condition_number) or condition_number > MAX_ALIGNMENT_CONDITION_NUMBER:
        raise ValueError("配对星几何分布过于集中，无法稳定求解源图模型。")

    try:
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(design, target_values, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("源图模型矩阵无法稳定求解，请检查配对星位置。") from exc
    if rank < design.shape[1]:
        raise ValueError("配对星几何分布过于集中，源图模型矩阵秩不足。")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("源图模型拟合结果包含无效系数。")
    return coefficients.astype(np.float64)


def _fit_polynomial_pair(source_points: np.ndarray, target_points: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    coeff_x = _fit_polynomial_coefficients(source_points, target_points[:, 0], degree)
    coeff_y = _fit_polynomial_coefficients(source_points, target_points[:, 1], degree)
    return coeff_x, coeff_y


def _preferred_polynomial_degree(pair_count: int) -> int:
    return 2 if pair_count >= 6 else 1


def _residual_summary(residual_vectors: np.ndarray) -> tuple[float, float, float]:
    if residual_vectors.size == 0:
        return float("nan"), float("nan"), float("nan")
    distances = np.linalg.norm(residual_vectors, axis=1)
    return (
        float(np.sqrt(np.mean(distances * distances))),
        float(np.median(distances)),
        float(np.max(distances)),
    )


def _polynomial_derivative_at_point(coeff: np.ndarray, degree: int, x_value: float, y_value: float) -> tuple[float, float]:
    coeff = np.asarray(coeff, dtype=np.float64)
    if degree >= 2 and coeff.size >= 6:
        d_dx = coeff[1] + 2.0 * coeff[3] * x_value + coeff[4] * y_value
        d_dy = coeff[2] + coeff[4] * x_value + 2.0 * coeff[5] * y_value
        return float(d_dx), float(d_dy)
    if coeff.size >= 3:
        return float(coeff[1]), float(coeff[2])
    raise ValueError("多项式系数数量不足。")


@dataclass(frozen=True)
class SourceAstrometricModel:
    image_width_px: int
    image_height_px: int
    pair_count: int
    sky_to_pixel_degree: int
    pixel_to_sky_degree: int
    center_vector: np.ndarray
    east_vector: np.ndarray
    north_vector: np.ndarray
    sky_to_pixel_coeff_x: np.ndarray
    sky_to_pixel_coeff_y: np.ndarray
    pixel_to_sky_coeff_u: np.ndarray
    pixel_to_sky_coeff_v: np.ndarray
    rms_px: float
    median_residual_px: float
    max_residual_px: float
    inverse_rms_arcsec: float
    inverse_median_arcsec: float
    inverse_max_arcsec: float
    wcs: dict[str, Any]

    def direction_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
        if ra_dec_array.ndim == 1:
            ra_dec_array = ra_dec_array.reshape(1, 2)
        if ra_dec_array.ndim != 2 or ra_dec_array.shape[1] != 2:
            raise ValueError("direction_to_pixel_points 需要 Nx2 的 RA/Dec 数组。")

        plane_points = project_radec_to_sky_plane(
            ra_dec_array[:, 0],
            ra_dec_array[:, 1],
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )
        x_values = evaluate_polynomial(plane_points, self.sky_to_pixel_coeff_x, self.sky_to_pixel_degree)
        y_values = evaluate_polynomial(plane_points, self.sky_to_pixel_coeff_y, self.sky_to_pixel_degree)
        return np.column_stack((x_values, y_values)).astype(np.float64)

    def pixel_to_radec_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixel_array = np.asarray(pixel_points, dtype=np.float64)
        if pixel_array.ndim == 1:
            pixel_array = pixel_array.reshape(1, 2)
        if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
            raise ValueError("pixel_to_radec_points 需要 Nx2 的像素坐标数组。")

        u_deg = evaluate_polynomial(pixel_array, self.pixel_to_sky_coeff_u, self.pixel_to_sky_degree)
        v_deg = evaluate_polynomial(pixel_array, self.pixel_to_sky_coeff_v, self.pixel_to_sky_degree)
        return sky_plane_to_radec(
            np.column_stack((u_deg, v_deg)),
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )

    def pixel_to_radec(self, x_px: float, y_px: float) -> tuple[float, float]:
        result = self.pixel_to_radec_points(np.asarray([[x_px, y_px]], dtype=np.float64))[0]
        return float(result[0]), float(result[1])

    def to_json_payload(
        self,
        *,
        source_image: dict[str, Any] | None = None,
        fit_pairs: list[dict[str, Any]] | None = None,
        mask: dict[str, Any] | None = None,
        matching: dict[str, Any] | None = None,
        reference_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        center_radec = unit_vectors_to_radec(self.center_vector)[0]
        payload: dict[str, Any] = {
            "format": SOURCE_MODEL_FORMAT,
            "version": SOURCE_MODEL_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "direction_frame": "ICRS",
            "pixel_convention": "0-based pixel coordinates at pixel centers",
            "model_type": "local_sky_plane_polynomial",
            "image": {
                "width_px": int(self.image_width_px),
                "height_px": int(self.image_height_px),
            },
            "projection_basis": {
                "plane_projection": "azimuthal_equidistant_local_tangent",
                "center_vector": _as_float_list(self.center_vector),
                "east_vector": _as_float_list(self.east_vector),
                "north_vector": _as_float_list(self.north_vector),
                "center_ra_deg": float(center_radec[0]),
                "center_dec_deg": float(center_radec[1]),
            },
            "sky_to_pixel": {
                "degree": int(self.sky_to_pixel_degree),
                "input_units": "deg in local sky plane",
                "coeff_x": _as_float_list(self.sky_to_pixel_coeff_x),
                "coeff_y": _as_float_list(self.sky_to_pixel_coeff_y),
            },
            "pixel_to_sky_plane": {
                "degree": int(self.pixel_to_sky_degree),
                "output_units": "deg in local sky plane",
                "coeff_u_deg": _as_float_list(self.pixel_to_sky_coeff_u),
                "coeff_v_deg": _as_float_list(self.pixel_to_sky_coeff_v),
            },
            "diagnostics": {
                "pair_count": int(self.pair_count),
                "rms_px": float(self.rms_px),
                "median_residual_px": float(self.median_residual_px),
                "max_residual_px": float(self.max_residual_px),
                "inverse_rms_arcsec": float(self.inverse_rms_arcsec),
                "inverse_median_arcsec": float(self.inverse_median_arcsec),
                "inverse_max_arcsec": float(self.inverse_max_arcsec),
            },
            "fits_wcs_compat": self.wcs,
        }
        if source_image is not None:
            payload["source_image"] = source_image
        if mask is not None:
            payload["mask"] = mask
        if matching is not None:
            payload["matching"] = matching
        if fit_pairs is not None:
            payload["fit_pairs"] = fit_pairs
        if reference_payload is not None:
            payload["reference_payload"] = reference_payload
        return payload


def _build_wcs_compat(
    *,
    model: SourceAstrometricModel,
    center_pixel: np.ndarray,
) -> dict[str, Any]:
    center_radec = unit_vectors_to_radec(model.center_vector)[0]
    x0 = float(center_pixel[0])
    y0 = float(center_pixel[1])
    du_dx, du_dy = _polynomial_derivative_at_point(model.pixel_to_sky_coeff_u, model.pixel_to_sky_degree, x0, y0)
    dv_dx, dv_dy = _polynomial_derivative_at_point(model.pixel_to_sky_coeff_v, model.pixel_to_sky_degree, x0, y0)
    header_cards = {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": float(center_radec[0]),
        "CRVAL2": float(center_radec[1]),
        "CRPIX1": x0 + 1.0,
        "CRPIX2": y0 + 1.0,
        "CD1_1": du_dx,
        "CD1_2": du_dy,
        "CD2_1": dv_dx,
        "CD2_2": dv_dy,
        "RADESYS": "ICRS",
        "EQUINOX": 2000.0,
    }
    return {
        "approximate": True,
        "standard": "FITS WCS TAN local approximation",
        "note": (
            "header_cards 是局部线性 TAN WCS 近似；完整高阶映射请使用 pixel_to_sky_plane "
            "多项式与 projection_basis。"
        ),
        "header_cards": header_cards,
        "crpix_is_one_based": True,
        "pixel_axis_order": ["x", "y"],
        "world_axis_order": ["RA", "Dec"],
    }


def fit_source_astrometric_model(
    ra_dec_points: np.ndarray,
    pixel_points: np.ndarray,
    image_size: tuple[int, int],
) -> SourceAstrometricModel:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    pixels = np.asarray(pixel_points, dtype=np.float64)
    if sky_radec.ndim != 2 or sky_radec.shape[1] != 2:
        raise ValueError("源图模型需要 Nx2 的 RA/Dec 点。")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("源图模型需要 Nx2 的像素点。")
    if sky_radec.shape[0] != pixels.shape[0]:
        raise ValueError("源图模型的 RA/Dec 点与像素点数量不一致。")

    finite_mask = _finite_point_mask(sky_radec, pixels)
    sky_radec = sky_radec[finite_mask]
    pixels = pixels[finite_mask]
    pair_count = int(sky_radec.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对星点才能拟合源图映射。")

    center, east, north = sky_plane_basis(sky_radec)
    sky_plane = project_radec_to_sky_plane(sky_radec[:, 0], sky_radec[:, 1], center, east, north)
    if not np.all(np.isfinite(sky_plane)):
        raise ValueError("源图模型的天球平面坐标包含无效数值。")

    preferred_degree = _preferred_polynomial_degree(pair_count)
    sky_to_pixel_degree = preferred_degree
    pixel_to_sky_degree = preferred_degree
    try:
        sky_to_pixel_coeff_x, sky_to_pixel_coeff_y = _fit_polynomial_pair(sky_plane, pixels, sky_to_pixel_degree)
    except ValueError:
        if preferred_degree <= 1:
            raise
        sky_to_pixel_degree = 1
        sky_to_pixel_coeff_x, sky_to_pixel_coeff_y = _fit_polynomial_pair(sky_plane, pixels, sky_to_pixel_degree)

    try:
        pixel_to_sky_coeff_u, pixel_to_sky_coeff_v = _fit_polynomial_pair(pixels, sky_plane, pixel_to_sky_degree)
    except ValueError:
        if preferred_degree <= 1:
            raise
        pixel_to_sky_degree = 1
        pixel_to_sky_coeff_u, pixel_to_sky_coeff_v = _fit_polynomial_pair(pixels, sky_plane, pixel_to_sky_degree)

    predicted_pixels = np.column_stack(
        (
            evaluate_polynomial(sky_plane, sky_to_pixel_coeff_x, sky_to_pixel_degree),
            evaluate_polynomial(sky_plane, sky_to_pixel_coeff_y, sky_to_pixel_degree),
        )
    )
    rms_px, median_px, max_px = _residual_summary(predicted_pixels - pixels)

    predicted_plane = np.column_stack(
        (
            evaluate_polynomial(pixels, pixel_to_sky_coeff_u, pixel_to_sky_degree),
            evaluate_polynomial(pixels, pixel_to_sky_coeff_v, pixel_to_sky_degree),
        )
    )
    inverse_rms_deg, inverse_median_deg, inverse_max_deg = _residual_summary(predicted_plane - sky_plane)
    center_pixel = np.asarray(
        [
            evaluate_polynomial(np.asarray([[0.0, 0.0]], dtype=np.float64), sky_to_pixel_coeff_x, sky_to_pixel_degree)[0],
            evaluate_polynomial(np.asarray([[0.0, 0.0]], dtype=np.float64), sky_to_pixel_coeff_y, sky_to_pixel_degree)[0],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(center_pixel)):
        center_pixel = np.asarray([image_size[0] / 2.0, image_size[1] / 2.0], dtype=np.float64)

    placeholder_wcs: dict[str, Any] = {}
    model = SourceAstrometricModel(
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        pair_count=pair_count,
        sky_to_pixel_degree=sky_to_pixel_degree,
        pixel_to_sky_degree=pixel_to_sky_degree,
        center_vector=center,
        east_vector=east,
        north_vector=north,
        sky_to_pixel_coeff_x=sky_to_pixel_coeff_x,
        sky_to_pixel_coeff_y=sky_to_pixel_coeff_y,
        pixel_to_sky_coeff_u=pixel_to_sky_coeff_u,
        pixel_to_sky_coeff_v=pixel_to_sky_coeff_v,
        rms_px=float(rms_px),
        median_residual_px=float(median_px),
        max_residual_px=float(max_px),
        inverse_rms_arcsec=float(inverse_rms_deg * 3600.0),
        inverse_median_arcsec=float(inverse_median_deg * 3600.0),
        inverse_max_arcsec=float(inverse_max_deg * 3600.0),
        wcs=placeholder_wcs,
    )
    object.__setattr__(model, "wcs", _build_wcs_compat(model=model, center_pixel=center_pixel))
    return model
