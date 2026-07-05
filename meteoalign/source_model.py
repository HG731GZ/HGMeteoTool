from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .alignment import (
    AnchorInterpolation2D,
    MIN_ALIGNMENT_PAIRS,
    ProjectionSkyAlignmentTransform,
    SKY_KNOWN_PROJECTION_MODELS,
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODELS,
    fit_anchor_interpolation,
    fit_projection_sky_alignment,
)
from .coordinates import project_radec_to_sky_plane, sky_plane_basis, sky_plane_to_radec, unit_vectors_to_radec


SOURCE_MODEL_FORMAT = "meteoalign_source_astrometric_model"
SOURCE_MODEL_VERSION = 2


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def _finite_point_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=bool)
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.all(np.isfinite(array), axis=1)
    return mask


def _residual_summary(residual_vectors: np.ndarray) -> tuple[float, float, float]:
    if residual_vectors.size == 0:
        return float("nan"), float("nan"), float("nan")
    distances = np.linalg.norm(residual_vectors, axis=1)
    return (
        float(np.sqrt(np.mean(distances * distances))),
        float(np.median(distances)),
        float(np.max(distances)),
    )


def _interpolation_payload(
    interpolation: AnchorInterpolation2D,
    *,
    input_units: str,
    output_units: str,
    input_axis_order: list[str],
    output_axis_order: list[str],
    weight_names: tuple[str, str],
    affine_names: tuple[str, str],
) -> dict[str, Any]:
    return {
        "kind": interpolation.kind,
        "input_units": input_units,
        "output_units": output_units,
        "input_axis_order": input_axis_order,
        "output_axis_order": output_axis_order,
        "normalization": {
            "origin_x": float(interpolation.origin_x),
            "origin_y": float(interpolation.origin_y),
            "scale_x": float(interpolation.scale_x),
            "scale_y": float(interpolation.scale_y),
        },
        "anchor_count": int(interpolation.anchor_points.shape[0]),
        "anchor_points_normalized": [
            _as_float_list(row) for row in np.asarray(interpolation.anchor_points, dtype=np.float64)
        ],
        weight_names[0]: _as_float_list(interpolation.tps_weights_x),
        weight_names[1]: _as_float_list(interpolation.tps_weights_y),
        affine_names[0]: _as_float_list(interpolation.tps_affine_x),
        affine_names[1]: _as_float_list(interpolation.tps_affine_y),
    }


def _known_projection_sky_to_pixel_payload() -> dict[str, Any]:
    return {
        "kind": "known_projection_with_residual_anchor_interpolation",
        "input": "ICRS RA/Dec direction",
        "projection": "see known_projection",
        "residual_correction": "see known_projection.residual_correction",
    }


@dataclass(frozen=True)
class SourceAstrometricModel:
    image_width_px: int
    image_height_px: int
    pair_count: int
    center_vector: np.ndarray
    east_vector: np.ndarray
    north_vector: np.ndarray
    sky_to_pixel_interpolation: AnchorInterpolation2D | None
    pixel_to_sky_plane_interpolation: AnchorInterpolation2D
    rms_px: float
    median_residual_px: float
    max_residual_px: float
    inverse_rms_arcsec: float
    inverse_median_arcsec: float
    inverse_max_arcsec: float
    wcs: dict[str, Any]
    model_type: str = "local_sky_plane_anchor_interpolation"
    projection_transform: ProjectionSkyAlignmentTransform | None = None

    def direction_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
        if ra_dec_array.ndim == 1:
            ra_dec_array = ra_dec_array.reshape(1, 2)
        if ra_dec_array.ndim != 2 or ra_dec_array.shape[1] != 2:
            raise ValueError("direction_to_pixel_points 需要 Nx2 的 RA/Dec 数组。")

        if self.projection_transform is not None:
            return self.projection_transform.transform_radec_points(ra_dec_array)
        if self.sky_to_pixel_interpolation is None:
            return np.full((ra_dec_array.shape[0], 2), np.nan, dtype=np.float64)

        plane_points = project_radec_to_sky_plane(
            ra_dec_array[:, 0],
            ra_dec_array[:, 1],
            self.center_vector,
            self.east_vector,
            self.north_vector,
        )
        return self.sky_to_pixel_interpolation.evaluate_points(plane_points)

    def pixel_to_radec_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixel_array = np.asarray(pixel_points, dtype=np.float64)
        if pixel_array.ndim == 1:
            pixel_array = pixel_array.reshape(1, 2)
        if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
            raise ValueError("pixel_to_radec_points 需要 Nx2 的像素坐标数组。")

        plane_points = self.pixel_to_sky_plane_interpolation.evaluate_points(pixel_array)
        return sky_plane_to_radec(
            plane_points,
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
        if self.projection_transform is None and self.sky_to_pixel_interpolation is None:
            raise ValueError("普适源图模型缺少 sky→pixel 锚点插值。")
        sky_to_pixel_payload = (
            _known_projection_sky_to_pixel_payload()
            if self.projection_transform is not None
            else _interpolation_payload(
                self.sky_to_pixel_interpolation,
                input_units="deg in local sky plane",
                output_units="px",
                input_axis_order=["u_deg", "v_deg"],
                output_axis_order=["x_px", "y_px"],
                weight_names=("tps_weights_x_px", "tps_weights_y_px"),
                affine_names=("tps_affine_x_px", "tps_affine_y_px"),
            )
        )
        payload: dict[str, Any] = {
            "format": SOURCE_MODEL_FORMAT,
            "version": SOURCE_MODEL_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "direction_frame": "ICRS",
            "pixel_convention": "0-based pixel coordinates at pixel centers",
            "model_type": self.model_type,
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
            "sky_to_pixel": sky_to_pixel_payload,
            "pixel_to_sky_plane": _interpolation_payload(
                self.pixel_to_sky_plane_interpolation,
                input_units="px",
                output_units="deg in local sky plane",
                input_axis_order=["x_px", "y_px"],
                output_axis_order=["u_deg", "v_deg"],
                weight_names=("tps_weights_u_deg", "tps_weights_v_deg"),
                affine_names=("tps_affine_u_deg", "tps_affine_v_deg"),
            ),
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
        if self.projection_transform is not None:
            payload["known_projection"] = _projection_transform_payload(self.projection_transform)
            payload["diagnostics"]["projection_rms_px_before_residual"] = float(
                self.projection_transform.projection_rms_px
            )
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


def _projection_transform_payload(transform: ProjectionSkyAlignmentTransform) -> dict[str, Any]:
    return {
        "lens_model": transform.lens_model,
        "fov_deg": None if transform.fov_deg is None else float(transform.fov_deg),
        "image_width_px": int(transform.image_width_px),
        "image_height_px": int(transform.image_height_px),
        "rotation_matrix_world_to_camera": [
            _as_float_list(row) for row in np.asarray(transform.rotation_matrix, dtype=np.float64)
        ],
        "principal_point_px": {
            "x": float(transform.center_x_px),
            "y": float(transform.center_y_px),
        },
        "scale_px": float(transform.scale_px),
        "residual_correction": {
            "kind": transform.residual_kind,
            "input": "raw projection pixel coordinates normalized by image size",
            "normalization": {
                "origin_x_px": float(transform.residual_origin_x_px),
                "origin_y_px": float(transform.residual_origin_y_px),
                "scale_x_px": float(transform.residual_scale_x_px),
                "scale_y_px": float(transform.residual_scale_y_px),
            },
            "anchor_count": int(transform.residual_anchor_points.shape[0]),
            "anchor_points_normalized": [
                _as_float_list(row) for row in np.asarray(transform.residual_anchor_points, dtype=np.float64)
            ],
            "tps_weights_dx_px": _as_float_list(transform.residual_tps_weights_x),
            "tps_weights_dy_px": _as_float_list(transform.residual_tps_weights_y),
            "tps_affine_dx_px": _as_float_list(transform.residual_tps_affine_x),
            "tps_affine_dy_px": _as_float_list(transform.residual_tps_affine_y),
        },
    }


def _finite_difference_pixel_to_sky_jacobian(
    interpolation: AnchorInterpolation2D,
    x_px: float,
    y_px: float,
) -> tuple[float, float, float, float]:
    step_px = max(min(float(interpolation.scale_x), float(interpolation.scale_y)) * 1e-4, 1.0)
    sample_points = np.asarray(
        (
            (x_px + step_px, y_px),
            (x_px - step_px, y_px),
            (x_px, y_px + step_px),
            (x_px, y_px - step_px),
        ),
        dtype=np.float64,
    )
    samples = interpolation.evaluate_points(sample_points)
    if not np.all(np.isfinite(samples)):
        return float("nan"), float("nan"), float("nan"), float("nan")
    du_dx, dv_dx = (samples[0] - samples[1]) / (2.0 * step_px)
    du_dy, dv_dy = (samples[2] - samples[3]) / (2.0 * step_px)
    return float(du_dx), float(du_dy), float(dv_dx), float(dv_dy)


def _build_wcs_compat(
    *,
    model: SourceAstrometricModel,
    center_pixel: np.ndarray,
) -> dict[str, Any]:
    center_radec = unit_vectors_to_radec(model.center_vector)[0]
    x0 = float(center_pixel[0])
    y0 = float(center_pixel[1])
    du_dx, du_dy, dv_dx, dv_dy = _finite_difference_pixel_to_sky_jacobian(
        model.pixel_to_sky_plane_interpolation,
        x0,
        y0,
    )
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
            "header_cards 是局部线性 TAN WCS 近似；完整映射请使用 pixel_to_sky_plane "
            "锚点插值与 projection_basis。"
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
    matching_model: str = SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    fisheye_fov_deg: float | None = None,
    initial_rotation_matrix: np.ndarray | None = None,
) -> SourceAstrometricModel:
    sky_radec = np.asarray(ra_dec_points, dtype=np.float64)
    pixels = np.asarray(pixel_points, dtype=np.float64)
    if sky_radec.ndim != 2 or sky_radec.shape[1] != 2:
        raise ValueError("源图模型需要 Nx2 的 RA/Dec 点。")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("源图模型需要 Nx2 的像素点。")
    if sky_radec.shape[0] != pixels.shape[0]:
        raise ValueError("源图模型的 RA/Dec 点与像素点数量不一致。")
    if matching_model not in SKY_MATCHING_MODELS:
        raise ValueError(f"不支持的源图匹配模型：{matching_model}")

    finite_mask = _finite_point_mask(sky_radec, pixels)
    sky_radec = sky_radec[finite_mask]
    pixels = pixels[finite_mask]
    pair_count = int(sky_radec.shape[0])
    if pair_count < MIN_ALIGNMENT_PAIRS:
        raise ValueError(f"至少需要 {MIN_ALIGNMENT_PAIRS} 对星点才能生成源图映射。")
    image_width = int(image_size[0])
    image_height = int(image_size[1])
    if image_width <= 0 or image_height <= 0:
        raise ValueError("源图尺寸无效，无法生成源图映射。")

    center, east, north = sky_plane_basis(sky_radec)
    sky_plane = project_radec_to_sky_plane(sky_radec[:, 0], sky_radec[:, 1], center, east, north)
    if not np.all(np.isfinite(sky_plane)):
        raise ValueError("源图模型的天球平面坐标包含无效数值。")

    pixel_to_sky_plane_interpolation = fit_anchor_interpolation(pixels, sky_plane)
    sky_to_pixel_interpolation: AnchorInterpolation2D | None = None
    projection_transform: ProjectionSkyAlignmentTransform | None = None
    if matching_model in SKY_KNOWN_PROJECTION_MODELS:
        projection_transform = fit_projection_sky_alignment(
            ra_dec_points=sky_radec,
            target_points=pixels,
            lens_model=matching_model,
            image_size=(image_width, image_height),
            fisheye_fov_deg=fisheye_fov_deg,
            initial_rotation_matrix=initial_rotation_matrix,
        )
        predicted_pixels = projection_transform.transform_radec_points(sky_radec)
        model_type = "known_projection_with_residual_interpolation"
    else:
        sky_to_pixel_interpolation = fit_anchor_interpolation(sky_plane, pixels)
        predicted_pixels = sky_to_pixel_interpolation.evaluate_points(sky_plane)
        model_type = "local_sky_plane_anchor_interpolation"
    rms_px, median_px, max_px = _residual_summary(predicted_pixels - pixels)

    predicted_plane = pixel_to_sky_plane_interpolation.evaluate_points(pixels)
    inverse_rms_deg, inverse_median_deg, inverse_max_deg = _residual_summary(predicted_plane - sky_plane)
    if projection_transform is not None:
        center_radec = sky_plane_to_radec(np.asarray([[0.0, 0.0]], dtype=np.float64), center, east, north)
        center_pixel = projection_transform.transform_radec_points(center_radec)[0]
    else:
        center_pixel = sky_to_pixel_interpolation.evaluate_points(np.asarray([[0.0, 0.0]], dtype=np.float64))[0]
    if not np.all(np.isfinite(center_pixel)):
        center_pixel = np.asarray([image_width / 2.0, image_height / 2.0], dtype=np.float64)

    placeholder_wcs: dict[str, Any] = {}
    model = SourceAstrometricModel(
        image_width_px=image_width,
        image_height_px=image_height,
        pair_count=pair_count,
        center_vector=center,
        east_vector=east,
        north_vector=north,
        sky_to_pixel_interpolation=sky_to_pixel_interpolation,
        pixel_to_sky_plane_interpolation=pixel_to_sky_plane_interpolation,
        rms_px=float(rms_px),
        median_residual_px=float(median_px),
        max_residual_px=float(max_px),
        inverse_rms_arcsec=float(inverse_rms_deg * 3600.0),
        inverse_median_arcsec=float(inverse_median_deg * 3600.0),
        inverse_max_arcsec=float(inverse_max_deg * 3600.0),
        wcs=placeholder_wcs,
        model_type=model_type,
        projection_transform=projection_transform,
    )
    object.__setattr__(model, "wcs", _build_wcs_compat(model=model, center_pixel=center_pixel))
    return model
