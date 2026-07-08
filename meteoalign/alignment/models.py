from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    RESIDUAL_CORRECTION_TPS,
    SKY_KNOWN_PROJECTION_DISPLAY_NAMES,
)
from .interpolation import AnchorInterpolation2D
from .projections import (
    _project_radec_to_sky_plane,
    _project_unit_vectors_with_known_projection,
    _radec_to_unit_vectors,
)
from .residuals import _apply_residual_correction
from .validation import (
    _alignment_coefficients_are_valid,
    _sky_transform_components_are_valid,
)


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
        from .fitting import _polynomial_terms

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
    pair_count: int
    center_vector: np.ndarray
    east_vector: np.ndarray
    north_vector: np.ndarray
    interpolation: AnchorInterpolation2D
    rms_px: float
    residual_soft_constraint_count: int = 0
    residual_soft_weight_min: float = 1.0
    residual_soft_weight_max: float = 1.0

    @property
    def display_name(self) -> str:
        if self.residual_soft_constraint_count > 0:
            return "普适平滑插值"
        return "普适锚点插值"

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
        transformed = self.interpolation.evaluate_points(sky_points)
        transformed[~np.all(np.isfinite(transformed), axis=1)] = np.nan
        return transformed

    def transform_radec(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        transformed = self.transform_radec_points(np.asarray([[ra_deg, dec_deg]], dtype=np.float64))[0]
        return float(transformed[0]), float(transformed[1])


@dataclass(frozen=True)
class ProjectionSkyAlignmentTransform:
    lens_model: str
    pair_count: int
    image_width_px: int
    image_height_px: int
    fov_deg: float | None
    rotation_matrix: np.ndarray
    center_x_px: float
    center_y_px: float
    scale_px: float
    residual_kind: str
    residual_origin_x_px: float
    residual_origin_y_px: float
    residual_scale_x_px: float
    residual_scale_y_px: float
    residual_anchor_points: np.ndarray
    residual_tps_weights_x: np.ndarray
    residual_tps_weights_y: np.ndarray
    residual_tps_affine_x: np.ndarray
    residual_tps_affine_y: np.ndarray
    residual_hard_anchor_count: int
    residual_soft_constraint_count: int
    residual_soft_weight_min: float
    residual_soft_weight_max: float
    projection_rms_px: float
    rms_px: float

    @property
    def display_name(self) -> str:
        projection_name = SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(self.lens_model, self.lens_model)
        if self.residual_soft_constraint_count > 0:
            return f"{projection_name}+平滑残差"
        if self.residual_kind == RESIDUAL_CORRECTION_TPS:
            return f"{projection_name}+锚点插值"
        return f"{projection_name}+残差插值"

    def raw_project_radec_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
        if ra_dec_array.ndim == 1:
            ra_dec_array = ra_dec_array.reshape(1, 2)
        if ra_dec_array.shape[1] != 2:
            raise ValueError("天球投影点必须是 Nx2 的 RA/Dec 数组。")
        vectors = _radec_to_unit_vectors(ra_dec_array[:, 0], ra_dec_array[:, 1])
        projected, valid = _project_unit_vectors_with_known_projection(
            vectors=vectors,
            rotation_matrix=self.rotation_matrix,
            center_x_px=self.center_x_px,
            center_y_px=self.center_y_px,
            scale_px=self.scale_px,
            lens_model=self.lens_model,
            strict_visibility=True,
        )
        projected[~valid] = np.nan
        return projected

    def transform_radec_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        projected = self.raw_project_radec_points(ra_dec_points)
        corrected = _apply_residual_correction(
            projected_points=projected,
            origin_x_px=self.residual_origin_x_px,
            origin_y_px=self.residual_origin_y_px,
            scale_x_px=self.residual_scale_x_px,
            scale_y_px=self.residual_scale_y_px,
            residual_kind=self.residual_kind,
            anchor_points=self.residual_anchor_points,
            tps_weights_x=self.residual_tps_weights_x,
            tps_weights_y=self.residual_tps_weights_y,
            tps_affine_x=self.residual_tps_affine_x,
            tps_affine_y=self.residual_tps_affine_y,
        )
        corrected[~np.all(np.isfinite(corrected), axis=1)] = np.nan
        return corrected.astype(np.float64)

    def transform_radec(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        transformed = self.transform_radec_points(np.asarray([[ra_deg, dec_deg]], dtype=np.float64))[0]
        return float(transformed[0]), float(transformed[1])

__all__ = [name for name in globals() if not name.startswith("__")]
