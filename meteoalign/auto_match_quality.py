"""自动扩展匹配专用的几何与测光质量指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


AUTO_MATCH_QUALITY_SCORE_KEY = "auto_match_quality_score"
AUTO_MATCH_GEOMETRY_SCORE_KEY = "auto_match_geometry_score"
AUTO_MATCH_POSITION_SCORE_KEY = "auto_match_position_score"
AUTO_MATCH_ASSIGNMENT_SCORE_KEY = "auto_match_assignment_score"
AUTO_MATCH_PREDICTION_OFFSET_PX_KEY = "auto_match_prediction_offset_px"
AUTO_MATCH_PREDICTION_OFFSET_RATIO_KEY = "auto_match_prediction_offset_ratio"
AUTO_MATCH_ASSIGNMENT_COST_KEY = "auto_match_assignment_cost"
AUTO_MATCH_ASSIGNMENT_ROW_MARGIN_KEY = "auto_match_assignment_row_margin"
AUTO_MATCH_ASSIGNMENT_SOURCE_MARGIN_KEY = "auto_match_assignment_source_margin"
AUTO_MATCH_OBSERVED_FLUX_KEY = "auto_match_observed_flux"
AUTO_MATCH_PSF_BRIGHTNESS_KEY = "auto_match_psf_brightness_proxy"
AUTO_MATCH_PHOTOMETRIC_RESIDUAL_MAG_KEY = "auto_match_photometric_residual_mag"
AUTO_MATCH_PHOTOMETRIC_SCORE_KEY = "auto_match_photometric_score"
AUTO_MATCH_PHOTOMETRIC_ZERO_POINT_MAG_KEY = "auto_match_photometric_zero_point_mag"
AUTO_MATCH_PHOTOMETRIC_SCATTER_MAG_KEY = "auto_match_photometric_scatter_mag"

AUTO_MATCH_QUALITY_FIELD_KEYS = frozenset(
    {
        AUTO_MATCH_QUALITY_SCORE_KEY,
        AUTO_MATCH_GEOMETRY_SCORE_KEY,
        AUTO_MATCH_POSITION_SCORE_KEY,
        AUTO_MATCH_ASSIGNMENT_SCORE_KEY,
        AUTO_MATCH_PREDICTION_OFFSET_PX_KEY,
        AUTO_MATCH_PREDICTION_OFFSET_RATIO_KEY,
        AUTO_MATCH_ASSIGNMENT_COST_KEY,
        AUTO_MATCH_ASSIGNMENT_ROW_MARGIN_KEY,
        AUTO_MATCH_ASSIGNMENT_SOURCE_MARGIN_KEY,
        AUTO_MATCH_OBSERVED_FLUX_KEY,
        AUTO_MATCH_PSF_BRIGHTNESS_KEY,
        AUTO_MATCH_PHOTOMETRIC_RESIDUAL_MAG_KEY,
        AUTO_MATCH_PHOTOMETRIC_SCORE_KEY,
        AUTO_MATCH_PHOTOMETRIC_ZERO_POINT_MAG_KEY,
        AUTO_MATCH_PHOTOMETRIC_SCATTER_MAG_KEY,
    }
)


@dataclass(frozen=True)
class AutoMatchPhotometrySample:
    """一颗自动匹配星的星表星等和图像亮度代理。"""

    star_id: str
    catalog_mag: float
    flux: float
    x_px: float
    y_px: float


@dataclass(frozen=True)
class AutoMatchPhotometricModel:
    """自动匹配批次使用的稳健星等—PSF 亮度模型。"""

    coefficients: np.ndarray
    spatial: bool
    scatter_mag: float
    sample_count: int
    image_width_px: int
    image_height_px: int

    def predicted_instrumental_mag(
        self,
        catalog_mag: float,
        x_px: float,
        y_px: float,
    ) -> float:
        design = _photometric_design(
            np.asarray([catalog_mag], dtype=np.float64),
            np.asarray([x_px], dtype=np.float64),
            np.asarray([y_px], dtype=np.float64),
            (self.image_width_px, self.image_height_px),
            spatial=self.spatial,
        )
        return float(design[0] @ self.coefficients)

    def zero_point_at(self, catalog_mag: float, x_px: float, y_px: float) -> float:
        """返回指定星等和画面位置上的等效测光零点。"""

        return self.predicted_instrumental_mag(catalog_mag, x_px, y_px) - float(catalog_mag)


@dataclass(frozen=True)
class AutoMatchPhotometricEvaluation:
    """单颗自动匹配星的亮度一致性结果。"""

    residual_mag: float
    quality_score: float | None
    reject: bool


def instrumental_magnitude(flux: float) -> float:
    """把正的图像亮度代理转换为相对仪器星等。"""

    if not math.isfinite(float(flux)) or float(flux) <= 0.0:
        return float("nan")
    return -2.5 * math.log10(float(flux))


def psf_brightness_proxy(amplitude: float, sigma_x: float, sigma_y: float) -> float:
    """用 PSF 峰值和椭圆面积估计亮度，保留饱和星的尺寸信息。"""

    values = (float(amplitude), float(sigma_x), float(sigma_y))
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        return float("nan")
    return float(values[0] * 2.0 * math.pi * values[1] * values[2])


def _photometric_design(
    catalog_magnitudes: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    image_size: tuple[int, int],
    *,
    spatial: bool,
) -> np.ndarray:
    width = max(float(image_size[0]), 1.0)
    height = max(float(image_size[1]), 1.0)
    normalized_x = (np.asarray(x_values, dtype=np.float64) - width * 0.5) / width
    normalized_y = (np.asarray(y_values, dtype=np.float64) - height * 0.5) / height
    catalog_values = np.asarray(catalog_magnitudes, dtype=np.float64)
    if not spatial:
        return np.column_stack(
            (
                np.ones(normalized_x.size, dtype=np.float64),
                catalog_values,
            )
        )
    radius_squared = normalized_x * normalized_x + normalized_y * normalized_y
    return np.column_stack(
        (
            np.ones(normalized_x.size, dtype=np.float64),
            catalog_values,
            normalized_x,
            normalized_y,
            radius_squared,
        )
    )


def _robust_scatter(values: np.ndarray, *, floor: float) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size <= 0:
        return float(floor)
    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    return max(float(floor), 1.4826 * mad)


def fit_auto_match_photometric_model(
    samples: list[AutoMatchPhotometrySample],
    image_size: tuple[int, int],
    *,
    minimum_samples: int = 5,
) -> AutoMatchPhotometricModel | None:
    """仅用自动扩展结果拟合亮度关系；样本足够时兼容渐晕和亮度梯度。"""

    usable = [
        sample
        for sample in samples
        if sample.star_id
        and math.isfinite(float(sample.catalog_mag))
        and math.isfinite(float(sample.flux))
        and float(sample.flux) > 0.0
        and math.isfinite(float(sample.x_px))
        and math.isfinite(float(sample.y_px))
    ]
    if len(usable) < max(3, int(minimum_samples)):
        return None

    x_values = np.asarray([sample.x_px for sample in usable], dtype=np.float64)
    y_values = np.asarray([sample.y_px for sample in usable], dtype=np.float64)
    catalog_magnitudes = np.asarray(
        [sample.catalog_mag for sample in usable],
        dtype=np.float64,
    )
    instrumental_magnitudes = np.asarray(
        [instrumental_magnitude(sample.flux) for sample in usable],
        dtype=np.float64,
    )
    spatial = len(usable) >= 12
    design = _photometric_design(
        catalog_magnitudes,
        x_values,
        y_values,
        image_size,
        spatial=spatial,
    )
    if spatial and int(np.linalg.matrix_rank(design)) < design.shape[1]:
        spatial = False
        design = _photometric_design(
            catalog_magnitudes,
            x_values,
            y_values,
            image_size,
            spatial=False,
        )
    if int(np.linalg.matrix_rank(design)) < design.shape[1]:
        return None

    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    # 以两两斜率的中位数初始化，避免一个错配样本先把整批亮度斜率拉偏。
    pairwise_slopes: list[float] = []
    for index in range(len(usable) - 1):
        magnitude_differences = catalog_magnitudes[index + 1 :] - catalog_magnitudes[index]
        usable_differences = np.abs(magnitude_differences) >= 0.20
        if not np.any(usable_differences):
            continue
        brightness_differences = (
            instrumental_magnitudes[index + 1 :] - instrumental_magnitudes[index]
        )
        pairwise_slopes.extend(
            np.asarray(
                brightness_differences[usable_differences]
                / magnitude_differences[usable_differences],
                dtype=np.float64,
            ).tolist()
        )
    if not pairwise_slopes:
        return None
    coefficients[1] = float(np.median(np.asarray(pairwise_slopes, dtype=np.float64)))
    coefficients[0] = float(
        np.median(instrumental_magnitudes - coefficients[1] * catalog_magnitudes)
    )
    for _iteration in range(8):
        residuals = instrumental_magnitudes - design @ coefficients
        scatter = _robust_scatter(residuals, floor=0.20)
        huber_limit = max(0.30, scatter * 1.5)
        absolute_residuals = np.abs(residuals)
        weights = np.ones(residuals.size, dtype=np.float64)
        outside = absolute_residuals > huber_limit
        weights[outside] = huber_limit / np.maximum(absolute_residuals[outside], 1e-9)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = instrumental_magnitudes * np.sqrt(weights)
        try:
            updated, *_unused = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(updated)):
            return None
        if float(np.max(np.abs(updated - coefficients))) < 1e-7:
            coefficients = updated.astype(np.float64)
            break
        coefficients = updated.astype(np.float64)

    residuals = instrumental_magnitudes - design @ coefficients
    # 把剩余中位偏差并回截距，避免少量异常值使整批结果整体偏移。
    coefficients[0] += float(np.median(residuals))
    residuals = instrumental_magnitudes - design @ coefficients
    # 响应斜率超出合理范围通常表示星等跨度不足或样本已被大量错配污染。
    if not 0.20 <= float(coefficients[1]) <= 2.00:
        return None
    scatter = _robust_scatter(residuals, floor=0.35)
    return AutoMatchPhotometricModel(
        coefficients=coefficients.astype(np.float64),
        spatial=spatial,
        scatter_mag=scatter,
        sample_count=len(usable),
        image_width_px=max(1, int(image_size[0])),
        image_height_px=max(1, int(image_size[1])),
    )


def evaluate_auto_match_photometry(
    sample: AutoMatchPhotometrySample,
    model: AutoMatchPhotometricModel,
    *,
    saturated: bool,
) -> AutoMatchPhotometricEvaluation | None:
    """评估星等与 PSF 亮度是否一致，饱和星使用更宽容的双边阈值。"""

    measured_mag = instrumental_magnitude(sample.flux)
    if not math.isfinite(measured_mag) or not math.isfinite(float(sample.catalog_mag)):
        return None
    residual = measured_mag - model.predicted_instrumental_mag(
        sample.catalog_mag,
        sample.x_px,
        sample.y_px,
    )
    score_scale = max(0.75 if saturated else 0.60, float(model.scatter_mag))
    quality_score = math.exp(-0.5 * (residual / score_scale) ** 2)
    scatter_multiplier = 4.5 if saturated else 3.5
    rejection_limit = max(2.0, float(model.scatter_mag) * scatter_multiplier)
    return AutoMatchPhotometricEvaluation(
        residual_mag=float(residual),
        quality_score=float(quality_score),
        reject=abs(residual) > rejection_limit,
    )


def auto_match_position_quality(distance_px: float, search_radius_px: float) -> float:
    """把加入模型前的预测偏差转换为 0 到 1 的几何质量。"""

    distance = max(0.0, float(distance_px))
    radius = max(1.0, float(search_radius_px))
    scale = max(2.0, radius * 0.60)
    return float(math.exp(-0.5 * (distance / scale) ** 2))


def combine_auto_match_quality(
    position_score: float,
    assignment_score: float,
    photometric_score: float | None,
) -> tuple[float, float]:
    """合成自动扩展匹配质量，并单独返回几何部分便于诊断。"""

    position = min(max(float(position_score), 0.0), 1.0)
    assignment = min(max(float(assignment_score), 0.0), 1.0)
    geometry_score = 0.35 * position + 0.65 * assignment
    if photometric_score is None or not math.isfinite(float(photometric_score)):
        return float(geometry_score), float(geometry_score)
    photometric = min(max(float(photometric_score), 0.0), 1.0)
    quality_score = 0.60 * geometry_score + 0.40 * photometric
    return float(quality_score), float(geometry_score)


__all__ = [
    "AUTO_MATCH_ASSIGNMENT_COST_KEY",
    "AUTO_MATCH_ASSIGNMENT_ROW_MARGIN_KEY",
    "AUTO_MATCH_ASSIGNMENT_SCORE_KEY",
    "AUTO_MATCH_ASSIGNMENT_SOURCE_MARGIN_KEY",
    "AUTO_MATCH_GEOMETRY_SCORE_KEY",
    "AUTO_MATCH_OBSERVED_FLUX_KEY",
    "AUTO_MATCH_PSF_BRIGHTNESS_KEY",
    "AUTO_MATCH_PHOTOMETRIC_RESIDUAL_MAG_KEY",
    "AUTO_MATCH_PHOTOMETRIC_SCATTER_MAG_KEY",
    "AUTO_MATCH_PHOTOMETRIC_SCORE_KEY",
    "AUTO_MATCH_PHOTOMETRIC_ZERO_POINT_MAG_KEY",
    "AUTO_MATCH_POSITION_SCORE_KEY",
    "AUTO_MATCH_PREDICTION_OFFSET_PX_KEY",
    "AUTO_MATCH_PREDICTION_OFFSET_RATIO_KEY",
    "AUTO_MATCH_QUALITY_FIELD_KEYS",
    "AUTO_MATCH_QUALITY_SCORE_KEY",
    "AutoMatchPhotometricEvaluation",
    "AutoMatchPhotometricModel",
    "AutoMatchPhotometrySample",
    "auto_match_position_quality",
    "combine_auto_match_quality",
    "evaluate_auto_match_photometry",
    "fit_auto_match_photometric_model",
    "instrumental_magnitude",
    "psf_brightness_proxy",
]
