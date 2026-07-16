"""不依赖 Qt 的恒星检测、去混叠和饱和兼容 PSF 测量。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sep
from scipy.optimize import least_squares

from .models import FittedStarPosition, StarSourceCandidate


_GAUSSIAN_FWHM_SCALE = 2.354820045


class StarFitError(ValueError):
    """带稳定错误码的星点检测或拟合异常。"""

    def __init__(self, message: str, *, code: str = "fit_failed") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _BackgroundModel:
    level: float
    x_slope: float
    y_slope: float
    noise: float
    plane: np.ndarray


@dataclass(frozen=True)
class _DetectionResult:
    candidates: tuple[StarSourceCandidate, ...]
    segmentation: np.ndarray
    background: _BackgroundModel
    saturation_level: float


def _robust_noise_from_differences(image: np.ndarray) -> float:
    """用相邻像素差估计噪声，降低星云、渐变和大亮星的影响。"""

    differences: list[np.ndarray] = []
    if image.shape[1] > 1:
        differences.append(np.diff(image, axis=1).ravel())
    if image.shape[0] > 1:
        differences.append(np.diff(image, axis=0).ravel())
    if not differences:
        return 1.0
    values = np.concatenate(differences)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad / math.sqrt(2.0), 0.35)


def _fit_background_plane(image: np.ndarray) -> _BackgroundModel:
    """从暗像素和边缘像素稳健估计局部线性背景。"""

    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    x_scale = max((width - 1.0) * 0.5, 1.0)
    y_scale = max((height - 1.0) * 0.5, 1.0)
    nx = (xx - (width - 1.0) * 0.5) / x_scale
    ny = (yy - (height - 1.0) * 0.5) / y_scale

    border_width = max(1, min(height, width) // 8)
    border = (
        (xx < border_width)
        | (xx >= width - border_width)
        | (yy < border_width)
        | (yy >= height - border_width)
    )
    low_limit = float(np.percentile(image, 62.0))
    mask = (image <= low_limit) | border
    design = np.column_stack((np.ones(mask.sum()), nx[mask], ny[mask]))
    values = image[mask]
    if values.size < 6:
        level = float(np.median(image))
        plane = np.full(image.shape, level, dtype=np.float64)
        return _BackgroundModel(level, 0.0, 0.0, _robust_noise_from_differences(image), plane)

    coefficients, *_unused = np.linalg.lstsq(design, values, rcond=None)
    for _iteration in range(3):
        residual = values - design @ coefficients
        residual_median = float(np.median(residual))
        residual_noise = max(1.4826 * float(np.median(np.abs(residual - residual_median))), 0.35)
        keep = (residual >= residual_median - 3.5 * residual_noise) & (
            residual <= residual_median + 2.0 * residual_noise
        )
        if int(np.count_nonzero(keep)) < 6:
            break
        coefficients, *_unused = np.linalg.lstsq(design[keep], values[keep], rcond=None)

    plane = coefficients[0] + coefficients[1] * nx + coefficients[2] * ny
    difference_noise = _robust_noise_from_differences(image - plane)
    return _BackgroundModel(
        level=float(coefficients[0]),
        x_slope=float(coefficients[1] / x_scale),
        y_slope=float(coefficients[2] / y_scale),
        noise=difference_noise,
        plane=np.asarray(plane, dtype=np.float64),
    )


def _infer_saturation_level(image: np.ndarray, saturation_level: float | None) -> float:
    if saturation_level is not None and math.isfinite(float(saturation_level)):
        return float(saturation_level)
    maximum = float(np.max(image))
    if maximum <= 1.5:
        return 1.0
    if maximum <= 255.5:
        return 255.0
    return maximum


def _resolve_array_saturation_level(array: np.ndarray, saturation_level: float | None) -> float | None:
    """在转换为浮点前保留整数图像的真实位深上限。"""

    if saturation_level is not None:
        return float(saturation_level)
    if np.issubdtype(array.dtype, np.integer):
        return float(np.iinfo(array.dtype).max)
    return None


def _detect_sources(
    image: np.ndarray,
    *,
    saturation_level: float | None = None,
    detection_sigma: float = 3.5,
) -> _DetectionResult:
    background = _fit_background_plane(image)
    signal = np.ascontiguousarray((image - background.plane).astype(np.float32))
    threshold = max(float(detection_sigma) * background.noise, 1.25)
    objects, segmentation = sep.extract(
        signal,
        threshold,
        minarea=3,
        deblend_nthresh=48,
        deblend_cont=0.001,
        clean=True,
        clean_param=1.0,
        segmentation_map=True,
    )
    saturation = _infer_saturation_level(image, saturation_level)
    saturation_cutoff = saturation - max(abs(saturation) * 0.002, 0.5)
    candidates: list[StarSourceCandidate] = []
    for object_index, source in enumerate(objects, start=1):
        x = float(source["x"])
        y = float(source["y"])
        major = max(float(source["a"]), 0.35)
        minor = max(float(source["b"]), 0.35)
        peak = max(float(source["peak"]), 0.0)
        flux = max(float(source["flux"]), 0.0)
        npix = max(int(source["npix"]), 1)
        source_mask = segmentation == object_index
        saturated_count = int(np.count_nonzero(source_mask & (image >= saturation_cutoff)))
        saturation_fraction = saturated_count / float(npix)
        snr = peak / max(background.noise, 1e-6)
        axis_ratio = major / max(minor, 1e-6)
        compactness_penalty = max(axis_ratio - 1.0, 0.0) * 0.10
        quality_score = snr / (1.0 + compactness_penalty)
        candidates.append(
            StarSourceCandidate(
                x=x,
                y=y,
                major_axis=major,
                minor_axis=minor,
                theta_rad=float(source["theta"]),
                flux=flux,
                peak=peak,
                snr=snr,
                npix=npix,
                label=object_index,
                saturated=saturated_count >= 3,
                saturation_fraction=saturation_fraction,
                blended=bool(int(source["flag"]) & 2),
                quality_score=quality_score,
            )
        )
    return _DetectionResult(tuple(candidates), segmentation, background, saturation)


def detect_star_candidates_from_array(
    luminance: np.ndarray,
    click_x: float,
    click_y: float,
    search_radius_px: int,
    *,
    saturation_level: float | None = None,
) -> list[StarSourceCandidate]:
    """检测搜索圆内的去混叠星源，并返回全图坐标。"""

    source_array = np.asarray(luminance)
    resolved_saturation = _resolve_array_saturation_level(source_array, saturation_level)
    image = np.asarray(source_array, dtype=np.float64)
    if image.ndim != 2:
        raise StarFitError("星点检测需要二维亮度图像。", code="invalid_image")
    if search_radius_px < 4:
        raise StarFitError("星点搜索半径过小。", code="invalid_radius")
    if not np.all(np.isfinite(image)):
        raise StarFitError("星点搜索窗口内存在无效像素。", code="invalid_image")
    height, width = image.shape
    if not (0.0 <= click_x < width and 0.0 <= click_y < height):
        raise StarFitError("点击位置不在真实图像范围内。", code="outside_image")

    margin = max(2, int(math.ceil(search_radius_px * 0.15)))
    crop_radius = int(search_radius_px + margin)
    center_x = int(round(click_x))
    center_y = int(round(click_y))
    x0 = max(0, center_x - crop_radius)
    x1 = min(width, center_x + crop_radius + 1)
    y0 = max(0, center_y - crop_radius)
    y1 = min(height, center_y + crop_radius + 1)
    patch = image[y0:y1, x0:x1]
    if min(patch.shape) < 7:
        raise StarFitError("点击位置太靠近图像边缘，无法检测星点。", code="edge")

    detection = _detect_sources(patch, saturation_level=resolved_saturation)
    candidates: list[StarSourceCandidate] = []
    for candidate in detection.candidates:
        image_x = candidate.x + x0
        image_y = candidate.y + y0
        distance = float(np.hypot(image_x - click_x, image_y - click_y))
        if distance > float(search_radius_px):
            continue
        candidates.append(
            StarSourceCandidate(
                x=image_x,
                y=image_y,
                major_axis=candidate.major_axis,
                minor_axis=candidate.minor_axis,
                theta_rad=candidate.theta_rad,
                flux=candidate.flux,
                peak=candidate.peak,
                snr=candidate.snr,
                npix=candidate.npix,
                label=candidate.label,
                saturated=candidate.saturated,
                saturation_fraction=candidate.saturation_fraction,
                blended=candidate.blended,
                quality_score=candidate.quality_score,
            )
        )
    candidates.sort(key=lambda item: (float(np.hypot(item.x - click_x, item.y - click_y)), -item.quality_score))
    return candidates


def _select_candidate(
    detection: _DetectionResult,
    click_x: float,
    click_y: float,
    search_radius_px: int,
    *,
    reject_ambiguous: bool,
    selection_mode: str,
) -> StarSourceCandidate:
    eligible: list[tuple[float, float, StarSourceCandidate]] = []
    for candidate in detection.candidates:
        distance = float(np.hypot(candidate.x - click_x, candidate.y - click_y))
        if distance > float(search_radius_px):
            continue
        axis_ratio = candidate.major_axis / max(candidate.minor_axis, 1e-6)
        if candidate.snr < 3.5 or axis_ratio > (5.0 if candidate.saturated else 3.8):
            continue
        if selection_mode == "manual":
            manual_distance_limit = max(
                5.0,
                candidate.major_axis * (2.2 if candidate.saturated else 1.6),
                min(float(search_radius_px) * 0.35, 12.0),
            )
            if distance > manual_distance_limit:
                continue
        distance_score = distance / max(float(search_radius_px), 1.0)
        shape_penalty = max(axis_ratio - 1.8, 0.0) * 0.12
        score = distance_score + shape_penalty
        eligible.append((score, distance, candidate))
    if not eligible:
        raise StarFitError("搜索范围内没有检测到形态可信的星点。", code="no_source")
    eligible.sort(key=lambda item: (item[0], item[1], -item[2].quality_score))
    best_score, best_distance, best = eligible[0]
    if reject_ambiguous and len(eligible) > 1:
        second_score, second_distance, _second = eligible[1]
        score_margin = second_score - best_score
        distance_margin = second_distance - best_distance
        if score_margin < 0.12 and distance_margin < max(1.5, search_radius_px * 0.12):
            raise StarFitError("搜索范围内有多颗距离相近的星点，无法可靠确定目标。", code="ambiguous")
    return best


def _candidate_neighbor_mask(
    detection: _DetectionResult,
    selected: StarSourceCandidate,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """屏蔽已去混叠邻星的主体和紧邻星翼。"""

    keep = (detection.segmentation == 0) | (detection.segmentation == selected.label)
    for candidate in detection.candidates:
        if candidate.label == selected.label:
            continue
        radius = max(2.5, min(12.0, candidate.major_axis * 2.8))
        keep &= (xx - candidate.x) ** 2 + (yy - candidate.y) ** 2 > radius**2
    return keep


def _moffat_fwhm(alpha: float, beta: float) -> float:
    return 2.0 * alpha * math.sqrt(max(2.0 ** (1.0 / max(beta, 1e-6)) - 1.0, 1e-9))


def _radial_quality(
    signal: np.ndarray,
    x: float,
    y: float,
    noise: float,
    radius: float,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    """返回径向单调性和方位覆盖率，用于排除地景边缘与纹理。"""

    yy, xx = np.indices(signal.shape, dtype=np.float64)
    distance = np.hypot(xx - x, yy - y)
    edges = np.linspace(0.0, max(radius, 3.0), 7)
    medians: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        annulus = valid_mask & (distance >= left) & (distance < right)
        medians.append(float(np.median(signal[annulus])) if np.any(annulus) else 0.0)
    decreasing = sum(medians[index] >= medians[index + 1] - noise for index in range(len(medians) - 1))
    monotonicity = decreasing / max(len(medians) - 1, 1)

    inner_radius = max(radius * 0.30, 1.5)
    outer_radius = max(radius * 0.85, inner_radius + 1.0)
    sector_positive = 0
    sector_total = 12
    angle = np.arctan2(yy - y, xx - x)
    for sector in range(sector_total):
        left = -math.pi + sector * 2.0 * math.pi / sector_total
        right = -math.pi + (sector + 1) * 2.0 * math.pi / sector_total
        sector_mask = (
            valid_mask
            & (distance >= inner_radius)
            & (distance <= outer_radius)
            & (angle >= left)
            & (angle < right)
        )
        if np.any(sector_mask) and float(np.percentile(signal[sector_mask], 65.0)) > noise:
            sector_positive += 1
    return monotonicity, sector_positive / float(sector_total)


def _fit_selected_candidate(
    image: np.ndarray,
    detection: _DetectionResult,
    selected: StarSourceCandidate,
    *,
    max_fit_radius_px: int,
) -> FittedStarPosition:
    height, width = image.shape
    adaptive_radius = int(
        math.ceil(
            max(
                5.0,
                selected.major_axis * (4.8 if selected.saturated else 3.5),
                math.sqrt(max(selected.npix, 1) / math.pi) * (2.2 if selected.saturated else 1.6),
            )
        )
    )
    fit_radius = min(max(adaptive_radius, 5), max(int(max_fit_radius_px), 5))
    center_x = int(round(selected.x))
    center_y = int(round(selected.y))
    x0 = max(0, center_x - fit_radius)
    x1 = min(width, center_x + fit_radius + 1)
    y0 = max(0, center_y - fit_radius)
    y1 = min(height, center_y + fit_radius + 1)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    background_plane = detection.background.plane[y0:y1, x0:x1]
    signal = patch - background_plane
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    local_selected = StarSourceCandidate(
        x=selected.x - x0,
        y=selected.y - y0,
        major_axis=selected.major_axis,
        minor_axis=selected.minor_axis,
        theta_rad=selected.theta_rad,
        flux=selected.flux,
        peak=selected.peak,
        snr=selected.snr,
        npix=selected.npix,
        label=selected.label,
        saturated=selected.saturated,
        saturation_fraction=selected.saturation_fraction,
        blended=selected.blended,
        quality_score=selected.quality_score,
    )
    local_candidates = tuple(
        StarSourceCandidate(
            x=candidate.x - x0,
            y=candidate.y - y0,
            major_axis=candidate.major_axis,
            minor_axis=candidate.minor_axis,
            theta_rad=candidate.theta_rad,
            flux=candidate.flux,
            peak=candidate.peak,
            snr=candidate.snr,
            npix=candidate.npix,
            label=candidate.label,
            saturated=candidate.saturated,
            saturation_fraction=candidate.saturation_fraction,
            blended=candidate.blended,
            quality_score=candidate.quality_score,
        )
        for candidate in detection.candidates
    )
    local_detection = _DetectionResult(
        local_candidates,
        detection.segmentation[y0:y1, x0:x1],
        detection.background,
        detection.saturation_level,
    )
    neighbor_keep = _candidate_neighbor_mask(local_detection, local_selected, xx, yy)
    circular = (xx - local_selected.x) ** 2 + (yy - local_selected.y) ** 2 <= fit_radius**2
    saturation_cutoff = detection.saturation_level - max(abs(detection.saturation_level) * 0.002, 0.5)
    unsaturated = patch < saturation_cutoff
    valid = circular & neighbor_keep & unsaturated & np.isfinite(patch)
    if int(np.count_nonzero(valid)) < 24:
        valid = circular & neighbor_keep & np.isfinite(patch)
    if int(np.count_nonzero(valid)) < 24:
        raise StarFitError("星点周围可用于测量的有效像素不足。", code="insufficient_pixels")

    amplitude0 = max(selected.peak, detection.background.noise * 4.0, 1.0)
    alpha_x0 = max(selected.major_axis * 1.35, 0.8)
    alpha_y0 = max(selected.minor_axis * 1.35, 0.8)
    beta0 = 2.5
    initial = np.asarray(
        [amplitude0, local_selected.x, local_selected.y, alpha_x0, alpha_y0, beta0, local_selected.theta_rad],
        dtype=np.float64,
    )
    center_limit = min(max(2.5, selected.major_axis * 0.75), max(fit_radius * 0.35, 2.5))
    lower = np.asarray(
        [0.0, local_selected.x - center_limit, local_selected.y - center_limit, 0.45, 0.45, 1.15, -math.pi],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            max(amplitude0 * 30.0, detection.saturation_level * 8.0, 255.0),
            local_selected.x + center_limit,
            local_selected.y + center_limit,
            max(fit_radius * 0.80, 1.0),
            max(fit_radius * 0.80, 1.0),
            8.0,
            math.pi,
        ],
        dtype=np.float64,
    )
    initial = np.minimum(np.maximum(initial, lower + 1e-6), upper - 1e-6)

    valid_x = xx[valid]
    valid_y = yy[valid]
    valid_signal = signal[valid]
    residual_scale = max(detection.background.noise * 2.0, 1.0)

    def residual(params: np.ndarray) -> np.ndarray:
        amplitude, fit_x, fit_y, alpha_x, alpha_y, beta, theta = params
        cosine = math.cos(theta)
        sine = math.sin(theta)
        dx = valid_x - fit_x
        dy = valid_y - fit_y
        rx = (cosine * dx + sine * dy) / alpha_x
        ry = (-sine * dx + cosine * dy) / alpha_y
        model = amplitude * (1.0 + rx * rx + ry * ry) ** (-beta)
        return model - valid_signal

    try:
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=residual_scale,
            max_nfev=320,
        )
    except (TypeError, ValueError) as exc:
        raise StarFitError("候选目标无法建立稳定的 PSF 初值。", code="invalid_initial_fit") from exc
    if not result.success or not np.all(np.isfinite(result.x)):
        raise StarFitError("PSF 拟合未收敛。", code="not_converged")

    amplitude, fit_x, fit_y, alpha_x, alpha_y, beta, theta = (float(value) for value in result.x)
    fwhm_x = _moffat_fwhm(alpha_x, beta)
    fwhm_y = _moffat_fwhm(alpha_y, beta)
    if fwhm_y > fwhm_x:
        fwhm_x, fwhm_y = fwhm_y, fwhm_x
        theta += math.pi * 0.5
    theta = (theta + math.pi) % math.pi
    sigma_x = fwhm_x / _GAUSSIAN_FWHM_SCALE
    sigma_y = fwhm_y / _GAUSSIAN_FWHM_SCALE
    fit_residual = residual(result.x)
    signal_scale = max(float(np.percentile(np.abs(valid_signal), 90.0)), detection.background.noise, 1.0)
    fit_error = float(np.sqrt(np.mean(np.square(fit_residual))) / signal_scale)
    axis_ratio = fwhm_x / max(fwhm_y, 1e-6)
    center_shift = float(np.hypot(fit_x - local_selected.x, fit_y - local_selected.y))
    monotonicity, angular_coverage = _radial_quality(
        signal,
        fit_x,
        fit_y,
        detection.background.noise,
        min(max(fwhm_x * 1.5, 3.0), fit_radius),
        circular & neighbor_keep,
    )
    snr = amplitude / max(detection.background.noise, 1e-6)

    if snr < 4.0:
        raise StarFitError("候选目标相对于局部背景过弱。", code="low_snr")
    if center_shift > max(2.2, selected.major_axis * 0.65):
        raise StarFitError("PSF 中心偏离检测星源，可能受到邻星或地景影响。", code="center_unstable")
    if axis_ratio > (5.0 if selected.saturated else 3.5):
        raise StarFitError("候选目标过于狭长，不符合可用星点形态。", code="elongated")
    if max(fwhm_x, fwhm_y) >= fit_radius * 1.35:
        raise StarFitError("PSF 尺寸触及拟合窗口边界，结果不可靠。", code="size_at_bound")
    residual_limit = 0.52 if selected.saturated else 0.42
    if fit_error > residual_limit:
        raise StarFitError("候选目标无法被稳定的恒星 PSF 描述。", code="poor_fit")
    if monotonicity < (0.55 if selected.saturated else 0.65):
        raise StarFitError("候选目标的径向亮度变化不像星点。", code="non_stellar_profile")
    if angular_coverage < (0.35 if selected.saturated else 0.45):
        raise StarFitError("候选目标缺少连续的星点外围轮廓。", code="non_stellar_profile")

    quality_score = max(
        0.0,
        min(
            1.0,
            0.35 * min(snr / 20.0, 1.0)
            + 0.25 * max(0.0, 1.0 - fit_error / residual_limit)
            + 0.20 * monotonicity
            + 0.20 * angular_coverage,
        ),
    )
    if quality_score < 0.62:
        raise StarFitError("候选目标的综合 PSF 质量不足。", code="poor_quality")
    image_x = float(x0 + fit_x)
    image_y = float(y0 + fit_y)
    background_x = int(np.clip(round(image_x), 0, width - 1))
    background_y = int(np.clip(round(image_y), 0, height - 1))
    return FittedStarPosition(
        x=image_x,
        y=image_y,
        amplitude=amplitude,
        background=float(detection.background.plane[background_y, background_x]),
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta_rad=theta,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        snr=snr,
        fit_error=fit_error,
        saturated=selected.saturated,
        saturation_fraction=selected.saturation_fraction,
        blended=selected.blended,
        quality_score=quality_score,
    )


def fit_star_position_from_array(
    luminance: np.ndarray,
    click_x: float,
    click_y: float,
    radius_px: int = 12,
    *,
    max_fit_radius_px: int | None = None,
    reject_ambiguous: bool = False,
    saturation_level: float | None = None,
    selection_mode: str = "manual",
) -> FittedStarPosition:
    """先检测/去混叠，再用独立自适应窗口拟合单颗恒星。"""

    source_array = np.asarray(luminance)
    resolved_saturation = _resolve_array_saturation_level(source_array, saturation_level)
    image = np.asarray(source_array, dtype=np.float64)
    if image.ndim != 2:
        raise StarFitError("星点拟合需要二维亮度图像。", code="invalid_image")
    if radius_px < 4:
        raise StarFitError("星点搜索半径过小。", code="invalid_radius")
    height, width = image.shape
    if not (0.0 <= click_x < width and 0.0 <= click_y < height):
        raise StarFitError("点击位置不在真实图像范围内。", code="outside_image")
    if not np.all(np.isfinite(image)):
        raise StarFitError("星点搜索窗口内存在无效像素。", code="invalid_image")

    detection = _detect_sources(image, saturation_level=resolved_saturation)
    selected = _select_candidate(
        detection,
        click_x,
        click_y,
        radius_px,
        reject_ambiguous=reject_ambiguous,
        selection_mode="predicted" if selection_mode == "predicted" else "manual",
    )
    fit_radius_limit = max_fit_radius_px
    if fit_radius_limit is None:
        fit_radius_limit = 48
    return _fit_selected_candidate(
        image,
        detection,
        selected,
        max_fit_radius_px=fit_radius_limit,
    )
