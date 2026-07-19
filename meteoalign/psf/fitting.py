"""不依赖 Qt 的恒星检测、去混叠和饱和兼容 PSF 测量。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sep
from scipy.ndimage import gaussian_filter, label as label_connected
from scipy.optimize import least_squares

from .models import FittedStarPosition, StarSourceCandidate


_GAUSSIAN_FWHM_SCALE = 2.354820045
_STRONG_DETECTED_AXIS_RATIO = 1.45
_MAX_MOMENT_TO_CORE_FWHM_RATIO = 2.0


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


@dataclass(frozen=True)
class _HalfMaximumCore:
    x: float
    y: float
    major_fwhm: float
    minor_fwhm: float
    theta_rad: float
    npix: int


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
    if source_array.ndim != 2:
        raise StarFitError("星点检测需要二维亮度图像。", code="invalid_image")
    if search_radius_px < 4:
        raise StarFitError("星点搜索半径过小。", code="invalid_radius")
    height, width = source_array.shape
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
    # 序列处理会传入整帧灰度图；先裁剪再转浮点，避免每颗星都复制整张照片。
    patch = np.asarray(source_array[y0:y1, x0:x1], dtype=np.float64)
    if min(patch.shape) < 7:
        raise StarFitError("点击位置太靠近图像边缘，无法检测星点。", code="edge")
    if not np.all(np.isfinite(patch)):
        raise StarFitError("星点搜索窗口内存在无效像素。", code="invalid_image")

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


def measure_star_candidate_fast(candidate: StarSourceCandidate) -> FittedStarPosition:
    """用 SEP 的亚像素矩心和二阶矩快速生成序列星点测量结果。"""

    if not all(
        math.isfinite(float(value))
        for value in (
            candidate.x,
            candidate.y,
            candidate.major_axis,
            candidate.minor_axis,
            candidate.theta_rad,
            candidate.peak,
            candidate.snr,
        )
    ):
        raise StarFitError("快速星点测量收到无效候选数据。", code="invalid_candidate")
    sigma_x = max(abs(float(candidate.major_axis)), 0.35)
    sigma_y = max(abs(float(candidate.minor_axis)), 0.35)
    theta = float(candidate.theta_rad)
    if sigma_y > sigma_x:
        sigma_x, sigma_y = sigma_y, sigma_x
        theta += math.pi * 0.5
    theta = (theta + math.pi) % math.pi
    axis_ratio = sigma_x / max(sigma_y, 1e-6)
    if float(candidate.snr) < 3.5:
        raise StarFitError("候选目标相对于局部背景过弱。", code="low_snr")
    if axis_ratio > (5.0 if candidate.saturated else 3.8):
        raise StarFitError("候选目标过于狭长，不符合可用星点形态。", code="elongated")
    snr_quality = min(max(float(candidate.snr) / 20.0, 0.0), 1.0)
    shape_quality = min(max(1.0 - max(axis_ratio - 1.0, 0.0) / 4.0, 0.0), 1.0)
    quality_score = 0.75 * snr_quality + 0.25 * shape_quality
    return FittedStarPosition(
        x=float(candidate.x),
        y=float(candidate.y),
        amplitude=max(float(candidate.peak), 0.0),
        background=0.0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta_rad=theta,
        fwhm_x=sigma_x * _GAUSSIAN_FWHM_SCALE,
        fwhm_y=sigma_y * _GAUSSIAN_FWHM_SCALE,
        snr=max(float(candidate.snr), 0.0),
        # 快速路径没有模型残差，沿用结果结构的兼容默认值。
        fit_error=0.0,
        saturated=bool(candidate.saturated),
        saturation_fraction=float(candidate.saturation_fraction),
        blended=bool(candidate.blended),
        quality_score=quality_score,
    )


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


def _select_forced_candidate(
    detection: _DetectionResult,
    click_x: float,
    click_y: float,
    search_radius_px: int,
) -> StarSourceCandidate | None:
    """手动强制测量时只按距离和信噪比选星，不再用形态门槛拒绝。"""

    eligible = [
        candidate
        for candidate in detection.candidates
        if float(np.hypot(candidate.x - click_x, candidate.y - click_y))
        <= float(search_radius_px)
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            float(np.hypot(candidate.x - click_x, candidate.y - click_y)),
            -candidate.snr,
        ),
    )


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


def _bic_score(
    residual: np.ndarray,
    parameter_count: int,
    *,
    residual_clip: float | None = None,
) -> float:
    """用稳健 BIC 比较 PSF，避免局部次峰或背景纹理制造长轴。"""

    sample_count = max(int(residual.size), 1)
    compared_residual = np.asarray(residual, dtype=np.float64)
    if (
        residual_clip is not None
        and math.isfinite(float(residual_clip))
        and float(residual_clip) > 0.0
    ):
        clip = float(residual_clip)
        compared_residual = np.clip(compared_residual, -clip, clip)
    mean_square = max(
        float(np.mean(np.square(compared_residual))),
        np.finfo(np.float64).tiny,
    )
    return sample_count * math.log(mean_square) + parameter_count * math.log(sample_count)


def _normalized_tolerance_multiplier(value: float) -> float:
    """把直接数值接口收到的非法容限倍率恢复为不改变旧行为的 1。"""

    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        return 1.0
    return multiplier if math.isfinite(multiplier) and multiplier > 0.0 else 1.0


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


def _half_maximum_core(
    image: np.ndarray,
    detection: _DetectionResult,
    selected: StarSourceCandidate,
) -> _HalfMaximumCore | None:
    """测量选中源的半峰值连通核心，用来识别被地景边缘撑大的分割源。"""

    source_mask = (detection.segmentation == selected.label) & np.isfinite(image)
    if not np.any(source_mask):
        return None
    peak_index = int(np.argmax(np.where(source_mask, image, -np.inf)))
    peak_y, peak_x = np.unravel_index(peak_index, image.shape)
    peak = float(image[peak_y, peak_x])
    background = float(detection.background.plane[peak_y, peak_x])
    contrast = peak - background
    if not math.isfinite(contrast) or contrast < detection.background.noise * 4.0:
        return None

    half_maximum = source_mask & (image >= background + contrast * 0.5)
    labels, _label_count = label_connected(
        half_maximum,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    peak_label = int(labels[peak_y, peak_x])
    if peak_label <= 0:
        return None
    component = labels == peak_label
    yy, xx = np.nonzero(component)
    if xx.size < 4:
        return None

    component_signal = np.maximum(
        image[component] - detection.background.plane[component],
        0.0,
    )
    weight_sum = float(np.sum(component_signal))
    if weight_sum > 0.0 and math.isfinite(weight_sum):
        center_x = float(np.sum(xx * component_signal) / weight_sum)
        center_y = float(np.sum(yy * component_signal) / weight_sum)
    else:
        center_x = float(np.mean(xx))
        center_y = float(np.mean(yy))

    shape_center_x = float(np.mean(xx))
    shape_center_y = float(np.mean(yy))
    dx = xx.astype(np.float64) - shape_center_x
    dy = yy.astype(np.float64) - shape_center_y
    covariance = np.asarray(
        [
            [float(np.mean(dx * dx)), float(np.mean(dx * dy))],
            [float(np.mean(dx * dy)), float(np.mean(dy * dy))],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.25**2)
    major_index = int(np.argmax(eigenvalues))
    minor_index = 1 - major_index
    major_fwhm = max(4.0 * math.sqrt(float(eigenvalues[major_index])), 1.0)
    minor_fwhm = max(4.0 * math.sqrt(float(eigenvalues[minor_index])), 1.0)
    major_vector = eigenvectors[:, major_index]
    theta = float(math.atan2(float(major_vector[1]), float(major_vector[0])) % math.pi)
    return _HalfMaximumCore(
        x=center_x,
        y=center_y,
        major_fwhm=major_fwhm,
        minor_fwhm=minor_fwhm,
        theta_rad=theta,
        npix=int(xx.size),
    )


def _stabilize_contaminated_candidate(
    image: np.ndarray,
    detection: _DetectionResult,
    selected: StarSourceCandidate,
) -> StarSourceCandidate:
    """当检测矩远大于半峰值核心时，用核心恢复星点中心和自适应窗口初值。"""

    core = _half_maximum_core(image, detection, selected)
    if core is None:
        return selected
    moment_fwhm = selected.major_axis * _GAUSSIAN_FWHM_SCALE
    if moment_fwhm <= core.major_fwhm * _MAX_MOMENT_TO_CORE_FWHM_RATIO:
        return selected
    return StarSourceCandidate(
        x=core.x,
        y=core.y,
        major_axis=max(core.major_fwhm / _GAUSSIAN_FWHM_SCALE, 0.35),
        minor_axis=max(core.minor_fwhm / _GAUSSIAN_FWHM_SCALE, 0.35),
        theta_rad=core.theta_rad,
        flux=selected.flux,
        peak=selected.peak,
        snr=selected.snr,
        npix=core.npix,
        label=selected.label,
        saturated=selected.saturated,
        saturation_fraction=selected.saturation_fraction,
        blended=selected.blended,
        quality_score=selected.quality_score,
    )


def _fit_selected_candidate(
    image: np.ndarray,
    detection: _DetectionResult,
    selected: StarSourceCandidate,
    *,
    max_fit_radius_px: int,
    use_local_background: bool = False,
    fit_error_limit: float | None = None,
    saturated_fit_error_limit: float | None = None,
    center_shift_tolerance_multiplier: float = 1.0,
    size_boundary_tolerance_multiplier: float = 1.0,
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
    fit_radius_limit = max(int(max_fit_radius_px), 5)
    fit_radius = min(max(adaptive_radius, 5), fit_radius_limit)
    center_x = int(round(selected.x))
    center_y = int(round(selected.y))
    x0 = max(0, center_x - fit_radius)
    x1 = min(width, center_x + fit_radius + 1)
    y0 = max(0, center_y - fit_radius)
    y1 = min(height, center_y + fit_radius + 1)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if use_local_background:
        local_background = _fit_background_plane(patch)
        background_plane = local_background.plane
        fit_noise = local_background.noise
    else:
        background_plane = detection.background.plane[y0:y1, x0:x1]
        fit_noise = detection.background.noise
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

    amplitude0 = max(selected.peak, fit_noise * 4.0, 1.0)
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
    residual_scale = max(fit_noise * 2.0, 1.0)

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
    fit_residual = residual(result.x)

    circular_initial = np.asarray(
        [amplitude, fit_x, fit_y, math.sqrt(max(alpha_x * alpha_y, 0.45**2)), beta],
        dtype=np.float64,
    )
    circular_lower = np.asarray(
        [lower[0], lower[1], lower[2], lower[3], lower[5]],
        dtype=np.float64,
    )
    circular_upper = np.asarray(
        [upper[0], upper[1], upper[2], max(upper[3], upper[4]), upper[5]],
        dtype=np.float64,
    )
    circular_initial = np.minimum(
        np.maximum(circular_initial, circular_lower + 1e-6),
        circular_upper - 1e-6,
    )

    def circular_residual(params: np.ndarray) -> np.ndarray:
        circular_amplitude, circular_x, circular_y, alpha, circular_beta = params
        radius_sq = (valid_x - circular_x) ** 2 + (valid_y - circular_y) ** 2
        model = circular_amplitude * (1.0 + radius_sq / (alpha * alpha)) ** (-circular_beta)
        return model - valid_signal

    circular_result = None
    raw_axis_ratio = max(alpha_x, alpha_y) / max(min(alpha_x, alpha_y), 1e-6)
    if raw_axis_ratio >= 1.10:
        try:
            circular_result = least_squares(
                circular_residual,
                circular_initial,
                bounds=(circular_lower, circular_upper),
                loss="soft_l1",
                f_scale=residual_scale,
                max_nfev=240,
            )
        except (TypeError, ValueError):
            pass
    if (
        circular_result is not None
        and circular_result.success
        and np.all(np.isfinite(circular_result.x))
    ):
        candidate_residual = circular_residual(circular_result.x)
        candidate_fwhm = _moffat_fwhm(
            float(circular_result.x[3]),
            float(circular_result.x[4]),
        )
        candidate_center_shift = float(
            np.hypot(
                float(circular_result.x[1]) - local_selected.x,
                float(circular_result.x[2]) - local_selected.y,
            )
        )
        candidate_size_limit = (
            fit_radius
            * 1.35
            * _normalized_tolerance_multiplier(size_boundary_tolerance_multiplier)
        )
        candidate_center_limit = max(
            2.2,
            selected.major_axis * 0.65,
        ) * _normalized_tolerance_multiplier(center_shift_tolerance_multiplier)
        detected_axis_ratio = selected.major_axis / max(selected.minor_axis, 1e-6)
        # 检测矩本身已有明显长轴时保留完整残差信息；近圆源则截掉局部次峰的支配作用。
        comparison_clip = (
            None
            if detected_axis_ratio >= _STRONG_DETECTED_AXIS_RATIO
            else fit_noise
        )
        # 椭圆模型多两个形状自由度；只有证据明显时才保留其长短轴与方向。
        if (
            candidate_fwhm < candidate_size_limit
            and candidate_center_shift <= candidate_center_limit
            and _bic_score(
                fit_residual,
                7,
                residual_clip=comparison_clip,
            )
            + 6.0
            >= _bic_score(
                candidate_residual,
                5,
                residual_clip=comparison_clip,
            )
        ):
            amplitude, fit_x, fit_y, alpha_x, beta = (
                float(value) for value in circular_result.x
            )
            alpha_y = alpha_x
            theta = 0.0
            fit_residual = candidate_residual

    fwhm_x = _moffat_fwhm(alpha_x, beta)
    fwhm_y = _moffat_fwhm(alpha_y, beta)
    if fwhm_y > fwhm_x:
        fwhm_x, fwhm_y = fwhm_y, fwhm_x
        theta += math.pi * 0.5
    theta = (theta + math.pi) % math.pi
    sigma_x = fwhm_x / _GAUSSIAN_FWHM_SCALE
    sigma_y = fwhm_y / _GAUSSIAN_FWHM_SCALE
    signal_scale = max(float(np.percentile(np.abs(valid_signal), 90.0)), fit_noise, 1.0)
    fit_error = float(np.sqrt(np.mean(np.square(fit_residual))) / signal_scale)
    axis_ratio = fwhm_x / max(fwhm_y, 1e-6)
    center_shift = float(np.hypot(fit_x - local_selected.x, fit_y - local_selected.y))
    monotonicity, angular_coverage = _radial_quality(
        signal,
        fit_x,
        fit_y,
        fit_noise,
        min(max(fwhm_x * 1.5, 3.0), fit_radius),
        circular & neighbor_keep,
    )
    snr = amplitude / max(fit_noise, 1e-6)

    if snr < 4.0:
        raise StarFitError("候选目标相对于局部背景过弱。", code="low_snr")
    center_shift_limit = max(2.2, selected.major_axis * 0.65) * _normalized_tolerance_multiplier(
        center_shift_tolerance_multiplier
    )
    if center_shift > center_shift_limit:
        raise StarFitError("PSF 中心偏离检测星源，可能受到邻星或地景影响。", code="center_unstable")
    if axis_ratio > (5.0 if selected.saturated else 3.5):
        raise StarFitError("候选目标过于狭长，不符合可用星点形态。", code="elongated")
    size_boundary_limit = (
        fit_radius
        * 1.35
        * _normalized_tolerance_multiplier(size_boundary_tolerance_multiplier)
    )
    if max(fwhm_x, fwhm_y) >= size_boundary_limit:
        raise StarFitError("PSF 尺寸触及拟合窗口边界，结果不可靠。", code="size_at_bound")
    default_residual_limit = 0.52 if selected.saturated else 0.42
    configured_residual_limit = (
        saturated_fit_error_limit if selected.saturated else fit_error_limit
    )
    residual_limit = (
        float(configured_residual_limit)
        if configured_residual_limit is not None
        and math.isfinite(float(configured_residual_limit))
        and float(configured_residual_limit) > 0.0
        else default_residual_limit
    )
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
    background_x = int(np.clip(round(fit_x), 0, patch.shape[1] - 1))
    background_y = int(np.clip(round(fit_y), 0, patch.shape[0] - 1))
    return FittedStarPosition(
        x=image_x,
        y=image_y,
        amplitude=amplitude,
        background=float(background_plane[background_y, background_x]),
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


def _forced_windowed_centroid(
    image: np.ndarray,
    detection: _DetectionResult,
    click_x: float,
    click_y: float,
    search_radius_px: int,
    max_fit_radius_px: int,
    selected: StarSourceCandidate | None,
    *,
    selected_was_stabilized: bool = False,
) -> FittedStarPosition:
    """以用户选星圈为先验，用高斯窗口矩心提供不依赖完整 PSF 的最终兜底。"""

    signal = np.asarray(image - detection.background.plane, dtype=np.float64)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    if selected is not None:
        center_x = float(selected.x)
        center_y = float(selected.y)
        source_scale = max(float(selected.major_axis), float(selected.minor_axis), 1.0)
    else:
        search_mask = (xx - click_x) ** 2 + (yy - click_y) ** 2 <= float(search_radius_px) ** 2
        smoothed = gaussian_filter(
            np.maximum(signal - detection.background.noise, 0.0),
            sigma=1.0,
            mode="nearest",
        )
        if not np.any(search_mask):
            raise StarFitError("选星圈内没有可用于强制测量的像素。", code="insufficient_pixels")
        peak_values = np.where(search_mask, smoothed, -np.inf)
        peak_index = int(np.argmax(peak_values))
        if not math.isfinite(float(peak_values.flat[peak_index])):
            raise StarFitError("选星圈内无法确定可靠亮度峰值。", code="no_source")
        center_y, center_x = (float(value) for value in np.unravel_index(peak_index, image.shape))
        source_scale = 2.0

    if selected_was_stabilized:
        # 半峰值核心说明原检测矩已被邻星或地景撑大；兜底时也必须限制窗口，
        # 否则会重新把刚排除的污染源累积进矩心。
        window_radius = min(
            max(int(max_fit_radius_px), 5),
            max(6, int(math.ceil(source_scale * 2.5))),
        )
        window_sigma = max(window_radius / 3.0, source_scale, 2.0)
    else:
        window_radius = min(
            max(int(max_fit_radius_px), 5),
            max(8, int(math.ceil(source_scale * 5.0))),
        )
        window_sigma = max(window_radius / 3.0, source_scale * 1.5, 2.0)
    # 去掉一倍局部噪声后再算矩心，避免大窗口中半数为正的背景起伏累积成假星翼。
    positive_signal = np.maximum(signal - detection.background.noise, 0.0)
    valid = np.isfinite(signal)
    for _iteration in range(8):
        distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        local = valid & (distance_sq <= window_radius**2)
        weights = np.where(
            local,
            positive_signal * np.exp(-0.5 * distance_sq / window_sigma**2),
            0.0,
        )
        weight_sum = float(np.sum(weights))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            raise StarFitError("选星圈内缺少可用于强制矩心测量的星点信号。", code="low_snr")
        next_x = float(np.sum(weights * xx) / weight_sum)
        next_y = float(np.sum(weights * yy) / weight_sum)
        if float(np.hypot(next_x - center_x, next_y - center_y)) < 1e-3:
            center_x, center_y = next_x, next_y
            break
        center_x, center_y = next_x, next_y

    distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    local = valid & (distance_sq <= window_radius**2)
    weights = np.where(
        local,
        positive_signal * np.exp(-0.5 * distance_sq / window_sigma**2),
        0.0,
    )
    weight_sum = max(float(np.sum(weights)), 1e-9)
    dx = xx - center_x
    dy = yy - center_y
    covariance = np.asarray(
        [
            [
                float(np.sum(weights * dx * dx) / weight_sum),
                float(np.sum(weights * dx * dy) / weight_sum),
            ],
            [
                float(np.sum(weights * dx * dy) / weight_sum),
                float(np.sum(weights * dy * dy) / weight_sum),
            ],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.35**2)
    # 强制路径只保证中心可用，二阶矩很容易被地平线纹理拉长，不把它伪装成可信形状。
    circular_sigma = float(math.sqrt(float(np.min(eigenvalues))))
    amplitude = max(float(np.max(positive_signal[local])) if np.any(local) else 0.0, 0.0)
    snr = amplitude / max(detection.background.noise, 1e-6)
    background_x = int(np.clip(round(center_x), 0, image.shape[1] - 1))
    background_y = int(np.clip(round(center_y), 0, image.shape[0] - 1))
    return FittedStarPosition(
        x=center_x,
        y=center_y,
        amplitude=amplitude,
        background=float(detection.background.plane[background_y, background_x]),
        sigma_x=circular_sigma,
        sigma_y=circular_sigma,
        theta_rad=0.0,
        fwhm_x=circular_sigma * _GAUSSIAN_FWHM_SCALE,
        fwhm_y=circular_sigma * _GAUSSIAN_FWHM_SCALE,
        snr=max(snr, 0.0),
        fit_error=1.0,
        saturated=bool(selected.saturated) if selected is not None else False,
        saturation_fraction=(
            float(selected.saturation_fraction) if selected is not None else 0.0
        ),
        blended=bool(selected.blended) if selected is not None else False,
        quality_score=0.35,
        forced=True,
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
    fit_error_limit: float | None = None,
    saturated_fit_error_limit: float | None = None,
    center_shift_tolerance_multiplier: float = 1.0,
    size_boundary_tolerance_multiplier: float = 1.0,
    force_reliable_source: bool = False,
) -> FittedStarPosition:
    """先严格拟合；手动可信模式失败时退回保守的加权矩心。"""

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
    fit_radius_limit = max_fit_radius_px
    if fit_radius_limit is None:
        fit_radius_limit = 48
    selected: StarSourceCandidate | None = None
    selected_was_stabilized = False
    try:
        selected = _select_candidate(
            detection,
            click_x,
            click_y,
            radius_px,
            reject_ambiguous=reject_ambiguous,
            selection_mode="predicted" if selection_mode == "predicted" else "manual",
        )
        stabilized_selected = _stabilize_contaminated_candidate(image, detection, selected)
        selected_was_stabilized = stabilized_selected is not selected
        selected = stabilized_selected
        return _fit_selected_candidate(
            image,
            detection,
            selected,
            max_fit_radius_px=fit_radius_limit,
            use_local_background=selected_was_stabilized,
            fit_error_limit=fit_error_limit,
            saturated_fit_error_limit=saturated_fit_error_limit,
            center_shift_tolerance_multiplier=center_shift_tolerance_multiplier,
            size_boundary_tolerance_multiplier=size_boundary_tolerance_multiplier,
        )
    except StarFitError:
        if not force_reliable_source:
            raise

    if selected is None:
        selected = _select_forced_candidate(detection, click_x, click_y, radius_px)
        if selected is not None:
            stabilized_selected = _stabilize_contaminated_candidate(image, detection, selected)
            selected_was_stabilized = stabilized_selected is not selected
            selected = stabilized_selected

    return _forced_windowed_centroid(
        image,
        detection,
        click_x,
        click_y,
        radius_px,
        fit_radius_limit,
        selected,
        selected_was_stabilized=selected_was_stabilized,
    )
