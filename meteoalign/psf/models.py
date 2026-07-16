"""PSF 底层模块使用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StarSourceCandidate:
    """去混叠后的一颗候选星源。"""

    x: float
    y: float
    major_axis: float
    minor_axis: float
    theta_rad: float
    flux: float
    peak: float
    snr: float
    npix: int
    label: int
    saturated: bool = False
    saturation_fraction: float = 0.0
    blended: bool = False
    quality_score: float = 0.0


@dataclass(frozen=True)
class FittedStarPosition:
    """恒星中心、PSF 尺寸和拟合质量。"""

    x: float
    y: float
    amplitude: float
    background: float
    sigma_x: float
    sigma_y: float
    theta_rad: float = 0.0
    fwhm_x: float = 0.0
    fwhm_y: float = 0.0
    snr: float = 0.0
    fit_error: float = 0.0
    saturated: bool = False
    saturation_fraction: float = 0.0
    blended: bool = False
    quality_score: float = 0.0

