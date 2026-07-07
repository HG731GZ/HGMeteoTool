from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .camera_calibration import CameraCalibrationProfile
from .fixed_camera_model import FixedCameraModel
from .frame_astrometry import FrameAstrometricModel, FrameLocalResidual, FramePose
from .simulator import ObserverSettings, compute_altaz_from_radec, local_vectors_from_altaz


ICRS_BASIS_RA_DEC = np.asarray(
    [
        [0.0, 0.0],
        [90.0, 0.0],
        [0.0, 90.0],
    ],
    dtype=np.float64,
)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def icrs_to_enu_rotation_matrix(observer: ObserverSettings) -> np.ndarray:
    """返回行向量约定下的 ICRS -> local ENU 矩阵。"""
    alt_deg, az_deg = compute_altaz_from_radec(
        ICRS_BASIS_RA_DEC[:, 0],
        ICRS_BASIS_RA_DEC[:, 1],
        observer,
    )
    enu_basis_rows = local_vectors_from_altaz(alt_deg, az_deg)
    return enu_basis_rows.T.astype(np.float64)


def frame_astrometric_model_from_fixed_camera(
    *,
    fixed_camera_model: FixedCameraModel,
    observer: ObserverSettings,
    fit_metadata: dict[str, object] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> FrameAstrometricModel:
    """把序列内部 ENU 固定相机模型折算为单帧 Pixel ↔ ICRS 模型。"""
    transform = fixed_camera_model.projection_transform
    icrs_to_camera = np.asarray(transform.rotation_matrix, dtype=np.float64) @ icrs_to_enu_rotation_matrix(observer)
    return FrameAstrometricModel(
        image_width_px=int(fixed_camera_model.image_width_px),
        image_height_px=int(fixed_camera_model.image_height_px),
        frame_pose=FramePose(icrs_to_camera),
        camera_calibration_profile=CameraCalibrationProfile.from_projection_transform(transform),
        frame_local_residual=FrameLocalResidual(),
        fit_metadata=dict(fit_metadata or {}),
        diagnostics=dict(diagnostics or {}),
    )


@dataclass(frozen=True)
class SequenceGeometryModel:
    """固定机位序列内部模型：Solver Observer + ENU 固定相机模型。"""

    solver_observer: ObserverSettings
    fixed_camera_model: FixedCameraModel

    @property
    def camera_calibration_profile(self) -> CameraCalibrationProfile:
        return CameraCalibrationProfile.from_projection_transform(self.fixed_camera_model.projection_transform)

    def project_radec_at_time(
        self,
        ra_dec_points: np.ndarray,
        observation_time_utc: datetime,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.fixed_camera_model.project_radec_at_time(
            ra_dec_points,
            observation_time_utc=_ensure_utc(observation_time_utc),
            latitude_deg=float(self.solver_observer.latitude_deg),
            longitude_deg=float(self.solver_observer.longitude_deg),
            elevation_m=float(self.solver_observer.elevation_m),
        )

    def frame_astrometric_model(
        self,
        observation_time_utc: datetime,
        *,
        fit_metadata: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> FrameAstrometricModel:
        observer = ObserverSettings(
            observation_time_utc=_ensure_utc(observation_time_utc),
            latitude_deg=float(self.solver_observer.latitude_deg),
            longitude_deg=float(self.solver_observer.longitude_deg),
            elevation_m=float(self.solver_observer.elevation_m),
        )
        return frame_astrometric_model_from_fixed_camera(
            fixed_camera_model=self.fixed_camera_model,
            observer=observer,
            fit_metadata=fit_metadata,
            diagnostics=diagnostics,
        )
