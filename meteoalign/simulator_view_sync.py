"""把星点求解得到的天球姿态转换为星空模拟取景。"""

from __future__ import annotations

import numpy as np

from .alignment.models import PreliminarySkyAlignmentTransform
from .coordinates import radec_to_unit_vectors, sky_plane_to_radec
from .domain.settings import ObserverSettings, ViewSettings
from .frame_astrometry import FramePose
from .projection.camera_models import camera_basis_from_view
from .sequence_geometry import icrs_to_enu_rotation_matrix


def _normalized_vector(vector: np.ndarray, field_name: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name}必须是有限的三维向量。")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError(f"{field_name}长度过小。")
    return (values / norm).astype(np.float64)


def preliminary_alignment_rotation_matrix(
    transform: PreliminarySkyAlignmentTransform,
    image_size: tuple[int, int],
) -> np.ndarray:
    """从两点或三点预配准估计图像中心的 ICRS→Camera 姿态。"""

    image_width, image_height = int(image_size[0]), int(image_size[1])
    if image_width <= 0 or image_height <= 0:
        raise ValueError("预配准图像尺寸无效。")

    linear_matrix = np.asarray(transform.linear_matrix, dtype=np.float64)
    offset_px = np.asarray(transform.offset_px, dtype=np.float64)
    if linear_matrix.shape != (2, 2) or offset_px.shape != (2,):
        raise ValueError("预配准相似变换形状无效。")
    try:
        inverse_linear = np.linalg.inv(linear_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("预配准相似变换不可逆。") from exc

    center_x = float(image_width) * 0.5
    center_y = float(image_height) * 0.5
    vertical_sample_px = max(1.0, min(float(image_width), float(image_height)) * 0.05)
    image_points = np.asarray(
        (
            (center_x, center_y),
            (center_x, center_y - vertical_sample_px),
        ),
        dtype=np.float64,
    )
    sky_plane_points = (image_points - offset_px) @ inverse_linear.T
    radec = sky_plane_to_radec(
        sky_plane_points,
        np.asarray(transform.center_vector, dtype=np.float64),
        np.asarray(transform.east_vector, dtype=np.float64),
        np.asarray(transform.north_vector, dtype=np.float64),
    )
    icrs_vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])
    forward = _normalized_vector(icrs_vectors[0], "预配准中心方向")
    up = icrs_vectors[1] - float(np.dot(icrs_vectors[1], forward)) * forward
    up = _normalized_vector(up, "预配准图像上方向")
    right = _normalized_vector(np.cross(forward, up), "预配准图像右方向")
    up = _normalized_vector(np.cross(right, forward), "预配准正交化上方向")
    return FramePose(np.vstack((right, up, forward))).icrs_to_camera.copy()


def view_settings_from_icrs_to_camera(
    icrs_to_camera: np.ndarray,
    observer: ObserverSettings,
) -> ViewSettings:
    """把 ICRS→Camera 姿态分解为当前观测者的 Az、Alt 和 Roll。"""

    pose = FramePose(np.asarray(icrs_to_camera, dtype=np.float64))
    icrs_to_enu = icrs_to_enu_rotation_matrix(observer)
    camera_from_enu = pose.icrs_to_camera @ icrs_to_enu.T
    forward = _normalized_vector(camera_from_enu[2], "相机中心方向")

    horizontal_norm = float(np.hypot(forward[0], forward[1]))
    if horizontal_norm <= 1e-10:
        center_az_deg = 0.0
    else:
        center_az_deg = float(np.rad2deg(np.arctan2(forward[0], forward[1]))) % 360.0
    center_alt_deg = float(np.rad2deg(np.arcsin(np.clip(forward[2], -1.0, 1.0))))

    base_right, base_up, _base_forward = camera_basis_from_view(
        ViewSettings(
            center_az_deg=center_az_deg,
            center_alt_deg=center_alt_deg,
            roll_deg=0.0,
        )
    )
    solved_right = camera_from_enu[0] - float(np.dot(camera_from_enu[0], forward)) * forward
    solved_right = _normalized_vector(solved_right, "相机右方向")
    roll_deg = float(
        np.rad2deg(
            np.arctan2(
                float(np.dot(solved_right, base_up)),
                float(np.dot(solved_right, base_right)),
            )
        )
    )
    roll_deg = (roll_deg + 180.0) % 360.0 - 180.0
    return ViewSettings(
        center_az_deg=center_az_deg,
        center_alt_deg=center_alt_deg,
        roll_deg=roll_deg,
    )
