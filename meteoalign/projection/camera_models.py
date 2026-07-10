"""相机投影、反投影和本地视图几何。"""

from __future__ import annotations

import numpy as np

from ..domain.settings import CameraSettings, ViewSettings


RECTILINEAR_LENS_MODEL = "rectilinear"
FISHEYE_EQUIDISTANT = "fisheye_equidistant"
FISHEYE_EQUISOLID = "fisheye_equisolid"
MERCATOR_LENS_MODEL = "mercator"
CYLINDRICAL_EQUIDISTANT_LENS_MODEL = "cylindrical_equidistant"
FISHEYE_LENS_MODELS = {FISHEYE_EQUIDISTANT, FISHEYE_EQUISOLID}
CYLINDRICAL_LENS_MODELS = {MERCATOR_LENS_MODEL, CYLINDRICAL_EQUIDISTANT_LENS_MODEL}
SUPPORTED_LENS_MODELS = {RECTILINEAR_LENS_MODEL, *FISHEYE_LENS_MODELS, *CYLINDRICAL_LENS_MODELS}


def horizontal_fov_deg(camera: CameraSettings) -> float:
    """计算相机水平方向视场角。"""

    if camera.lens_model in FISHEYE_LENS_MODELS or camera.lens_model in CYLINDRICAL_LENS_MODELS:
        return float(camera.fisheye_fov_deg)
    return float(np.degrees(2.0 * np.arctan(camera.sensor_width_mm / (2.0 * camera.focal_length_mm))))


def vertical_fov_deg(camera: CameraSettings) -> float:
    """计算相机垂直方向视场角。"""

    if camera.lens_model in FISHEYE_LENS_MODELS or camera.lens_model in CYLINDRICAL_LENS_MODELS:
        aspect_scale = camera.image_height_px / max(float(camera.image_width_px), 1.0)
        return float(camera.fisheye_fov_deg * min(aspect_scale, 1.0))
    return float(np.degrees(2.0 * np.arctan(camera.sensor_height_mm / (2.0 * camera.focal_length_mm))))


def _local_vectors_from_altaz(alt_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)
    cos_alt = np.cos(alt)
    return np.column_stack((cos_alt * np.sin(az), cos_alt * np.cos(az), np.sin(alt)))


def local_vectors_from_altaz(alt_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    """把高度角、方位角转换为本地 ENU 单位方向。"""

    return _local_vectors_from_altaz(alt_deg, az_deg)


def _project_vectors_onto_camera_basis(
    vectors: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right, up, forward = basis
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    return (
        (x * right[0] + y * right[1] + z * right[2]).astype(np.float64),
        (x * up[0] + y * up[1] + z * up[2]).astype(np.float64),
        (x * forward[0] + y * forward[1] + z * forward[2]).astype(np.float64),
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def _camera_basis(view: ViewSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _local_vectors_from_altaz(
        np.asarray([view.center_alt_deg], dtype=np.float64),
        np.asarray([view.center_az_deg], dtype=np.float64),
    )[0]
    forward = _normalize(forward)
    reference_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, reference_up)
    if np.linalg.norm(right) < 1e-8:
        reference_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, reference_up)
    right = _normalize(right)
    up = _normalize(np.cross(right, forward))
    roll = np.deg2rad(view.roll_deg)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    return right * cos_roll + up * sin_roll, -right * sin_roll + up * cos_roll, forward


def camera_basis_from_view(view: ViewSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算取景对应的右、上、前相机基向量。"""

    return _camera_basis(view)


def _project_altaz_points(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    camera: CameraSettings,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        return _project_altaz_points_rectilinear(alt_deg, az_deg, camera, basis)
    if camera.lens_model in FISHEYE_LENS_MODELS:
        return _project_altaz_points_fisheye(alt_deg, az_deg, camera, basis)
    if camera.lens_model in CYLINDRICAL_LENS_MODELS:
        return _project_altaz_points_cylindrical(alt_deg, az_deg, camera, basis)
    raise ValueError(f"Unsupported lens model: {camera.lens_model}")


def _project_altaz_points_rectilinear(alt_deg: np.ndarray, az_deg: np.ndarray, camera: CameraSettings, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)
    finite_depth = np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z) & (cam_z > 1e-6)
    x_px = np.full_like(cam_z, np.nan, dtype=np.float64)
    y_px = np.full_like(cam_z, np.nan, dtype=np.float64)
    if np.any(finite_depth):
        x_mm = camera.focal_length_mm * cam_x[finite_depth] / cam_z[finite_depth]
        y_mm = camera.focal_length_mm * cam_y[finite_depth] / cam_z[finite_depth]
        x_px[finite_depth] = camera.image_width_px * 0.5 + (x_mm / camera.sensor_width_mm) * camera.image_width_px
        y_px[finite_depth] = camera.image_height_px * 0.5 - (y_mm / camera.sensor_height_mm) * camera.image_height_px
    margin_x = camera.image_width_px * 2.0
    margin_y = camera.image_height_px * 2.0
    valid = finite_depth & np.isfinite(x_px) & np.isfinite(y_px) & (x_px >= -margin_x) & (x_px <= camera.image_width_px + margin_x) & (y_px >= -margin_y) & (y_px <= camera.image_height_px + margin_y)
    return x_px.astype(np.float64), y_px.astype(np.float64), valid


def _fisheye_radius_ratio(theta: np.ndarray, theta_max: float, lens_model: str) -> np.ndarray:
    if theta_max <= 0.0:
        raise ValueError("Fisheye FOV must be positive")
    if lens_model == FISHEYE_EQUIDISTANT:
        return theta / theta_max
    if lens_model == FISHEYE_EQUISOLID:
        denominator = np.sin(theta_max / 2.0)
        if abs(float(denominator)) <= 1e-12:
            raise ValueError("Fisheye FOV is too small")
        return np.sin(theta / 2.0) / denominator
    raise ValueError(f"Unsupported fisheye lens model: {lens_model}")


def _fisheye_theta_from_radius_ratio(rho: np.ndarray, theta_max: float, lens_model: str) -> np.ndarray:
    if lens_model == FISHEYE_EQUIDISTANT:
        return rho * theta_max
    if lens_model == FISHEYE_EQUISOLID:
        return 2.0 * np.arcsin(np.clip(rho * np.sin(theta_max / 2.0), -1.0, 1.0))
    raise ValueError(f"Unsupported fisheye lens model: {lens_model}")


def _project_altaz_points_fisheye(alt_deg: np.ndarray, az_deg: np.ndarray, camera: CameraSettings, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)
    theta = np.arccos(np.clip(cam_z, -1.0, 1.0))
    theta_max = np.deg2rad(camera.fisheye_fov_deg * 0.5)
    rho = _fisheye_radius_ratio(theta, theta_max, camera.lens_model)
    r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
    r_px = r_limit * rho
    plane_norm = np.hypot(cam_x, cam_y)
    unit_x = np.divide(cam_x, plane_norm, out=np.zeros_like(cam_x), where=plane_norm > 1e-12)
    unit_y = np.divide(cam_y, plane_norm, out=np.zeros_like(cam_y), where=plane_norm > 1e-12)
    x_px = camera.image_width_px * 0.5 + unit_x * r_px
    y_px = camera.image_height_px * 0.5 - unit_y * r_px
    margin = r_limit * 0.1
    valid = (theta <= theta_max + 1e-9) & np.isfinite(r_px) & np.isfinite(x_px) & np.isfinite(y_px) & (r_px <= r_limit + margin) & (x_px >= -margin) & (x_px <= camera.image_width_px + margin) & (y_px >= -margin) & (y_px <= camera.image_height_px + margin)
    return x_px.astype(np.float64), y_px.astype(np.float64), valid


def _projection_horizontal_scale_px(camera: CameraSettings) -> float:
    fov_deg = max(1.0, min(360.0, float(camera.fisheye_fov_deg)))
    return float(camera.image_width_px) / max(np.deg2rad(fov_deg), 1e-6)


def _project_altaz_points_cylindrical(alt_deg: np.ndarray, az_deg: np.ndarray, camera: CameraSettings, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)
    norm = np.sqrt(cam_x * cam_x + cam_y * cam_y + cam_z * cam_z)
    unit_y = np.divide(cam_y, norm, out=np.zeros_like(cam_y), where=norm > 1e-12)
    longitude = np.arctan2(cam_x, cam_z)
    latitude = np.arcsin(np.clip(unit_y, -1.0, 1.0))
    scale_px = _projection_horizontal_scale_px(camera)
    plane_y = latitude
    valid = np.isfinite(longitude) & np.isfinite(latitude) & (norm > 1e-12)
    if camera.lens_model == MERCATOR_LENS_MODEL:
        valid &= np.abs(latitude) < (np.pi * 0.5 - 1e-8)
        plane_y = np.arctanh(np.clip(np.sin(latitude), -1.0 + 1e-12, 1.0 - 1e-12))
    x_px = camera.image_width_px * 0.5 + scale_px * longitude
    y_px = camera.image_height_px * 0.5 - scale_px * plane_y
    margin_x = camera.image_width_px * 0.1
    margin_y = camera.image_height_px * 0.1
    valid &= np.isfinite(x_px) & np.isfinite(y_px) & (x_px >= -margin_x) & (x_px <= camera.image_width_px + margin_x) & (y_px >= -margin_y) & (y_px <= camera.image_height_px + margin_y)
    return x_px.astype(np.float64), y_px.astype(np.float64), valid.astype(bool)


def image_points_to_local_vectors(x_px: np.ndarray, y_px: np.ndarray, camera: CameraSettings, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """把目标画布像素反解为本地 ENU 单位方向。"""

    right, up, forward = basis
    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        x_mm = (x_px - camera.image_width_px * 0.5) * camera.sensor_width_mm / camera.image_width_px
        y_mm = (camera.image_height_px * 0.5 - y_px) * camera.sensor_height_mm / camera.image_height_px
        cam_x, cam_y, cam_z = x_mm / camera.focal_length_mm, y_mm / camera.focal_length_mm, np.ones_like(x_mm, dtype=np.float64)
        valid = np.isfinite(cam_x) & np.isfinite(cam_y)
    elif camera.lens_model in FISHEYE_LENS_MODELS:
        screen_x, screen_y = x_px - camera.image_width_px * 0.5, camera.image_height_px * 0.5 - y_px
        r_px = np.hypot(screen_x, screen_y)
        r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
        rho = np.divide(r_px, r_limit, out=np.full_like(r_px, np.inf), where=r_limit > 1e-12)
        theta = _fisheye_theta_from_radius_ratio(np.clip(rho, 0.0, 1.0), np.deg2rad(camera.fisheye_fov_deg * 0.5), camera.lens_model)
        plane_norm = np.sin(theta)
        cam_x = np.divide(screen_x, r_px, out=np.zeros_like(screen_x), where=r_px > 1e-12) * plane_norm
        cam_y = np.divide(screen_y, r_px, out=np.zeros_like(screen_y), where=r_px > 1e-12) * plane_norm
        cam_z = np.cos(theta)
        valid = (rho <= 1.0 + 1e-9) & np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z)
    elif camera.lens_model in CYLINDRICAL_LENS_MODELS:
        scale_px = _projection_horizontal_scale_px(camera)
        longitude = (x_px - camera.image_width_px * 0.5) / max(scale_px, 1e-12)
        plane_y = (camera.image_height_px * 0.5 - y_px) / max(scale_px, 1e-12)
        if camera.lens_model == MERCATOR_LENS_MODEL:
            latitude = np.arcsin(np.clip(np.tanh(plane_y), -1.0, 1.0))
            valid = np.isfinite(longitude) & np.isfinite(latitude)
        else:
            latitude = plane_y
            valid = np.isfinite(longitude) & np.isfinite(latitude) & (np.abs(latitude) <= np.pi * 0.5 + 1e-8)
        cos_lat = np.cos(latitude)
        cam_x, cam_y, cam_z = cos_lat * np.sin(longitude), np.sin(latitude), cos_lat * np.cos(longitude)
    else:
        raise ValueError(f"Unsupported lens model: {camera.lens_model}")
    local_x = cam_x * right[0] + cam_y * up[0] + cam_z * forward[0]
    local_y = cam_x * right[1] + cam_y * up[1] + cam_z * forward[1]
    local_z = cam_x * right[2] + cam_y * up[2] + cam_z * forward[2]
    norm = np.sqrt(local_x * local_x + local_y * local_y + local_z * local_z)
    valid &= norm > 1e-12
    vectors = np.column_stack((np.divide(local_x, norm, out=np.full_like(local_x, np.nan), where=norm > 1e-12), np.divide(local_y, norm, out=np.full_like(local_y, np.nan), where=norm > 1e-12), np.divide(local_z, norm, out=np.full_like(local_z, np.nan), where=norm > 1e-12)))
    vectors[~valid] = np.nan
    return vectors.astype(np.float64), valid.astype(bool)


def local_vectors_to_altaz(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把本地 ENU 单位方向转为高度角和方位角。"""

    vector_array = np.asarray(vectors, dtype=np.float64)
    if vector_array.ndim == 1:
        vector_array = vector_array.reshape(1, 3)
    if vector_array.ndim != 2 or vector_array.shape[1] != 3:
        raise ValueError("本地 ENU 方向必须是 Nx3 数组。")
    norm = np.linalg.norm(vector_array, axis=1)
    valid = np.all(np.isfinite(vector_array), axis=1) & np.isfinite(norm) & (norm > 1e-12)
    unit = np.full_like(vector_array, np.nan, dtype=np.float64)
    unit[valid] = vector_array[valid] / norm[valid, None]
    alt_deg = np.rad2deg(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))
    az_deg = np.rad2deg(np.arctan2(unit[:, 0], unit[:, 1])) % 360.0
    alt_deg[~valid] = np.nan
    az_deg[~valid] = np.nan
    return alt_deg.astype(np.float64), az_deg.astype(np.float64), valid.astype(bool)


def _camera_longitudes_from_altaz(alt_deg: np.ndarray, az_deg: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    vectors = _local_vectors_from_altaz(alt_deg, az_deg)
    cam_x, _cam_y, cam_z = _project_vectors_onto_camera_basis(vectors, basis)
    return np.arctan2(cam_x, cam_z).astype(np.float64)
