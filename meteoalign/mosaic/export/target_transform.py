"""目标 ICRS 与自由投影输出像素之间的变换。"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ...coordinates import normalize_vector, radec_to_unit_vectors
from ...domain.settings import CameraSettings, ObserverSettings, ViewSettings
from ...projection.camera_models import (
    CYLINDRICAL_LENS_MODELS,
    FISHEYE_LENS_MODELS,
    MERCATOR_LENS_MODEL,
    RECTILINEAR_LENS_MODEL,
    _fisheye_radius_ratio,
    _fisheye_theta_from_radius_ratio,
    _projection_horizontal_scale_px,
)
from .geometry import MosaicExportGeometry


MOSAIC_TARGET_ICRS_TO_PIXEL_VERSION = 1


def build_target_icrs_to_pixel_transform_payload(
    *,
    camera: CameraSettings,
    view: ViewSettings,
    observer: ObserverSettings,
    geometry: MosaicExportGeometry,
) -> dict[str, object]:
    """导出源图无关的 ICRS 到全景图像素变换。"""

    right, up, forward = _icrs_camera_basis_from_view(view, observer)
    return {
        "version": MOSAIC_TARGET_ICRS_TO_PIXEL_VERSION,
        "type": "icrs_to_cropped_output_pixel",
        "pixel_convention": "0-based_pixel_center",
        "boundary_width_px": int(geometry.boundary_width_px),
        "boundary_height_px": int(geometry.boundary_height_px),
        "crop_left_px": int(geometry.crop_left_px),
        "crop_top_px": int(geometry.crop_top_px),
        "output_width_px": int(geometry.output_width_px),
        "output_height_px": int(geometry.output_height_px),
        "camera": {
            "lens_model": str(camera.lens_model),
            "sensor_width_mm": float(camera.sensor_width_mm),
            "sensor_height_mm": float(camera.sensor_height_mm),
            "image_width_px": int(camera.image_width_px),
            "image_height_px": int(camera.image_height_px),
            "focal_length_mm": float(camera.focal_length_mm),
            "fisheye_fov_deg": float(camera.fisheye_fov_deg),
        },
        "icrs_camera_basis": {
            "right": [float(value) for value in right],
            "up": [float(value) for value in up],
            "forward": [float(value) for value in forward],
        },
    }


def target_icrs_to_pixel_transform_payload_matches(
    payload: object,
    *,
    geometry: MosaicExportGeometry,
) -> bool:
    """检查 ICRS 到全景图像素变换是否匹配当前裁剪输出几何。"""

    if not isinstance(payload, dict):
        return False
    if int(payload.get("version", 0) or 0) != MOSAIC_TARGET_ICRS_TO_PIXEL_VERSION:
        return False
    if str(payload.get("type") or "") != "icrs_to_cropped_output_pixel":
        return False
    expected = {
        "boundary_width_px": geometry.boundary_width_px,
        "boundary_height_px": geometry.boundary_height_px,
        "crop_left_px": geometry.crop_left_px,
        "crop_top_px": geometry.crop_top_px,
        "output_width_px": geometry.output_width_px,
        "output_height_px": geometry.output_height_px,
    }
    for key, expected_value in expected.items():
        try:
            actual_value = int(payload.get(key, -1))
        except (TypeError, ValueError):
            return False
        if actual_value != int(expected_value):
            return False
    return isinstance(payload.get("camera"), dict) and isinstance(payload.get("icrs_camera_basis"), dict)


def target_icrs_vectors_to_output_pixel_points(
    vectors: np.ndarray,
    target_icrs_to_pixel_payload: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """把 ICRS 单位方向投到裁剪后全景图像素坐标。"""

    vector_array = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(vector_array, axis=1)
    valid = np.all(np.isfinite(vector_array), axis=1) & np.isfinite(norm) & (norm > 1e-12)
    normalized = np.full_like(vector_array, np.nan, dtype=np.float64)
    normalized[valid] = vector_array[valid] / norm[valid, None]
    right, up, forward = _target_transform_basis(target_icrs_to_pixel_payload)
    camera_vectors = np.column_stack((normalized @ right, normalized @ up, normalized @ forward))
    camera = _target_transform_camera(target_icrs_to_pixel_payload)
    full_pixels, projection_valid = target_camera_vectors_to_image_points(camera_vectors, camera)
    output_pixels = full_pixels.copy()
    output_pixels[:, 0] -= float(target_icrs_to_pixel_payload.get("crop_left_px", 0.0))
    output_pixels[:, 1] -= float(target_icrs_to_pixel_payload.get("crop_top_px", 0.0))
    valid &= projection_valid & np.all(np.isfinite(output_pixels), axis=1)
    output_pixels[~valid] = np.nan
    return output_pixels.astype(np.float64), valid.astype(bool)


def target_camera_vectors_to_image_points(
    camera_vectors: np.ndarray,
    camera: CameraSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """把目标相机坐标系方向投影到完整全景图边界像素。"""

    vectors = np.asarray(camera_vectors, dtype=np.float64)
    cam_x = vectors[:, 0]
    cam_y = vectors[:, 1]
    cam_z = vectors[:, 2]
    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        valid = np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z) & (cam_z > 1e-6)
        x_px = np.full_like(cam_z, np.nan, dtype=np.float64)
        y_px = np.full_like(cam_z, np.nan, dtype=np.float64)
        if np.any(valid):
            x_mm = camera.focal_length_mm * cam_x[valid] / cam_z[valid]
            y_mm = camera.focal_length_mm * cam_y[valid] / cam_z[valid]
            x_px[valid] = camera.image_width_px * 0.5 + (x_mm / camera.sensor_width_mm) * camera.image_width_px
            y_px[valid] = camera.image_height_px * 0.5 - (y_mm / camera.sensor_height_mm) * camera.image_height_px
    elif camera.lens_model in FISHEYE_LENS_MODELS:
        norm = np.linalg.norm(vectors, axis=1)
        unit_z = np.divide(cam_z, norm, out=np.full_like(cam_z, np.nan), where=norm > 1e-12)
        theta = np.arccos(np.clip(unit_z, -1.0, 1.0))
        theta_max = np.deg2rad(camera.fisheye_fov_deg * 0.5)
        rho = _fisheye_radius_ratio(theta, theta_max, camera.lens_model)
        r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
        r_px = r_limit * rho
        plane_norm = np.hypot(cam_x, cam_y)
        unit_x = np.divide(cam_x, plane_norm, out=np.zeros_like(cam_x), where=plane_norm > 1e-12)
        unit_y = np.divide(cam_y, plane_norm, out=np.zeros_like(cam_y), where=plane_norm > 1e-12)
        x_px = camera.image_width_px * 0.5 + unit_x * r_px
        y_px = camera.image_height_px * 0.5 - unit_y * r_px
        valid = (theta <= theta_max + 1e-9) & np.isfinite(x_px) & np.isfinite(y_px)
    elif camera.lens_model in CYLINDRICAL_LENS_MODELS:
        norm = np.linalg.norm(vectors, axis=1)
        unit_y = np.divide(cam_y, norm, out=np.zeros_like(cam_y), where=norm > 1e-12)
        longitude = np.arctan2(cam_x, cam_z)
        latitude = np.arcsin(np.clip(unit_y, -1.0, 1.0))
        scale_px = _projection_horizontal_scale_px(camera)
        if camera.lens_model == MERCATOR_LENS_MODEL:
            valid = np.isfinite(longitude) & np.isfinite(latitude) & (np.abs(latitude) < (np.pi * 0.5 - 1e-8))
            plane_y = np.arctanh(np.clip(np.sin(latitude), -1.0 + 1e-12, 1.0 - 1e-12))
        else:
            valid = np.isfinite(longitude) & np.isfinite(latitude)
            plane_y = latitude
        x_px = camera.image_width_px * 0.5 + scale_px * longitude
        y_px = camera.image_height_px * 0.5 - scale_px * plane_y
    else:
        raise ValueError(f"不支持的目标投影模型：{camera.lens_model}")
    pixels = np.column_stack((x_px, y_px)).astype(np.float64)
    valid &= np.all(np.isfinite(pixels), axis=1)
    pixels[~valid] = np.nan
    return pixels, valid.astype(bool)


def target_image_points_to_icrs_vectors(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    camera: CameraSettings,
    icrs_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """把全景图像素直接反解为 ICRS 单位方向。"""

    camera_vectors, valid = target_image_points_to_camera_vectors(x_px, y_px, camera)
    right, up, forward = icrs_basis
    icrs_x = camera_vectors[:, 0, None] * right[None, :]
    icrs_y = camera_vectors[:, 1, None] * up[None, :]
    icrs_z = camera_vectors[:, 2, None] * forward[None, :]
    vectors = icrs_x + icrs_y + icrs_z
    norm = np.linalg.norm(vectors, axis=1)
    valid &= np.isfinite(norm) & (norm > 1e-12)
    vectors[valid] /= norm[valid, None]
    vectors[~valid] = np.nan
    return vectors.astype(np.float64), valid.astype(bool)


def target_image_points_to_camera_vectors(
    x_px: np.ndarray,
    y_px: np.ndarray,
    camera: CameraSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """把全景图像素反解为目标相机坐标系下的方向。"""

    x_values = np.asarray(x_px, dtype=np.float64)
    y_values = np.asarray(y_px, dtype=np.float64)
    if x_values.shape != y_values.shape:
        raise ValueError("全景图像素 x/y 数组形状必须一致。")
    if camera.lens_model == RECTILINEAR_LENS_MODEL:
        x_mm = (x_values - camera.image_width_px * 0.5) * camera.sensor_width_mm / camera.image_width_px
        y_mm = (camera.image_height_px * 0.5 - y_values) * camera.sensor_height_mm / camera.image_height_px
        cam_x = x_mm / camera.focal_length_mm
        cam_y = y_mm / camera.focal_length_mm
        cam_z = np.ones_like(cam_x, dtype=np.float64)
        valid = np.isfinite(cam_x) & np.isfinite(cam_y)
    elif camera.lens_model in FISHEYE_LENS_MODELS:
        center_x = camera.image_width_px * 0.5
        center_y = camera.image_height_px * 0.5
        screen_x = x_values - center_x
        screen_y = center_y - y_values
        r_px = np.hypot(screen_x, screen_y)
        r_limit = min(camera.image_width_px, camera.image_height_px) * 0.5 - 0.5
        rho = np.divide(r_px, r_limit, out=np.full_like(r_px, np.inf), where=r_limit > 1e-12)
        theta_max = np.deg2rad(camera.fisheye_fov_deg * 0.5)
        theta = _fisheye_theta_from_radius_ratio(np.clip(rho, 0.0, 1.0), theta_max, camera.lens_model)
        plane_norm = np.sin(theta)
        unit_x = np.divide(screen_x, r_px, out=np.zeros_like(screen_x), where=r_px > 1e-12)
        unit_y = np.divide(screen_y, r_px, out=np.zeros_like(screen_y), where=r_px > 1e-12)
        cam_x = unit_x * plane_norm
        cam_y = unit_y * plane_norm
        cam_z = np.cos(theta)
        valid = (rho <= 1.0 + 1e-9) & np.isfinite(cam_x) & np.isfinite(cam_y) & np.isfinite(cam_z)
    elif camera.lens_model in CYLINDRICAL_LENS_MODELS:
        center_x = camera.image_width_px * 0.5
        center_y = camera.image_height_px * 0.5
        scale_px = _projection_horizontal_scale_px(camera)
        longitude = (x_values - center_x) / max(scale_px, 1e-12)
        plane_y = (center_y - y_values) / max(scale_px, 1e-12)
        if camera.lens_model == MERCATOR_LENS_MODEL:
            latitude = np.arcsin(np.clip(np.tanh(plane_y), -1.0, 1.0))
            valid = np.isfinite(longitude) & np.isfinite(latitude)
        else:
            latitude = plane_y
            valid = np.isfinite(longitude) & np.isfinite(latitude) & (np.abs(latitude) <= np.pi * 0.5 + 1e-8)
        cos_lat = np.cos(latitude)
        cam_x = cos_lat * np.sin(longitude)
        cam_y = np.sin(latitude)
        cam_z = cos_lat * np.cos(longitude)
    else:
        raise ValueError(f"不支持的目标投影模型：{camera.lens_model}")
    vectors = np.column_stack((cam_x, cam_y, cam_z)).astype(np.float64)
    norm = np.linalg.norm(vectors, axis=1)
    valid &= np.isfinite(norm) & (norm > 1e-12)
    vectors[valid] /= norm[valid, None]
    vectors[~valid] = np.nan
    return vectors.astype(np.float64), valid.astype(bool)


def _target_transform_camera(payload: dict[str, object]) -> CameraSettings:
    camera_payload = payload.get("camera")
    if not isinstance(camera_payload, dict):
        raise ValueError("取景 JSON 缺少 target_icrs_to_pixel.camera。")
    return CameraSettings(
        sensor_width_mm=float(camera_payload.get("sensor_width_mm", 36.0)),
        sensor_height_mm=float(camera_payload.get("sensor_height_mm", 24.0)),
        image_width_px=int(camera_payload.get("image_width_px", payload.get("boundary_width_px", 0))),
        image_height_px=int(camera_payload.get("image_height_px", payload.get("boundary_height_px", 0))),
        focal_length_mm=float(camera_payload.get("focal_length_mm", 24.0)),
        lens_model=str(camera_payload.get("lens_model", RECTILINEAR_LENS_MODEL)),
        fisheye_fov_deg=float(camera_payload.get("fisheye_fov_deg", 180.0)),
    )


def _target_transform_basis(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_payload = payload.get("icrs_camera_basis")
    if not isinstance(basis_payload, dict):
        raise ValueError("取景 JSON 缺少 target_icrs_to_pixel.icrs_camera_basis。")
    return (
        _payload_vector3(basis_payload.get("right"), "target_icrs_to_pixel.icrs_camera_basis.right"),
        _payload_vector3(basis_payload.get("up"), "target_icrs_to_pixel.icrs_camera_basis.up"),
        _payload_vector3(basis_payload.get("forward"), "target_icrs_to_pixel.icrs_camera_basis.forward"),
    )


def _payload_vector3(value: object, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"取景 JSON 字段 {field_name} 必须是 3 个有限数值。")
    return vector.astype(np.float64)


def _icrs_camera_basis_from_view(
    view: ViewSettings,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, zenith, north = _reference_icrs_vectors_from_view(view, observer)
    right = np.cross(center, zenith)
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.cross(center, north)
    right = normalize_vector(right)
    up = normalize_vector(np.cross(right, center))
    roll = np.deg2rad(view.roll_deg)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    rolled_right = right * cos_roll + up * sin_roll
    rolled_up = -right * sin_roll + up * cos_roll
    return (
        normalize_vector(rolled_right),
        normalize_vector(rolled_up),
        normalize_vector(center),
    )


def _reference_icrs_vectors_from_view(
    view: ViewSettings,
    observer: ObserverSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alt_deg = np.asarray([view.center_alt_deg, 90.0, 0.0], dtype=np.float64)
    az_deg = np.asarray([view.center_az_deg, view.center_az_deg, 0.0], dtype=np.float64)
    location = EarthLocation.from_geodetic(
        lon=observer.longitude_deg * u.deg,
        lat=observer.latitude_deg * u.deg,
        height=observer.elevation_m * u.m,
    )
    altaz = SkyCoord(
        alt=alt_deg * u.deg,
        az=az_deg * u.deg,
        frame=AltAz(obstime=Time(observer.observation_time_utc), location=location),
    )
    icrs = altaz.icrs
    vectors = radec_to_unit_vectors(icrs.ra.degree, icrs.dec.degree)
    return normalize_vector(vectors[0]), normalize_vector(vectors[1]), normalize_vector(vectors[2])
