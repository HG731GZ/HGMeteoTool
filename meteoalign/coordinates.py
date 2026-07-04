from __future__ import annotations

import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("无法归一化零长度向量。")
    return values / norm


def radec_to_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra_rad = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec_rad = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec_rad)
    return np.column_stack((cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad))).astype(np.float64)


def unit_vectors_to_radec(vectors: np.ndarray) -> np.ndarray:
    vector_array = np.asarray(vectors, dtype=np.float64)
    if vector_array.ndim == 1:
        vector_array = vector_array.reshape(1, 3)
    if vector_array.ndim != 2 or vector_array.shape[1] != 3:
        raise ValueError("天球方向必须是 Nx3 单位向量数组。")

    norms = np.linalg.norm(vector_array, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("天球方向中包含无效或零长度向量。")

    normalized = vector_array / norms[:, None]
    ra_deg = np.rad2deg(np.arctan2(normalized[:, 1], normalized[:, 0])) % 360.0
    dec_deg = np.rad2deg(np.arcsin(np.clip(normalized[:, 2], -1.0, 1.0)))
    return np.column_stack((ra_deg, dec_deg)).astype(np.float64)


def sky_plane_basis(ra_dec_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra_dec_array = np.asarray(ra_dec_points, dtype=np.float64)
    if ra_dec_array.ndim != 2 or ra_dec_array.shape[1] != 2:
        raise ValueError("天球平面基底需要 Nx2 的 RA/Dec 点。")

    vectors = radec_to_unit_vectors(ra_dec_array[:, 0], ra_dec_array[:, 1])
    center = vectors.mean(axis=0)
    if float(np.linalg.norm(center)) <= 1e-8:
        center = vectors[0]
    center = normalize_vector(center)

    celestial_north = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(celestial_north, center)
    if float(np.linalg.norm(east)) <= 1e-8:
        east = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    east = normalize_vector(east)
    north = normalize_vector(np.cross(center, east))
    return center, east, north


def project_radec_to_sky_plane(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_vector: np.ndarray,
    east_vector: np.ndarray,
    north_vector: np.ndarray,
) -> np.ndarray:
    ra_array = np.asarray(ra_deg, dtype=np.float64)
    dec_array = np.asarray(dec_deg, dtype=np.float64)
    if ra_array.shape != dec_array.shape:
        raise ValueError("RA 与 Dec 数组形状必须一致。")

    center = normalize_vector(center_vector)
    east = normalize_vector(east_vector)
    north = normalize_vector(north_vector)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        vectors = radec_to_unit_vectors(ra_array, dec_array)
        center_component = np.clip(vectors @ center, -1.0, 1.0)
        east_component = vectors @ east
        north_component = vectors @ north
        theta = np.arccos(center_component)
        sin_theta = np.sin(theta)
        scale = np.divide(theta, sin_theta, out=np.ones_like(theta), where=np.abs(sin_theta) > 1e-12)
        radians_to_degrees = 180.0 / np.pi
        projected = np.column_stack(
            (east_component * scale * radians_to_degrees, north_component * scale * radians_to_degrees)
        )
    projected[~np.all(np.isfinite(projected), axis=1)] = np.nan
    return projected.astype(np.float64)


def sky_plane_to_radec(
    plane_points: np.ndarray,
    center_vector: np.ndarray,
    east_vector: np.ndarray,
    north_vector: np.ndarray,
) -> np.ndarray:
    plane_array = np.asarray(plane_points, dtype=np.float64)
    if plane_array.ndim == 1:
        plane_array = plane_array.reshape(1, 2)
    if plane_array.ndim != 2 or plane_array.shape[1] != 2:
        raise ValueError("天球平面坐标必须是 Nx2 数组。")

    center = normalize_vector(center_vector)
    east = normalize_vector(east_vector)
    north = normalize_vector(north_vector)
    radius_deg = np.linalg.norm(plane_array, axis=1)
    radius_rad = np.deg2rad(radius_deg)

    # 这里使用和 project_radec_to_sky_plane 对偶的方位等距局部平面反变换。
    direction_in_plane = np.zeros((plane_array.shape[0], 3), dtype=np.float64)
    valid_radius = radius_deg > 1e-12
    if np.any(valid_radius):
        direction_in_plane[valid_radius] = (
            plane_array[valid_radius, 0, None] * east[None, :]
            + plane_array[valid_radius, 1, None] * north[None, :]
        ) / radius_deg[valid_radius, None]

    vectors = center[None, :] * np.cos(radius_rad)[:, None]
    vectors += direction_in_plane * np.sin(radius_rad)[:, None]
    vectors[~valid_radius] = center
    return unit_vectors_to_radec(vectors)
