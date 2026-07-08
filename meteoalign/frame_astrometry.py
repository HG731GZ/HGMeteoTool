from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .alignment.interpolation import AnchorInterpolation2D
from .camera_calibration import CameraCalibrationProfile, _interpolation_from_payload, _interpolation_payload
from .coordinates import radec_to_unit_vectors, unit_vectors_to_radec


SOURCE_MODEL_SCHEMA = "hgmeteo_source_astrometric_model"
SOURCE_MODEL_VERSION = 3
FRAME_LOCAL_RESIDUAL_PIXEL_TPS = "pixel_tps_bidirectional"
FRAME_LOCAL_RESIDUAL_DEFAULT_PADDING_PX = 96.0


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def _json_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"FrameAstrometricModel 字段 {field_name} 必须是对象。")
    return value


def _json_int(value: object, field_name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"FrameAstrometricModel 缺少字段：{field_name}")
        return int(default)
    return int(value)


def _json_float_array(
    value: object,
    field_name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"FrameAstrometricModel 字段 {field_name} 形状应为 {shape}，实际为 {array.shape}。")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"FrameAstrometricModel 字段 {field_name} 包含无效数值。")
    return array.astype(np.float64)


def _orthonormalized_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Frame Pose 需要有限的 3x3 旋转矩阵。")
    input_determinant = float(np.linalg.det(rotation))
    if not np.isfinite(input_determinant) or abs(input_determinant) <= 1e-12:
        raise ValueError("Frame Pose 旋转矩阵接近奇异。")
    target_handedness = -1.0 if input_determinant < 0.0 else 1.0
    try:
        u_matrix, _values, vt_matrix = np.linalg.svd(rotation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Frame Pose 旋转矩阵无法正交化。") from exc
    orthonormal = u_matrix @ vt_matrix
    if float(np.linalg.det(orthonormal)) * target_handedness < 0.0:
        u_matrix[:, -1] *= -1.0
        orthonormal = u_matrix @ vt_matrix
    return orthonormal.astype(np.float64)


@dataclass(frozen=True)
class FramePose:
    """当前帧 ICRS 方向到相机射线坐标的姿态。"""

    icrs_to_camera: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "icrs_to_camera", _orthonormalized_rotation_matrix(self.icrs_to_camera))

    def icrs_vectors_to_camera(self, vectors: np.ndarray) -> np.ndarray:
        vector_array = np.asarray(vectors, dtype=np.float64)
        if vector_array.ndim == 1:
            vector_array = vector_array.reshape(1, 3)
        if vector_array.ndim != 2 or vector_array.shape[1] != 3:
            raise ValueError("ICRS 方向必须是 Nx3 数组。")
        return (vector_array @ self.icrs_to_camera.T).astype(np.float64)

    def camera_vectors_to_icrs(self, vectors: np.ndarray) -> np.ndarray:
        vector_array = np.asarray(vectors, dtype=np.float64)
        if vector_array.ndim == 1:
            vector_array = vector_array.reshape(1, 3)
        if vector_array.ndim != 2 or vector_array.shape[1] != 3:
            raise ValueError("相机方向必须是 Nx3 数组。")
        return (vector_array @ self.icrs_to_camera).astype(np.float64)

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "type": "rotation_matrix",
            "icrs_to_camera": [
                _as_float_list(row) for row in np.asarray(self.icrs_to_camera, dtype=np.float64)
            ],
        }

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> "FramePose":
        if not isinstance(payload, dict):
            raise ValueError("frame_pose JSON 必须是对象。")
        if str(payload.get("type", "")) != "rotation_matrix":
            raise ValueError("当前仅支持 rotation_matrix 类型的 frame_pose。")
        return cls(
            icrs_to_camera=_json_float_array(
                payload.get("icrs_to_camera"),
                "frame_pose.icrs_to_camera",
                shape=(3, 3),
            )
        )


@dataclass(frozen=True)
class FrameLocalResidual:
    """当前帧局部残差，用于在冻结 Profile 后吸收少量局部误差。"""

    enabled: bool = False
    residual_type: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    base_to_corrected_interpolation: AnchorInterpolation2D | None = None
    corrected_to_base_interpolation: AnchorInterpolation2D | None = None

    def apply_forward(self, pixel_points: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        if not self.enabled:
            return pixels
        if self.residual_type != FRAME_LOCAL_RESIDUAL_PIXEL_TPS or self.base_to_corrected_interpolation is None:
            raise NotImplementedError("当前 frame_local_residual 类型尚未实现正向修正。")
        corrected = self.base_to_corrected_interpolation.evaluate_points(pixels)
        return self._blend_outside_coverage(
            input_pixels=pixels,
            corrected_pixels=corrected,
            bbox_key="base_coverage_bbox_px",
        )

    def apply_inverse(self, pixel_points: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        if not self.enabled:
            return pixels
        if self.residual_type != FRAME_LOCAL_RESIDUAL_PIXEL_TPS or self.corrected_to_base_interpolation is None:
            raise NotImplementedError("当前 frame_local_residual 类型尚未实现反向修正。")
        corrected = self.corrected_to_base_interpolation.evaluate_points(pixels)
        return self._blend_outside_coverage(
            input_pixels=pixels,
            corrected_pixels=corrected,
            bbox_key="corrected_coverage_bbox_px",
        )

    def _blend_outside_coverage(
        self,
        *,
        input_pixels: np.ndarray,
        corrected_pixels: np.ndarray,
        bbox_key: str,
    ) -> np.ndarray:
        result = np.asarray(corrected_pixels, dtype=np.float64).copy()
        finite_result = np.all(np.isfinite(result), axis=1)
        inside = self._inside_coverage_bbox(input_pixels, bbox_key)
        fallback = (~inside) | (~finite_result)
        result[fallback] = input_pixels[fallback]
        return result.astype(np.float64)

    def _inside_coverage_bbox(self, pixel_points: np.ndarray, bbox_key: str) -> np.ndarray:
        bbox = self.parameters.get(bbox_key)
        points = np.asarray(pixel_points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, 2)
        if not isinstance(bbox, dict):
            return np.all(np.isfinite(points), axis=1)
        min_x = float(bbox.get("min_x_px", -np.inf))
        max_x = float(bbox.get("max_x_px", np.inf))
        min_y = float(bbox.get("min_y_px", -np.inf))
        max_y = float(bbox.get("max_y_px", np.inf))
        padding = float(self.parameters.get("coverage_padding_px", FRAME_LOCAL_RESIDUAL_DEFAULT_PADDING_PX))
        return (
            np.all(np.isfinite(points), axis=1)
            & (points[:, 0] >= min_x - padding)
            & (points[:, 0] <= max_x + padding)
            & (points[:, 1] >= min_y - padding)
            & (points[:, 1] <= max_y + padding)
        )

    def to_json_payload(self) -> dict[str, Any]:
        parameters = dict(self.parameters)
        if self.enabled and self.residual_type == FRAME_LOCAL_RESIDUAL_PIXEL_TPS:
            if self.base_to_corrected_interpolation is None or self.corrected_to_base_interpolation is None:
                raise ValueError("frame_local_residual 缺少双向插值数据。")
            parameters.update(
                {
                    "base_to_corrected": _interpolation_payload(
                        self.base_to_corrected_interpolation,
                        input_units="px before frame local residual",
                        output_units="px after frame local residual",
                        input_axis_order=["x_px", "y_px"],
                        output_axis_order=["x_px", "y_px"],
                        weight_names=("tps_weights_x_px", "tps_weights_y_px"),
                        affine_names=("tps_affine_x_px", "tps_affine_y_px"),
                    ),
                    "corrected_to_base": _interpolation_payload(
                        self.corrected_to_base_interpolation,
                        input_units="px after frame local residual",
                        output_units="px before frame local residual",
                        input_axis_order=["x_px", "y_px"],
                        output_axis_order=["x_px", "y_px"],
                        weight_names=("tps_weights_x_px", "tps_weights_y_px"),
                        affine_names=("tps_affine_x_px", "tps_affine_y_px"),
                    ),
                }
            )
        return {
            "enabled": bool(self.enabled),
            "type": self.residual_type,
            "parameters": parameters,
        }

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any] | None) -> "FrameLocalResidual":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("frame_local_residual JSON 必须是对象。")
        enabled = bool(payload.get("enabled", False))
        residual_type = None if payload.get("type") is None else str(payload.get("type"))
        parameters = dict(payload.get("parameters")) if isinstance(payload.get("parameters"), dict) else {}
        if enabled and residual_type == FRAME_LOCAL_RESIDUAL_PIXEL_TPS:
            return cls(
                enabled=True,
                residual_type=residual_type,
                parameters=parameters,
                base_to_corrected_interpolation=_interpolation_from_payload(
                    _json_mapping(parameters.get("base_to_corrected"), "frame_local_residual.parameters.base_to_corrected"),
                    "frame_local_residual.parameters.base_to_corrected",
                    weight_names=("tps_weights_x_px", "tps_weights_y_px"),
                    affine_names=("tps_affine_x_px", "tps_affine_y_px"),
                ),
                corrected_to_base_interpolation=_interpolation_from_payload(
                    _json_mapping(parameters.get("corrected_to_base"), "frame_local_residual.parameters.corrected_to_base"),
                    "frame_local_residual.parameters.corrected_to_base",
                    weight_names=("tps_weights_x_px", "tps_weights_y_px"),
                    affine_names=("tps_affine_x_px", "tps_affine_y_px"),
                ),
            )
        return cls(enabled=enabled, residual_type=residual_type, parameters=parameters)


@dataclass(frozen=True)
class FrameAstrometricModel:
    """单帧 Pixel ↔ ICRS 模型，由 frame pose + embedded CameraCalibrationProfile 组成。"""

    image_width_px: int
    image_height_px: int
    frame_pose: FramePose
    camera_calibration_profile: CameraCalibrationProfile
    frame_local_residual: FrameLocalResidual = field(default_factory=FrameLocalResidual)
    fit_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def sky_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        radec = np.asarray(ra_dec_points, dtype=np.float64)
        if radec.ndim == 1:
            radec = radec.reshape(1, 2)
        if radec.ndim != 2 or radec.shape[1] != 2:
            raise ValueError("sky_to_pixel_points 需要 Nx2 的 RA/Dec 数组。")
        vectors = radec_to_unit_vectors(radec[:, 0], radec[:, 1])
        return self.icrs_vectors_to_pixel_points(vectors)

    def icrs_vectors_to_pixel_points(self, vectors: np.ndarray) -> np.ndarray:
        """把 ICRS 单位方向直接投影到当前帧像素坐标。"""

        vector_array = np.asarray(vectors, dtype=np.float64)
        if vector_array.ndim == 1:
            vector_array = vector_array.reshape(1, 3)
        if vector_array.ndim != 2 or vector_array.shape[1] != 3:
            raise ValueError("icrs_vectors_to_pixel_points 需要 Nx3 的 ICRS 方向数组。")
        norm = np.linalg.norm(vector_array, axis=1)
        valid = np.all(np.isfinite(vector_array), axis=1) & np.isfinite(norm) & (norm > 1e-12)
        normalized = np.full_like(vector_array, np.nan, dtype=np.float64)
        normalized[valid] = vector_array[valid] / norm[valid, None]
        normalized[~valid] = np.nan
        camera_vectors = self.frame_pose.icrs_vectors_to_camera(normalized)
        pixels = self.camera_calibration_profile.camera_ray_to_pixel_points(camera_vectors)
        return self._apply_frame_local_residual_forward(pixels)

    def direction_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        return self.sky_to_pixel_points(ra_dec_points)

    def pixel_to_sky_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixel_to_sky_points 需要 Nx2 的像素坐标数组。")
        profile_pixels = self._apply_frame_local_residual_inverse(pixels)
        camera_vectors = self.camera_calibration_profile.pixel_to_camera_ray_points(profile_pixels)
        icrs_vectors = self.frame_pose.camera_vectors_to_icrs(camera_vectors)
        norm = np.linalg.norm(icrs_vectors, axis=1)
        valid = np.all(np.isfinite(icrs_vectors), axis=1) & np.isfinite(norm) & (norm > 1e-12)
        normalized = np.full_like(icrs_vectors, np.nan, dtype=np.float64)
        normalized[valid] = icrs_vectors[valid] / norm[valid, None]
        radec = np.full((pixels.shape[0], 2), np.nan, dtype=np.float64)
        if np.any(valid):
            radec[valid] = unit_vectors_to_radec(normalized[valid])
        return radec.astype(np.float64)

    def pixel_to_radec_points(self, pixel_points: np.ndarray) -> np.ndarray:
        return self.pixel_to_sky_points(pixel_points)

    def sky_to_pixel(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        pixel = self.sky_to_pixel_points(np.asarray([[ra_deg, dec_deg]], dtype=np.float64))[0]
        return float(pixel[0]), float(pixel[1])

    def pixel_to_sky(self, x_px: float, y_px: float) -> tuple[float, float]:
        radec = self.pixel_to_sky_points(np.asarray([[x_px, y_px]], dtype=np.float64))[0]
        return float(radec[0]), float(radec[1])

    def _apply_frame_local_residual_forward(self, pixel_points: np.ndarray) -> np.ndarray:
        return self.frame_local_residual.apply_forward(pixel_points)

    def _apply_frame_local_residual_inverse(self, pixel_points: np.ndarray) -> np.ndarray:
        return self.frame_local_residual.apply_inverse(pixel_points)

    def to_json_payload(
        self,
        *,
        source_image: dict[str, Any] | None = None,
        fit_pairs: list[dict[str, Any]] | None = None,
        mask: dict[str, Any] | None = None,
        matching: dict[str, Any] | None = None,
        reference_payload: dict[str, Any] | None = None,
        generated_at_utc: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SOURCE_MODEL_SCHEMA,
            "version": SOURCE_MODEL_VERSION,
            "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        }
        if source_image is not None:
            payload["source_image"] = source_image
        if mask is not None:
            payload["mask"] = mask
        payload.update(
            {
                "image_geometry": {
                    "width_px": int(self.image_width_px),
                    "height_px": int(self.image_height_px),
                    "pixel_convention": "0-based_pixel_center",
                },
                "direction_frame": "ICRS",
                "frame_pose": self.frame_pose.to_json_payload(),
                "camera_calibration_profile": self.camera_calibration_profile.to_json_payload(),
                "frame_local_residual": self.frame_local_residual.to_json_payload(),
                "fit_metadata": dict(self.fit_metadata),
                "diagnostics": dict(self.diagnostics),
            }
        )
        if matching is not None:
            payload["matching"] = matching
        if fit_pairs is not None:
            payload["fit_pairs"] = fit_pairs
        if reference_payload is not None:
            payload["reference_payload"] = reference_payload
        return payload

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> "FrameAstrometricModel":
        if not isinstance(payload, dict):
            raise ValueError("源图模型 JSON 根对象必须是对象。")
        schema = str(payload.get("schema", ""))
        if schema != SOURCE_MODEL_SCHEMA:
            raise ValueError("不支持的源图模型 JSON：需要新的 Pixel↔ICRS FrameAstrometricModel。")
        version = _json_int(payload.get("version"), "version")
        if version != SOURCE_MODEL_VERSION:
            raise ValueError(f"不支持的源图模型 version: {version}")
        if str(payload.get("direction_frame", "")) != "ICRS":
            raise ValueError("源图模型 direction_frame 必须是 ICRS。")

        image_geometry = _json_mapping(payload.get("image_geometry"), "image_geometry")
        width = _json_int(image_geometry.get("width_px"), "image_geometry.width_px")
        height = _json_int(image_geometry.get("height_px"), "image_geometry.height_px")
        if width <= 0 or height <= 0:
            raise ValueError("源图模型 image_geometry 尺寸无效。")

        fit_metadata = payload.get("fit_metadata")
        diagnostics = payload.get("diagnostics")
        return cls(
            image_width_px=width,
            image_height_px=height,
            frame_pose=FramePose.from_json_payload(
                _json_mapping(payload.get("frame_pose"), "frame_pose"),
            ),
            camera_calibration_profile=CameraCalibrationProfile.from_json_payload(
                _json_mapping(payload.get("camera_calibration_profile"), "camera_calibration_profile"),
            ),
            frame_local_residual=FrameLocalResidual.from_json_payload(
                payload.get("frame_local_residual") if isinstance(payload.get("frame_local_residual"), dict) else None
            ),
            fit_metadata=dict(fit_metadata) if isinstance(fit_metadata, dict) else {},
            diagnostics=dict(diagnostics) if isinstance(diagnostics, dict) else {},
        )

    @classmethod
    def from_json_file(cls, path: str) -> "FrameAstrometricModel":
        import json

        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return cls.from_json_payload(payload)
