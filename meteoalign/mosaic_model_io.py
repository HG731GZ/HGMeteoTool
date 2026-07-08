from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QPointF

from .app_constants import AUTO_MATCH_CONSTRAINT_SOFT
from .frame_astrometry import FrameAstrometricModel
from .geometry2d import expand_polygon_radially
from .mosaic_common import MOSAIC_MODEL_REFIT_MIN_PAIRS
from .simulator import ObserverSettings


@dataclass(frozen=True)
class MosaicModelFitData:
    ra_dec_points: np.ndarray
    pixel_points: np.ndarray
    point_weights: np.ndarray
    residual_anchor_mask: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.ra_dec_points.shape[0])


@dataclass(frozen=True)
class MosaicSourceModel:
    json_path: Path
    source_image_path: Path | None
    source_image_text: str
    model: FrameAstrometricModel
    observer: ObserverSettings
    utc_offset_hours: float
    fit_data: MosaicModelFitData | None
    image_width_px: int
    image_height_px: int
    pair_count: int
    rms_px: float


@dataclass(frozen=True)
class MosaicCoverageCache:
    grid_rows: int
    grid_columns: int
    grid_x_px: np.ndarray
    grid_y_px: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class MosaicSourceTextureCache:
    source_image_path: Path
    source_rgb: np.ndarray
    source_scale_x: float
    source_scale_y: float
    source_width_px: int
    source_height_px: int


def _parse_datetime_utc(value: object, field_name: str) -> datetime:
    if value is None:
        raise ValueError(f"模型 JSON 缺少时间字段：{field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"模型 JSON 时间字段为空：{field_name}")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"模型 JSON 字段 {field_name} 必须是对象。")
    return value


def _payload_float(value: object, field_name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"模型 JSON 缺少字段：{field_name}")
        return float(default)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"模型 JSON 字段 {field_name} 不是有效数值。")
    return result


def _payload_optional_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(result) if np.isfinite(result) else float(default)


def _payload_observer(payload: dict[str, object]) -> ObserverSettings:
    dynamic: dict[str, object] | None = None
    fit_metadata = payload.get("fit_metadata")
    if isinstance(fit_metadata, dict) and isinstance(fit_metadata.get("scene_observer_hint"), dict):
        dynamic = fit_metadata.get("scene_observer_hint")
    elif isinstance(payload.get("dynamic_sky_conversion"), dict):
        dynamic = payload.get("dynamic_sky_conversion")
    elif isinstance(payload.get("reference_payload"), dict):
        reference_payload = payload.get("reference_payload")
        observer_payload = reference_payload.get("observer") if isinstance(reference_payload, dict) else None
        if isinstance(observer_payload, dict):
            dynamic = observer_payload
    if dynamic is None:
        dynamic = {}

    time_value = None
    time_field = ""
    for candidate in (
        "frame_effective_time_utc",
        "observation_time_utc",
        "frame_nominal_time_utc",
        "capture_time_utc",
        "first_frame_nominal_time_utc",
    ):
        if dynamic.get(candidate):
            time_value = dynamic.get(candidate)
            time_field = f"dynamic_sky_conversion.{candidate}"
            break
    if time_value is None:
        source_image = payload.get("source_image")
        if isinstance(source_image, dict) and source_image.get("capture_time_utc"):
            time_value = source_image.get("capture_time_utc")
            time_field = "source_image.capture_time_utc"
    if time_value is None:
        time_value = datetime.now(timezone.utc).isoformat()
        time_field = "generated_default_scene_observer_time"
    return ObserverSettings(
        observation_time_utc=_parse_datetime_utc(time_value, time_field or "observation_time_utc"),
        latitude_deg=_payload_float(dynamic.get("latitude_deg"), "scene_observer.latitude_deg", default=0.0),
        longitude_deg=_payload_float(dynamic.get("longitude_deg"), "scene_observer.longitude_deg", default=0.0),
        elevation_m=_payload_float(dynamic.get("elevation_m"), "scene_observer.elevation_m", default=0.0),
    )


def _payload_utc_offset_hours(payload: dict[str, object]) -> float:
    fit_metadata = payload.get("fit_metadata")
    if isinstance(fit_metadata, dict) and isinstance(fit_metadata.get("scene_observer_hint"), dict):
        return _payload_optional_float(fit_metadata["scene_observer_hint"].get("utc_offset_hours"), 0.0)
    dynamic = payload.get("dynamic_sky_conversion")
    if isinstance(dynamic, dict):
        return _payload_optional_float(dynamic.get("utc_offset_hours"), 0.0)
    reference_payload = payload.get("reference_payload")
    if isinstance(reference_payload, dict):
        observer_payload = reference_payload.get("observer")
        if isinstance(observer_payload, dict):
            return _payload_optional_float(observer_payload.get("utc_offset_hours"), 0.0)
    return 0.0


def _expanded_polygon_points(xs: np.ndarray, ys: np.ndarray, padding_px: float) -> list[QPointF]:
    """让覆盖面片相互轻微重叠，避免逐格填充时出现内部接缝。"""

    points = expand_polygon_radially(np.column_stack((xs, ys)), padding_px)
    return [QPointF(float(x_value), float(y_value)) for x_value, y_value in points]


def _load_mosaic_fit_data(payload: dict[str, object]) -> MosaicModelFitData | None:
    pair_payload = payload.get("fit_pairs")
    if not isinstance(pair_payload, list):
        pair_payload = payload.get("pairs")
    if not isinstance(pair_payload, list):
        return None

    ra_dec_points: list[tuple[float, float]] = []
    pixel_points: list[tuple[float, float]] = []
    point_weights: list[float] = []
    residual_anchor_mask: list[bool] = []
    for record in pair_payload:
        if not isinstance(record, dict):
            continue
        ra_deg = _payload_optional_float(record.get("ra_deg"), float("nan"))
        dec_deg = _payload_optional_float(record.get("dec_deg"), float("nan"))
        image_x_px = _payload_optional_float(record.get("image_x_px"), float("nan"))
        image_y_px = _payload_optional_float(record.get("image_y_px"), float("nan"))
        if not all(np.isfinite(value) for value in (ra_deg, dec_deg, image_x_px, image_y_px)):
            continue
        fit_weight = _payload_optional_float(record.get("fit_weight"), 1.0)
        constraint_mode = str(record.get("fit_constraint_mode") or "").strip()
        ra_dec_points.append((float(ra_deg), float(dec_deg)))
        pixel_points.append((float(image_x_px), float(image_y_px)))
        point_weights.append(float(fit_weight))
        residual_anchor_mask.append(constraint_mode != AUTO_MATCH_CONSTRAINT_SOFT)

    if len(ra_dec_points) < MOSAIC_MODEL_REFIT_MIN_PAIRS:
        return None
    return MosaicModelFitData(
        ra_dec_points=np.asarray(ra_dec_points, dtype=np.float64),
        pixel_points=np.asarray(pixel_points, dtype=np.float64),
        point_weights=np.asarray(point_weights, dtype=np.float64),
        residual_anchor_mask=np.asarray(residual_anchor_mask, dtype=bool),
    )


def _resolve_source_image_path(payload: dict[str, object], json_path: Path) -> tuple[Path | None, str]:
    source_image = payload.get("source_image")
    if not isinstance(source_image, dict):
        return None, "未记录源图路径"
    raw_path = str(source_image.get("path") or "").strip()
    if raw_path:
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = (json_path.parent / image_path).resolve()
        return image_path, image_path.name
    relative_path = str(source_image.get("relative_path") or "").strip()
    if relative_path:
        image_path = (json_path.parent / relative_path).resolve()
        return image_path, image_path.name
    file_name = str(source_image.get("file_name") or "").strip()
    if file_name:
        return None, file_name
    return None, "未记录源图路径"


def _load_mosaic_source_model(json_path: Path) -> MosaicSourceModel:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("模型 JSON 根对象必须是对象。")
    model = FrameAstrometricModel.from_json_payload(payload)
    observer = _payload_observer(payload)
    utc_offset_hours = _payload_utc_offset_hours(payload)
    fit_data = _load_mosaic_fit_data(payload)
    source_image_path, source_image_text = _resolve_source_image_path(payload, json_path)
    diagnostics_mapping = model.diagnostics
    fit_metadata = model.fit_metadata
    diagnostics_pair_count = int(
        diagnostics_mapping.get("pair_count", 0)
        or fit_metadata.get("control_point_count", 0)
        or 0
    )
    return MosaicSourceModel(
        json_path=json_path,
        source_image_path=source_image_path,
        source_image_text=source_image_text,
        model=model,
        observer=observer,
        utc_offset_hours=utc_offset_hours,
        fit_data=fit_data,
        image_width_px=int(model.image_width_px),
        image_height_px=int(model.image_height_px),
        pair_count=diagnostics_pair_count or (0 if fit_data is None else fit_data.pair_count),
        rms_px=float(diagnostics_mapping.get("rms_px", float("nan"))),
    )




__all__ = [name for name in globals() if not name.startswith("__")]
