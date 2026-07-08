from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .simulator import ObserverSettings, ReferenceStar, compute_altaz_from_radec
from .star_fitting import FittedStarPosition


PAIR_ORIGIN_MANUAL = "manual"
PAIR_ORIGIN_AUTO_MATCH = "auto_match"
CONSTRAINT_ANCHOR = "anchor"
CONSTRAINT_SOFT = "soft"


def _payload_float(payload: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _payload_text(payload: dict[str, object], key: str, default: str = "") -> str:
    return str(payload.get(key, default) or default).strip()


@dataclass(frozen=True)
class PsfFit:
    """星点 PSF 拟合结果。"""

    x: float
    y: float
    amplitude: float = 0.0
    background: float = 0.0
    sigma_x: float = 0.0
    sigma_y: float = 0.0

    @classmethod
    def from_fitted_position(cls, fitted: FittedStarPosition) -> "PsfFit":
        return cls(
            x=float(fitted.x),
            y=float(fitted.y),
            amplitude=float(fitted.amplitude),
            background=float(fitted.background),
            sigma_x=float(fitted.sigma_x),
            sigma_y=float(fitted.sigma_y),
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        *,
        fallback_x: float,
        fallback_y: float,
    ) -> "PsfFit | None":
        values: dict[str, float] = {}
        for key in ("amplitude", "background", "sigma_x", "sigma_y"):
            value = _payload_float(payload, key)
            if math.isfinite(value):
                values[key] = value
        if not values:
            return None
        x_value = _payload_float(payload, "x", fallback_x)
        y_value = _payload_float(payload, "y", fallback_y)
        return cls(
            x=x_value,
            y=y_value,
            amplitude=float(values.get("amplitude", 0.0)),
            background=float(values.get("background", 0.0)),
            sigma_x=float(values.get("sigma_x", 0.0)),
            sigma_y=float(values.get("sigma_y", 0.0)),
        )

    def to_fitted_position(self) -> FittedStarPosition:
        return FittedStarPosition(
            x=float(self.x),
            y=float(self.y),
            amplitude=float(self.amplitude),
            background=float(self.background),
            sigma_x=float(self.sigma_x),
            sigma_y=float(self.sigma_y),
        )

    def to_table_payload(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "amplitude": float(self.amplitude),
            "background": float(self.background),
            "sigma_x": float(self.sigma_x),
            "sigma_y": float(self.sigma_y),
        }

    def to_json_payload(self) -> dict[str, float]:
        return {
            "amplitude": float(self.amplitude),
            "background": float(self.background),
            "sigma_x": float(self.sigma_x),
            "sigma_y": float(self.sigma_y),
        }


@dataclass(frozen=True)
class StarPairRecord:
    """一条星点配对业务记录，不依赖具体表格行。"""

    reference_star: ReferenceStar
    image_x_px: float
    image_y_px: float
    psf: PsfFit | None = None
    pair_origin: str = PAIR_ORIGIN_MANUAL
    group_id: str | None = None
    group_name: str | None = None
    fit_constraint_mode: str = CONSTRAINT_ANCHOR
    fit_weight: float = 1.0
    residual_dx_px: float | None = None
    residual_dy_px: float | None = None
    residual_px: float | None = None
    enabled: bool = True
    extra_fields: dict[str, object] = field(default_factory=dict)

    @property
    def star_id(self) -> str:
        return self.reference_star.star_id.strip()

    @property
    def position(self) -> tuple[float, float]:
        return float(self.image_x_px), float(self.image_y_px)

    @property
    def fitted_position(self) -> FittedStarPosition:
        if self.psf is not None:
            return self.psf.to_fitted_position()
        return FittedStarPosition(
            x=float(self.image_x_px),
            y=float(self.image_y_px),
            amplitude=0.0,
            background=0.0,
            sigma_x=0.0,
            sigma_y=0.0,
        )

    @property
    def is_auto_match(self) -> bool:
        return self.pair_origin == PAIR_ORIGIN_AUTO_MATCH

    def is_catalog_pair(self) -> bool:
        star_id = self.star_id
        return self.reference_star.object_type == "star" and bool(star_id) and not star_id.startswith("solar_system:")

    def is_valid_for_fit(self) -> bool:
        values = (
            self.reference_star.ra_deg,
            self.reference_star.dec_deg,
            self.image_x_px,
            self.image_y_px,
        )
        return self.enabled and self.is_catalog_pair() and all(math.isfinite(float(value)) for value in values)

    def to_json_payload(self, *, reference_index: int | None = None) -> dict[str, object]:
        reference_star = self.reference_star
        payload: dict[str, object] = {
            "reference_index": int(reference_index if reference_index is not None else reference_star.index),
            "star_id": reference_star.star_id,
            "name": reference_star.name,
            "display_name": reference_star.display_name,
            "common_name": reference_star.common_name,
            "ra_deg": float(reference_star.ra_deg),
            "dec_deg": float(reference_star.dec_deg),
            "mag_v": float(reference_star.mag_v),
            "image_x_px": float(self.image_x_px),
            "image_y_px": float(self.image_y_px),
            "sim_x": float(reference_star.sim_x),
            "sim_y": float(reference_star.sim_y),
            "alt_deg": float(reference_star.alt_deg),
            "az_deg": float(reference_star.az_deg),
            "object_type": reference_star.object_type,
            "pair_origin": self.pair_origin,
            "fit_constraint_mode": self.fit_constraint_mode,
            "fit_weight": float(self.fit_weight),
        }
        if reference_star.index_label:
            payload["index_label"] = reference_star.index_label
        if self.group_id:
            payload["auto_match_group_id"] = self.group_id
        if self.group_name:
            payload["auto_match_group_name"] = self.group_name
        if self.psf is not None:
            payload.update(self.psf.to_json_payload())
        if self.residual_px is not None:
            payload["residual_dx_px"] = float(self.residual_dx_px or 0.0)
            payload["residual_dy_px"] = float(self.residual_dy_px or 0.0)
            payload["residual_px"] = float(self.residual_px)
        payload.update(self.extra_fields)
        return payload


def reference_star_from_pair_payload(
    payload: object,
    *,
    observer: ObserverSettings | None = None,
    output_index: int = 0,
) -> ReferenceStar | None:
    """从配对 JSON 记录恢复参考星信息。"""

    if not isinstance(payload, dict):
        return None
    object_type = _payload_text(payload, "object_type", "star")
    star_id = _payload_text(payload, "star_id")
    if object_type != "star" or not star_id or star_id.startswith("solar_system:"):
        return None
    ra_deg = _payload_float(payload, "ra_deg")
    dec_deg = _payload_float(payload, "dec_deg")
    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg):
        return None

    alt_deg = _payload_float(payload, "alt_deg")
    az_deg = _payload_float(payload, "az_deg")
    if observer is not None and (not math.isfinite(alt_deg) or not math.isfinite(az_deg)):
        try:
            alt_values, az_values = compute_altaz_from_radec(
                np.asarray([ra_deg], dtype=np.float64),
                np.asarray([dec_deg], dtype=np.float64),
                observer,
            )
            alt_deg = float(alt_values[0])
            az_deg = float(az_values[0])
        except Exception:  # noqa: BLE001 - 导入旧记录时缺少地平坐标不应阻断整个会话。
            pass

    display_name = _payload_text(payload, "display_name")
    common_name = _payload_text(payload, "common_name")
    name = _payload_text(payload, "name", common_name or display_name or star_id)
    sim_x = _payload_float(payload, "sim_x", _payload_float(payload, "theoretical_x_px", _payload_float(payload, "image_x_px")))
    sim_y = _payload_float(payload, "sim_y", _payload_float(payload, "theoretical_y_px", _payload_float(payload, "image_y_px")))
    return ReferenceStar(
        index=int(output_index),
        star_id=star_id,
        name=name or star_id,
        display_name=display_name or name or star_id,
        common_name=common_name,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        mag_v=_payload_float(payload, "mag_v"),
        sim_x=sim_x,
        sim_y=sim_y,
        alt_deg=alt_deg,
        az_deg=az_deg,
        object_type=object_type,
        index_label=_payload_text(payload, "index_label"),
    )


def star_pair_record_from_payload(
    payload: object,
    *,
    observer: ObserverSettings | None = None,
    output_index: int = 0,
) -> StarPairRecord | None:
    """从 JSON payload 恢复星点配对记录。"""

    if not isinstance(payload, dict):
        return None
    reference_star = reference_star_from_pair_payload(payload, observer=observer, output_index=output_index)
    if reference_star is None:
        return None
    image_x = _payload_float(payload, "image_x_px")
    image_y = _payload_float(payload, "image_y_px")
    if not math.isfinite(image_x) or not math.isfinite(image_y):
        return None

    mode = _payload_text(payload, "fit_constraint_mode", CONSTRAINT_ANCHOR)
    if mode not in (CONSTRAINT_ANCHOR, CONSTRAINT_SOFT):
        mode = CONSTRAINT_ANCHOR
    fit_weight = _payload_float(payload, "fit_weight", 1.0)
    if mode == CONSTRAINT_SOFT:
        fit_weight = max(0.01, min(1.0, fit_weight))
    else:
        fit_weight = 1.0

    known_keys = {
        "reference_index",
        "star_id",
        "name",
        "display_name",
        "common_name",
        "ra_deg",
        "dec_deg",
        "mag_v",
        "image_x_px",
        "image_y_px",
        "sim_x",
        "sim_y",
        "alt_deg",
        "az_deg",
        "object_type",
        "pair_origin",
        "auto_match_group_id",
        "auto_match_group_name",
        "fit_constraint_mode",
        "fit_weight",
        "amplitude",
        "background",
        "sigma_x",
        "sigma_y",
        "x",
        "y",
        "residual_dx_px",
        "residual_dy_px",
        "residual_px",
        "index_label",
    }
    extras = {key: value for key, value in payload.items() if key not in known_keys}
    residual_px = _payload_float(payload, "residual_px")
    return StarPairRecord(
        reference_star=reference_star,
        image_x_px=image_x,
        image_y_px=image_y,
        psf=PsfFit.from_payload(payload, fallback_x=image_x, fallback_y=image_y),
        pair_origin=_payload_text(payload, "pair_origin", PAIR_ORIGIN_MANUAL),
        group_id=_payload_text(payload, "auto_match_group_id") or None,
        group_name=_payload_text(payload, "auto_match_group_name") or None,
        fit_constraint_mode=mode,
        fit_weight=fit_weight,
        residual_dx_px=_payload_float(payload, "residual_dx_px") if math.isfinite(residual_px) else None,
        residual_dy_px=_payload_float(payload, "residual_dy_px") if math.isfinite(residual_px) else None,
        residual_px=residual_px if math.isfinite(residual_px) else None,
        extra_fields=extras,
    )


def star_pair_records_from_payloads(
    payloads: list[object],
    *,
    observer: ObserverSettings | None = None,
) -> list[StarPairRecord]:
    records: list[StarPairRecord] = []
    for output_index, payload in enumerate(payloads, start=1):
        record = star_pair_record_from_payload(payload, observer=observer, output_index=output_index)
        if record is not None:
            records.append(record)
    return records
