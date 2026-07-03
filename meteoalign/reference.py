from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .simulator import CameraSettings, ObserverSettings, ProjectedStarMap, ReferenceStar, ViewSettings


def build_reference_payload(
    star_map: ProjectedStarMap,
    reference_stars: tuple[ReferenceStar, ...],
    observer: ObserverSettings,
    camera: CameraSettings,
    view: ViewSettings,
    visible_mag_limit: float,
    utc_offset_hours: float = 0.0,
    reference_label_mode: str = "fixed_count",
    reference_mag_limit: float | None = None,
    generated_at_utc: datetime | None = None,
) -> dict[str, object]:
    generated_time = generated_at_utc or datetime.now(timezone.utc)
    if generated_time.tzinfo is None:
        generated_time = generated_time.replace(tzinfo=timezone.utc)
    generated_time = generated_time.astimezone(timezone.utc)
    local_timezone = timezone(timedelta(hours=utc_offset_hours))
    observation_time_local = observer.observation_time_utc.astimezone(local_timezone)

    return {
        "format": "meteoalign_phase1_reference",
        "version": 1,
        "generated_at_utc": generated_time.isoformat(),
        "catalog": {
            "source_name": star_map.source_name,
            "catalog_count": star_map.catalog_count,
            "visible_count": len(star_map),
            "above_horizon_count": star_map.above_horizon_count,
        },
        "observer": {
            "observation_time_utc": observer.observation_time_utc.astimezone(timezone.utc).isoformat(),
            "observation_time_local": observation_time_local.isoformat(),
            "utc_offset_hours": utc_offset_hours,
            "latitude_deg": observer.latitude_deg,
            "longitude_deg": observer.longitude_deg,
            "elevation_m": observer.elevation_m,
        },
        "camera": {
            "sensor_width_mm": camera.sensor_width_mm,
            "sensor_height_mm": camera.sensor_height_mm,
            "image_width_px": camera.image_width_px,
            "image_height_px": camera.image_height_px,
            "focal_length_mm": camera.focal_length_mm,
            "lens_model": camera.lens_model,
            "fisheye_fov_deg": camera.fisheye_fov_deg,
        },
        "view": {
            "center_az_deg": view.center_az_deg,
            "center_alt_deg": view.center_alt_deg,
            "roll_deg": view.roll_deg,
        },
        "render": {
            "visible_mag_limit": visible_mag_limit,
            "reference_label_mode": reference_label_mode,
            "reference_mag_limit": reference_mag_limit,
            "reference_star_count": len(reference_stars),
        },
        "stars": [
            {
                "index": star.index,
                "star_id": star.star_id,
                "name": star.name,
                "display_name": star.display_name,
                "common_name": star.common_name,
                "ra_deg": star.ra_deg,
                "dec_deg": star.dec_deg,
                "mag_v": star.mag_v,
                "sim_x": star.sim_x,
                "sim_y": star.sim_y,
                "alt_deg": star.alt_deg,
                "az_deg": star.az_deg,
            }
            for star in reference_stars
        ],
    }


def save_reference_outputs(image, payload: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "reference_star_map.png"
    json_path = output_dir / "reference_star_list.json"

    if not image.save(str(image_path)):
        raise OSError(f"无法保存参考星图：{image_path}")

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return image_path, json_path
