from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from meteoalign.star_fitting import FittedStarPosition
from meteoalign.star_pair_model import (
    PAIR_ORIGIN_AUTO_MATCH,
    PsfFit,
    StarPairRecord,
    star_pair_record_from_payload,
)
from meteoalign.simulator import ObserverSettings, ReferenceStar


def _star() -> ReferenceStar:
    return ReferenceStar(
        index=3,
        star_id="HR123",
        name="测试星",
        display_name="HR123",
        common_name="",
        ra_deg=12.5,
        dec_deg=34.5,
        mag_v=2.1,
        sim_x=101.0,
        sim_y=202.0,
        alt_deg=45.0,
        az_deg=180.0,
    )


def test_star_pair_record_round_trips_json_payload() -> None:
    fitted = FittedStarPosition(
        x=123.4,
        y=234.5,
        amplitude=50.0,
        background=3.0,
        sigma_x=1.2,
        sigma_y=1.4,
        theta_rad=0.25,
        fwhm_x=3.3,
        fwhm_y=2.8,
        snr=18.0,
        fit_error=0.12,
        saturated=True,
        saturation_fraction=0.2,
        blended=True,
        quality_score=0.88,
    )
    record = StarPairRecord(
        reference_star=_star(),
        image_x_px=fitted.x,
        image_y_px=fitted.y,
        psf=PsfFit.from_fitted_position(fitted),
        pair_origin=PAIR_ORIGIN_AUTO_MATCH,
        group_id="B",
        group_name="自动匹配 B",
        fit_constraint_mode="soft",
        fit_weight=0.35,
        residual_dx_px=1.0,
        residual_dy_px=-2.0,
        residual_px=float(np.hypot(1.0, -2.0)),
        extra_fields={
            "theoretical_x_px": 120.0,
            "auto_match_quality_score": 0.76,
        },
    )

    payload = record.to_json_payload()
    restored = star_pair_record_from_payload(payload)

    assert payload["auto_match_quality_score"] == 0.76
    assert restored is not None
    assert restored.star_id == record.star_id
    assert restored.position == record.position
    assert restored.psf is not None
    assert restored.psf.to_table_payload()["x"] == fitted.x
    assert restored.psf.theta_rad == fitted.theta_rad
    assert restored.psf.saturated
    assert restored.psf.blended
    assert restored.psf.quality_score == fitted.quality_score
    assert "quality_score" not in restored.extra_fields
    assert restored.fit_constraint_mode == "soft"
    assert restored.fit_weight == 0.35
    assert restored.group_id == "B"
    assert restored.extra_fields["theoretical_x_px"] == 120.0
    assert restored.extra_fields["auto_match_quality_score"] == 0.76


def test_star_pair_record_payload_can_recover_missing_altaz_from_observer() -> None:
    star = replace(_star(), alt_deg=float("nan"), az_deg=float("nan"))
    payload = StarPairRecord(star, image_x_px=10.0, image_y_px=20.0).to_json_payload()
    observer = ObserverSettings(
        observation_time_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        latitude_deg=40.0,
        longitude_deg=116.0,
        elevation_m=50.0,
    )

    restored = star_pair_record_from_payload(payload, observer=observer)

    assert restored is not None
    assert np.isfinite(restored.reference_star.alt_deg)
    assert np.isfinite(restored.reference_star.az_deg)
