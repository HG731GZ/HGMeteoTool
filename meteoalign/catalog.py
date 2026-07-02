from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StarCatalog:
    source_name: str
    star_ids: np.ndarray
    display_names: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag_v: np.ndarray
    color_index_bv: np.ndarray
    spectral_type: np.ndarray
    common_names: np.ndarray

    def with_mag_limit(self, mag_limit: float) -> "StarCatalog":
        mask = self.mag_v <= mag_limit
        return StarCatalog(
            source_name=self.source_name,
            star_ids=self.star_ids[mask],
            display_names=self.display_names[mask],
            ra_deg=self.ra_deg[mask],
            dec_deg=self.dec_deg[mask],
            mag_v=self.mag_v[mask],
            color_index_bv=self.color_index_bv[mask],
            spectral_type=self.spectral_type[mask],
            common_names=self.common_names[mask],
        )

    def __len__(self) -> int:
        return int(self.ra_deg.size)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_yale_bsc_path() -> Path:
    return project_root() / "catalog" / "yale_bsc" / "catalog.gz"


def _parse_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    return int(text)


def _parse_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    return float(text)


def _ra_hms_to_deg(hours: int, minutes: int, seconds: float) -> float:
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def _dec_dms_to_deg(sign_text: str, degrees: int, minutes: int, seconds: int) -> float:
    sign = -1.0 if sign_text == "-" else 1.0
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


BRIGHT_STAR_COMMON_NAMES = {
    472: "Achernar",
    1457: "Aldebaran",
    1708: "Capella",
    1713: "Rigel",
    1790: "Bellatrix",
    2061: "Betelgeuse",
    2326: "Canopus",
    2491: "Sirius",
    2943: "Procyon",
    2990: "Pollux",
    3165: "Miaplacidus",
    4730: "Acrux",
    4853: "Mimosa",
    5056: "Spica",
    5267: "Hadar",
    5340: "Arcturus",
    5459: "Rigil Kentaurus",
    6134: "Antares",
    7001: "Vega",
    7557: "Altair",
    8728: "Fomalhaut",
}


def load_yale_bsc(path: Path | None = None, mag_limit: float | None = 6.5) -> StarCatalog:
    """Load the Yale Bright Star Catalog fixed-width file.

    Phase 1 extracts HR id, display name, J2000 RA/Dec, Johnson V magnitude,
    B-V color index, spectral type, and a small common-name compatibility map.
    """

    catalog_path = path or default_yale_bsc_path()
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Yale Bright Star Catalog not found: {catalog_path}. "
            "Run scripts/download_catalogs.py first."
        )

    star_ids: list[str] = []
    display_names: list[str] = []
    ra_values: list[float] = []
    dec_values: list[float] = []
    mag_values: list[float] = []
    color_values: list[float] = []
    spectral_types: list[str] = []
    common_names: list[str] = []

    with gzip.open(catalog_path, "rt", encoding="latin-1") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            try:
                hr = _parse_int(line[0:4])
                ra_h = _parse_int(line[75:77])
                ra_m = _parse_int(line[77:79])
                ra_s = _parse_float(line[79:83])
                dec_d = _parse_int(line[84:86])
                dec_m = _parse_int(line[86:88])
                dec_s = _parse_int(line[88:90])
                mag_v = _parse_float(line[102:107])
            except ValueError:
                continue

            try:
                color_index = _parse_float(line[109:114])
            except ValueError:
                color_index = None

            spectral_type = line[127:147].strip()

            if (
                hr is None
                or ra_h is None
                or ra_m is None
                or ra_s is None
                or dec_d is None
                or dec_m is None
                or dec_s is None
                or mag_v is None
            ):
                continue
            if mag_limit is not None and mag_v > mag_limit:
                continue

            name = line[4:14].strip()
            var_id = line[51:60].strip()
            display_name = name or var_id or f"HR {hr}"

            star_ids.append(f"HR{hr}")
            display_names.append(display_name)
            ra_values.append(_ra_hms_to_deg(ra_h, ra_m, ra_s))
            dec_values.append(_dec_dms_to_deg(line[83:84], dec_d, dec_m, dec_s))
            mag_values.append(mag_v)
            color_values.append(np.nan if color_index is None else color_index)
            spectral_types.append(spectral_type)
            common_names.append(BRIGHT_STAR_COMMON_NAMES.get(hr, ""))

    if not star_ids:
        raise ValueError(f"No usable stars loaded from {catalog_path}")

    return StarCatalog(
        source_name="Yale Bright Star Catalog",
        star_ids=np.asarray(star_ids, dtype=object),
        display_names=np.asarray(display_names, dtype=object),
        ra_deg=np.asarray(ra_values, dtype=np.float64),
        dec_deg=np.asarray(dec_values, dtype=np.float64),
        mag_v=np.asarray(mag_values, dtype=np.float64),
        color_index_bv=np.asarray(color_values, dtype=np.float64),
        spectral_type=np.asarray(spectral_types, dtype=object),
        common_names=np.asarray(common_names, dtype=object),
    )


def load_default_catalog(mag_limit: float | None = 6.5) -> StarCatalog:
    return load_yale_bsc(mag_limit=mag_limit)
