from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from .runtime_paths import runtime_catalog_dir, source_project_root


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


@dataclass(frozen=True)
class ConstellationLine:
    """一条按 HIP 编号依次连接的星座折线。"""

    hip_ids: tuple[int, ...]
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag_v: np.ndarray


@dataclass(frozen=True)
class ConstellationDefinition:
    """单个星座的中文名称及其全部连线。"""

    constellation_id: str
    abbreviation: str
    chinese_name: str
    lines: tuple[ConstellationLine, ...]


@dataclass(frozen=True)
class ConstellationCatalog:
    """从 Stellarium 连线文件和 Hipparcos 主表组成的星座数据。"""

    source_name: str
    constellations: tuple[ConstellationDefinition, ...]


def project_root() -> Path:
    return source_project_root()


def default_catalog_dir() -> Path:
    return runtime_catalog_dir()


def default_yale_bsc_path() -> Path:
    return default_catalog_dir() / "yale_bsc" / "catalog.gz"


def default_iau_csn_path() -> Path:
    return default_catalog_dir() / "iau_csn" / "modern_iau_star_names.html"


def default_hipparcos_path() -> Path:
    return default_catalog_dir() / "hipparcos_i239" / "hip_main.dat"


def default_constellation_path() -> Path:
    return default_catalog_dir() / "constellation.json"


def default_chinese_star_names_path() -> Path:
    return default_catalog_dir() / "star_names.zh_CN.fab"


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


class _IauCsnTableParser(HTMLParser):
    """只抓取 IAU CSN 主表中的单元格文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self._in_target_table = False
        self._table_depth = 0
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == "table_1":
            self._in_target_table = True
            self._table_depth = 1
            return

        if not self._in_target_table:
            return

        if tag == "table":
            self._table_depth += 1
        elif tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
        elif tag in {"br", "p", "div"} and self._current_cell is not None:
            self._current_cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_target_table:
            return

        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(" ".join("".join(self._current_cell).split()))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(tuple(self._current_row))
            self._current_row = None
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth <= 0:
                self._in_target_table = False


def _extract_hr_id(text: str) -> int | None:
    match = re.search(r"\bHR\s*(\d+)\b", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


@lru_cache(maxsize=4)
def _load_iau_star_names_by_hr_cached(path_text: str) -> dict[int, str]:
    path = Path(path_text)
    if not path.exists():
        return {}

    parser = _IauCsnTableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))

    names: dict[int, str] = {}
    for row in parser.rows:
        if len(row) < 16 or row[0] == "proper names":
            continue

        hr = _extract_hr_id(row[2])
        display_name = row[5].strip() or row[0].strip()
        if hr is not None and display_name and hr not in names:
            names[hr] = display_name

    if parser.rows and not names:
        raise ValueError(f"未能从 IAU CSN 表解析出 HR 星名：{path}")
    return names


def load_iau_star_names_by_hr(path: Path | None = None) -> dict[int, str]:
    return dict(_load_iau_star_names_by_hr_cached(str(path or default_iau_csn_path())))


_CHINESE_STAR_NAME_PATTERN = re.compile(r'^\s*(\d+)\s*\|\s*_\("((?:\\.|[^"])*)"\)')


def _clean_chinese_star_name(raw_name: str) -> str:
    return raw_name.replace(r"\"", '"').replace(r"\\", "\\").strip()


@lru_cache(maxsize=4)
def _load_chinese_star_names_by_hip_cached(path_text: str) -> dict[int, str]:
    path = Path(path_text)
    if not path.exists():
        return {}

    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _CHINESE_STAR_NAME_PATTERN.match(line)
        if match is None:
            continue

        hip_id = int(match.group(1))
        # 同一个 HIP 可能有多个传统名，按星表文件中的第一项作为显示名。
        if hip_id not in names:
            names[hip_id] = _clean_chinese_star_name(match.group(2))
    return names


def load_chinese_star_names_by_hip(path: Path | None = None) -> dict[int, str]:
    return dict(_load_chinese_star_names_by_hip_cached(str(path or default_chinese_star_names_path())))


@lru_cache(maxsize=4)
def _load_hip_ids_by_hd_cached(path_text: str) -> dict[int, int]:
    path = Path(path_text)
    if not path.exists():
        return {}

    hip_ids_by_hd: dict[int, int] = {}
    with path.open("rt", encoding="latin-1") as handle:
        for line in handle:
            try:
                hip_id = _parse_int(line[8:14])
                hd_id = _parse_int(line[390:396])
            except ValueError:
                continue

            # 少数 HD 可能对应多个分量；这里保留 Hipparcos 主表中第一次出现的 HIP。
            if hip_id is not None and hd_id is not None and hd_id not in hip_ids_by_hd:
                hip_ids_by_hd[hd_id] = hip_id
    return hip_ids_by_hd


def load_hip_ids_by_hd(path: Path | None = None) -> dict[int, int]:
    return dict(_load_hip_ids_by_hd_cached(str(path or default_hipparcos_path())))


@lru_cache(maxsize=4)
def _load_hipparcos_positions_cached(path_text: str) -> dict[int, tuple[float, float, float]]:
    """读取 Hipparcos 的 HIP、ICRS 坐标和 V 星等。"""

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"未找到 Hipparcos 主表：{path}")

    positions: dict[int, tuple[float, float, float]] = {}
    with path.open("rt", encoding="latin-1") as handle:
        for line in handle:
            try:
                hip_id = _parse_int(line[8:14])
                ra_deg = _parse_float(line[51:63])
                dec_deg = _parse_float(line[64:76])
                mag_v = _parse_float(line[41:46])
            except ValueError:
                continue
            if hip_id is None or ra_deg is None or dec_deg is None:
                continue
            positions[hip_id] = (
                float(ra_deg),
                float(dec_deg),
                np.nan if mag_v is None else float(mag_v),
            )
    return positions


def load_constellation_catalog(
    path: Path | None = None,
    hipparcos_path: Path | None = None,
) -> ConstellationCatalog:
    """加载 Stellarium 星座折线，并用 Hipparcos 主表补全节点坐标。"""

    constellation_path = path or default_constellation_path()
    if not constellation_path.exists():
        raise FileNotFoundError(f"未找到星座连线文件：{constellation_path}")
    try:
        payload = json.loads(constellation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"星座连线文件不是有效 JSON：{constellation_path}") from exc

    raw_constellations = payload.get("constellations", []) if isinstance(payload, dict) else []
    if not isinstance(raw_constellations, list):
        raise ValueError(f"星座连线文件缺少 constellations 数组：{constellation_path}")

    positions = _load_hipparcos_positions_cached(str(hipparcos_path or default_hipparcos_path()))
    definitions: list[ConstellationDefinition] = []
    for raw_constellation in raw_constellations:
        if not isinstance(raw_constellation, dict):
            continue
        constellation_id = str(raw_constellation.get("id", "")).strip()
        abbreviation = constellation_id.split()[-1] if constellation_id else ""
        common_name = raw_constellation.get("common_name", {})
        chinese_name = str(common_name.get("chinese", "")).strip() if isinstance(common_name, dict) else ""
        if not chinese_name:
            raise ValueError(f"星座 {abbreviation or constellation_id!r} 缺少 common_name.chinese")

        lines: list[ConstellationLine] = []
        raw_lines = raw_constellation.get("lines", [])
        if not isinstance(raw_lines, list):
            continue
        for raw_line in raw_lines:
            if not isinstance(raw_line, list):
                continue
            try:
                hip_ids = tuple(int(value) for value in raw_line)
            except (TypeError, ValueError):
                continue
            if len(hip_ids) < 2:
                continue

            # 缺坐标时按缺失节点切开，避免错误地跨节点补线。
            current_ids: list[int] = []
            for hip_id in (*hip_ids, -1):
                if hip_id in positions:
                    current_ids.append(hip_id)
                    continue
                if len(current_ids) >= 2:
                    values = [positions[value] for value in current_ids]
                    lines.append(
                        ConstellationLine(
                            hip_ids=tuple(current_ids),
                            ra_deg=np.asarray([value[0] for value in values], dtype=np.float64),
                            dec_deg=np.asarray([value[1] for value in values], dtype=np.float64),
                            mag_v=np.asarray([value[2] for value in values], dtype=np.float64),
                        )
                    )
                current_ids = []

        definitions.append(
            ConstellationDefinition(
                constellation_id=constellation_id,
                abbreviation=abbreviation,
                chinese_name=chinese_name,
                lines=tuple(lines),
            )
        )

    return ConstellationCatalog(
        source_name=f"Stellarium constellation lines ({constellation_path.name})",
        constellations=tuple(definitions),
    )


def load_chinese_star_names_by_hr(
    yale_bsc_path: Path | None = None,
    hipparcos_path: Path | None = None,
    chinese_names_path: Path | None = None,
) -> dict[int, str]:
    hip_ids_by_hd = load_hip_ids_by_hd(hipparcos_path)
    names_by_hip = load_chinese_star_names_by_hip(chinese_names_path)
    if not hip_ids_by_hd or not names_by_hip:
        return {}

    names_by_hr: dict[int, str] = {}
    with gzip.open(yale_bsc_path or default_yale_bsc_path(), "rt", encoding="latin-1") as handle:
        for line in handle:
            try:
                hr = _parse_int(line[0:4])
                hd = _parse_int(line[25:31])
            except ValueError:
                continue
            if hr is None or hd is None:
                continue

            hip_id = hip_ids_by_hd.get(hd)
            chinese_name = names_by_hip.get(hip_id) if hip_id is not None else None
            if chinese_name:
                names_by_hr[hr] = chinese_name
    return names_by_hr


def load_yale_bsc(path: Path | None = None, mag_limit: float | None = 6.5) -> StarCatalog:
    """加载 Yale Bright Star Catalog 固定宽度星表。

    这里提取 HR 编号、显示名、J2000 赤经赤纬、V 星等、B-V 色指数和光谱型。
    中文星名通过 HR→HD→HIP 链接优先补全 common_names，找不到时再回退到 IAU 英文名；
    显示字段不参与坐标、星等或后续解算。
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
    chinese_display_names_by_hr = load_chinese_star_names_by_hr(yale_bsc_path=catalog_path)
    iau_display_names_by_hr = load_iau_star_names_by_hr()

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
            common_names.append(
                chinese_display_names_by_hr.get(hr)
                or iau_display_names_by_hr.get(hr)
                or BRIGHT_STAR_COMMON_NAMES.get(hr, "")
            )

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
