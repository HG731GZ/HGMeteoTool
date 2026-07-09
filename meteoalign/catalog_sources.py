from __future__ import annotations

import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CatalogFile:
    label: str
    url: str
    relative_path: Path
    expected_size: int | None = None
    minimum_size: int = 1
    required_text: tuple[str, ...] = ()
    text_encoding: str = "utf-8"


CATALOG_FILES = (
    CatalogFile(
        label="Yale Bright Star Catalog",
        url="https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz",
        relative_path=Path("yale_bsc/catalog.gz"),
        expected_size=573_921,
    ),
    CatalogFile(
        label="Yale Bright Star Catalog ReadMe",
        url="https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe",
        relative_path=Path("yale_bsc/ReadMe"),
        expected_size=11_571,
    ),
    CatalogFile(
        label="Hipparcos main catalog",
        url="https://cdsarc.cds.unistra.fr/ftp/I/239/hip_main.dat",
        relative_path=Path("hipparcos_i239/hip_main.dat"),
        expected_size=53_316_318,
    ),
    CatalogFile(
        label="Hipparcos main catalog ReadMe",
        url="https://cdsarc.cds.unistra.fr/ftp/I/239/ReadMe",
        relative_path=Path("hipparcos_i239/ReadMe"),
        expected_size=69_008,
    ),
    CatalogFile(
        label="IAU Catalog of Star Names (CSN)",
        url="https://exopla.net/star-names/modern-iau-star-names/",
        relative_path=Path("iau_csn/modern_iau_star_names.html"),
        minimum_size=100_000,
        required_text=(
            "IAU-Catalog of Star Names",
            "proper names",
            "Simbad spelling",
            "Sirius",
        ),
    ),
    CatalogFile(
        label="d3-celestial Milky Way GeoJSON",
        url="https://cdn.jsdelivr.net/npm/d3-celestial@0.7.35/data/mw.json",
        relative_path=Path("d3_celestial/mw.json"),
        expected_size=534_254,
        required_text=(
            '"FeatureCollection"',
            '"MultiPolygon"',
            '"ol1"',
        ),
    ),
    CatalogFile(
        label="JPL DE440s Solar System Ephemeris",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
        relative_path=Path("de440s.bsp"),
        expected_size=32_726_016,
        minimum_size=30_000_000,
    ),
)


def default_catalog_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "catalog"


def file_is_complete(path: Path, expected_size: int | None) -> bool:
    return path.exists() and path.is_file() and (expected_size is None or path.stat().st_size == expected_size)


def catalog_file_is_complete(item: CatalogFile, catalog_dir: Path) -> bool:
    target = catalog_dir / item.relative_path
    if not file_is_complete(target, item.expected_size):
        return False

    if target.stat().st_size < item.minimum_size:
        return False

    if not item.required_text:
        return True

    try:
        text = target.read_text(encoding=item.text_encoding, errors="replace")
    except OSError:
        return False
    return all(fragment in text for fragment in item.required_text)


def incomplete_catalog_files(catalog_dir: Path | None = None) -> list[CatalogFile]:
    base_dir = catalog_dir or default_catalog_dir()
    if not base_dir.exists() or not base_dir.is_dir():
        return list(CATALOG_FILES)

    return [item for item in CATALOG_FILES if not catalog_file_is_complete(item, base_dir)]


def ensure_catalog_dir(catalog_dir: Path) -> None:
    if not catalog_dir.exists():
        print(f"创建星表目录：{catalog_dir}")
        catalog_dir.mkdir(parents=True)
    elif not catalog_dir.is_dir():
        raise NotADirectoryError(f"星表路径不是目录：{catalog_dir}")


DownloadProgressCallback = Callable[[int, int | None, float], None]


def _content_length(response) -> int | None:  # type: ignore[no-untyped-def]
    text = response.headers.get("Content-Length")
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def download_file(
    item: CatalogFile,
    catalog_dir: Path,
    force: bool,
    progress_callback: DownloadProgressCallback | None = None,
) -> None:
    target = catalog_dir / item.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force and catalog_file_is_complete(item, catalog_dir):
        print(f"跳过：{item.label} -> {target}")
        return

    if target.exists() and not force:
        print(f"重新下载：{target} 不完整或校验未通过")

    print(f"下载：{item.label}")
    print(f"  来源：{item.url}")
    print(f"  保存：{target}")

    with tempfile.NamedTemporaryFile(
        prefix=f"{target.name}.", suffix=".part", dir=target.parent, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            request = urllib.request.Request(
                item.url, headers={"User-Agent": "HoshinoPanoAssistant catalog downloader"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                total_bytes = item.expected_size or _content_length(response)
                downloaded_bytes = 0
                started_at = time.monotonic()
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    if progress_callback is not None:
                        elapsed = max(time.monotonic() - started_at, 1e-6)
                        progress_callback(downloaded_bytes, total_bytes, downloaded_bytes / elapsed)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    temp_path.replace(target)
    if not catalog_file_is_complete(item, catalog_dir):
        actual_size = target.stat().st_size
        target.unlink(missing_ok=True)
        if item.expected_size is None:
            raise RuntimeError(f"{item.label} 下载后未通过完整性校验")
        raise RuntimeError(
            f"{item.label} 下载后大小为 {actual_size} 字节，"
            f"预期为 {item.expected_size} 字节"
        )


def download_all_catalogs(catalog_dir: Path | None = None, force: bool = False) -> None:
    base_dir = (catalog_dir or default_catalog_dir()).resolve()
    ensure_catalog_dir(base_dir)
    for item in CATALOG_FILES:
        download_file(item, base_dir, force)
