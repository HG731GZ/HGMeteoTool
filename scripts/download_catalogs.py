"""Download the star catalogs used by MeteoAlign.

Run with:
    conda run -n hgastro python scripts/download_catalogs.py

The script creates the catalog directory if it does not exist. Existing files
with the expected size are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CatalogFile:
    label: str
    url: str
    relative_path: Path
    expected_size: int


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
)


def default_catalog_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "catalog"


def file_is_complete(path: Path, expected_size: int) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size == expected_size


def download_file(item: CatalogFile, catalog_dir: Path, force: bool) -> None:
    target = catalog_dir / item.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force and file_is_complete(target, item.expected_size):
        print(f"skip: {item.label} -> {target}")
        return

    if target.exists() and not force:
        print(f"redownload: size mismatch for {target}")

    print(f"download: {item.label}")
    print(f"  from: {item.url}")
    print(f"  to:   {target}")

    with tempfile.NamedTemporaryFile(
        prefix=f"{target.name}.", suffix=".part", dir=target.parent, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            request = urllib.request.Request(
                item.url, headers={"User-Agent": "MeteoAlign catalog downloader"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temp_file)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    actual_size = temp_path.stat().st_size
    if actual_size != item.expected_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{item.label} downloaded {actual_size} bytes, "
            f"expected {item.expected_size} bytes"
        )

    temp_path.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=default_catalog_dir(),
        help="Destination catalog directory. Defaults to ./catalog in the project root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when an existing file has the expected size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_dir = args.catalog_dir.resolve()

    if not catalog_dir.exists():
        print(f"create catalog directory: {catalog_dir}")
        catalog_dir.mkdir(parents=True)
    elif not catalog_dir.is_dir():
        raise NotADirectoryError(f"catalog path is not a directory: {catalog_dir}")

    try:
        for item in CATALOG_FILES:
            download_file(item, catalog_dir, args.force)
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("catalog downloads OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
