"""下载 MeteoAlign 使用的离线星表。

运行方式：
    conda run -n hgastro python scripts/download_catalogs.py

脚本会自动创建 catalog 目录；已有且完整的文件会跳过，除非传入 --force。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meteoalign.catalog_sources import (
    CATALOG_FILES,
    default_catalog_dir,
    download_all_catalogs,
    download_file,
    file_is_complete,
)

__all__ = (
    "CATALOG_FILES",
    "default_catalog_dir",
    "download_all_catalogs",
    "download_file",
    "file_is_complete",
    "main",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=default_catalog_dir(),
        help="星表保存目录。默认是项目根目录下的 ./catalog。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已有文件完整，也重新下载。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        download_all_catalogs(args.catalog_dir, args.force)
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("星表下载完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
