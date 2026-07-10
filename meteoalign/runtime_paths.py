from __future__ import annotations

import sys
from pathlib import Path


def source_project_root() -> Path:
    """源码运行时的项目根目录。"""

    return Path(__file__).resolve().parents[1]


def is_frozen_app() -> bool:
    """判断当前进程是否来自打包后的可执行程序。"""

    return bool(getattr(sys, "frozen", False))


def frozen_app_sibling_dir() -> Path:
    """返回冻结程序外侧目录：Windows 为 exe 同级，macOS 为 app 同级。"""

    executable_path = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        for parent in executable_path.parents:
            if parent.suffix == ".app":
                return parent.parent
    return executable_path.parent


def frozen_resource_roots() -> tuple[Path, ...]:
    """返回打包工具可能放置内部资源的目录。"""

    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass).resolve())

    executable_path = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        for parent in executable_path.parents:
            if parent.suffix == ".app":
                roots.extend(
                    (
                        parent / "Contents" / "Resources",
                        parent / "Contents" / "Frameworks",
                        parent / "Contents" / "MacOS",
                    )
                )
                break
    roots.append(executable_path.parent)
    return _unique_paths(roots)


def runtime_catalog_dir() -> Path:
    """定位 catalog 目录，优先外置，失败时回退到包内资源。"""

    if is_frozen_app():
        external_catalog = frozen_app_sibling_dir() / "catalog"
        if external_catalog.exists():
            return external_catalog
        for root in frozen_resource_roots():
            bundled_catalog = root / "catalog"
            if bundled_catalog.exists():
                return bundled_catalog
        return external_catalog

    return source_project_root() / "catalog"


def runtime_icon_path() -> Path:
    """定位应用图标，兼容源码目录与打包后的资源目录。"""

    if not is_frozen_app():
        return source_project_root() / "icon256.png"

    for root in frozen_resource_roots():
        icon_path = root / "icon256.png"
        if icon_path.exists():
            return icon_path

    return frozen_app_sibling_dir() / "icon256.png"


def _unique_paths(paths: list[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return tuple(unique)
