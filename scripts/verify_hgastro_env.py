"""Smoke-test the hgastro conda environment.

Run with:
    conda run -n hgastro python scripts/verify_hgastro_env.py
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import site
import sys


MODULES = {
    "PyQt5": "PyQt5",
    "pyqtgraph": "pyqtgraph",
    "numpy": "numpy",
    "scipy": "scipy",
    "astropy": "astropy",
    "skyfield": "skyfield",
    "opencv-python-headless": "cv2",
    "photutils": "photutils",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "pillow": "PIL",
    "sep": "sep",
    "gwcs": "gwcs",
    "asdf": "asdf",
    "PyYAML": "yaml",
    "pytest": "pytest",
    "pyinstaller": "PyInstaller",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def check_paths() -> None:
    if not sys.prefix.endswith("/envs/hgastro"):
        fail(f"expected hgastro prefix, got {sys.prefix}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        fail("PYTHONNOUSERSITE must be 1")
    if site.ENABLE_USER_SITE:
        fail("user site-packages is enabled")

    bad_tokens = ("/.local/lib/python", "/opt/ros/", "/usr/lib/python")
    bad_paths = [path for path in sys.path if any(token in path for token in bad_tokens)]
    if bad_paths:
        fail(f"external Python package paths detected: {bad_paths}")


def check_imports() -> None:
    for package_name, module_name in MODULES.items():
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", "")
        if module_path and not module_path.startswith(sys.prefix):
            fail(f"{module_name} imported from outside hgastro: {module_path}")
        version = metadata.version(package_name)
        print(f"{package_name}=={version}")


def check_qt() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
    from PyQt5.QtWidgets import QApplication, QLabel

    app = QApplication.instance() or QApplication([])
    label = QLabel("MeteoAlign")
    if label.text() != "MeteoAlign":
        fail("Qt QLabel smoke test failed")
    print(f"Qt=={QT_VERSION_STR}")
    print(f"PyQt=={PYQT_VERSION_STR}")
    app.quit()


def main() -> None:
    print(f"python={sys.executable}")
    print(f"prefix={sys.prefix}")
    check_paths()
    check_imports()
    check_qt()
    print("hgastro environment OK")


if __name__ == "__main__":
    main()
