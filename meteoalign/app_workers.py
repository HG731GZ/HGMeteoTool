from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .app_utils import (
    _image_with_binary_mask,
    _qimage_to_binary_mask,
    _resolve_star_pair_session_real_image_path,
)
from .image_preview import ImagePreview, load_image_preview


class ImagePreviewLoadWorker(QObject):
    """后台线程 Worker：加载图像预览。"""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str | Path, max_long_side_px: int | None) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.max_long_side_px = max_long_side_px

    def run(self) -> None:
        try:
            preview = load_image_preview(self.file_path, max_long_side_px=self.max_long_side_px)
            self.finished.emit(preview)
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有读取错误传回界面层。
            self.failed.emit(str(exc))


class SkyMaskLoadWorker(QObject):
    """后台线程 Worker：加载天空蒙版图像。"""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        file_path: str | Path,
        expected_size: tuple[int, int],
        source_image,
        source_path: Path,
    ) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.expected_size = expected_size
        self.source_image = source_image
        self.source_path = source_path

    def run(self) -> None:
        try:
            preview = load_image_preview(self.file_path, max_long_side_px=None)
            expected_width, expected_height = self.expected_size
            if preview.image.width() != expected_width or preview.image.height() != expected_height:
                raise ValueError(
                    "蒙版尺寸必须与真实图像一致：真实图像 {image_width} x {image_height} px，"
                    "蒙版 {mask_width} x {mask_height} px。".format(
                        image_width=expected_width,
                        image_height=expected_height,
                        mask_width=preview.image.width(),
                        mask_height=preview.image.height(),
                    )
                )

            mask = _qimage_to_binary_mask(preview.image)
            if not np.any(mask):
                raise ValueError("蒙版中没有任何非零像素，无法参与星点匹配。")

            masked_image = _image_with_binary_mask(self.source_image, mask)
            self.finished.emit((preview.path, self.source_path, mask, masked_image))
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有蒙版读取错误传回界面层。
            self.failed.emit(str(exc))


class ReferenceJsonImportWorker(QObject):
    """后台线程 Worker：导入预览 JSON。"""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self.file_path = Path(file_path)

    def run(self) -> None:
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            self.finished.emit((self.file_path, payload))
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有 JSON 读取错误传回界面层。
            self.failed.emit(str(exc))


class StarPairSessionImportWorker(QObject):
    """后台线程 Worker：导入星点配对会话 JSON。"""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self.file_path = Path(file_path)

    def run(self) -> None:
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            image_path = self._real_image_path(payload)
            preview = load_image_preview(image_path, max_long_side_px=None)
            self.finished.emit((self.file_path, payload, preview))
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有 JSON/图像读取错误传回界面层。
            self.failed.emit(str(exc))

    def _real_image_path(self, payload: object) -> Path:
        return _resolve_star_pair_session_real_image_path(payload, self.file_path)
