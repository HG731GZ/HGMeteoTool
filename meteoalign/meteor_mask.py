"""流星检测蒙版的后台加载 Worker。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .binary_mask import image_with_binary_mask, qimage_to_binary_mask, scale_binary_mask_nearest
from .image_preview import load_image_preview


class MeteorMaskLoadWorker(QObject):
    """读取流星检测蒙版并生成当前图片的缓存预览。"""

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
            if preview.original_width != expected_width or preview.original_height != expected_height:
                raise ValueError(
                    "蒙版尺寸必须与流星原图一致：原图 {image_width} x {image_height} px，"
                    "蒙版 {mask_width} x {mask_height} px。".format(
                        image_width=expected_width,
                        image_height=expected_height,
                        mask_width=preview.original_width,
                        mask_height=preview.original_height,
                    )
                )

            mask = qimage_to_binary_mask(preview.image)
            if not np.any(mask):
                raise ValueError("蒙版中没有任何非零像素，无法参与流星检测。")
            preview_mask = scale_binary_mask_nearest(
                mask,
                self.source_image.width(),
                self.source_image.height(),
            )
            masked_image = image_with_binary_mask(self.source_image, preview_mask)
            self.finished.emit((preview.path, self.source_path, mask, masked_image))
        except Exception as exc:  # noqa: BLE001 - 后台线程需要把所有读取错误传回界面层。
            self.failed.emit(str(exc))
