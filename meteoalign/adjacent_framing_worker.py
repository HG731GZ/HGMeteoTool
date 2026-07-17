"""参考图像粗略取景的 Qt 后台任务封装。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from .adjacent_alignment import calculate_adjacent_rough_framing, localized_adjacent_alignment_error


class AdjacentFramingWorker(QObject):
    """后台计算参考图像 A→B 配准与当前图像粗略取景。"""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        model_json_path: str | Path,
        image_b_path: str | Path,
        mode: str,
        image_a_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.model_json_path = Path(model_json_path)
        self.image_b_path = Path(image_b_path)
        self.image_a_path = Path(image_a_path) if image_a_path is not None else None
        self.mode = str(mode)

    def run(self) -> None:
        try:
            result = calculate_adjacent_rough_framing(
                self.model_json_path,
                self.image_b_path,
                self.mode,
                image_a_path=self.image_a_path,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - 后台计算错误需要返回主线程弹窗展示。
            self.failed.emit(localized_adjacent_alignment_error(exc))
