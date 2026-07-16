"""可供各业务页面复用的图像预览窗口。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QRectF, QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QGraphicsScene

from ..image_preview import ImagePreview
from ..ui.ui_image_preview_dialog import Ui_ImagePreviewDialog


class ImagePreviewDialog(QDialog):
    """在应用级独立窗口中显示并刷新任意图像预览。"""

    def __init__(self) -> None:
        super().__init__(None)
        # 使用正常顶层窗口类型，确保预览不从属于呼出它的对话框并可显示在任务栏。
        window_flags = (self.windowFlags() & ~Qt.WindowType_Mask) | Qt.Window
        self.setWindowFlags(window_flags)
        self.ui = Ui_ImagePreviewDialog()
        self.ui.setupUi(self)
        self.setModal(False)
        self._image_path: Path | None = None
        self._scene = QGraphicsScene(self)
        self.ui.graphicsViewImagePreview.setScene(self._scene)

    @property
    def image_path(self) -> Path | None:
        """返回窗口当前显示的图像路径。"""

        return self._image_path

    def set_preview(self, preview: ImagePreview) -> None:
        """替换当前图像，并让画面完整适配窗口。"""

        image_path = preview.path.expanduser().resolve()
        self._image_path = image_path
        self.ui.labelImageName.setText(image_path.name)
        self.ui.labelImageName.setToolTip(str(image_path))
        self.setWindowTitle(f"{image_path.name} 预览")
        self._scene.clear()
        pixmap = QPixmap.fromImage(preview.image)
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._fit_image()

    def show_preview(self, preview: ImagePreview) -> None:
        """刷新图像并将当前窗口提到前台。"""

        self.set_preview(preview)
        self.show()
        self.raise_()
        self.activateWindow()

    def clear_preview(self) -> None:
        """清空图像并恢复窗口的初始状态。"""

        self._image_path = None
        self._scene.clear()
        self._scene.setSceneRect(QRectF())
        self.ui.graphicsViewImagePreview.resetTransform()
        self.ui.labelImageName.setText("未加载图像")
        self.ui.labelImageName.setToolTip("")
        self.setWindowTitle("图像预览")

    def _fit_image(self) -> None:
        """保持完整图像可见，并保留原始宽高比。"""

        if not self._scene.sceneRect().isEmpty():
            self.ui.graphicsViewImagePreview.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        QDialog.showEvent(self, event)
        QTimer.singleShot(0, self._fit_image)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        QDialog.resizeEvent(self, event)
        QTimer.singleShot(0, self._fit_image)


__all__ = ["ImagePreviewDialog"]
