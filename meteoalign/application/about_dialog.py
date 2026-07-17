from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QLabel, QWidget

from ..runtime_paths import runtime_qrcode_path
from ..ui.ui_about_dialog import Ui_AboutDialog


class AboutDialog(QDialog):
    """显示软件简介、二维码和项目链接。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_AboutDialog()
        self.ui.setupUi(self)
        self.setModal(False)
        self._load_qrcode(self.ui.labelOfficialAccountQrCode, "公众号.jpg")
        self._load_qrcode(self.ui.labelAlipayQrCode, "支付宝.png")
        self.ui.pushButtonCloseAbout.clicked.connect(self.close)

    @staticmethod
    def _load_qrcode(label: QLabel, filename: str) -> None:
        """加载并缩放二维码；资源缺失时在原位置给出明确提示。"""

        image_path = runtime_qrcode_path(filename)
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            label.setText(f"二维码加载失败\n{filename}")
            label.setToolTip(str(image_path))
            return

        label.setPixmap(
            pixmap.scaled(
                label.maximumSize(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        label.setToolTip(filename)


__all__ = ["AboutDialog"]
