from __future__ import annotations

import urllib.error
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox

from scripts.download_catalogs import CATALOG_FILES, default_catalog_dir, download_file, file_is_complete

from .ui.ui_catalog_download_dialog import Ui_CatalogDownloadDialog


def incomplete_catalog_labels(catalog_dir: Path | None = None) -> list[str]:
    base_dir = catalog_dir or default_catalog_dir()
    if not base_dir.exists() or not base_dir.is_dir():
        return [item.label for item in CATALOG_FILES]

    incomplete: list[str] = []
    for item in CATALOG_FILES:
        target = base_dir / item.relative_path
        if not file_is_complete(target, item.expected_size):
            incomplete.append(item.label)
    return incomplete


def catalog_is_complete(catalog_dir: Path | None = None) -> bool:
    return not incomplete_catalog_labels(catalog_dir)


class CatalogDownloadWorker(QObject):
    progress_changed = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, catalog_dir: Path | None = None) -> None:
        super().__init__()
        self.catalog_dir = catalog_dir or default_catalog_dir()

    def run(self) -> None:
        total = len(CATALOG_FILES)
        try:
            self.catalog_dir.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(CATALOG_FILES, start=1):
                self.progress_changed.emit(index - 1, total, f"正在下载：{item.label}")
                download_file(item, self.catalog_dir, force=False)
                self.progress_changed.emit(index, total, f"已完成：{item.label}")
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            self.finished.emit(False, f"星表下载失败：{exc}")
            return

        self.finished.emit(True, "星表下载完成。请关闭窗口后重新运行 main.py。")


class CatalogDownloadDialog(QDialog):
    def __init__(self, parent=None, catalog_dir: Path | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.ui = Ui_CatalogDownloadDialog()
        self.ui.setupUi(self)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.ui.pushButtonClose.clicked.connect(self.accept)

        self.worker = CatalogDownloadWorker(catalog_dir)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)

    def start(self) -> int:
        self.thread.start()
        return int(self.exec_())

    def reject(self) -> None:
        if self.ui.pushButtonClose.isEnabled():
            super().reject()

    def _on_progress_changed(self, value: int, total: int, message: str) -> None:
        self.ui.progressBar.setMaximum(total)
        self.ui.progressBar.setValue(value)
        self.ui.labelStatus.setText(message)

    def _on_finished(self, success: bool, message: str) -> None:
        self.ui.progressBar.setValue(self.ui.progressBar.maximum())
        self.ui.labelStatus.setText(message)
        self.ui.pushButtonClose.setEnabled(True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.show()
        if success:
            self.setWindowTitle("下载完成")
        else:
            self.setWindowTitle("下载失败")


def prompt_for_catalog_download(parent=None) -> bool:  # type: ignore[no-untyped-def]
    missing = incomplete_catalog_labels()
    message = "检测到 catalog 目录不存在或星表文件不完整。"
    if missing:
        message += "\n\n缺失或不完整：\n" + "\n".join(f"- {label}" for label in missing)
    message += "\n\n是否现在下载星表？"

    box = QMessageBox(parent)
    box.setWindowTitle("星表数据缺失")
    box.setIcon(QMessageBox.Warning)
    box.setText(message)
    download_button = box.addButton("下载星表", QMessageBox.AcceptRole)
    box.addButton("直接退出", QMessageBox.RejectRole)
    box.exec_()
    return box.clickedButton() is download_button


def ensure_catalogs_ready_or_handle(parent=None) -> bool:  # type: ignore[no-untyped-def]
    if catalog_is_complete():
        return True

    if not prompt_for_catalog_download(parent):
        return False

    dialog = CatalogDownloadDialog(parent)
    dialog.start()
    return False
