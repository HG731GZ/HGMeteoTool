from __future__ import annotations

import urllib.error
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox

from .catalog_sources import default_catalog_dir, download_file, incomplete_catalog_files

from .ui.ui_catalog_download_dialog import Ui_CatalogDownloadDialog


def incomplete_catalog_labels(catalog_dir: Path | None = None) -> list[str]:
    return [item.label for item in incomplete_catalog_files(catalog_dir)]


def catalog_is_complete(catalog_dir: Path | None = None) -> bool:
    return not incomplete_catalog_labels(catalog_dir)


class CatalogDownloadWorker(QObject):
    progress_changed = pyqtSignal(object)
    finished = pyqtSignal(bool, str)

    def __init__(self, catalog_dir: Path | None = None) -> None:
        super().__init__()
        self.catalog_dir = catalog_dir or default_catalog_dir()

    def run(self) -> None:
        try:
            self.catalog_dir.mkdir(parents=True, exist_ok=True)
            items = incomplete_catalog_files(self.catalog_dir)
            if not items:
                self.finished.emit(True, "星表已经完整。")
                return

            total_estimated_bytes = sum(self._estimated_download_size(item) for item in items)
            completed_bytes = 0
            total_count = len(items)

            for index, item in enumerate(items, start=1):
                current_total_bytes = self._estimated_download_size(item)
                current_total_known = item.expected_size is not None
                current_downloaded_bytes = 0
                last_speed_bps = 0.0

                def handle_file_progress(downloaded_bytes: int, known_total_bytes: int | None, speed_bps: float) -> None:
                    nonlocal current_total_bytes, current_total_known, current_downloaded_bytes
                    nonlocal total_estimated_bytes, last_speed_bps
                    current_downloaded_bytes = downloaded_bytes
                    last_speed_bps = speed_bps
                    if known_total_bytes is not None:
                        current_total_known = True
                    adjusted_total_bytes = known_total_bytes if known_total_bytes is not None else downloaded_bytes
                    if adjusted_total_bytes > current_total_bytes:
                        total_estimated_bytes += adjusted_total_bytes - current_total_bytes
                        current_total_bytes = adjusted_total_bytes
                    self.progress_changed.emit(
                        self._progress_payload(
                            item_label=item.label,
                            index=index,
                            total_count=total_count,
                            current_downloaded_bytes=current_downloaded_bytes,
                            current_total_bytes=current_total_bytes,
                            current_total_known=current_total_known,
                            total_downloaded_bytes=completed_bytes + current_downloaded_bytes,
                            total_estimated_bytes=total_estimated_bytes,
                            speed_bps=last_speed_bps,
                            status=f"正在下载：{item.label}",
                        )
                    )

                self.progress_changed.emit(
                    self._progress_payload(
                        item_label=item.label,
                        index=index,
                        total_count=total_count,
                        current_downloaded_bytes=0,
                        current_total_bytes=current_total_bytes,
                        current_total_known=current_total_known,
                        total_downloaded_bytes=completed_bytes,
                        total_estimated_bytes=total_estimated_bytes,
                        speed_bps=0.0,
                        status=f"正在下载：{item.label}",
                    )
                )
                download_file(item, self.catalog_dir, force=False, progress_callback=handle_file_progress)

                target = self.catalog_dir / item.relative_path
                actual_bytes = target.stat().st_size if target.exists() else current_downloaded_bytes
                if actual_bytes > current_total_bytes:
                    total_estimated_bytes += actual_bytes - current_total_bytes
                    current_total_bytes = actual_bytes
                current_total_known = True
                completed_bytes += max(actual_bytes, current_total_bytes)
                self.progress_changed.emit(
                    self._progress_payload(
                        item_label=item.label,
                        index=index,
                        total_count=total_count,
                        current_downloaded_bytes=current_total_bytes,
                        current_total_bytes=current_total_bytes,
                        current_total_known=current_total_known,
                        total_downloaded_bytes=min(completed_bytes, total_estimated_bytes),
                        total_estimated_bytes=total_estimated_bytes,
                        speed_bps=last_speed_bps,
                        status=f"已完成：{item.label}",
                    )
                )
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            self.finished.emit(False, f"星表下载失败：{exc}")
            return

        self.finished.emit(True, "星表下载完成，正在启动主窗口。")

    def _estimated_download_size(self, item) -> int:  # type: ignore[no-untyped-def]
        # 没有精确大小的网页星表用最低完整大小作为初始估计，下载时会按响应头修正。
        return int(item.expected_size or item.minimum_size or 1)

    def _progress_payload(
        self,
        item_label: str,
        index: int,
        total_count: int,
        current_downloaded_bytes: int,
        current_total_bytes: int,
        current_total_known: bool,
        total_downloaded_bytes: int,
        total_estimated_bytes: int,
        speed_bps: float,
        status: str,
    ) -> dict[str, object]:
        return {
            "item_label": item_label,
            "index": index,
            "total_count": total_count,
            "current_downloaded_bytes": max(0, int(current_downloaded_bytes)),
            "current_total_bytes": max(1, int(current_total_bytes)),
            "current_total_known": bool(current_total_known),
            "total_downloaded_bytes": max(0, int(total_downloaded_bytes)),
            "total_estimated_bytes": max(1, int(total_estimated_bytes)),
            "speed_bps": max(0.0, float(speed_bps)),
            "status": status,
        }


class CatalogDownloadDialog(QDialog):
    def __init__(self, parent=None, catalog_dir: Path | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.ui = Ui_CatalogDownloadDialog()
        self.ui.setupUi(self)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.ui.pushButtonClose.clicked.connect(self.reject)
        self.download_succeeded = False

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

    def _on_progress_changed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        item_label = str(payload.get("item_label", "-"))
        index = int(payload.get("index", 0))
        total_count = int(payload.get("total_count", 0))
        current_downloaded = int(payload.get("current_downloaded_bytes", 0))
        current_total = max(1, int(payload.get("current_total_bytes", 1)))
        current_total_known = bool(payload.get("current_total_known", True))
        total_downloaded = int(payload.get("total_downloaded_bytes", 0))
        total_estimated = max(1, int(payload.get("total_estimated_bytes", 1)))
        speed_bps = float(payload.get("speed_bps", 0.0))

        self.ui.labelStatus.setText(str(payload.get("status", "")))
        if current_total_known:
            self.ui.labelCurrentTitle.setText(
                f"当前星表（{index}/{total_count}）：{item_label}  "
                f"{_format_bytes(current_downloaded)} / {_format_bytes(current_total)}"
            )
            self.ui.progressBarCurrent.setRange(0, current_total)
            self.ui.progressBarCurrent.setValue(min(current_downloaded, current_total))
        else:
            self.ui.labelCurrentTitle.setText(
                f"当前星表（{index}/{total_count}）：{item_label}  已下载 {_format_bytes(current_downloaded)}"
            )
            self.ui.progressBarCurrent.setRange(0, 0)
        self.ui.labelTotalTitle.setText(
            f"总进度：{_format_bytes(total_downloaded)} / {_format_bytes(total_estimated)}"
        )
        self.ui.progressBarTotal.setRange(0, total_estimated)
        self.ui.progressBarTotal.setValue(min(total_downloaded, total_estimated))
        self.ui.labelSpeed.setText(f"下载速度：{_format_speed(speed_bps)}")

    def _on_finished(self, success: bool, message: str) -> None:
        self.download_succeeded = success
        if success:
            self.ui.progressBarCurrent.setValue(self.ui.progressBarCurrent.maximum())
            self.ui.progressBarTotal.setValue(self.ui.progressBarTotal.maximum())
        self.ui.labelStatus.setText(message)
        if success:
            self.setWindowTitle("下载完成")
            self.ui.pushButtonClose.setEnabled(False)
            QTimer.singleShot(600, self.accept)
        else:
            self.setWindowTitle("下载失败")
            self.ui.pushButtonClose.setEnabled(True)
            self.setWindowFlag(Qt.WindowCloseButtonHint, True)
            self.show()


def ensure_catalogs_ready_or_handle(parent=None) -> bool:  # type: ignore[no-untyped-def]
    if catalog_is_complete():
        return True

    if not prompt_for_catalog_download(parent):
        return False

    dialog = CatalogDownloadDialog(parent)
    result = dialog.start()
    return bool(result == QDialog.Accepted and dialog.download_succeeded and catalog_is_complete())


def prompt_for_catalog_download(parent=None, catalog_dir: Path | None = None) -> bool:  # type: ignore[no-untyped-def]
    missing = incomplete_catalog_labels(catalog_dir)
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


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _format_speed(bytes_per_second: float) -> str:
    if bytes_per_second <= 0.0:
        return "-"
    return f"{_format_bytes(int(bytes_per_second))}/s"
