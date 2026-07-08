from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable

from PyQt5.QtCore import QObject, QThread, QTimer, Qt
from PyQt5.QtWidgets import QProgressDialog


@dataclass(frozen=True)
class WorkerTaskHandle:
    """后台 worker 任务的 Qt 对象引用，防止线程运行时被回收。"""

    thread: QThread
    worker: QObject
    progress: QProgressDialog | None = None


def create_progress_dialog(
    parent,
    *,
    title: str,
    label_text: str,
    minimum: int = 0,
    maximum: int = 100,
) -> QProgressDialog:
    """创建项目内统一样式的不可取消进度对话框。"""

    dialog = QProgressDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setLabelText(label_text)
    dialog.setRange(int(minimum), int(maximum))
    dialog.setValue(int(minimum))
    dialog.setCancelButton(None)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.show()
    return dialog


def start_qt_worker_task(
    *,
    parent,
    worker: QObject,
    finished_signal,
    failed_signal,
    on_finished: Callable,
    on_failed: Callable[[str], None],
    progress_signal=None,
    on_progress: Callable[[int, str], None] | None = None,
    on_cleanup: Callable[[], None] | None = None,
    progress_dialog: QProgressDialog | None = None,
    run_slot: Callable[[], None] | None = None,
    start_delay_ms: int = 0,
    minimum_visible_ms: int = 0,
) -> WorkerTaskHandle:
    """启动 QThread worker，并统一处理成功、失败和清理路径。"""

    thread = QThread(parent)
    started_at = monotonic()

    def dispatch_after_minimum_visible(callback: Callable, *args) -> None:
        delay_ms = 0
        if minimum_visible_ms > 0:
            elapsed_ms = int((monotonic() - started_at) * 1000)
            delay_ms = max(0, int(minimum_visible_ms) - elapsed_ms)

        def run_callback() -> None:
            callback(*args)
            thread.quit()

        if delay_ms > 0:
            QTimer.singleShot(delay_ms, run_callback)
        else:
            run_callback()

    worker.moveToThread(thread)
    thread.started.connect(run_slot or getattr(worker, "run"))
    if progress_signal is not None and on_progress is not None:
        progress_signal.connect(on_progress)
    finished_signal.connect(lambda *args: dispatch_after_minimum_visible(on_finished, *args))
    failed_signal.connect(lambda error_message: dispatch_after_minimum_visible(on_failed, error_message))
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    if on_cleanup is not None:
        thread.finished.connect(on_cleanup)
    if start_delay_ms > 0:
        QTimer.singleShot(int(start_delay_ms), thread.start)
    else:
        thread.start()
    return WorkerTaskHandle(thread=thread, worker=worker, progress=progress_dialog)
