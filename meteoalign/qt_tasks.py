from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyQt5.QtCore import QObject, QThread, Qt
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
) -> WorkerTaskHandle:
    """启动 QThread worker，并统一处理成功、失败和清理路径。"""

    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(run_slot or getattr(worker, "run"))
    if progress_signal is not None and on_progress is not None:
        progress_signal.connect(on_progress)
    finished_signal.connect(on_finished)
    failed_signal.connect(on_failed)
    finished_signal.connect(thread.quit)
    failed_signal.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    if on_cleanup is not None:
        thread.finished.connect(on_cleanup)
    thread.start()
    return WorkerTaskHandle(thread=thread, worker=worker, progress=progress_dialog)
