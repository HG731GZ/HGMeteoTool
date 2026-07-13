"""使用 QProcess 管理 MetDet JSON Lines worker。"""

from __future__ import annotations

import json
from uuid import uuid4

from PyQt5.QtCore import QObject, QProcess, pyqtSignal

from ..meteor_detection import MeteorDetectionOptions, resolve_meteor_worker_invocation


class MetDetWorkerClient(QObject):
    """解析 worker 输出并以 Qt 信号转发结构化协议消息。"""

    ready = pyqtSignal(dict)
    messageReceived = pyqtSignal(dict)
    workerError = pyqtSignal(str)
    workerStopped = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._read_standard_output)
        self._process.readyReadStandardError.connect(self._read_standard_error)
        self._process.errorOccurred.connect(self._handle_process_error)
        self._process.finished.connect(self._handle_process_finished)
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._ready = False
        self._stopping = False
        self._last_error_text = ""

    @property
    def is_ready(self) -> bool:
        return self._ready and self._process.state() == QProcess.Running

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    def start(self, engine_path: str = "") -> None:
        """启动配置指定的 worker；已运行时先将旧进程关闭。"""

        self.stop()
        invocation = resolve_meteor_worker_invocation(engine_path)
        self._stdout_buffer.clear()
        self._stderr_buffer.clear()
        self._last_error_text = ""
        self._ready = False
        self._stopping = False
        self._process.setWorkingDirectory(invocation.working_directory)
        self._process.setProgram(invocation.program)
        self._process.setArguments(list(invocation.arguments))
        self._process.start()

    def detect(self, image_paths: list[str], options: MeteorDetectionOptions) -> str:
        """向已就绪 worker 发送串行批量检测请求。"""

        if not self.is_ready:
            raise RuntimeError("流星检测引擎尚未就绪。")
        request_id = f"meteor-{uuid4().hex}"
        self._write_message(
            {
                "command": "detect",
                "request_id": request_id,
                "image_paths": image_paths,
                "options": options.worker_options(),
            }
        )
        return request_id

    def stop(self) -> None:
        """优先请求正常退出，短超时后终止进程。"""

        if self._process.state() == QProcess.NotRunning:
            self._ready = False
            return
        self._stopping = True
        if self.is_ready:
            try:
                self._write_message({"command": "shutdown", "request_id": f"close-{uuid4().hex}"})
                if self._process.waitForFinished(500):
                    self._ready = False
                    return
            except RuntimeError:
                pass
        self._process.terminate()
        if not self._process.waitForFinished(700):
            self._process.kill()
            self._process.waitForFinished(700)
        self._ready = False

    def cancel_active_job(self) -> None:
        """按协议终止正在推理的 worker；调用方可随后重新启动。"""

        if self._process.state() == QProcess.NotRunning:
            return
        self._stopping = True
        self._process.terminate()
        if not self._process.waitForFinished(400):
            self._process.kill()
        self._ready = False

    def _write_message(self, payload: dict[str, object]) -> None:
        if self._process.state() != QProcess.Running:
            raise RuntimeError("流星检测引擎进程未运行。")
        # Windows 上的 Qt 路径可能包含 UTF-16 代理项，ASCII 转义可确保协议始终是有效 UTF-8。
        line = json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
        if self._process.write(line.encode("utf-8")) < 0:
            raise RuntimeError("无法向流星检测引擎发送请求。")

    def _read_standard_output(self) -> None:
        self._stdout_buffer.extend(bytes(self._process.readAllStandardOutput()))
        while b"\n" in self._stdout_buffer:
            raw_line, _separator, remainder = self._stdout_buffer.partition(b"\n")
            self._stdout_buffer = bytearray(remainder)
            self._handle_protocol_line(raw_line)

    def _handle_protocol_line(self, raw_line: bytes) -> None:
        if not raw_line.strip():
            return
        try:
            payload = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.workerError.emit(f"流星检测引擎返回了无效协议消息：{exc}")
            return
        if not isinstance(payload, dict):
            self.workerError.emit("流星检测引擎返回的协议消息不是 JSON 对象。")
            return
        if payload.get("type") == "ready":
            if payload.get("protocol") != "metdet.jsonl" or payload.get("protocol_version") != 1:
                self.workerError.emit("流星检测引擎协议版本不兼容，需要 metdet.jsonl v1。")
                return
            self._ready = True
            self.ready.emit(payload)
        self.messageReceived.emit(payload)

    def _read_standard_error(self) -> None:
        self._stderr_buffer.extend(bytes(self._process.readAllStandardError()))
        if len(self._stderr_buffer) > 16384:
            del self._stderr_buffer[:-16384]
        text = self._stderr_buffer.decode("utf-8", errors="replace").strip()
        if text:
            self._last_error_text = text

    def _handle_process_error(self, _error: QProcess.ProcessError) -> None:
        if self._stopping:
            return
        detail = self._process.errorString().strip()
        self.workerError.emit(f"流星检测引擎进程错误：{detail}")

    def _handle_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        was_stopping = self._stopping
        self._ready = False
        self._stopping = False
        self.workerStopped.emit()
        if was_stopping or exit_code == 0:
            return
        detail = self._last_error_text or f"退出码 {exit_code}"
        self.workerError.emit(f"流星检测引擎异常退出：{detail}")


__all__ = ["MetDetWorkerClient"]
