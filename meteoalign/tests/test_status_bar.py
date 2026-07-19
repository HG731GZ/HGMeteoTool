"""状态栏消息颜色、路径和常驻行为测试。"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication

from meteoalign.application.app_meteor_selection import MeteorSelectionMixin
from meteoalign.application.status_bar import (
    STATUS_MESSAGE_ERROR,
    STATUS_MESSAGE_HINT,
    STATUS_MESSAGE_NORMAL,
    AppStatusBar,
)


def test_status_bar_applies_semantic_message_colors() -> None:
    """普通、失败和操作提示必须使用各自的统一底色。"""

    _app = QApplication.instance() or QApplication([])
    statusbar = AppStatusBar()

    statusbar.showMessage("图像导入完成。")
    assert statusbar.property("messageKind") == STATUS_MESSAGE_NORMAL

    statusbar.showMessage("手动匹配失败：PSF 拟合失败。")
    assert statusbar.property("messageKind") == STATUS_MESSAGE_ERROR
    assert statusbar.palette().color(QPalette.Window).name() == "#f8d7da"

    statusbar.showMessage("已强制记录 昴宿一 (3486.94,2091.72)")
    assert statusbar.property("messageKind") == STATUS_MESSAGE_ERROR
    assert statusbar.palette().color(QPalette.Window).name() == "#f8d7da"

    statusbar.showMessage("请按 Ctrl+左键确认星点。")
    assert statusbar.property("messageKind") == STATUS_MESSAGE_HINT
    assert statusbar.palette().color(QPalette.Window).name() == "#fff3cd"

    statusbar.showMessage("测试星的理论位置在真实图像外。")
    assert statusbar.property("messageKind") == STATUS_MESSAGE_HINT

    statusbar.showMessage("无法聚焦理论位置：当前配准模型无效。")
    assert statusbar.property("messageKind") == STATUS_MESSAGE_HINT

    statusbar.showMessage("已记录 昴宿一 (3487.23,2091.27)")
    assert statusbar.property("messageKind") == STATUS_MESSAGE_NORMAL


def test_status_bar_hides_quoted_import_paths() -> None:
    """导入错误中的绝对路径只能显示文件名。"""

    _app = QApplication.instance() or QApplication([])
    statusbar = AppStatusBar()
    statusbar.showMessage("导入图像失败：[Errno 2] 文件不存在：'/Users/demo/星空/IMG_001.CR3'")

    assert statusbar.currentMessage() == "导入图像失败：[Errno 2] 文件不存在：'IMG_001.CR3'"


def test_meteor_engine_ready_message_has_no_timeout() -> None:
    """检测引擎状态必须常驻到下一条状态消息覆盖。"""

    calls: list[tuple[object, ...]] = []
    host = SimpleNamespace(
        ui=SimpleNamespace(statusbar=SimpleNamespace(showMessage=lambda *args: calls.append(args))),
        _meteor_detection_engine_status="",
        _update_meteor_selection_controls=lambda: None,
    )

    MeteorSelectionMixin._handle_meteor_detection_worker_ready(
        host,
        {"available_providers": ["CUDAExecutionProvider"]},
    )

    assert calls == [("检测引擎已就绪；可用 Provider：CUDAExecutionProvider",)]
