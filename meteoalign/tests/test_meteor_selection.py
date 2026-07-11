"""流星框选数据和视图交互测试。"""

from __future__ import annotations

import json
import os

# 让无显示服务器的 CI 也能创建 Qt 视图。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QImage, QMouseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow

from meteoalign.application.app_meteor_selection import MeteorSelectionMixin
from meteoalign.application.meteor_selection_view import MeteorSelectionView
from meteoalign.meteor_selection import MeteorBox, load_meteor_selection, meteor_json_path, save_meteor_selection
from meteoalign.ui.ui_main_window import Ui_MainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


class _MeteorSelectionHost(QMainWindow, MeteorSelectionMixin):
    """为批量保存行为提供最小化的页面宿主。"""

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._init_meteor_selection_page()


def test_meteor_selection_json_uses_image_sibling_name_and_original_pixels(tmp_path) -> None:
    """保存结果应使用指定文件名，且保留原始像素坐标。"""

    image_path = tmp_path / "IMG_1234.TIF"
    output_path = save_meteor_selection(
        image_path,
        6000,
        4000,
        [MeteorBox(120.7, 30.25, 5000.75, 3900.5)],
    )

    assert output_path == tmp_path / "IMG_1234_Meteor.json"
    assert output_path == meteor_json_path(image_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_image"] == "IMG_1234.TIF"
    assert payload["image_size_px"] == {"width": 6000, "height": 4000}
    assert payload["meteor_boxes"] == [
        {
            "top_left": {"x": 121, "y": 30},
            "bottom_right": {"x": 5001, "y": 3900},
        }
    ]
    assert load_meteor_selection(image_path) == [MeteorBox(121.0, 30.0, 5001.0, 3900.0)]


def test_meteor_selection_view_creates_boxes_in_original_image_coordinates() -> None:
    """缩略图显示时，Ctrl 拖拽得到的仍应是原图坐标。"""

    app = _application()
    view = MeteorSelectionView()
    view.resize(800, 500)
    preview = QImage(200, 100, QImage.Format_RGB32)
    preview.fill(Qt.black)
    view.set_image(preview, 1000, 500)
    view.show()
    app.processEvents()
    view.fit_image()

    start = view.mapFromScene(QPointF(125.0, 80.0))
    end = view.mapFromScene(QPointF(700.0, 420.0))
    view.mousePressEvent(
        QMouseEvent(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton, Qt.ControlModifier)
    )
    view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton, Qt.ControlModifier))
    view.mouseReleaseEvent(
        QMouseEvent(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton, Qt.ControlModifier)
    )

    boxes = view.boxes()
    assert len(boxes) == 1
    box = boxes[0]
    assert abs(box.left - 125.0) < 2.0
    assert abs(box.top - 80.0) < 2.0
    assert abs(box.right - 700.0) < 2.0
    assert abs(box.bottom - 420.0) < 2.0

    view.clear_boxes()
    assert view.boxes() == []
    view.close()


def test_save_all_meteor_boxes_only_writes_images_with_boxes(tmp_path) -> None:
    """批量保存应跳过流星数为零的图像。"""

    app = _application()
    host = _MeteorSelectionHost()
    image_with_box = tmp_path / "IMG_0001.TIF"
    image_without_box = tmp_path / "IMG_0002.TIF"
    host._meteor_selection_paths = [image_with_box, image_without_box]
    host._meteor_selection_boxes_by_path = {
        image_with_box: [MeteorBox(12.4, 24.6, 100.2, 200.8)],
        image_without_box: [],
    }
    host._meteor_selection_image_sizes = {
        image_with_box: (6000, 4000),
        image_without_box: (6000, 4000),
    }
    host.save_all_meteor_boxes()

    assert meteor_json_path(image_with_box).exists()
    assert not meteor_json_path(image_without_box).exists()
    assert load_meteor_selection(image_with_box) == [MeteorBox(12.0, 25.0, 100.0, 201.0)]
    host.close()
    app.processEvents()
