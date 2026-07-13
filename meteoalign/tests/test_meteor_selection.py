"""流星框选数据和视图交互测试。"""

from __future__ import annotations

import json
import os

# 让无显示服务器的 CI 也能创建 Qt 视图。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QContextMenuEvent, QImage, QMouseEvent, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu

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
    assert view._box_items[0].pen().color().getRgb()[:3] == (52, 211, 153)

    view.clear_boxes()
    assert view.boxes() == []
    view.close()


def test_meteor_selection_view_supports_touchpad_pinch_zoom() -> None:
    """右侧流星预览应以窗口中心为锚点响应原生触控板缩放。"""

    class NativeZoomEvent:
        def __init__(self, gesture_type, value: float = 0.5) -> None:  # type: ignore[no-untyped-def]
            self._gesture_type = gesture_type
            self._value = value

        def type(self):  # type: ignore[no-untyped-def]
            return QEvent.NativeGesture

        def gestureType(self):  # type: ignore[no-untyped-def]
            return self._gesture_type

        def value(self):  # type: ignore[no-untyped-def]
            return self._value

    app = _application()
    view = MeteorSelectionView()
    view.resize(800, 500)
    image = QImage(200, 100, QImage.Format_RGB32)
    image.fill(Qt.black)
    view.set_image(image, 4000, 2000)
    view.show()
    app.processEvents()
    view.fit_image()
    view.scale(1.6, 1.6)
    view.centerOn(QPointF(1400.0, 650.0))
    center_before = view.mapToScene(view.viewport().rect().center())

    scale_before = view.transform().m11()
    assert not view._handle_native_gesture(NativeZoomEvent(Qt.BeginNativeGesture))
    assert view._handle_native_gesture(NativeZoomEvent(Qt.ZoomNativeGesture))
    assert view.transform().m11() > scale_before
    center_after = view.mapToScene(view.viewport().rect().center())
    # QGraphicsView 的滚动条以整数视口像素定位，换回原图坐标时会有少量量化误差。
    assert abs(center_after.x() - center_before.x()) < 6.0
    assert abs(center_after.y() - center_before.y()) < 6.0
    assert not view._handle_native_gesture(NativeZoomEvent(Qt.EndNativeGesture))
    assert view._native_zoom_center is None

    view.set_touchpad_pinch_zoom_enabled(False)
    scale_before = view.transform().m11()
    assert not view._handle_native_gesture(NativeZoomEvent(Qt.ZoomNativeGesture))
    assert view.transform().m11() == scale_before
    view.close()


def test_meteor_selection_view_right_click_deletes_only_hit_box(monkeypatch) -> None:
    """框内右键菜单应只删除命中的单个框，并发出新的内存框选。"""

    app = _application()
    view = MeteorSelectionView()
    view.resize(800, 500)
    image = QImage(200, 100, QImage.Format_RGB32)
    image.fill(Qt.black)
    view.set_image(image, 1000, 500)
    view.set_boxes(
        [
            MeteorBox(100, 100, 300, 300),
            MeteorBox(600, 100, 850, 350),
        ]
    )
    view.show()
    app.processEvents()
    view.fit_image()
    emitted_boxes: list[list[MeteorBox]] = []
    view.boxesChanged.connect(emitted_boxes.append)
    monkeypatch.setattr(QMenu, "exec_", lambda menu, _position: menu.actions()[0])

    view_position = view.mapFromScene(QPointF(200, 200))
    event = QContextMenuEvent(
        QContextMenuEvent.Mouse,
        view_position,
        view.viewport().mapToGlobal(view_position),
    )
    view.contextMenuEvent(event)

    assert view.boxes() == [MeteorBox(600, 100, 850, 350)]
    assert emitted_boxes[-1] == [MeteorBox(600, 100, 850, 350)]
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
    assert host.ui.tableWidgetMeteorSelectionImages.item(0, 1).background().color().getRgb()[:3] == (220, 252, 231)
    table_palette = host.ui.tableWidgetMeteorSelectionImages.palette()
    assert table_palette.color(QPalette.Inactive, QPalette.Highlight) == table_palette.color(
        QPalette.Active,
        QPalette.Highlight,
    )
    host.close()
    app.processEvents()


def test_deleted_last_boxes_remain_in_memory_until_save_then_remove_json(tmp_path) -> None:
    """连续删除多张图的最后一个框时，应等统一保存后才删除各自 JSON。"""

    app = _application()
    host = _MeteorSelectionHost()
    first_path = tmp_path / "IMG_0001.TIF"
    second_path = tmp_path / "IMG_0002.TIF"
    original_box = [MeteorBox(10, 20, 100, 200)]
    save_meteor_selection(first_path, 6000, 4000, original_box)
    save_meteor_selection(second_path, 6000, 4000, original_box)
    host._meteor_selection_paths = [first_path, second_path]
    host._meteor_selection_boxes_by_path = {
        first_path: list(original_box),
        second_path: list(original_box),
    }

    host._meteor_selection_current_index = 0
    host._handle_meteor_boxes_changed([])
    host._meteor_selection_current_index = 1
    host._handle_meteor_boxes_changed([])

    assert meteor_json_path(first_path).exists()
    assert meteor_json_path(second_path).exists()
    assert host._meteor_selection_dirty_paths == {first_path, second_path}
    assert host.ui.pushButtonSaveAllMeteorBoxes.isEnabled()

    host.save_all_meteor_boxes()

    assert not meteor_json_path(first_path).exists()
    assert not meteor_json_path(second_path).exists()
    assert host._meteor_selection_dirty_paths == set()
    assert not host.ui.pushButtonSaveAllMeteorBoxes.isEnabled()
    host.close()
    app.processEvents()
