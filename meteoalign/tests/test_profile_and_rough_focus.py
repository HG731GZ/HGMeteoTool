from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from meteoalign.application.app_alignment import AlignmentMixin
from meteoalign.application.app_star_pair_annotations import StarPairAnnotationsMixin
from meteoalign.camera_calibration import CameraCalibrationProfile


class _ProfileLabelHarness(AlignmentMixin):
    """验证导入 Profile 标签包含其对应原图名称。"""

    def __init__(self) -> None:
        self._imported_camera_calibration_profile = CameraCalibrationProfile(
            image_width_px=4000,
            image_height_px=3000,
            base_projection_type="rectilinear",
            principal_point_x_px=2000.0,
            principal_point_y_px=1500.0,
            scale_x_px=2500.0,
            scale_y_px=2500.0,
            global_distortion_type="none",
        )
        self._imported_camera_calibration_profile_path = Path("/tmp/IMG_0063_model.json")
        self._imported_camera_calibration_image_name = "IMG_0063.TIF"


def test_profile_label_includes_source_image_name() -> None:
    """导出的 model.json 应优先显示 source_image.file_name。"""

    harness = _ProfileLabelHarness()

    assert AlignmentMixin._profile_source_image_name(
        {"source_image": {"file_name": "IMG_0063.TIF"}}
    ) == "IMG_0063.TIF"
    label_text, tooltip = harness._imported_profile_label_text()
    assert label_text.startswith("IMG_0063.TIF  rectilinear")
    assert "对应图像：IMG_0063.TIF" in tooltip


class _StatusBar:
    """记录聚焦提示的最小状态栏替身。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.messages.append(message)


class _RoughTransform:
    """提供固定理论位置的粗略取景变换替身。"""

    def transform_radec(self, _ra_deg: float, _dec_deg: float) -> tuple[float, float]:
        return 120.0, 80.0


class _RoughFocusHarness(StarPairAnnotationsMixin):
    """验证不足四对时可用粗略取景进行双击聚焦。"""

    def __init__(self) -> None:
        self.ui = SimpleNamespace(statusbar=_StatusBar())
        self._rough_alignment_transform = object()
        self._sky_alignment_transform = _RoughTransform()
        self.current_image_preview = SimpleNamespace(image=SimpleNamespace(width=lambda: 400, height=lambda: 300))
        self.focused_position: tuple[int, float, float] | None = None

    def _star_pair_position_count(self) -> int:
        return 0

    def _reference_star_for_row(self, _row: int):  # type: ignore[no-untyped-def]
        return SimpleNamespace(ra_deg=10.0, dec_deg=20.0)

    def _focus_star_pair_image_point(self, row: int, image_x: float, image_y: float) -> None:
        self.focused_position = (row, image_x, image_y)

    def _star_pair_label(self, _row: int) -> str:
        return "测试星"


def test_rough_framing_allows_double_click_focus_before_four_matches() -> None:
    """已有粗略取景时，零个手工匹配也应允许聚焦理论位置。"""

    harness = _RoughFocusHarness()

    harness._focus_star_pair_theoretical_position(2)

    assert harness.focused_position == (2, 120.0, 80.0)
    assert "已聚焦理论位置" in harness.ui.statusbar.messages[-1]
