from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meteoalign.application.app_image import ImageMixin


class _Label:
    """记录状态栏右侧文字的最小标签替身。"""

    def __init__(self) -> None:
        self.text = ""
        self.tooltip = ""

    def setText(self, text: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.text = text

    def setToolTip(self, tooltip: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.tooltip = tooltip


class _StatusBar:
    """记录普通状态提示的最小状态栏替身。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.messages.append(message)


class _ProjectionComboBox:
    """提供当前投影模型名称的最小下拉框替身。"""

    def currentText(self) -> str:  # noqa: N802 - 保持 Qt 接口名称。
        return "普通透视镜头(TAN)"


class _ImageMaskHarness(ImageMixin):
    """隔离验证同名蒙版发现和状态栏上下文。"""

    def __init__(self, image_path: Path) -> None:
        self.ui = SimpleNamespace(
            labelStatusImageContext=_Label(),
            statusbar=_StatusBar(),
            comboBoxSkyAlignmentModel=_ProjectionComboBox(),
        )
        self.current_image_preview = SimpleNamespace(path=image_path)
        self.current_sky_mask = None
        self.current_sky_mask_path = None
        self._mask_import_thread = None
        self.auto_imported_mask_paths: list[Path] = []

    def start_sky_mask_import(self, file_path: str | Path) -> None:
        self.auto_imported_mask_paths.append(Path(file_path))


def test_companion_sky_mask_uses_fixed_image_name_convention(tmp_path: Path) -> None:
    """自动发现应匹配“原图名_Mask.后缀”，并将其交给蒙版导入流程。"""

    image_path = tmp_path / "IMG_0071.TIF"
    mask_path = tmp_path / "IMG_0071_Mask.png"
    image_path.touch()
    mask_path.touch()
    harness = _ImageMaskHarness(image_path)

    assert harness._companion_sky_mask_path(image_path) == mask_path
    assert harness._maybe_auto_import_sky_mask_for_image(image_path)
    assert harness.auto_imported_mask_paths == [mask_path]
    assert "发现同名蒙版" in harness.ui.statusbar.messages[-1]


def test_status_image_context_shows_file_name_and_mask_state(tmp_path: Path) -> None:
    """状态栏右侧信息刷新时应给出当前图像文件名和蒙版是否生效。"""

    image_path = tmp_path / "IMG_0071.TIF"
    mask_path = tmp_path / "IMG_0071_Mask.tif"
    harness = _ImageMaskHarness(image_path)

    harness._update_status_image_context()
    assert harness.ui.labelStatusImageContext.text == "图像：IMG_0071.TIF  |  蒙版：未使用  |  投影：普通透视镜头(TAN)"

    harness.current_sky_mask = np.ones((2, 2), dtype=bool)
    harness.current_sky_mask_path = mask_path
    harness._update_status_image_context()
    assert harness.ui.labelStatusImageContext.text == "图像：IMG_0071.TIF  |  蒙版：已使用  |  投影：普通透视镜头(TAN)"
    assert str(mask_path) in harness.ui.labelStatusImageContext.tooltip
