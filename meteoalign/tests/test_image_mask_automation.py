from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meteoalign.application import app_image
from meteoalign.application.app_image import ImageMixin
from meteoalign.image_path_resolution import companion_sky_mask_path, is_reserved_mask_path


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

    assert companion_sky_mask_path(image_path) == mask_path
    assert harness._maybe_auto_import_sky_mask_for_image(image_path)
    assert harness.auto_imported_mask_paths == [mask_path]
    assert "发现同名蒙版" in harness.ui.statusbar.messages[-1]


def test_project_28mm_images_follow_mask_name_convention() -> None:
    """项目测试图像应按 _Mask 规则找到 IMG_0067 蒙版，并确认 IMG_0116 没有蒙版。"""

    image_dir = Path(__file__).resolve().parents[2] / "testimages" / "28mm测试"
    first_image = image_dir / "IMG_0067.TIF"
    second_image = image_dir / "IMG_0116.TIF"

    assert companion_sky_mask_path(first_image) == (image_dir / "IMG_0067_Mask.tif").resolve()
    assert companion_sky_mask_path(second_image) is None
    assert is_reserved_mask_path(image_dir / "IMG_0067_Mask.tif")
    assert is_reserved_mask_path(image_dir / "lowercase_mask.PNG")
    assert not is_reserved_mask_path(image_dir / "IMG_Mask_preview.tif")


class _ImageImportSelectionHarness(ImageMixin):
    """记录“导入图像”和“导入蒙版”文件选择结果。"""

    def __init__(self, current_image_path: Path) -> None:
        self._image_import_thread = None
        self._mask_import_thread = None
        self.current_image_preview = SimpleNamespace(path=current_image_path)
        self.remembered_paths: list[object] = []
        self.image_group_paths: tuple[Path, ...] = ()
        self.started_image_imports: list[tuple[Path, bool]] = []
        self.started_mask_imports: list[Path] = []

    def _import_dialog_directory(self, fallback: str | Path) -> Path:
        return Path(fallback)

    def _remember_import_path(self, selected: object) -> None:
        self.remembered_paths.append(selected)

    def _set_image_group_paths(self, file_paths: list[str] | tuple[str, ...]) -> tuple[Path, ...]:
        self.image_group_paths = tuple(Path(path).expanduser().resolve() for path in file_paths)
        return self.image_group_paths

    def start_single_image_import(
        self,
        file_path: str | Path,
        *,
        preserve_image_group_status: bool = False,
    ) -> None:
        self.started_image_imports.append(
            (Path(file_path).expanduser().resolve(), preserve_image_group_status)
        )

    def start_sky_mask_import(self, file_path: str | Path) -> None:
        self.started_mask_imports.append(Path(file_path).expanduser().resolve())


def test_import_images_rejects_a_reserved_mask_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """单独通过“导入图像”选择 _Mask 文件时应提示并停止导入。"""

    mask_path = tmp_path / "frame_Mask.tif"
    harness = _ImageImportSelectionHarness(tmp_path / "frame.tif")
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_image,
        "get_multiple_open_file_names",
        lambda *_args, **_kwargs: ([str(mask_path)], ""),
    )
    monkeypatch.setattr(
        app_image.QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )

    harness.import_images()

    assert harness.image_group_paths == ()
    assert harness.started_image_imports == []
    assert messages and "_Mask" in messages[0][1]
    assert "导入蒙版" in messages[0][1]


def test_import_image_group_excludes_masks_and_keeps_originals(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """图像组同时选中原图和蒙版时，蒙版不得成为图像组成员。"""

    first_image = tmp_path / "first.tif"
    first_mask = tmp_path / "first_Mask.tif"
    second_image = tmp_path / "second.tif"
    harness = _ImageImportSelectionHarness(first_image)
    monkeypatch.setattr(
        app_image,
        "get_multiple_open_file_names",
        lambda *_args, **_kwargs: (
            [str(first_image), str(first_mask), str(second_image)],
            "",
        ),
    )
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_image.QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )

    harness.import_images()

    assert harness.image_group_paths == (first_image.resolve(), second_image.resolve())
    assert harness.started_image_imports == [(first_image.resolve(), True)]
    assert messages == []


def test_import_sky_mask_accepts_any_file_name(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """“导入蒙版”入口不得要求文件名包含 _Mask。"""

    image_path = tmp_path / "frame.tif"
    freely_named_mask = tmp_path / "sky-region.png"
    harness = _ImageImportSelectionHarness(image_path)
    monkeypatch.setattr(
        app_image,
        "get_open_file_name",
        lambda *_args, **_kwargs: (str(freely_named_mask), ""),
    )

    harness.import_sky_mask()

    assert harness.started_mask_imports == [freely_named_mask.resolve()]


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
