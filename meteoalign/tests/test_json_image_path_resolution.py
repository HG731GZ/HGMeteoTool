from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtGui import QImage

from meteoalign.adjacent_alignment import resolve_model_source_image_path
from meteoalign.application import app_star_pair_session
from meteoalign.application.app_star_pair_session import StarPairSessionMixin
from meteoalign.application.app_utils import (
    _resolve_star_pair_session_real_image_path,
    _validate_star_pair_session_current_image,
)
from meteoalign.application.app_workers import StarPairSessionImportWorker
from meteoalign.image_path_resolution import associated_image_candidates, image_size_matches
from meteoalign.image_sequence import read_image_capture_time
from meteoalign.mosaic_model_io import _load_mosaic_source_model, _resolve_source_image_path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def _write_image(path: Path, width: int, height: int) -> Path:
    """写入可由 QImageReader 读取尺寸的测试图像。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(0)
    assert image.save(str(path))
    return path


def test_star_pair_session_image_lookup_prefers_same_dir_name_then_relative_then_absolute(tmp_path: Path) -> None:
    json_path = tmp_path / "session.json"
    same_dir_image = _touch(tmp_path / "same.jpg")
    relative_image = _touch(tmp_path / "images" / "relative.jpg")
    absolute_image = _touch(tmp_path / "elsewhere" / "absolute.jpg")
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_name": same_dir_image.name,
            "relative_path": "images/relative.jpg",
            "path": str(absolute_image),
        },
    }

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == same_dir_image.resolve()

    same_dir_image.unlink()

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == relative_image.resolve()

    relative_image.unlink()

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == absolute_image.resolve()


def test_star_pair_session_accepts_current_image_when_recorded_paths_are_stale(tmp_path: Path) -> None:
    """已有图像与 JSON 的名称、尺寸一致时，不应再用失效路径阻拦导入。"""

    current_image = _write_image(tmp_path / "images" / "scene.tif", 120, 80)
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "path": "/old/computer/archive/scene.jpg",
            "relative_path": "../wrong/place/scene.jpg",
            "file_name": "scene.jpg",
            "file_stem": "scene",
            "original_width_px": 120,
            "original_height_px": 80,
        },
    }

    assert _validate_star_pair_session_current_image(payload, current_image, (120, 80)) == current_image.resolve()

    worker = StarPairSessionImportWorker(
        tmp_path / "elsewhere" / "scene_starpairs.json",
        current_image_path=current_image,
        current_image_size=(120, 80),
    )
    assert worker._real_image_path(payload) == current_image.resolve()


def test_star_pair_session_rejects_current_image_with_different_stem(tmp_path: Path) -> None:
    """已有图像的无后缀文件名不同时必须阻拦导入。"""

    current_image = _write_image(tmp_path / "other.tif", 120, 80)
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_stem": "scene",
            "original_width_px": 120,
            "original_height_px": 80,
        },
    }

    with pytest.raises(ValueError, match="图像名称"):
        _validate_star_pair_session_current_image(payload, current_image, (120, 80))


def test_star_pair_session_rejects_current_image_with_different_size(tmp_path: Path) -> None:
    """已有图像的宽高与 JSON 记录不同时必须阻拦导入。"""

    current_image = _write_image(tmp_path / "scene.tif", 120, 80)
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_stem": "scene",
            "original_width_px": 240,
            "original_height_px": 160,
        },
    }

    with pytest.raises(ValueError, match="图像尺寸"):
        _validate_star_pair_session_current_image(payload, current_image, (120, 80))


def test_star_pair_session_can_ignore_loaded_image_when_importing_new_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """导入新序列首帧时，后台任务应自行解析 JSON 中记录的图像。"""

    created_workers: list[object] = []

    class _FakeWorker:
        def __init__(
            self,
            file_path: Path,
            *,
            current_image_path: Path | None = None,
            current_image_size: tuple[int, int] | None = None,
        ) -> None:
            self.file_path = file_path
            self.current_image_path = current_image_path
            self.current_image_size = current_image_size
            self.finished = object()
            self.failed = object()
            created_workers.append(self)

    def _fake_start_task(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(thread=object(), worker=kwargs["worker"])

    class _StatusBar:
        def showMessage(self, _message: str) -> None:  # noqa: N802 - Qt 风格 API。
            return

    class _Harness(StarPairSessionMixin):
        def __init__(self) -> None:
            self.ui = SimpleNamespace(statusbar=_StatusBar())
            self.current_image_preview = SimpleNamespace(
                path=tmp_path / "上一序列.tif",
                original_width=6000,
                original_height=4000,
            )
            self._json_import_thread = None
            self._json_import_worker = None

        def _set_json_import_controls_enabled(self, _enabled: bool) -> None:
            return

        def _cleanup_json_import(self) -> None:
            return

    monkeypatch.setattr(app_star_pair_session, "StarPairSessionImportWorker", _FakeWorker)
    monkeypatch.setattr(app_star_pair_session, "start_qt_worker_task", _fake_start_task)

    harness = _Harness()
    json_path = tmp_path / "新序列_starpairs.json"
    harness.load_star_pair_session(json_path, show_progress=False, reuse_current_image=False)

    assert len(created_workers) == 1
    worker = created_workers[0]
    assert worker.file_path == json_path
    assert worker.current_image_path is None
    assert worker.current_image_size is None


def test_star_pair_export_path_is_always_next_to_current_image(tmp_path: Path) -> None:
    """导入外部目录的 JSON 后，默认导出位置仍应跟随当前图像。"""

    class _Harness(StarPairSessionMixin):
        pass

    image_path = tmp_path / "images" / "scene.tif"
    imported_json_path = tmp_path / "metadata" / "scene_starpairs.json"
    harness = _Harness()
    harness.current_image_preview = SimpleNamespace(path=image_path)

    export_path = harness._default_star_pair_session_path()
    assert export_path == image_path.parent / "scene_starpairs.json"
    assert export_path != imported_json_path


def test_mosaic_source_image_lookup_prefers_same_dir_name_then_relative_then_absolute(tmp_path: Path) -> None:
    json_path = tmp_path / "model.json"
    same_dir_image = _touch(tmp_path / "same.jpg")
    relative_image = _touch(tmp_path / "images" / "relative.jpg")
    absolute_image = _touch(tmp_path / "elsewhere" / "absolute.jpg")
    payload = {
        "source_image": {
            "file_name": same_dir_image.name,
            "relative_path": "images/relative.jpg",
            "path": str(absolute_image),
        },
    }

    assert _resolve_source_image_path(payload, json_path)[0] == same_dir_image.resolve()

    same_dir_image.unlink()

    assert _resolve_source_image_path(payload, json_path)[0] == relative_image.resolve()

    relative_image.unlink()

    assert _resolve_source_image_path(payload, json_path)[0] == absolute_image.resolve()


def test_relative_and_absolute_lookup_require_exact_recorded_file_name(tmp_path: Path) -> None:
    """相对与绝对路径目录中存在同主名其他后缀时，不得发生跨后缀误匹配。"""

    json_path = tmp_path / "metadata" / "session.json"
    relative_tif = _touch(tmp_path / "metadata" / "relative" / "scene.tif")
    absolute_png = _touch(tmp_path / "absolute" / "scene.png")
    absolute_jpg = tmp_path / "absolute" / "scene.jpg"
    metadata = {
        "file_name": "scene.jpg",
        "file_stem": "scene",
        "relative_path": "relative/scene.jpg",
        "path": str(absolute_jpg),
    }

    candidates = associated_image_candidates(metadata, json_path)

    assert relative_tif.resolve() not in candidates
    assert absolute_png.resolve() not in candidates
    assert candidates == [
        (tmp_path / "metadata" / "relative" / "scene.jpg").resolve(),
        absolute_jpg.resolve(),
    ]


def test_missing_exact_relative_path_falls_back_to_exact_absolute_path(tmp_path: Path) -> None:
    """相对路径缺少精确文件时，应跳过同主名转换图并使用绝对路径精确文件。"""

    json_path = tmp_path / "metadata" / "session.json"
    _touch(tmp_path / "metadata" / "relative" / "scene.tif")
    absolute_image = _touch(tmp_path / "absolute" / "scene.jpg")
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_name": "scene.jpg",
            "file_stem": "scene",
            "relative_path": "relative/scene.jpg",
            "path": str(absolute_image),
        },
    }

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == absolute_image.resolve()


def test_mosaic_missing_candidates_do_not_return_stale_absolute_path(tmp_path: Path) -> None:
    """所有候选均不存在时，不得把旧绝对路径伪装成已连接的源图。"""

    json_path = tmp_path / "metadata" / "model.json"
    stale_absolute_path = tmp_path / "old" / "scene.jpg"
    payload = {
        "source_image": {
            "file_name": "scene.jpg",
            "relative_path": "images/scene.jpg",
            "path": str(stale_absolute_path),
        },
    }

    image_path, image_text = _resolve_source_image_path(payload, json_path)

    assert image_path is None
    assert image_text == "scene.jpg"


def test_tiff_size_check_falls_back_when_qt_cannot_read_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第三方重写的 TIFF 即使 Qt 头部读取失败，也应使用 tifffile 校验尺寸。"""

    import meteoalign.image_path_resolution as path_resolution

    tiff_path = _write_image(tmp_path / "scene.tif", 120, 80)

    class _UnreadableQtImageReader:
        def __init__(self, _path: str) -> None:
            pass

        @staticmethod
        def size():
            return SimpleNamespace(isValid=lambda: False)

    monkeypatch.setattr(path_resolution, "QImageReader", _UnreadableQtImageReader)

    assert image_size_matches(tiff_path, (120, 80))
    assert not image_size_matches(tiff_path, (60, 40))


def test_relative_path_accepts_windows_separator_on_other_platforms(tmp_path: Path) -> None:
    """相对路径使用 Windows 分隔符时，跨平台读取仍应定位到精确文件。"""

    json_path = tmp_path / "metadata" / "session.json"
    relative_image = _touch(tmp_path / "metadata" / "images" / "scene.jpg")
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_name": "scene.jpg",
            "relative_path": r"images\scene.jpg",
        },
    }

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == relative_image.resolve()


def test_json_image_lookup_uses_tif_png_jpg_priority_and_skips_wrong_size(tmp_path: Path) -> None:
    """同主文件名多格式共存时应按优先级选择，并排除尺寸不符的 TIFF。"""

    json_path = tmp_path / "session.json"
    jpg_path = _write_image(tmp_path / "scene.jpg", 120, 80)
    png_path = _write_image(tmp_path / "scene.png", 120, 80)
    tif_path = _write_image(tmp_path / "scene.tif", 120, 80)
    payload = {
        "format": "meteoalign_star_pair_session",
        "real_image": {
            "file_name": jpg_path.name,
            "file_stem": "scene",
            "original_width_px": 120,
            "original_height_px": 80,
        },
    }

    assert _resolve_star_pair_session_real_image_path(payload, json_path) == tif_path.resolve()

    tif_path.unlink()
    _write_image(tif_path, 60, 40)
    assert _resolve_star_pair_session_real_image_path(payload, json_path) == png_path.resolve()


def test_source_model_and_adjacent_lookup_accept_converted_extension(tmp_path: Path) -> None:
    """源模型仅记录旧 JPG 名称时，也应按主文件名找到尺寸一致的 TIFF。"""

    json_path = tmp_path / "model.json"
    converted_path = _write_image(tmp_path / "frame.tif", 160, 90)
    payload = {
        "source_image": {
            "file_name": "frame.jpg",
            "file_stem": "frame",
            "original_width_px": 160,
            "original_height_px": 90,
        },
        "image_geometry": {"width_px": 160, "height_px": 90},
    }

    assert _resolve_source_image_path(payload, json_path)[0] == converted_path.resolve()
    assert resolve_model_source_image_path(payload, json_path) == converted_path.resolve()


def test_standard_json_lookup_does_not_add_raw_candidates(tmp_path: Path) -> None:
    """流星框选以外的 JSON 解析不得把 RAW 加入普通图片候选。"""

    raw_path = _touch(tmp_path / "frame.dng")
    metadata = {"file_name": "frame.dng", "file_stem": "frame"}

    assert raw_path.resolve() not in associated_image_candidates(metadata, tmp_path / "model.json")


def test_meteor_json_candidate_order_places_raw_between_tif_and_png(tmp_path: Path) -> None:
    """流星关联图像候选应使用 TIFF、RAW、PNG、JPG 的顺序。"""

    for suffix in (".jpg", ".png", ".dng", ".tif"):
        _touch(tmp_path / f"meteor{suffix}")
    candidates = associated_image_candidates(
        {"file_name": "meteor.jpg", "file_stem": "meteor"},
        tmp_path / "meteor_Meteor.json",
        include_raw=True,
    )

    assert [path.suffix for path in candidates[:4]] == [".tif", ".dng", ".png", ".jpg"]


def test_starless_model_uses_sibling_processed_tiff_and_json_observer() -> None:
    """无拍摄时间 EXIF 的 Starless TIFF 应连接成功，并使用 model.json 中的观测信息。"""

    model_path = (
        Path(__file__).resolve().parents[2]
        / "testimages"
        / "28mm测试"
        / "Starless"
        / "IMG_0116_model.json"
    )
    if not model_path.exists():
        pytest.skip("仓库中没有 Starless 真实测试文件。")

    with pytest.raises(ValueError, match="没有可用的原始拍摄时间"):
        read_image_capture_time(model_path.with_name("IMG_0116.TIF"))

    source_model = _load_mosaic_source_model(model_path)

    assert source_model.source_image_path == model_path.with_name("IMG_0116.TIF").resolve()
    assert source_model.image_width_px == 5472
    assert source_model.image_height_px == 3648
    assert source_model.observer.observation_time_utc.isoformat() == "2025-12-14T19:45:07.290000+00:00"
    assert source_model.observer.latitude_deg == 25.0
    assert source_model.observer.longitude_deg == 102.0
    assert source_model.observer.elevation_m == 200.0
